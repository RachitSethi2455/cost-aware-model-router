"""Outcome logging from your own traffic ("shadow" calls).

The shipped router learned from other models' outcomes (SPROUT). To learn
which questions *your* large model actually helps on, the service can, for a
sampled share of answered requests, also ask the tier it did not serve, after
the user already has their answer, and log both answers. Offline,
evals/label_shadow.py grades each pair (an LLM judge, outside the request
path) and `evals/train_outcome_router.py --own-traffic` trains on them.

Off by default (SHADOW_RATE=0): every sampled request pays for a second call,
capped at SHADOW_MAX_PER_DAY. The log holds question and answer text, so it
stays on local disk and should only be enabled where users have agreed to
that. An escalated request already has both answers and is logged at no extra
cost. Multi-turn requests and calculator answers are skipped.
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from .config import TIERS
from .features import extract_v2


class ShadowLogger:
    def __init__(self, client_factory, rate: float, log_path: str | Path | None, max_per_day: int,
                 metrics=None, rng: random.Random | None = None, executor=None):
        self.client_factory = client_factory  # returns the LLMClient to call with
        self.rate, self.max_per_day, self.metrics = rate, max_per_day, metrics
        self.path = Path(log_path) if log_path else None
        self.rng = rng or random.Random()
        self.executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow")
        self._lock = threading.Lock()
        self._day, self._used = "", 0

    def _take_slot(self) -> bool:
        """One of today's SHADOW_MAX_PER_DAY paid shadow calls, if any are left."""
        with self._lock:
            today = date.today().isoformat()
            if today != self._day:
                self._day, self._used = today, 0
            if self._used >= self.max_per_day:
                return False
            self._used += 1
            return True

    def consider(self, query: str, system: str | None, result, history=None) -> str | None:
        """Called after a request was answered. Returns "escalation" when both
        answers were already there, "sampled" when a shadow call was queued,
        or None."""
        if self.path is None or history or result.tool:
            return None
        answers = {c.tier: c for c in result.calls if not c.error and c.text.strip()}
        if {"small", "large"} <= answers.keys():
            self._write(query, answers["small"], answers["large"], result, "escalation")
            return "escalation"
        served = answers.get(result.tier_served)
        if served is None or self.rate <= 0 or self.rng.random() >= self.rate or not self._take_slot():
            return None
        other = "large" if result.tier_served == "small" else "small"
        self.executor.submit(self._shadow_call, query, system, other, served, result)
        return "sampled"

    def _shadow_call(self, query, system, other, served, result) -> None:
        try:
            call = self.client_factory().complete(TIERS[other], query, system=system)
        except Exception as exc:  # noqa: BLE001 — shadow logging must never hurt serving
            print(f"[shadow] call failed: {exc}", file=sys.stderr)
            return
        if self.metrics is not None:
            self.metrics.inc("router_shadow_calls_total", tier=other)
            self.metrics.inc("router_shadow_cost_usd_total", call.cost_usd, tier=other)
        if not call.error and call.text.strip():
            small, large = (served, call) if other == "large" else (call, served)
            self._write(query, small, large, result, "sampled")

    def _write(self, query, small, large, result, source: str) -> None:
        entry = {"ts": round(time.time(), 3), "source": source, "query": query,
                 "router_score": result.router_confidence, "tier_chosen": result.tier_chosen,
                 "features_v2": extract_v2(query).to_dict()}
        for tier, call in (("small", small), ("large", large)):
            entry[tier] = {"model": call.model_id, "answer": call.text, "cost_usd": call.cost_usd,
                           "input_tokens": call.input_tokens, "output_tokens": call.output_tokens}
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[shadow] log disabled: {exc}", file=sys.stderr)
            self.path = None
