"""Monitoring: Prometheus-style metrics and a per-request decision log.

Metrics (GET /metrics) are in-process counters and histograms in the
Prometheus text format, written by hand so the service needs no metrics
library. They reset when the process restarts, as Prometheus expects.

The decision log is one JSON line per answered request: which tier the
router chose and served, its score and threshold, escalation, cost,
latency, and the features it decided on. That is what you need to answer
"why did this request cost so much?" or "how often does escalation fire?"
after the fact. It records no question text unless LOG_QUERY_TEXT=1, and
on a read-only filesystem (Vercel) it switches itself off.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

LATENCY_BUCKETS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
HELP = {
    "router_requests_total": "Answered requests by router mode and tier chosen / served.",
    "router_escalations_total": "Small-model answers retried on the large model, by reason.",
    "router_risk_overrides_total": "Requests sent to the large model by a risk rule.",
    "router_code_tool_answers_total": "Requests answered by the sandboxed calculator.",
    "router_errors_total": "Failed requests by HTTP status.",
    "router_cost_usd_total": "Model spend in USD by tier of the call.",
    "router_tokens_total": "Tokens billed by tier and direction.",
    "router_explain_requests_total": "Routing decisions served by /explain (no model call).",
    "router_shadow_calls_total": "Background calls to the other tier for outcome logging.",
    "router_shadow_cost_usd_total": "Spend on those background calls, in USD.",
    "router_request_latency_seconds": "Wall-clock time to answer a request.",
    "router_score": "Router score of answered requests (the tier is large at or above the threshold).",
}


def _labels(labels: dict) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}"


class Metrics:
    """Thread-safe counters and histograms; render() gives the text format."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: dict[str, tuple[tuple[float, ...], dict]] = {}

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        with self._lock:
            self._counters[name][tuple(sorted(labels.items()))] += value

    def observe(self, name: str, value: float, buckets: tuple[float, ...]) -> None:
        with self._lock:
            _, state = self._histograms.setdefault(
                name, (buckets, {"counts": [0] * len(buckets), "sum": 0.0, "count": 0}))
            for i, bound in enumerate(buckets):
                if value <= bound:
                    state["counts"][i] += 1
            state["sum"] += value
            state["count"] += 1

    def value(self, name: str, **labels) -> float:
        with self._lock:
            return self._counters[name].get(tuple(sorted(labels.items())), 0.0)

    def render(self) -> str:
        lines = []
        with self._lock:
            for name in sorted(self._counters):
                lines += [f"# HELP {name} {HELP.get(name, name)}", f"# TYPE {name} counter"]
                for key, value in sorted(self._counters[name].items()):
                    lines.append(f"{name}{_labels(dict(key))} {value:g}")
            for name in sorted(self._histograms):
                buckets, state = self._histograms[name]
                lines += [f"# HELP {name} {HELP.get(name, name)}", f"# TYPE {name} histogram"]
                for bound, count in zip(buckets, state["counts"], strict=True):
                    lines.append(f'{name}_bucket{{le="{bound:g}"}} {count}')
                lines += [f'{name}_bucket{{le="+Inf"}} {state["count"]}',
                          f"{name}_sum {state['sum']:g}", f"{name}_count {state['count']}"]
        return "\n".join(lines) + "\n"


class DecisionLog:
    """Appends one JSON line per decision; disables itself if it cannot write."""

    def __init__(self, path: str | Path | None, log_query_text: bool = False):
        self.path = Path(path) if path else None
        self.log_query_text = log_query_text
        self._lock = threading.Lock()

    def write(self, entry: dict, query: str | None = None) -> None:
        if self.path is None:
            return
        entry = {"ts": round(time.time(), 3), **entry}
        if query is not None:
            entry["query_chars"] = len(query)
            if self.log_query_text:
                entry["query"] = query
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:  # read-only filesystem: keep serving, stop logging
            print(f"[telemetry] decision log disabled: {exc}", file=sys.stderr)
            self.path = None


def record_routed(metrics: Metrics, log: DecisionLog, *, request_id: str, mode: str,
                  threshold: float, result, query: str) -> None:
    """Metrics and a decision-log line for one answered request (a RoutedResponse)."""
    metrics.inc("router_requests_total", mode=mode, chosen=result.tier_chosen,
                served=result.tier_served)
    if result.escalated:
        metrics.inc("router_escalations_total", reason=result.escalation_reason or "unknown")
    if result.risk_override:
        metrics.inc("router_risk_overrides_total")
    if result.tool:
        metrics.inc("router_code_tool_answers_total")
    for call in result.calls:
        metrics.inc("router_cost_usd_total", call.cost_usd, tier=call.tier)
        metrics.inc("router_tokens_total", call.input_tokens, tier=call.tier, direction="input")
        metrics.inc("router_tokens_total", call.output_tokens, tier=call.tier, direction="output")
    metrics.observe("router_request_latency_seconds", result.total_latency_s, LATENCY_BUCKETS)
    metrics.observe("router_score", result.router_confidence, SCORE_BUCKETS)
    log.write({
        "request_id": request_id, "status": 200, "mode": mode,
        "router_score": round(result.router_confidence, 4), "threshold": threshold,
        "tier_chosen": result.tier_chosen, "tier_served": result.tier_served,
        "escalated": result.escalated, "escalation_reason": result.escalation_reason,
        "risk_override": result.risk_override, "code_tool": bool(result.tool),
        "cost_usd": round(result.total_cost_usd, 8), "latency_s": round(result.total_latency_s, 3),
        "calls": [{"tier": c.tier, "model": c.model_id, "input_tokens": c.input_tokens,
                   "output_tokens": c.output_tokens, "cached": c.cached} for c in result.calls],
        "features": result.features,
    }, query)


def record_failure(metrics: Metrics, log: DecisionLog, *, request_id: str, mode: str,
                   status: int, query: str) -> None:
    metrics.inc("router_errors_total", status=status)
    log.write({"request_id": request_id, "status": status, "mode": mode}, query)
