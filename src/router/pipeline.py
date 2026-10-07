"""End-to-end routing pipeline.

    query -> features -> router -> model call -> escalation check -> response

Supports four modes so the benchmark can run all arms through identical code
(important: if the baselines went through a different code path, the
comparison would be confounded).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from . import heuristic
from .classifier import ClassifierRouter
from .config import LARGE, ROUTE_THRESHOLD, TIERS
from .escalation import should_escalate
from .features import extract
from .llm import LLMClient
from .types import CallResult

RouterMode = Literal["heuristic", "classifier", "always_small", "always_large"]


@dataclass
class RoutedResponse:
    answer: str
    tier_chosen: str          # what the router picked
    tier_served: str          # what actually produced the final answer
    escalated: bool
    escalation_reason: str | None
    router_confidence: float
    total_cost_usd: float
    total_latency_s: float
    calls: list[CallResult] = field(default_factory=list)
    features: dict = field(default_factory=dict)


class RouterPipeline:
    def __init__(
        self,
        mode: RouterMode = "classifier",
        client: LLMClient | None = None,
        threshold: float = ROUTE_THRESHOLD,
        enable_escalation: bool = True,
    ):
        self.mode = mode
        self.threshold = threshold
        self.enable_escalation = enable_escalation
        self.client = client or LLMClient()
        self._clf = ClassifierRouter() if mode == "classifier" else None

    def _decide(self, query: str, threshold: float | None = None) -> tuple[str, float]:
        t = self.threshold if threshold is None else threshold
        if self.mode == "always_small":
            return "small", 0.0
        if self.mode == "always_large":
            return "large", 1.0
        if self.mode == "heuristic":
            return heuristic.route(query, t)
        return self._clf.route(query, t)

    def run(
        self,
        query: str,
        system: str | None = None,
        threshold: float | None = None,
        history: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> RoutedResponse:
        """threshold overrides the pipeline default for this call only.

        Pipelines are shared across API requests, so a per-request setting
        must never be stored on self.

        history: earlier conversation turns. Routing looks at `query` (the
        latest user message) only; both model tiers see the full history.
        """
        started = time.perf_counter()
        tier, confidence = self._decide(query, threshold)
        spec = TIERS[tier]
        gen = {"system": system, "history": history}
        if max_tokens is not None:
            gen["max_tokens"] = max_tokens

        first = self.client.complete(spec, query, **gen)
        calls = [first]
        escalated = False
        reason = None
        final = first

        # Only a small-model answer is ever escalated; there is no tier above large.
        if self.enable_escalation and tier == "small":
            decision = should_escalate(first, query)
            if decision.escalate:
                escalated = True
                reason = decision.reason
                retry = self.client.complete(LARGE, query, **gen)
                calls.append(retry)
                final = retry

        return RoutedResponse(
            answer=final.text,
            tier_chosen=tier,
            tier_served=final.tier,
            escalated=escalated,
            escalation_reason=reason,
            router_confidence=confidence,
            total_cost_usd=sum(c.cost_usd for c in calls),
            total_latency_s=time.perf_counter() - started,
            calls=calls,
            features=extract(query).to_dict(),
        )
