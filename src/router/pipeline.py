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
from .config import ENABLE_CODE_TOOL, ENABLE_RISK_RULES, LARGE, ROUTE_THRESHOLD, SMALL, TIERS
from .escalation import should_escalate
from .features import extract
from .llm import LLMClient
from .risk import silent_failure_risks
from .tools import looks_computable, solve_with_code
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
    # Risk rules that sent a would-be small query to the large model.
    risk_override: list[str] = field(default_factory=list)
    # Set when the answer was computed by code: {"expression", "answer"}.
    tool: dict | None = None


class RouterPipeline:
    def __init__(
        self,
        mode: RouterMode = "classifier",
        client: LLMClient | None = None,
        threshold: float = ROUTE_THRESHOLD,
        enable_escalation: bool = True,
        enable_risk_rules: bool = ENABLE_RISK_RULES,
        enable_code_tool: bool = ENABLE_CODE_TOOL,
    ):
        self.mode = mode
        self.threshold = threshold
        self.enable_escalation = enable_escalation
        self.enable_risk_rules = enable_risk_rules
        self.enable_code_tool = enable_code_tool
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

    def route(self, query: str, threshold: float | None = None) -> tuple[str, float, list[str]]:
        """(tier, router confidence, risk rules that overrode a small decision).

        Escalation cannot see a fluent wrong answer, so question shapes the
        small model fails silently (router/risk.py) skip it. Fixed-tier modes
        are left alone so the benchmark baselines stay pure.
        """
        tier, confidence = self._decide(query, threshold)
        if self.enable_risk_rules and self.mode in ("heuristic", "classifier") and tier == "small":
            risks = silent_failure_risks(query)
            if risks:
                return "large", confidence, risks
        return tier, confidence, []

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

        # Exact-answer questions: let the small model write an expression and
        # compute it (router/tools.py). Single-turn only, so earlier context
        # can't be lost. Anything but a computed answer falls through to
        # normal routing; the extra call still counts toward cost.
        tool_calls: list[CallResult] = []
        if (self.enable_code_tool and self.mode in ("heuristic", "classifier")
                and not history and looks_computable(query)):
            code = solve_with_code(self.client, SMALL, query)
            if code.call is not None:
                tool_calls.append(code.call)
            if code.status == "answered":
                _, confidence = self._decide(query, threshold)
                return RoutedResponse(
                    answer=code.answer,
                    tier_chosen="small",
                    tier_served="small",
                    escalated=False,
                    escalation_reason=None,
                    router_confidence=confidence,
                    total_cost_usd=sum(c.cost_usd for c in tool_calls),
                    total_latency_s=time.perf_counter() - started,
                    calls=tool_calls,
                    features=extract(query).to_dict(),
                    tool={"expression": code.expression, "answer": code.answer},
                )

        tier, confidence, risks = self.route(query, threshold)
        spec = TIERS[tier]
        gen = {"system": system, "history": history}
        if max_tokens is not None:
            gen["max_tokens"] = max_tokens

        first = self.client.complete(spec, query, **gen)
        calls = [*tool_calls, first]
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
            risk_override=risks,
        )
