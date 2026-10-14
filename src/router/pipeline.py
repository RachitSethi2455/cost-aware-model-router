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
from .outcome import OutcomeRouter
from .risk import silent_failure_risks
from .tools import looks_computable, solve_with_code
from .types import CallResult

RouterMode = Literal["outcome", "classifier", "heuristic", "always_small", "always_large"]
# Modes where a router decides per query; the others are fixed-tier baselines.
# "outcome" (trained on recorded outcomes) is the default; "classifier" is the
# first router, trained on 60 hand labels, kept for comparison.
ROUTED_MODES = ("outcome", "classifier", "heuristic")


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
    # A spending guardrail kept this request off the large model (router/budget.py).
    budget_limited: bool = False


class RouterPipeline:
    def __init__(
        self,
        mode: RouterMode = "outcome",
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
        routers = {"classifier": ClassifierRouter, "outcome": OutcomeRouter}
        self._router = routers[mode]() if mode in routers else None

    def _decide(self, query: str, threshold: float | None = None) -> tuple[str, float]:
        t = self.threshold if threshold is None else threshold
        if self.mode == "always_small":
            return "small", 0.0
        if self.mode == "always_large":
            return "large", 1.0
        if self.mode == "heuristic":
            return heuristic.route(query, t)
        return self._router.route(query, t)

    def route(self, query: str, threshold: float | None = None) -> tuple[str, float, list[str]]:
        """(tier, router confidence, risk rules that overrode a small decision).

        Escalation cannot see a fluent wrong answer, so question shapes the
        small model fails silently (router/risk.py) skip it. Fixed-tier modes
        are left alone so the benchmark baselines stay pure.
        """
        tier, confidence = self._decide(query, threshold)
        if self.enable_risk_rules and self.mode in ROUTED_MODES and tier == "small":
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
        allow_large: bool = True,
    ) -> RoutedResponse:
        """threshold overrides the pipeline default for this call only.

        allow_large=False (a spending guardrail is active for this client or
        the day) serves everything with the small model and skips escalation.

        Pipelines are shared across API requests, so a per-request setting
        must never be stored on self.

        history: earlier conversation turns. Routing looks at `query` (the
        latest user message) only; both model tiers see the full history.
        """
        for kind, payload in self.stream(query, system, threshold, history, max_tokens, allow_large,
                                         live=False):
            if kind == "result":
                return payload
        raise RuntimeError("the pipeline ended without a result")  # pragma: no cover

    def stream(
        self,
        query: str,
        system: str | None = None,
        threshold: float | None = None,
        history: list[dict] | None = None,
        max_tokens: int | None = None,
        allow_large: bool = True,
        live: bool = True,
    ):
        """The routing pipeline as events: ("start", {"tier", "model_id"}) when
        the answering model is known, ("delta", text) pieces of the answer,
        and finally ("result", RoutedResponse). run() is this with live=False,
        so streamed and unstreamed requests make the same decisions.

        A routed-small answer that may be escalated is checked whole before
        any of it is sent, since a streamed answer cannot be taken back; the
        large model, or a small one with no escalation possible, streams
        token by token.
        """
        started = time.perf_counter()

        # Exact-answer questions: let the small model write an expression and
        # compute it (router/tools.py). Single-turn only, so earlier context
        # can't be lost. Anything but a computed answer falls through to
        # normal routing; the extra call still counts toward cost.
        tool_calls: list[CallResult] = []
        if (self.enable_code_tool and self.mode in ROUTED_MODES
                and not history and looks_computable(query)):
            code = solve_with_code(self.client, SMALL, query)
            if code.call is not None:
                tool_calls.append(code.call)
            if code.status == "answered":
                _, confidence = self._decide(query, threshold)
                yield "start", {"tier": "small", "model_id": "calculator"}
                if live:
                    yield "delta", code.answer
                yield "result", RoutedResponse(
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
                return

        tier, confidence, risks = self.route(query, threshold)
        budget_limited = not allow_large and tier == "large"
        if budget_limited:
            tier, risks = "small", []
        gen = {"system": system, "history": history}
        if max_tokens is not None:
            gen["max_tokens"] = max_tokens

        calls = list(tool_calls)
        escalated = False
        reason = None
        # Only a small-model answer is ever escalated; there is no tier above large.
        if self.enable_escalation and allow_large and tier == "small":
            first = self.client.complete(SMALL, query, **gen)
            calls.append(first)
            decision = should_escalate(first, query)
            if decision.escalate:
                escalated, reason = True, decision.reason
                yield "start", {"tier": "large", "model_id": LARGE.model_id}
                final = yield from self._call(LARGE, query, gen, live)
                calls.append(final)
            else:
                final = first
                yield "start", {"tier": "small", "model_id": first.model_id}
                if live and first.text:
                    yield "delta", first.text
        else:
            spec = TIERS[tier]
            yield "start", {"tier": tier, "model_id": spec.model_id}
            final = yield from self._call(spec, query, gen, live)
            calls.append(final)

        yield "result", RoutedResponse(
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
            budget_limited=budget_limited,
        )

    def _call(self, spec, query: str, gen: dict, live: bool):
        """One model call: streamed as ("delta", text) events when live.
        Returns the CallResult (use with `yield from`)."""
        if not live:
            return self.client.complete(spec, query, **gen)
        pieces = self.client.stream(spec, query, **gen)
        while True:
            try:
                piece = next(pieces)
            except StopIteration as done:
                return done.value
            yield "delta", piece
