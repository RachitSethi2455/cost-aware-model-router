"""FastAPI service exposing the router.

Endpoints:
    POST /route    - route a query, return answer + cost/tier metadata
    POST /explain  - routing decision only, no model call (free, fast)
    GET  /health   - liveness + whether a trained router is loaded
    GET  /stats    - cumulative cost/tier counters for this process
    GET  /metrics  - Prometheus text format: requests, escalations, cost, tokens,
                     latency and router-score histograms (router/telemetry.py)
    GET  /         - interactive demo page (uses /explain only)
    POST /v1/chat/completions - OpenAI-compatible; model="auto" routes,
                   "small"/"large" force a tier
    GET  /v1/models          - the model names above

Error contract for /route:
    503 - the router is not configured (missing API key, SDK not installed).
          Nothing was sent to any model.
    502 - the model provider failed (bad model id, rate limit, network error)
          and there was no successful answer to fall back on.
    403 - PUBLIC_DEMO=1: model calls are disabled on a public deployment.

/explain returns 429 past EXPLAIN_RATE_LIMIT requests per minute per client.
"""

from __future__ import annotations

import itertools
import json
import os
import time
import uuid
from collections import Counter, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from router import heuristic
from router.budget import SpendGuard
from router.classifier import ClassifierRouter
from router.config import (
    DAILY_SPEND_LIMIT_USD,
    DECISION_LOG,
    ENABLE_RISK_RULES,
    LARGE,
    LARGE_CALLS_PER_CLIENT_PER_HOUR,
    LOG_QUERY_TEXT,
    ROUTE_THRESHOLD,
    SHADOW_LOG,
    SHADOW_MAX_PER_DAY,
    SHADOW_RATE,
    SMALL,
)
from router.features import FEATURE_SETS, extract
from router.llm import LLMClient
from router.outcome import OutcomeRouter
from router.pipeline import ROUTED_MODES, RoutedResponse, RouterMode, RouterPipeline
from router.risk import silent_failure_risks
from router.shadow import ShadowLogger
from router.telemetry import DecisionLog, Metrics, record_failure, record_routed
from router.tools import looks_computable

# Read after the router imports so values from .env are already loaded.
# A public deployment must not let strangers spend the owner's API key.
PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "0") == "1"
EXPLAIN_RATE_LIMIT = int(os.getenv("EXPLAIN_RATE_LIMIT", "60"))

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Adaptive LLM Router",
    description="Per-request model selection with escalation and cost accounting.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_pipelines: dict[str, RouterPipeline] = {}
_counters = Counter()
_cost_total = {"usd": 0.0}
_classifier: ClassifierRouter | None = None
_outcome: OutcomeRouter | None = None
_shadow_client: LLMClient | None = None

METRICS = Metrics()
DECISIONS = DecisionLog(DECISION_LOG or None, log_query_text=LOG_QUERY_TEXT)


def shadow_client() -> LLMClient:
    global _shadow_client
    if _shadow_client is None:
        _shadow_client = LLMClient()
    return _shadow_client


GUARD = SpendGuard(LARGE_CALLS_PER_CLIENT_PER_HOUR, DAILY_SPEND_LIMIT_USD)
SHADOW = ShadowLogger(shadow_client, rate=SHADOW_RATE, log_path=SHADOW_LOG if SHADOW_RATE > 0 else None,
                      max_per_day=SHADOW_MAX_PER_DAY, metrics=METRICS)


def get_pipeline(mode: RouterMode) -> RouterPipeline:
    if mode not in _pipelines:
        _pipelines[mode] = RouterPipeline(mode=mode)
    return _pipelines[mode]


def get_classifier() -> ClassifierRouter:
    """Loaded once; /explain must not touch the LLM client or its cache."""
    global _classifier
    if _classifier is None:
        _classifier = ClassifierRouter()
    return _classifier


def get_outcome_router() -> OutcomeRouter:
    global _outcome
    if _outcome is None:
        _outcome = OutcomeRouter()
    return _outcome


def learned_router(mode: str) -> ClassifierRouter | OutcomeRouter | None:
    if mode == "outcome":
        return get_outcome_router()
    return get_classifier() if mode == "classifier" else None


class RateLimiter:
    """Sliding one-minute window per client, in memory.

    Enough to stop one visitor hammering a single-process demo; not a
    substitute for a gateway limit on a multi-instance deployment.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self.hits.setdefault(key, deque())
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        if len(self.hits) > 10_000:  # drop idle clients so memory stays bounded
            self.hits = {k: v for k, v in self.hits.items() if v and now - v[-1] <= 60}
        return True


_explain_limiter = RateLimiter(EXPLAIN_RATE_LIMIT)


def client_key(request: Request) -> str:
    # Hosted platforms put the app behind a proxy, so the socket peer is the
    # proxy. Trust X-Forwarded-For only in that deployment mode.
    forwarded = request.headers.get("x-forwarded-for")
    if PUBLIC_DEMO and forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20000)
    mode: RouterMode = "outcome"
    system: str | None = None
    threshold: float | None = Field(None, ge=0.0, le=1.0)


class RouteResponse(BaseModel):
    answer: str
    tier_chosen: str
    tier_served: str
    escalated: bool
    escalation_reason: str | None
    router_confidence: float
    cost_usd: float
    latency_s: float
    n_calls: int
    # The served answer hit the token ceiling. Only a large-model answer can
    # be returned truncated; a truncated small answer is escalated instead.
    truncated: bool
    # Risk rules that sent a would-be small query to the large model.
    risk_override: list[str] = []
    # Set when the answer was computed by code: {"expression", "answer"}.
    tool: dict | None = None
    # A spending guardrail kept this request on the small model (router/budget.py).
    budget_limited: bool = False


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "router": "outcome",
        "router_trained": get_outcome_router().trained,
        "default_threshold": ROUTE_THRESHOLD,
        "public_demo": PUBLIC_DEMO,
        "budget": GUARD.status(),
    }


@app.post("/explain")
def explain(req: RouteRequest, request: Request) -> dict:
    """Routing decision without spending a token. Useful for demos and debugging."""
    if not _explain_limiter.allow(client_key(request)):
        raise HTTPException(status_code=429, detail="Too many requests; try again in a minute.")

    threshold = req.threshold if req.threshold is not None else ROUTE_THRESHOLD
    router = learned_router(req.mode)
    if req.mode == "always_small":
        tier, confidence = "small", 0.0
    elif req.mode == "always_large":
        tier, confidence = "large", 1.0
    elif req.mode == "heuristic":
        tier, confidence = heuristic.route(req.query, threshold)
    else:
        tier, confidence = router.route(req.query, threshold)

    # Same override as RouterPipeline.route: shapes the small model fails
    # silently skip it, whatever the router's probability says.
    METRICS.inc("router_explain_requests_total", tier=tier)
    risks = silent_failure_risks(req.query)
    overridden_by = []
    if ENABLE_RISK_RULES and req.mode in ROUTED_MODES and tier == "small" and risks:
        tier, overridden_by = "large", risks

    # Which router actually decided: a learned router falls back to the
    # heuristic when its trained artifact is missing.
    decided_by = req.mode
    if router is not None and not router.trained:
        decided_by = "heuristic_fallback"

    return {
        "tier": tier,
        "confidence": round(confidence, 4),
        "threshold": threshold,
        # The features the deciding router read: v2 for the outcome router.
        "features": (FEATURE_SETS[router.feature_set][1] if req.mode == "outcome" and router.trained
                     else extract)(req.query).to_dict(),
        "mode": req.mode,
        "router": decided_by,
        "contributions": router.contributions(req.query) if router is not None else None,
        "risk_rules": risks,
        "overridden_by": overridden_by,
        # Whether /route would try the code tool (ENABLE_CODE_TOOL) first.
        "computable": looks_computable(req.query),
    }


class RoutingFailed(Exception):
    """A /route or /v1/chat/completions call that cannot return an answer.

    Each endpoint renders it in its own error format.
    """

    def __init__(self, status: int, message: str, detail: dict | str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.detail = detail if detail is not None else message


DEMO_BLOCKED = ("Model calls are disabled on the public demo. Use /explain, "
                "or run the project locally with your own API key.")


def run_routed(mode: RouterMode, query: str, client: str = "local", **run_kwargs) -> RoutedResponse:
    """Shared by both answering endpoints: demo guard, spending guardrails,
    errors, counters, metrics, the decision log, and (if enabled) outcome
    logging."""
    request_id, limit = uuid.uuid4().hex, _apply_guard(client, run_kwargs)
    try:
        if PUBLIC_DEMO:
            raise RoutingFailed(403, DEMO_BLOCKED)
        try:
            result = get_pipeline(mode).run(query, **run_kwargs)
        except RuntimeError as exc:
            # Raised while building a provider client, before any model call.
            # This is a configuration problem, not a server crash.
            raise RoutingFailed(503, str(exc)) from exc
        if failure := _failed_call(result):
            raise failure
    except RoutingFailed as exc:
        record_failure(METRICS, DECISIONS, request_id=request_id, mode=mode, status=exc.status,
                       query=query)
        raise
    _book(mode, query, client, request_id, limit, result, run_kwargs)
    return result


def stream_routed(mode: RouterMode, query: str, client: str = "local", **run_kwargs):
    """Streaming counterpart of run_routed: returns the pipeline's events.

    It runs until the first piece of the answer (or the result) first, so a
    failure known by then (no key, provider error, public demo) still comes
    back as an ordinary error response rather than a broken stream. A failure
    after that arrives as an ("error", RoutingFailed) event."""
    request_id, limit = uuid.uuid4().hex, _apply_guard(client, run_kwargs)
    try:
        if PUBLIC_DEMO:
            raise RoutingFailed(403, DEMO_BLOCKED)
        events, head = get_pipeline(mode).stream(query, **run_kwargs), []
        try:
            for event in events:
                head.append(event)
                if event[0] in ("delta", "result"):
                    break
        except RuntimeError as exc:
            raise RoutingFailed(503, str(exc)) from exc
        if head and head[-1][0] == "result" and (failure := _failed_call(head[-1][1])):
            raise failure
    except RoutingFailed as exc:
        record_failure(METRICS, DECISIONS, request_id=request_id, mode=mode, status=exc.status,
                       query=query)
        raise

    def rest():
        for kind, payload in itertools.chain(head, events):
            if kind == "result":
                if failure := _failed_call(payload):
                    record_failure(METRICS, DECISIONS, request_id=request_id, mode=mode, status=502,
                                   query=query)
                    yield "error", failure
                    return
                _book(mode, query, client, request_id, limit, payload, run_kwargs)
            yield kind, payload
    return rest()


def _apply_guard(client: str, run_kwargs: dict) -> str | None:
    """Keep this request off the large model if a spending limit applies."""
    limit = GUARD.check(client)
    if limit:
        run_kwargs["allow_large"] = False
        METRICS.inc("router_budget_limited_total", reason=limit)
    return limit


def _failed_call(result: RoutedResponse) -> RoutingFailed | None:
    # The final call is the one whose answer we would return. If it failed
    # (and escalation could not rescue it), returning 200 with an empty answer
    # would hide the failure from the caller. A calculator answer parsed
    # straight from the question has no model call at all.
    final = result.calls[-1] if result.calls else None
    if final is None or not final.error:
        return None
    return RoutingFailed(502, "Model provider call failed.", {
        "message": "Model provider call failed.",
        "model_id": final.model_id,
        "error": final.error,
        "escalated": result.escalated,
    })


def _book(mode, query, client, request_id, limit, result, run_kwargs) -> None:
    """Counters, guardrail spend, metrics, decision log and outcome logging
    for an answered request."""
    _counters[result.tier_served] += 1
    if result.escalated:
        _counters["escalations"] += 1
    _cost_total["usd"] += result.total_cost_usd
    GUARD.record(client, result)
    threshold = run_kwargs.get("threshold")
    record_routed(METRICS, DECISIONS, request_id=request_id, mode=mode, result=result, query=query,
                  threshold=ROUTE_THRESHOLD if threshold is None else threshold)
    if limit is None:  # a second, paid call is the last thing to make over a budget
        SHADOW.consider(query, run_kwargs.get("system"), result, history=run_kwargs.get("history"))


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest, request: Request) -> RouteResponse:
    try:
        result = run_routed(req.mode, req.query, client=client_key(request), system=req.system,
                            threshold=req.threshold)
    except RoutingFailed as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
    final = result.calls[-1] if result.calls else None

    return RouteResponse(
        answer=result.answer,
        tier_chosen=result.tier_chosen,
        tier_served=result.tier_served,
        escalated=result.escalated,
        escalation_reason=result.escalation_reason,
        router_confidence=round(result.router_confidence, 4),
        cost_usd=round(result.total_cost_usd, 6),
        latency_s=round(result.total_latency_s, 3),
        n_calls=len(result.calls),
        truncated=final is not None and final.stop_reason == "max_tokens",
        risk_override=result.risk_override,
        tool=result.tool,
        budget_limited=result.budget_limited,
    )


# ---- OpenAI-compatible endpoint -------------------------------------------
# Point any OpenAI client at this server (base_url=".../v1") and send
# model="auto": the router picks the tier per request. "small" and "large"
# force a tier. Routing reads the latest user message; both tiers receive
# the whole conversation.

CHAT_MODELS: dict[str, RouterMode] = {
    "auto": "outcome",
    "small": "always_small",
    "large": "always_large",
}
FINISH_REASONS = {"max_tokens": "length", "refusal": "content_filter"}


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[dict] | None = None


class ChatRequest(BaseModel):
    # Other OpenAI parameters (temperature, top_p, ...) are accepted and
    # ignored, so existing client code works unchanged.
    model_config = {"extra": "allow"}

    model: str = "auto"
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    stream: bool = False


def openai_error(status: int, message: str, code: str | None = None) -> JSONResponse:
    kind = "invalid_request_error" if status < 500 else "api_error"
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": kind, "param": None, "code": code},
    })


def _text(message: ChatMessage) -> str:
    """Plain text of a message; content may be a string or a list of parts."""
    if message.content is None or isinstance(message.content, str):
        return message.content or ""
    parts = message.content
    if any(p.get("type") != "text" for p in parts):
        raise ValueError("Only text content is supported.")
    return "".join(p.get("text", "") for p in parts)


@app.get("/v1/models")
def list_models() -> dict:
    targets = {"auto": "routed per request", "small": SMALL.model_id, "large": LARGE.model_id}
    return {"object": "list", "data": [
        {"id": name, "object": "model", "created": 0, "owned_by": "llm-router",
         "routes_to": target}
        for name, target in targets.items()
    ]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, request: Request):
    mode = CHAT_MODELS.get(req.model)
    if mode is None:
        return openai_error(400, f"Unknown model {req.model!r}. Use one of: "
                                 f"{', '.join(CHAT_MODELS)}.", "model_not_found")
    try:
        texts = [_text(m) for m in req.messages]
    except ValueError as exc:
        return openai_error(400, str(exc), "unsupported_content")
    if req.messages[-1].role != "user":
        return openai_error(400, "The last message must have role 'user'.")

    system = "\n\n".join(t for m, t in zip(req.messages, texts, strict=True)
                         if m.role in ("system", "developer")) or None
    turns = [{"role": m.role, "content": t}
             for m, t in zip(req.messages[:-1], texts[:-1], strict=True)
             if m.role in ("user", "assistant")]
    kwargs = {"client": client_key(request), "system": system, "history": turns or None,
              "max_tokens": req.max_completion_tokens or req.max_tokens}

    try:
        if req.stream:
            return StreamingResponse(_sse(stream_routed(mode, texts[-1], **kwargs)),
                                     media_type="text/event-stream")
        result = run_routed(mode, texts[-1], **kwargs)
    except RoutingFailed as exc:
        return openai_error(exc.status, _error_message(exc))

    # No calls at all when the calculator parsed the question itself.
    final = result.calls[-1] if result.calls else None
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": final.model_id if final else "calculator",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.answer},
            "finish_reason": _finish_reason(result),
        }],
        "usage": _usage(result),
        "router": _router_info(result),
    }


def _error_message(exc: RoutingFailed) -> str:
    return exc.message if exc.status != 502 else f"{exc.message} {exc.detail['error']}"


def _finish_reason(result: RoutedResponse) -> str:
    final = result.calls[-1] if result.calls else None
    return FINISH_REASONS.get(final.stop_reason, "stop") if final else "stop"


def _usage(result: RoutedResponse) -> dict:
    """Billed tokens across every call, including an escalation retry."""
    prompt_tokens = sum(c.input_tokens for c in result.calls)
    completion_tokens = sum(c.output_tokens for c in result.calls)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


def _router_info(result: RoutedResponse) -> dict:
    """Extension field; standard clients ignore unknown keys."""
    return {
        "tier_chosen": result.tier_chosen,
        "tier_served": result.tier_served,
        "escalated": result.escalated,
        "escalation_reason": result.escalation_reason,
        "confidence": round(result.router_confidence, 4),
        "cost_usd": round(result.total_cost_usd, 6),
        "risk_override": result.risk_override,
        "tool": result.tool,
        "budget_limited": result.budget_limited,
    }


def _sse(events):
    """OpenAI's chat.completion.chunk stream: a role chunk when the answering
    model is known, content chunks, a final chunk with finish_reason, usage and
    the router extension, then [DONE]. A failure after the first token is sent
    as an error event, which the OpenAI SDK raises."""
    stream_id, created, model = f"chatcmpl-{uuid.uuid4().hex}", int(time.time()), "auto"

    def chunk(delta: dict, finish: str | None = None, **extra) -> str:
        body = {"id": stream_id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
        return f"data: {json.dumps(body)}\n\n"

    for kind, payload in events:
        if kind == "start":
            model = payload["model_id"]
            yield chunk({"role": "assistant", "content": ""})
        elif kind == "delta":
            yield chunk({"content": payload})
        elif kind == "error":
            error = {"message": _error_message(payload), "type": "api_error", "code": None}
            yield f"data: {json.dumps({'error': error})}\n\n"
            return
        elif kind == "result":
            yield chunk({}, _finish_reason(payload), usage=_usage(payload), router=_router_info(payload))
    yield "data: [DONE]\n\n"


@app.get("/metrics", include_in_schema=False)
def metrics() -> PlainTextResponse:
    """Prometheus text format: request, escalation, cost, token and latency
    counters for this process (router/telemetry.py)."""
    return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")


@app.get("/stats")
def stats() -> dict:
    served = _counters["small"] + _counters["large"]
    return {
        "requests_served": served,
        "by_tier": {"small": _counters["small"], "large": _counters["large"]},
        "escalations": _counters["escalations"],
        "pct_small": round(100 * _counters["small"] / served, 1) if served else 0.0,
        "total_cost_usd": round(_cost_total["usd"], 6),
    }
