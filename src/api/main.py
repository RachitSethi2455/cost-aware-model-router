"""FastAPI service exposing the router.

Endpoints:
    POST /route    - route a query, return answer + cost/tier metadata
    POST /explain  - routing decision only, no model call (free, fast)
    GET  /health   - liveness + whether a trained router is loaded
    GET  /stats    - cumulative cost/tier counters for this process
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

import os
import time
import uuid
from collections import Counter, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from router import heuristic
from router.classifier import ClassifierRouter
from router.config import ENABLE_RISK_RULES, LARGE, ROUTE_THRESHOLD, SMALL
from router.features import extract
from router.pipeline import RoutedResponse, RouterMode, RouterPipeline
from router.risk import silent_failure_risks

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
    mode: RouterMode = "classifier"
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


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "router_trained": get_classifier().trained,
        "default_threshold": ROUTE_THRESHOLD,
        "public_demo": PUBLIC_DEMO,
    }


@app.post("/explain")
def explain(req: RouteRequest, request: Request) -> dict:
    """Routing decision without spending a token. Useful for demos and debugging."""
    if not _explain_limiter.allow(client_key(request)):
        raise HTTPException(status_code=429, detail="Too many requests; try again in a minute.")

    threshold = req.threshold if req.threshold is not None else ROUTE_THRESHOLD
    clf = get_classifier()
    if req.mode == "always_small":
        tier, confidence = "small", 0.0
    elif req.mode == "always_large":
        tier, confidence = "large", 1.0
    elif req.mode == "heuristic":
        tier, confidence = heuristic.route(req.query, threshold)
    else:
        tier, confidence = clf.route(req.query, threshold)

    # Same override as RouterPipeline.route: shapes the small model fails
    # silently skip it, whatever the router's probability says.
    risks = silent_failure_risks(req.query)
    overridden_by = []
    if ENABLE_RISK_RULES and req.mode in ("heuristic", "classifier") and tier == "small" and risks:
        tier, overridden_by = "large", risks

    # Which router actually decided: the classifier falls back to the
    # heuristic when no trained artifact is loaded.
    decided_by = req.mode
    if req.mode == "classifier" and not clf.trained:
        decided_by = "heuristic_fallback"

    return {
        "tier": tier,
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "features": extract(req.query).to_dict(),
        "mode": req.mode,
        "router": decided_by,
        "contributions": clf.contributions(req.query) if req.mode == "classifier" else None,
        "risk_rules": risks,
        "overridden_by": overridden_by,
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


def run_routed(mode: RouterMode, query: str, **run_kwargs) -> RoutedResponse:
    """Shared by both answering endpoints: demo guard, errors, counters."""
    if PUBLIC_DEMO:
        raise RoutingFailed(403, DEMO_BLOCKED)
    try:
        result = get_pipeline(mode).run(query, **run_kwargs)
    except RuntimeError as exc:
        # Raised while building a provider client, before any model call.
        # This is a configuration problem, not a server crash.
        raise RoutingFailed(503, str(exc)) from exc

    # The final call is the one whose answer we would return. If it failed
    # (and escalation could not rescue it), returning 200 with an empty answer
    # would hide the failure from the caller.
    final = result.calls[-1]
    if final.error:
        raise RoutingFailed(502, "Model provider call failed.", {
            "message": "Model provider call failed.",
            "model_id": final.model_id,
            "error": final.error,
            "escalated": result.escalated,
        })

    _counters[result.tier_served] += 1
    if result.escalated:
        _counters["escalations"] += 1
    _cost_total["usd"] += result.total_cost_usd
    return result


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    try:
        result = run_routed(req.mode, req.query, system=req.system, threshold=req.threshold)
    except RoutingFailed as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
    final = result.calls[-1]

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
        truncated=final.stop_reason == "max_tokens",
        risk_override=result.risk_override,
    )


# ---- OpenAI-compatible endpoint -------------------------------------------
# Point any OpenAI client at this server (base_url=".../v1") and send
# model="auto": the router picks the tier per request. "small" and "large"
# force a tier. Routing reads the latest user message; both tiers receive
# the whole conversation.

CHAT_MODELS: dict[str, RouterMode] = {
    "auto": "classifier",
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
def chat_completions(req: ChatRequest):
    if req.stream:
        return openai_error(400, "Streaming is not supported yet; send stream=false.",
                            "stream_unsupported")
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

    try:
        result = run_routed(
            mode, texts[-1], system=system, history=turns or None,
            max_tokens=req.max_completion_tokens or req.max_tokens,
        )
    except RoutingFailed as exc:
        return openai_error(exc.status, exc.message if exc.status != 502
                            else f"{exc.message} {exc.detail['error']}")

    final = result.calls[-1]
    prompt_tokens = sum(c.input_tokens for c in result.calls)
    completion_tokens = sum(c.output_tokens for c in result.calls)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": final.model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.answer},
            "finish_reason": FINISH_REASONS.get(final.stop_reason, "stop"),
        }],
        # Billed tokens across every call, including an escalation retry.
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        # Extension field; standard clients ignore unknown keys.
        "router": {
            "tier_chosen": result.tier_chosen,
            "tier_served": result.tier_served,
            "escalated": result.escalated,
            "escalation_reason": result.escalation_reason,
            "confidence": round(result.router_confidence, 4),
            "cost_usd": round(result.total_cost_usd, 6),
            "risk_override": result.risk_override,
        },
    }


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
