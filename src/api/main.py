"""FastAPI service exposing the router.

Endpoints:
    POST /route    - route a query, return answer + cost/tier metadata
    POST /explain  - routing decision only, no model call (free, fast)
    GET  /health   - liveness + whether a trained router is loaded
    GET  /stats    - cumulative cost/tier counters for this process
    GET  /         - interactive demo page (uses /explain only)

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
from collections import Counter, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from router import heuristic
from router.classifier import ClassifierRouter
from router.config import ROUTE_THRESHOLD
from router.features import extract
from router.pipeline import RouterPipeline, RouterMode

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
    }


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    if PUBLIC_DEMO:
        raise HTTPException(
            status_code=403,
            detail="Model calls are disabled on the public demo. Use /explain, "
                   "or run the project locally with your own API key.",
        )
    pipe = get_pipeline(req.mode)
    try:
        result = pipe.run(req.query, system=req.system, threshold=req.threshold)
    except RuntimeError as exc:
        # Raised while building a provider client, before any model call.
        # This is a configuration problem, not a server crash.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # The final call is the one whose answer we would return. If it failed
    # (and escalation could not rescue it), returning 200 with an empty answer
    # would hide the failure from the caller.
    final = result.calls[-1]
    if final.error:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Model provider call failed.",
                "model_id": final.model_id,
                "error": final.error,
                "escalated": result.escalated,
            },
        )

    _counters[result.tier_served] += 1
    if result.escalated:
        _counters["escalations"] += 1
    _cost_total["usd"] += result.total_cost_usd

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
    )


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
