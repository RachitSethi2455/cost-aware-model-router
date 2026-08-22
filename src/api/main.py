"""FastAPI service exposing the router.

Endpoints:
    POST /route    - route a query, return answer + cost/tier metadata
    POST /explain  - routing decision only, no model call (free, fast)
    GET  /health   - liveness + whether a trained router is loaded
    GET  /stats    - cumulative cost/tier counters for this process
"""

from __future__ import annotations

from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from router.classifier import ClassifierRouter
from router.config import ROUTE_THRESHOLD
from router.features import extract
from router.pipeline import RouterPipeline, RouterMode

app = FastAPI(
    title="Adaptive LLM Router",
    description="Per-request model selection with escalation and cost accounting.",
    version="0.1.0",
)

_pipelines: dict[str, RouterPipeline] = {}
_counters = Counter()
_cost_total = {"usd": 0.0}


def get_pipeline(mode: RouterMode) -> RouterPipeline:
    if mode not in _pipelines:
        _pipelines[mode] = RouterPipeline(mode=mode)
    return _pipelines[mode]


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


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "router_trained": ClassifierRouter().trained,
        "default_threshold": ROUTE_THRESHOLD,
    }


@app.post("/explain")
def explain(req: RouteRequest) -> dict:
    """Routing decision without spending a token. Useful for demos and debugging."""
    pipe = get_pipeline(req.mode)
    tier, confidence = pipe._decide(req.query)
    return {
        "tier": tier,
        "confidence": round(confidence, 4),
        "features": extract(req.query).to_dict(),
        "mode": req.mode,
    }


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    pipe = get_pipeline(req.mode)
    if req.threshold is not None:
        pipe.threshold = req.threshold
    try:
        result = pipe.run(req.query, system=req.system)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
