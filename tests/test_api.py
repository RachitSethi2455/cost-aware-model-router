"""API error-contract tests — offline, the pipeline is replaced with fakes."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from api import main
from router.pipeline import RoutedResponse
from router.types import CallResult


def call(text: str, tier: str, error: str | None = None) -> CallResult:
    return CallResult(
        text=text, model_id=f"{tier}-model", tier=tier, input_tokens=5,
        output_tokens=5, cost_usd=0.001, latency_s=0.1,
        stop_reason=None if error else "end_turn", error=error,
    )


class FakePipeline:
    def __init__(self, outcome):
        self.outcome = outcome
        self.threshold = 0.5

    def run(self, query, system=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def routed(calls, escalated=False) -> RoutedResponse:
    return RoutedResponse(
        answer=calls[-1].text, tier_chosen=calls[0].tier, tier_served=calls[-1].tier,
        escalated=escalated, escalation_reason="api_error" if escalated else None,
        router_confidence=0.2, total_cost_usd=sum(c.cost_usd for c in calls),
        total_latency_s=0.2, calls=calls,
    )


@pytest.fixture
def client(monkeypatch):
    def use(outcome):
        monkeypatch.setattr(main, "get_pipeline", lambda mode: FakePipeline(outcome))
        return TestClient(main.app)
    return use


def test_missing_key_is_503_not_500(client):
    c = client(RuntimeError("GROQ_API_KEY is not set. Add it to your .env file."))
    r = c.post("/route", json={"query": "hi"})
    assert r.status_code == 503
    assert "GROQ_API_KEY" in r.json()["detail"]


def test_provider_failure_after_escalation_is_502(client):
    calls = [call("", "small", error="NotFoundError: model"),
             call("", "large", error="NotFoundError: model")]
    r = client(routed(calls, escalated=True)).post("/route", json={"query": "hi"})
    assert r.status_code == 502
    body = r.json()["detail"]
    assert body["model_id"] == "large-model" and body["escalated"] is True


def test_escalation_rescues_small_model_failure(client):
    calls = [call("", "small", error="RateLimitError"),
             call("Paris is the capital of France.", "large")]
    r = client(routed(calls, escalated=True)).post("/route", json={"query": "hi"})
    assert r.status_code == 200
    assert r.json()["tier_served"] == "large" and r.json()["escalated"] is True


def test_success_is_200(client):
    r = client(routed([call("Tokyo is the capital of Japan.", "small")])).post(
        "/route", json={"query": "capital of Japan?"})
    assert r.status_code == 200
    assert r.json()["answer"].startswith("Tokyo")
