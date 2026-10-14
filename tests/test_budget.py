"""Spending guardrails (router/budget.py) and their effect on routing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from api import main  # noqa: E402
from router.budget import SpendGuard  # noqa: E402
from router.pipeline import RoutedResponse, RouterPipeline  # noqa: E402
from router.telemetry import Metrics  # noqa: E402
from router.types import CallResult  # noqa: E402


def call(tier, cost=0.01, text="A complete enough answer."):
    return CallResult(text=text, model_id=f"{tier}-model", tier=tier, input_tokens=10,
                      output_tokens=10, cost_usd=cost, latency_s=0.1, stop_reason="end_turn")


def result(*calls):
    return RoutedResponse(answer=calls[-1].text, tier_chosen=calls[0].tier, tier_served=calls[-1].tier,
                          escalated=len(calls) > 1, escalation_reason=None, router_confidence=0.9,
                          total_cost_usd=sum(c.cost_usd for c in calls), total_latency_s=0.2,
                          calls=list(calls))


class Clock:
    def __init__(self):
        self.t, self.day = 1_000.0, "2026-10-14"


def test_per_client_cap_counts_large_calls_in_a_sliding_hour():
    clock = Clock()
    guard = SpendGuard(large_calls_per_hour=2, daily_limit_usd=0, clock=lambda: clock.t,
                       today=lambda: clock.day)
    guard.record("a", result(call("small"), call("large")))  # an escalation uses the large model
    guard.record("a", result(call("large")))
    assert guard.check("a") == "client_large_call_limit"
    assert guard.check("b") is None  # other clients are unaffected
    clock.t += 3601
    assert guard.check("a") is None


def test_daily_spend_limit_resets_with_the_date():
    clock = Clock()
    guard = SpendGuard(large_calls_per_hour=0, daily_limit_usd=0.05, clock=lambda: clock.t,
                       today=lambda: clock.day)
    for _ in range(5):
        guard.record("a", result(call("small", cost=0.01)))
    assert guard.check("anyone") == "daily_spend_limit"
    assert guard.status()["spent_today_usd"] == 0.05
    clock.day = "2026-10-15"
    assert guard.check("anyone") is None


def test_zero_turns_limits_off():
    guard = SpendGuard(large_calls_per_hour=0, daily_limit_usd=0)
    for _ in range(50):
        guard.record("a", result(call("large", cost=1.0)))
    assert guard.check("a") is None


class FakeLLM:
    def __init__(self):
        self.tiers = []

    def complete(self, spec, prompt, **kwargs):
        self.tiers.append(spec.name)
        return call(spec.name, text="")  # empty: escalation would fire if allowed


def test_pipeline_without_large_serves_small_and_skips_escalation():
    llm = FakeLLM()
    pipe = RouterPipeline(mode="always_large", client=llm, enable_code_tool=False)
    out = pipe.run("Explain the CAP theorem in depth.", allow_large=False)
    assert out.tier_served == "small" and out.budget_limited and not out.escalated
    assert llm.tiers == ["small"]
    llm = FakeLLM()
    out = RouterPipeline(mode="always_small", client=llm, enable_code_tool=False).run("hi", allow_large=False)
    assert llm.tiers == ["small"] and not out.budget_limited  # nothing was taken away


def test_api_downgrades_a_client_over_its_cap(monkeypatch):
    seen = []

    class Pipe:
        def run(self, query, **kwargs):
            seen.append(kwargs.get("allow_large", True))
            return result(call("large"))

    monkeypatch.setattr(main, "get_pipeline", lambda mode: Pipe())
    monkeypatch.setattr(main, "GUARD", SpendGuard(large_calls_per_hour=1, daily_limit_usd=0))
    monkeypatch.setattr(main, "METRICS", Metrics())
    c = TestClient(main.app)
    for _ in range(2):
        assert c.post("/route", json={"query": "hi"}).status_code == 200
    assert seen == [True, False]
    assert 'router_budget_limited_total{reason="client_large_call_limit"} 1' in c.get("/metrics").text
    assert c.get("/health").json()["budget"]["large_calls_per_client_per_hour"] == 1
