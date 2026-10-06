"""Benchmark retry behaviour — offline, the underlying client is faked."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evals")]

import run_benchmark  # noqa: E402
from router.llm import LLMClient  # noqa: E402
from router.types import CallResult  # noqa: E402


def result(error=None):
    return CallResult(text="" if error else "ok", model_id="m", tier="large",
                      input_tokens=1, output_tokens=1, cost_usd=0.0,
                      latency_s=0.0, error=error)


@pytest.fixture
def client(monkeypatch):
    """A RetryingClient whose underlying calls return a scripted sequence."""
    monkeypatch.setattr(run_benchmark.time, "sleep", lambda s: None)

    def make(outcomes):
        calls = iter(outcomes)
        monkeypatch.setattr(LLMClient, "complete", lambda self, *a, **k: next(calls))
        return run_benchmark.RetryingClient(cache=object())
    return make


def test_transient_error_is_retried(client):
    c = client([result("InternalServerError: 503 high demand"), result()])
    assert c.complete("spec", "q").error is None


def test_gives_up_after_retries(client):
    c = client([result("RateLimitError: 429 per minute")] * 4)
    assert c.complete("spec", "q").error


def test_daily_quota_stops_the_run(client):
    c = client([result("429 GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    with pytest.raises(run_benchmark.DailyQuotaExhausted):
        c.complete("spec", "q")


def test_limit_interleaves_categories():
    ids = [r["id"] for r in run_benchmark.load_dataset(8)]
    assert {i.rstrip("0123456789") for i in ids} == {"s", "m", "c", "a"}


def test_router_analysis_leave_one_category_out_counts():
    """Every query is tested exactly once across the held-out categories."""
    import analyze_router
    queries, y, cats = analyze_router.load()
    loco = analyze_router.leave_one_category_out(queries, y, cats)
    for name, per_cat in loco.items():
        assert set(per_cat) == set(cats)
        for c, correct in per_cat.items():
            assert 0 <= correct <= (cats == c).sum()
