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
    for per_cat in loco.values():
        assert set(per_cat) == set(cats)
        for c, correct in per_cat.items():
            assert 0 <= correct <= (cats == c).sum()


def test_cached_only_never_calls_the_model(monkeypatch):
    """--cached-only must skip uncached references without spending a request."""
    calls = []
    monkeypatch.setattr(LLMClient, "complete", lambda self, *a, **k: calls.append(1))
    cached = {"q1": result()}
    monkeypatch.setattr(LLMClient, "cached",
                        lambda self, spec, prompt, *a, **k: cached.get(prompt))
    c = run_benchmark.RetryingClient(cache=object())
    refs = run_benchmark.build_references(
        c, [{"id": "a", "query": "q1"}, {"id": "b", "query": "q2"}], cached_only=True)
    assert refs == {"a": "ok"} and calls == []


def test_paraphrase_check_rejects_rewrites_that_change_the_question():
    """A model rewrite must keep every number and quoted word, or its answer is wrong."""
    from exact_answer_eval import _paraphrase_keeps_meaning as keeps
    original = "How many times does the letter 'e' appear in the word 'nevertheless'?"
    assert keeps(original, "Count the e's in 'nevertheless' for me.")
    assert not keeps(original, "Count the e's in 'neverthless' for me.")      # word changed
    arith = "What is 486 * 83 - 4267 / 17 + 449?"
    assert keeps(arith, "Can you work out 486 times 83, minus 4267 over 17, plus 449?")
    assert not keeps(arith, "Can you work out 486 times 38, minus 4267 over 17, plus 449?")


def test_wilson_interval_matches_known_values():
    from report_stats import wilson
    lo, hi = wilson(40, 40)
    assert round(lo, 3) == 0.912 and hi == 1.0
    lo, hi = wilson(23, 40)
    assert 0.41 < lo < 0.43 and 0.71 < hi < 0.73


def test_report_stats_runs_on_committed_results():
    """The README's intervals must stay reproducible from results/."""
    from pathlib import Path

    from report_stats import benchmark_report, exact_answer_report
    results = Path(__file__).resolve().parents[1] / "results"
    table = benchmark_report(results / "benchmark_gemini_free16.json")
    assert "n = 14 questions" in table and "| heuristic | 0.929" in table
    assert "40/40 = 100%" in exact_answer_report(results / "exact_answer_eval_seed19.json")


class TestRoutingMetrics:
    """RouteLLM metrics on small hand-checkable cases."""

    def test_oracle_random_and_perfect_curves(self):
        import numpy as np
        from routing_metrics import apgr, cpt, pgr_curve, quality_curve, summarize

        small = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=float)
        large = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=float)
        gain = large - small            # 4 queries where large helps
        oracle = summarize(gain, small, large)
        assert oracle["cpt50"] == 0.25 and oracle["cpt80"] == 0.38
        assert oracle["apgr"] > 0.7

        # A router that ranks the helpful queries last is worse than random.
        anti = summarize(-gain, small, large)
        assert anti["apgr"] < 0.5

        # Random routing averages APGR 0.5 - 1/(2n) once calls are whole queries.
        from routing_metrics import random_apgr
        assert abs(random_apgr(8) - 0.4375) < 0.01 and abs(random_apgr(6637) - 0.5) < 1e-3
        rng = np.random.default_rng(1)
        randoms = [apgr(pgr_curve(quality_curve(rng.random(8), small, large), small, large))
                   for _ in range(2000)]
        assert abs(np.mean(randoms) - random_apgr(8)) < 0.01
        assert cpt(np.array([0.0, 0.4, 0.6]), 0.5, np.array([0.0, 0.5, 1.0])) == 1.0

    def test_auroc_matches_sklearn(self):
        import numpy as np
        from routing_metrics import auroc
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p = np.round(rng.random(200), 1)  # many ties
        assert abs(auroc(y, p) - roc_auc_score(y, p)) < 1e-9
