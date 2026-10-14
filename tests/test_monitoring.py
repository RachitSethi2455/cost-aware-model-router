"""Monitoring (router/telemetry.py) and own-traffic logging (router/shadow.py)."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from api import main  # noqa: E402
from router.pipeline import RoutedResponse  # noqa: E402
from router.shadow import ShadowLogger  # noqa: E402
from router.telemetry import DecisionLog, Metrics  # noqa: E402
from router.types import CallResult  # noqa: E402


def call(text, tier, cost=0.001, error=None):
    return CallResult(text=text, model_id=f"{tier}-model", tier=tier, input_tokens=10,
                      output_tokens=20, cost_usd=cost, latency_s=0.3,
                      stop_reason=None if error else "end_turn", error=error)


def result(calls, escalated=False, tool=None):
    return RoutedResponse(answer=calls[-1].text if calls else "42", tier_chosen=calls[0].tier if calls else "small",
                          tier_served=calls[-1].tier if calls else "small", escalated=escalated,
                          escalation_reason="hedged" if escalated else None, router_confidence=0.3,
                          total_cost_usd=sum(c.cost_usd for c in calls), total_latency_s=0.7,
                          calls=calls, features={"n_words": 4}, tool=tool)


class Inline:
    """Runs shadow calls at once, so tests need no threads."""

    def submit(self, fn, *args):
        fn(*args)


class FakeClient:
    def __init__(self):
        self.calls = []

    def complete(self, spec, prompt, system=None):
        self.calls.append(spec.name)
        return call(f"{spec.name} answer", spec.name, cost=0.01)


def test_metrics_render_counters_and_cumulative_histograms():
    m = Metrics()
    m.inc("router_requests_total", mode="outcome", chosen="small", served="small")
    m.inc("router_cost_usd_total", 0.25, tier="large")
    for v in (0.1, 0.4, 3.0):
        m.observe("router_request_latency_seconds", v, (0.25, 0.5, 1.0))
    text = m.render()
    assert 'router_requests_total{chosen="small",mode="outcome",served="small"} 1' in text
    assert 'router_cost_usd_total{tier="large"} 0.25' in text
    assert 'router_request_latency_seconds_bucket{le="0.25"} 1' in text
    assert 'router_request_latency_seconds_bucket{le="0.5"} 2' in text
    assert 'router_request_latency_seconds_bucket{le="+Inf"} 3' in text
    assert "# TYPE router_request_latency_seconds histogram" in text


def test_route_updates_metrics_and_logs_a_decision_without_query_text(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "METRICS", Metrics())
    monkeypatch.setattr(main, "DECISIONS", DecisionLog(tmp_path / "decisions.jsonl"))
    outcome = result([call("A cheap but complete answer.", "small"), call("Better.", "large")],
                     escalated=True)
    monkeypatch.setattr(main, "get_pipeline", lambda mode: type("P", (), {"run": lambda self, q, **k: outcome})())
    c = TestClient(main.app)
    assert c.post("/route", json={"query": "my private question"}).status_code == 200
    metrics = c.get("/metrics").text
    assert 'router_requests_total{chosen="small",mode="outcome",served="large"} 1' in metrics
    assert 'router_escalations_total{reason="hedged"} 1' in metrics
    entry = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert entry["tier_served"] == "large" and entry["escalated"] and entry["status"] == 200
    assert entry["query_chars"] == len("my private question") and "query" not in entry


def test_failed_requests_are_counted(monkeypatch):
    monkeypatch.setattr(main, "METRICS", Metrics())
    monkeypatch.setattr(main, "PUBLIC_DEMO", True)
    c = TestClient(main.app)
    assert c.post("/route", json={"query": "hi"}).status_code == 403
    assert 'router_errors_total{status="403"} 1' in c.get("/metrics").text


def shadow(tmp_path, rate=1.0, max_per_day=10):
    client = FakeClient()
    logger = ShadowLogger(lambda: client, rate=rate, log_path=tmp_path / "shadow.jsonl",
                          max_per_day=max_per_day, rng=random.Random(0), executor=Inline())
    return logger, client


def test_shadow_calls_the_other_tier_and_logs_both_answers(tmp_path):
    logger, client = shadow(tmp_path)
    assert logger.consider("Why is the sky blue?", None, result([call("Rayleigh.", "small")])) == "sampled"
    assert client.calls == ["large"]
    entry = json.loads((tmp_path / "shadow.jsonl").read_text())
    assert entry["small"]["answer"] == "Rayleigh." and entry["large"]["answer"] == "large answer"
    assert entry["source"] == "sampled" and "is_multiple_choice" in entry["features_v2"]


def test_escalations_are_logged_without_an_extra_call(tmp_path):
    logger, client = shadow(tmp_path, rate=0.0)
    both = result([call("Not sure.", "small"), call("Here it is.", "large")], escalated=True)
    assert logger.consider("q", None, both) == "escalation" and client.calls == []


def test_shadow_skips_what_it_should(tmp_path):
    logger, client = shadow(tmp_path, max_per_day=1)
    one = result([call("ok answer", "small")])
    assert logger.consider("q", None, one, history=[{"role": "user", "content": "hi"}]) is None
    assert logger.consider("q", None, result([], tool={"expression": "6*7", "answer": "42"})) is None
    assert logger.consider("q", None, one) == "sampled"
    assert logger.consider("q", None, one) is None  # today's cap of 1 is used
    off, _ = shadow(tmp_path, rate=0.0)
    assert off.consider("q", None, one) is None
    assert client.calls == ["large"]


def test_label_shadow_turns_judge_scores_into_gains():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
    from label_shadow import label_entries
    entry = {"query": "q", "source": "sampled", "ts": 1.0, "features_v2": {"n_words": 1},
             "small": {"answer": "small says", "model": "s"}, "large": {"answer": "large says", "model": "l"}}
    scores = iter([2, 0, None, 1])
    rows, failed = label_entries([entry] * 4, lambda q, ref, cand: {"score": next(scores)})
    assert [r["gain"] for r in rows] == [0.0, 1.0, 0.5] and failed == 1
