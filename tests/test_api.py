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
    assert r.json()["truncated"] is False


def test_truncated_large_answer_is_flagged(client):
    cut = call("The CAP theorem says", "large")
    cut.stop_reason = "max_tokens"
    r = client(routed([cut])).post("/route", json={"query": "Explain CAP"})
    assert r.status_code == 200
    assert r.json()["truncated"] is True


class TestOpenAICompatUsage:
    """Cost accounting against a fake OpenAI-compatible client."""

    def make_provider(self, usage):
        from types import SimpleNamespace as NS
        from router.providers.openai_compat import OpenAICompatProvider

        p = OpenAICompatProvider(base_url="http://localhost:11434/v1", api_key_env="UNSET_KEY")
        resp = NS(
            choices=[NS(message=NS(content="answer"), finish_reason="stop")],
            usage=usage,
        )
        p.client = NS(chat=NS(completions=NS(create=lambda **kw: resp)))
        return p

    def test_hidden_reasoning_tokens_count_as_output(self):
        from types import SimpleNamespace as NS
        p = self.make_provider(NS(prompt_tokens=19, completion_tokens=40, total_tokens=1039))
        _, in_tok, out_tok, stop, err = p.complete("m", "q", None, 1024)
        assert (in_tok, out_tok, stop, err) == (19, 1020, "end_turn", None)

    def test_plain_usage_unchanged(self):
        from types import SimpleNamespace as NS
        p = self.make_provider(NS(prompt_tokens=10, completion_tokens=20, total_tokens=30))
        assert p.complete("m", "q", None, 1024)[1:3] == (10, 20)


class TestDemo:
    """The public demo page and the endpoints it relies on."""

    @pytest.fixture
    def app_client(self):
        return TestClient(main.app)

    def test_page_is_served(self, app_client):
        r = app_client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Adaptive LLM Router" in r.text

    def test_recorded_examples_are_valid(self, app_client):
        items = app_client.get("/static/examples.json").json()
        assert len(items) >= 2 and {e["tier"] for e in items} == {"small", "large"}
        for e in items:
            assert e["answer"].strip() and e["cost_usd"] > 0 and e["model_id"]

    def test_contributions_explain_the_probability(self, app_client, monkeypatch, tmp_path):
        import math
        from router.classifier import ClassifierRouter, save, train
        queries = ["What is the capital of France?", "Define an API.",
                   "Explain why and compare the tradeoffs of two designs in depth.",
                   "Write a Python function to parse logs and explain each step."]
        path = tmp_path / "clf.joblib"
        save(train(queries, [0, 0, 1, 1]), path)
        monkeypatch.setattr(main, "_classifier", ClassifierRouter(path))

        body = app_client.post("/explain", json={"query": "Write a Python function to sort a list."}).json()
        c = body["contributions"]
        logit = c["intercept"] + sum(f["contribution"] for f in c["features"])
        assert body["router"] == "classifier"
        assert abs(1 / (1 + math.exp(-logit)) - body["confidence"]) < 1e-3

    def test_public_demo_blocks_model_calls(self, app_client, monkeypatch):
        monkeypatch.setattr(main, "PUBLIC_DEMO", True)
        r = app_client.post("/route", json={"query": "hi"})
        assert r.status_code == 403

    def test_explain_is_rate_limited(self, app_client, monkeypatch):
        monkeypatch.setattr(main, "_explain_limiter", main.RateLimiter(2))
        codes = [app_client.post("/explain", json={"query": "hi"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
