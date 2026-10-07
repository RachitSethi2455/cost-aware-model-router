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

    def run(self, query, **kwargs):
        self.seen = {"query": query, **kwargs}
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


def test_vercel_entrypoint_is_safe_and_self_training():
    """app.py must default to demo mode and train in memory with no artifact.

    Runs in a subprocess: the entrypoint sets PUBLIC_DEMO at import time,
    which must not leak into the other tests.
    """
    import os
    import subprocess
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys; sys.path.insert(0, 'src')\n"
        "import router.classifier as c\n"
        "c.load = lambda path=None: None  # simulate a fresh deploy: no artifact\n"
        "import app\n"
        "clf = app.main.get_classifier()\n"
        "print(app.main.PUBLIC_DEMO, clf.trained,"
        " clf.route('Write a Python function to merge two sorted lists.')[0])\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PUBLIC_DEMO"}
    env["OPENBLAS_NUM_THREADS"] = "1"
    out = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["True", "True", "large"]


def test_request_threshold_does_not_leak_into_later_requests(monkeypatch):
    """Pipelines are shared across requests; a per-request threshold must not stick."""
    from router.pipeline import RouterPipeline

    class FakeLLM:
        def complete(self, spec, prompt, **kwargs):
            return call("Tokyo is the capital city of Japan.", spec.name)

    pipe = RouterPipeline(mode="heuristic", client=FakeLLM())
    monkeypatch.setattr(main, "get_pipeline", lambda mode: pipe)
    c = TestClient(main.app)
    q = {"query": "What is the capital of Japan?"}

    assert c.post("/route", json={**q, "threshold": 0.0}).json()["tier_chosen"] == "large"
    assert c.post("/route", json=q).json()["tier_chosen"] == "small"


def test_sdk_retries_follow_config():
    """Stacked retries burn quota; the SDK must use the configured count."""
    from router.config import PROVIDER_MAX_RETRIES
    from router.providers.openai_compat import OpenAICompatProvider
    p = OpenAICompatProvider(base_url="http://localhost:11434/v1", api_key_env="UNSET_KEY")
    assert p.client.max_retries == PROVIDER_MAX_RETRIES


class TestOpenAICompatible:
    """/v1/chat/completions must look like OpenAI's API to existing clients."""

    @pytest.fixture
    def chat(self, monkeypatch):
        def use(outcome, modes=None):
            pipe = FakePipeline(outcome)

            def get(mode):
                if modes is not None:
                    modes.append(mode)
                return pipe
            monkeypatch.setattr(main, "get_pipeline", get)
            return TestClient(main.app), pipe
        return use

    def test_response_shape_and_conversation_passthrough(self, chat):
        c, pipe = chat(routed([call("Paris is the capital of France.", "small")]))
        body = c.post("/v1/chat/completions", json={
            "model": "auto",
            "temperature": 0.2,  # ignored, must not be rejected
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": [{"type": "text", "text": "Capital of France?"}]},
            ],
        }).json()

        assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
        assert body["model"] == "small-model"
        choice = body["choices"][0]
        assert choice["message"] == {"role": "assistant", "content": "Paris is the capital of France."}
        assert choice["finish_reason"] == "stop"
        assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        assert body["router"]["tier_served"] == "small"
        # Routing sees the latest user message; the model gets the whole conversation.
        assert pipe.seen["query"] == "Capital of France?"
        assert pipe.seen["system"] == "Be brief."
        assert pipe.seen["history"] == [{"role": "user", "content": "Hi"},
                                        {"role": "assistant", "content": "Hello!"}]

    def test_model_names_select_the_mode(self, chat):
        modes = []
        c, _ = chat(routed([call("ok, a long enough answer", "large")]), modes)
        for model in ("auto", "small", "large"):
            c.post("/v1/chat/completions",
                   json={"model": model, "messages": [{"role": "user", "content": "hi"}]})
        assert modes == ["classifier", "always_small", "always_large"]

    @pytest.mark.parametrize("payload, code", [
        ({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}, "model_not_found"),
        ({"stream": True, "messages": [{"role": "user", "content": "hi"}]}, "stream_unsupported"),
        ({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, "unsupported_content"),
        ({"messages": [{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": "yo"}]}, None),
    ])
    def test_bad_requests_get_openai_style_400s(self, chat, payload, code):
        c, _ = chat(routed([call("unused answer text here", "small")]))
        r = c.post("/v1/chat/completions", json=payload)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == code and r.json()["error"]["message"]

    def test_truncation_and_provider_failure(self, chat):
        cut = call("The CAP theorem", "large")
        cut.stop_reason = "max_tokens"
        c, _ = chat(routed([cut]))
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        assert r.json()["choices"][0]["finish_reason"] == "length"

        c, _ = chat(routed([call("", "small", error="RateLimitError: 429"),
                            call("", "large", error="RateLimitError: 429")], escalated=True))
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 502 and "429" in r.json()["error"]["message"]

    def test_public_demo_blocks_chat(self, chat, monkeypatch):
        c, _ = chat(routed([call("unused answer text here", "small")]))
        monkeypatch.setattr(main, "PUBLIC_DEMO", True)
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 403 and r.json()["error"]["message"]

    def test_models_list(self):
        ids = [m["id"] for m in TestClient(main.app).get("/v1/models").json()["data"]]
        assert ids == ["auto", "small", "large"]


def test_providers_send_history_before_the_prompt(monkeypatch):
    from types import SimpleNamespace as NS

    from router.providers.anthropic_provider import AnthropicProvider
    from router.providers.openai_compat import OpenAICompatProvider

    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    sent = {}

    oa = OpenAICompatProvider(base_url="http://localhost:11434/v1", api_key_env="UNSET_KEY")
    oa_resp = NS(choices=[NS(message=NS(content="ok"), finish_reason="stop")],
                 usage=NS(prompt_tokens=1, completion_tokens=1, total_tokens=2))
    oa.client = NS(chat=NS(completions=NS(create=lambda **kw: sent.update(oa=kw) or oa_resp)))
    oa.complete("m", "Next?", "Be brief.", 64, history=history)
    assert [m["role"] for m in sent["oa"]["messages"]] == ["system", "user", "assistant", "user"]
    assert sent["oa"]["messages"][-1]["content"] == "Next?"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    an = AnthropicProvider()
    an_resp = NS(stop_reason="end_turn", content=[NS(type="text", text="ok")],
                 usage=NS(input_tokens=1, output_tokens=1))
    an.client = NS(messages=NS(create=lambda **kw: sent.update(an=kw) or an_resp))
    an.complete("m", "Next?", "Be brief.", 64, history=history)
    assert sent["an"]["messages"] == [*history, {"role": "user", "content": "Next?"}]
    assert sent["an"]["system"] == "Be brief."


def test_cache_key_unchanged_for_single_turn_calls():
    """Existing cached answers must stay reachable after adding history."""
    from router.config import SMALL
    from router.llm import LLMClient
    key = LLMClient._cache_key(SMALL, "q", None, 8192)
    assert key == f"{SMALL.provider}||||q||8192"
    assert LLMClient._cache_key(SMALL, "q", None, 8192, [{"role": "user", "content": "a"}]) != key
