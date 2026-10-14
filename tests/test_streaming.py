"""Streaming: the pipeline's events and the OpenAI-style SSE endpoint."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from api import main  # noqa: E402
from router.config import LARGE  # noqa: E402
from router.pipeline import RoutedResponse, RouterPipeline  # noqa: E402
from router.telemetry import Metrics  # noqa: E402
from router.types import CallResult  # noqa: E402


def call(tier, text, error=None):
    return CallResult(text=text, model_id=f"{tier}-model", tier=tier, input_tokens=4,
                      output_tokens=6, cost_usd=0.001, latency_s=0.1,
                      stop_reason=None if error else "end_turn", error=error)


class FakeLLM:
    """complete() answers whole; stream() yields words, then returns the result."""

    def __init__(self, answers):
        self.answers, self.log = answers, []

    def complete(self, spec, prompt, **kwargs):
        self.log.append(("complete", spec.name))
        return call(spec.name, self.answers[spec.name])

    def stream(self, spec, prompt, **kwargs):
        self.log.append(("stream", spec.name))
        text = self.answers[spec.name]
        for word in text.split(" "):
            yield word + " "
        return call(spec.name, text)


def events(pipe, query="Explain consensus algorithms."):
    out = list(pipe.stream(query))
    return out, out[-1][1]


def test_large_answers_stream_token_by_token():
    llm = FakeLLM({"large": "Raft elects a leader and replicates a log."})
    out, result = events(RouterPipeline(mode="always_large", client=llm, enable_code_tool=False))
    assert out[0] == ("start", {"tier": "large", "model_id": LARGE.model_id})
    deltas = [p for k, p in out if k == "delta"]
    assert len(deltas) > 1 and "".join(deltas).strip() == result.answer
    assert llm.log == [("stream", "large")]


def test_a_small_answer_is_checked_whole_before_it_is_sent():
    llm = FakeLLM({"small": "Paxos and Raft keep replicas in agreement."})
    out, result = events(RouterPipeline(mode="always_small", client=llm, enable_code_tool=False))
    assert [k for k, _ in out] == ["start", "delta", "result"] and not result.escalated
    assert llm.log == [("complete", "small")]


def test_an_escalated_answer_streams_from_the_large_model():
    llm = FakeLLM({"small": "", "large": "Here is a full answer about consensus."})
    out, result = events(RouterPipeline(mode="always_small", client=llm, enable_code_tool=False))
    assert result.escalated and result.tier_served == "large"
    assert out[0] == ("start", {"tier": "large", "model_id": LARGE.model_id})
    assert llm.log == [("complete", "small"), ("stream", "large")]


def test_run_and_stream_reach_the_same_result():
    answers = {"small": "", "large": "A complete answer."}
    ran = RouterPipeline(mode="always_small", client=FakeLLM(answers), enable_code_tool=False).run("q")
    _, streamed = events(RouterPipeline(mode="always_small", client=FakeLLM(answers), enable_code_tool=False))
    assert (ran.answer, ran.tier_served, ran.escalated) == (streamed.answer, streamed.tier_served,
                                                            streamed.escalated)


class StreamPipe:
    def __init__(self, evts):
        self.evts = evts

    def stream(self, query, **kwargs):
        yield from self.evts


def result(calls):
    return RoutedResponse(answer=calls[-1].text, tier_chosen="large", tier_served="large", escalated=False,
                          escalation_reason=None, router_confidence=0.8,
                          total_cost_usd=sum(c.cost_usd for c in calls), total_latency_s=0.3, calls=calls)


def sse_chunks(response):
    lines = [line[len("data: "):] for line in response.text.split("\n") if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def test_endpoint_streams_openai_chunks_and_books_the_request(monkeypatch):
    final = call("large", "Hello there.")
    pipe = StreamPipe([("start", {"tier": "large", "model_id": "large-model"}), ("delta", "Hello "),
                       ("delta", "there."), ("result", result([final]))])
    monkeypatch.setattr(main, "get_pipeline", lambda mode: pipe)
    monkeypatch.setattr(main, "METRICS", Metrics())
    c = TestClient(main.app)
    r = c.post("/v1/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    chunks = sse_chunks(r)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(ch["choices"][0]["delta"].get("content") or "" for ch in chunks) == "Hello there."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and chunks[-1]["usage"]["total_tokens"] == 10
    assert chunks[-1]["model"] == "large-model" and chunks[-1]["router"]["tier_served"] == "large"
    assert 'router_requests_total{chosen="large",mode="outcome",served="large"} 1' in c.get("/metrics").text


def test_a_failure_before_the_first_token_is_an_ordinary_error(monkeypatch):
    failed = call("large", "", error="RateLimitError: 429")
    monkeypatch.setattr(main, "get_pipeline", lambda mode: StreamPipe([
        ("start", {"tier": "large", "model_id": "large-model"}), ("result", result([failed]))]))
    c = TestClient(main.app)
    r = c.post("/v1/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502 and "429" in r.json()["error"]["message"]


def test_a_failure_mid_stream_ends_with_an_error_event(monkeypatch):
    failed = call("large", "partial", error="APIConnectionError: dropped")
    monkeypatch.setattr(main, "get_pipeline", lambda mode: StreamPipe([
        ("start", {"tier": "large", "model_id": "large-model"}), ("delta", "partial"),
        ("result", result([failed]))]))
    r = TestClient(main.app).post("/v1/chat/completions",
                                  json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    payloads = [line[len("data: "):] for line in r.text.split("\n") if line.startswith("data: ")]
    assert "dropped" in json.loads(payloads[-1])["error"]["message"] and "[DONE]" not in payloads


def test_public_demo_blocks_streaming_too(monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_DEMO", True)
    r = TestClient(main.app).post("/v1/chat/completions",
                                  json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 403
