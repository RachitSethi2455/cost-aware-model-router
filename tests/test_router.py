"""Unit tests — all offline, no API key needed.

Keeping the deterministic logic testable without network access is deliberate:
it means CI can run on every push for free.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router import heuristic
from router.escalation import should_escalate
from router.features import FEATURE_NAMES, extract
from router.types import CallResult


def make_result(text: str, stop_reason: str = "end_turn", error: str | None = None):
    return CallResult(
        text=text, model_id="test", tier="small", input_tokens=10,
        output_tokens=10, cost_usd=0.0, latency_s=0.1,
        stop_reason=stop_reason, error=error,
    )


class TestFeatures:
    def test_vector_length_matches_names(self):
        assert len(extract("hello world").to_vector()) == len(FEATURE_NAMES)

    def test_word_count(self):
        assert extract("one two three").n_words == 3

    def test_detects_code(self):
        assert extract("def foo(): pass").has_code == 1
        assert extract("what is the capital of France").has_code == 0

    def test_reasoning_markers(self):
        assert extract("Explain why this fails and compare alternatives").reasoning_marker_count >= 2

    def test_simple_markers(self):
        assert extract("What is a pointer? Define it.").simple_marker_count >= 1

    def test_detects_code_request_without_code(self):
        assert extract("Write a Python function to merge two sorted lists.").asks_for_code == 1
        assert extract("Implement an LRU cache with O(1) get and put.").asks_for_code == 1
        assert extract("What is the capital of Japan?").asks_for_code == 0
        assert extract("Write a one-line git command to discard changes.").asks_for_code == 0

    def test_counts_constraints(self):
        q = "Do this:\n- first thing\n- second thing\nMake sure it compiles."
        assert extract(q).n_constraints >= 3


class TestHeuristicRouter:
    def test_short_lookup_goes_small(self):
        tier, _ = heuristic.route("What is the capital of Japan?")
        assert tier == "small"

    def test_long_reasoning_goes_large(self):
        q = ("Explain why distributed consensus is hard and compare Raft to Paxos, "
             "analyzing the tradeoffs each makes and why one is easier to implement "
             "correctly than the other in a production system with partial failures.")
        tier, _ = heuristic.route(q)
        assert tier == "large"

    def test_confidence_in_range(self):
        for q in ["hi", "Explain " * 100]:
            _, conf = heuristic.route(q)
            assert 0.0 <= conf <= 1.0


class TestEscalation:
    def test_escalates_on_empty(self):
        assert should_escalate(make_result(""), "anything").escalate

    def test_escalates_on_refusal(self):
        d = should_escalate(make_result("", stop_reason="refusal"), "q")
        assert d.escalate and d.reason == "refusal"

    def test_escalates_on_truncation(self):
        long_text = "a valid looking answer " * 10
        d = should_escalate(make_result(long_text, stop_reason="max_tokens"), "q")
        assert d.escalate and d.reason == "truncated"

    def test_escalates_on_hedge(self):
        d = should_escalate(make_result("I'm not sure what you mean by that at all."), "q")
        assert d.escalate and d.reason == "hedged"

    def test_escalates_when_code_expected_but_absent(self):
        d = should_escalate(
            make_result("You should probably use a loop and iterate over things."),
            "Write a Python function to reverse a list",
        )
        assert d.escalate and d.reason == "expected_code_missing"

    def test_escalates_on_api_error(self):
        d = should_escalate(make_result("", error="rate limited"), "q")
        assert d.escalate and d.reason == "api_error"

    def test_passes_good_answer(self):
        good = "The capital of Japan is Tokyo, which has been the seat of government since 1868."
        assert not should_escalate(make_result(good), "What is the capital of Japan?").escalate

    def test_code_answer_passes(self):
        answer = "Here you go:\n```python\ndef rev(x):\n    return x[::-1]\n```"
        assert not should_escalate(make_result(answer), "Write a function to reverse a list").escalate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestClassifierArtifact:
    def test_stale_artifact_falls_back_to_heuristic(self, tmp_path):
        import numpy as np
        from router.classifier import ClassifierRouter, build_pipeline, save

        # A model trained on fewer features than the code now extracts.
        stale = build_pipeline().fit(np.array([[0.0] * 3, [1.0] * 3]), [0, 1])
        path = tmp_path / "old.joblib"
        save(stale, path)

        router = ClassifierRouter(path)
        assert router.trained is False
        tier, _ = router.route("What is the capital of Japan?")
        assert tier == "small"


def test_response_cache_survives_concurrent_threads(tmp_path):
    """The API shares one cache across its thread pool."""
    from concurrent.futures import ThreadPoolExecutor
    from router.cache import ResponseCache

    cache = ResponseCache(tmp_path / "cache.sqlite")

    def work(i):
        cache.set("m", f"prompt {i}", {"text": str(i)})
        return cache.get("m", f"prompt {i}")["text"]

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(work, range(400)))
    assert results == [str(i) for i in range(400)]
    assert cache.stats() == {"m": 400}
