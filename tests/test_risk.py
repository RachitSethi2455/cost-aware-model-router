"""Silent-failure risk rules and the routing override — offline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router.pipeline import RouterPipeline
from router.risk import silent_failure_risks
from router.types import CallResult


@pytest.mark.parametrize("query, rule", [
    ("What is 17 * 24 + 3^7 - 1024 / 8?", "exact_arithmetic"),
    ("What is 120 divided by 8?", "exact_arithmetic"),
    ("How many r's are in the word 'strawberry'?", "character_counting"),
    ("How many times does the letter 'e' appear in 'nevertheless'?", "character_counting"),
    ("Count the vowels in 'onomatopoeia'.", "character_counting"),
    ("Is 1729 prime?", "number_theory"),
    ("Is 7919 a prime number?", "number_theory"),
    ("Is 1001 divisible by 7?", "number_theory"),
    ("Reverse the letters of the word 'algorithm'.", "string_transform"),
    ("List the countries that border exactly three other countries.", "exhaustive_list"),
    ("Name all the elements which are liquid at room temperature.", "exhaustive_list"),
])
def test_rules_flag_their_shapes(query, rule):
    assert rule in silent_failure_risks(query)


@pytest.mark.parametrize("query", [
    "What is the capital of Australia?",
    "The release is due 2026-10-07; what day of the week is that?",  # date, not subtraction
    "List the verbs in this sentence: the engineer refactored the module.",  # closed-world
    "Extract all the email addresses from this text: a@b.com, c@d.com",
    "How many continents are there?",
    "What does 'prime' mean in 'prime minister'?",
    "List three primary colours.",
    "Spell out the number 4728 in English words.",
])
def test_rules_ignore_lookalikes(query):
    assert silent_failure_risks(query) == []


class FakeLLM:
    def complete(self, spec, prompt, **kwargs):
        return CallResult(text="A plausible answer of reasonable length.", model_id=spec.model_id,
                          tier=spec.name, input_tokens=1, output_tokens=1, cost_usd=0.0,
                          latency_s=0.0, stop_reason="end_turn")


def test_risk_rules_override_a_small_decision():
    on = RouterPipeline(mode="heuristic", client=FakeLLM(), enable_risk_rules=True)
    off = RouterPipeline(mode="heuristic", client=FakeLLM(), enable_risk_rules=False)
    q = "Is 1729 prime?"

    assert off.run(q).tier_chosen == "small"
    result = on.run(q)
    assert result.tier_chosen == "large" and result.tier_served == "large"
    assert result.risk_override == ["number_theory"]


def test_fixed_tier_baselines_ignore_risk_rules():
    pipe = RouterPipeline(mode="always_small", client=FakeLLM(), enable_risk_rules=True)
    result = pipe.run("Is 1729 prime?")
    assert result.tier_chosen == "small" and result.risk_override == []


def test_explain_reports_risks_and_overrides_only_when_enabled(monkeypatch):
    from fastapi.testclient import TestClient

    from api import main
    c = TestClient(main.app)
    q = {"query": "Is 1729 prime?", "mode": "heuristic"}

    monkeypatch.setattr(main, "ENABLE_RISK_RULES", False)
    body = c.post("/explain", json=q).json()
    assert body["tier"] == "small" and body["risk_rules"] == ["number_theory"]
    assert body["overridden_by"] == []

    monkeypatch.setattr(main, "ENABLE_RISK_RULES", True)
    body = c.post("/explain", json=q).json()
    assert body["tier"] == "large" and body["overridden_by"] == ["number_theory"]

    plain = c.post("/explain", json={"query": "What is the capital of Peru?"}).json()
    assert plain["overridden_by"] == [] and plain["risk_rules"] == []


def test_risk_rules_are_opt_in_by_default():
    from router.config import ENABLE_RISK_RULES
    assert ENABLE_RISK_RULES is False or __import__("os").getenv("ENABLE_RISK_RULES") == "1"
