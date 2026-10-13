"""Exact-match grading for the live benchmark (evals/exact_match.py)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from exact_match import (  # noqa: E402
    extract_answer,
    gold_answer,
    is_correct,
    last_boxed,
    parse_number,
)


@pytest.mark.parametrize("text, value", [
    ("42", Fraction(42)), ("-7", Fraction(-7)), ("0.5", Fraction(1, 2)), (".25", Fraction(1, 4)),
    ("3/4", Fraction(3, 4)), ("\\frac{3}{4}", Fraction(3, 4)), ("-\\dfrac{1}{2}", Fraction(-1, 2)),
    ("$12$", Fraction(12)), ("1,000", Fraction(1000)), ("45^\\circ", Fraction(45)), ("12.", Fraction(12)),
    ("x+1", None), ("(1,2)", None), ("\\sqrt{2}", None), ("1/0", None), ("", None),
])
def test_parse_number(text, value):
    assert parse_number(text) == value


def test_last_boxed_balances_braces():
    assert last_boxed("so \\boxed{\\frac{1}{2}} and then \\boxed{7}") == "7"
    assert last_boxed("x = \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert last_boxed("no box") is None and last_boxed("\\boxed{unclosed") is None


def test_gold_answers():
    assert gold_answer("choice", "G: to which we need give no precise answer.") == "G"
    assert gold_answer("number", "...so the answer is $\\boxed{-2}$.") == Fraction(-2)
    assert gold_answer("number", "the interval $\\boxed{(1,2]}$") is None  # not a plain number


@pytest.mark.parametrize("response, letter", [
    ("Reasoning...\nFinal answer: B", "B"),
    ("Final answer: (C)", "C"),
    ("**Final answer: D.**", "D"),
    ("I think so.\nFinal answer: I", "I"),
    ("Step by step... The answer is (E).", "E"),
    ("The answer is A at first, but the answer is (F)", "F"),
    ("Final answer: I think B is right", None),  # no explicit form: not guessed
    ("no answer here", None),
])
def test_choice_extraction(response, letter):
    assert extract_answer("choice", response) == letter


@pytest.mark.parametrize("response, value", [
    ("work...\nFinal answer: 12", Fraction(12)),
    ("Final answer: $\\boxed{\\frac{5}{2}}$", Fraction(5, 2)),
    ("Final answer: 2.5 meters", Fraction(5, 2)),
    ("so we get \\boxed{-3}", Fraction(-3)),
    ("Final answer: x = 4 or x = 5", None),  # two numbers: ambiguous, not guessed
    ("I could not solve it.", None),
])
def test_number_extraction(response, value):
    assert extract_answer("number", response) == value


def test_is_correct_compares_exact_values():
    gold = "Thus the answer is $\\boxed{\\frac{1}{2}}$."
    assert is_correct("number", "Final answer: 0.5", gold)
    assert not is_correct("number", "Final answer: 0.49", gold)
    assert is_correct("choice", "Final answer: G", "G: something")
    assert not is_correct("number", "Final answer: 3", "answer $\\boxed{x+1}$")  # unusable gold


def fake_record(small_ok, large_ok, outcome_score, escalate=False):
    tier = lambda ok, cost: {"correct": ok, "cost_usd": cost, "latency_s": 1.0,  # noqa: E731
                             "escalate": escalate, "escalation_reason": None}
    return {"kind": "number", "n_words": 10, "small": tier(small_ok, 0.001), "large": tier(large_ok, 0.01),
            "scores": {"outcome": outcome_score, "classifier": 0.5, "heuristic": 0.2}}


def test_replay_applies_escalation_and_charges_both_calls():
    import numpy as np
    from live_exact_benchmark import serve
    records = [fake_record(False, True, 0.9), fake_record(False, True, 0.1, escalate=True),
               fake_record(True, True, 0.1)]
    arm = serve(records, np.array([True, False, False]))
    assert arm["correct"].tolist() == [1.0, 1.0, 1.0]
    assert arm["escalated"].tolist() == [False, True, False]
    assert np.allclose(arm["cost"], [0.01, 0.011, 0.001])


def test_analyze_reports_routers_against_baselines():
    from live_exact_benchmark import analyze
    # The large model helps on the first two; the outcome router ranks them first.
    records = [fake_record(False, True, 0.9), fake_record(False, True, 0.8),
               fake_record(True, True, 0.2), fake_record(True, True, 0.1)]
    s = analyze(records, threshold=0.5)
    arms = s["arms"]
    assert arms["always_small"]["accuracy"] == 0.5 and arms["always_large"]["accuracy"] == 1.0
    assert arms["outcome"]["accuracy"] == 1.0 and arms["outcome"]["large_share"] == 0.5
    assert arms["outcome"]["gain_over_random"] == 0.25 and arms["outcome"]["pgr"] == 1.0
    assert arms["always_small"]["escalations"] == 0
    assert s["curves"]["outcome"]["apgr"] > s["curves"]["classifier"]["apgr"]


def test_daily_budget_persists_across_runs_and_stops_the_run(tmp_path):
    from live_exact_benchmark import DailyBudget, DailyLimit
    path = tmp_path / "usage.json"
    first = DailyBudget(path, limit=1000)
    first.check("m")
    first.add("m", 900)
    second = DailyBudget(path, limit=1000)  # a later run on the same day
    second.check("m")
    second.add("m", 200)
    with pytest.raises(DailyLimit):
        DailyBudget(path, limit=1000).check("m")
    DailyBudget(path, limit=1000).check("other-model")  # budgets are per model
