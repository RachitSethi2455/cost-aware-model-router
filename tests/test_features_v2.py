"""The v2 features must not read formatting as difficulty (router/features.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router.features import FEATURE_NAMES_V2, extract, extract_v2  # noqa: E402

QUESTION = "Which gas makes up most of Earth's atmosphere? Pick one."
STACKED = QUESTION + "\n(A): Oxygen.\n(B): Nitrogen.\n(C): Argon.\n(D): Carbon dioxide."
INLINE = QUESTION + " (A) Oxygen (B) Nitrogen (C) Argon (D) Carbon dioxide"


def test_options_on_separate_lines_are_not_sentences():
    stacked, inline = extract_v2(STACKED), extract_v2(INLINE)
    assert stacked.n_sentences == inline.n_sentences == extract_v2(QUESTION).n_sentences == 2
    assert stacked.is_multiple_choice == inline.is_multiple_choice == 1
    assert extract_v2(QUESTION).is_multiple_choice == 0
    assert extract(STACKED).n_sentences == 6  # what v1 saw: every line a sentence


def test_line_breaks_inside_a_sentence_do_not_count():
    assert extract_v2("Compare REST\nand GraphQL\nfor a mobile backend").n_sentences == 1


def test_latex_is_maths_not_code():
    maths = "Let $f(x) = \\frac{x^2}{\\sqrt{x+1}}$. Find $f(3)$."
    v2 = extract_v2(maths)
    assert v2.has_code == 0 and v2.has_math == 1
    assert extract(maths).has_code == 1  # v1 took the braces for code
    assert extract_v2("What is 17 * 24?").has_math == 1
    assert extract_v2("What is the capital of Peru?").has_math == 0


def test_real_code_is_still_code():
    assert extract_v2("Fix this:\n```python\nprint(1/0)\n```").has_code == 1
    assert extract_v2("Why is this slow?\ndef f(n):\n    return f(n - 1)").has_code == 1
    assert extract_v2("Optimise SELECT * FROM orders WHERE total > 10").has_code == 1
    assert extract_v2("Explain the difference between a list and a tuple.").has_code == 0


def test_bullets_count_as_constraints_but_options_do_not():
    assert extract_v2("Write a plan:\n- under 200 words\n- formal tone\n1. include a budget").n_constraints == 3
    assert extract_v2(STACKED).n_constraints == 0


def test_vector_order_matches_names():
    vector = extract_v2(STACKED).to_vector()
    assert len(vector) == len(FEATURE_NAMES_V2)
    assert vector[FEATURE_NAMES_V2.index("is_multiple_choice")] == 1.0
