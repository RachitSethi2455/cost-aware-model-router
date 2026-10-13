"""Exact-match grading for the live benchmark: no LLM judge.

Two kinds of question, both with an answer that can be checked by code:
- "choice": multiple choice (MMLU-Pro), graded on the option letter.
- "number": a maths problem (MATH) whose reference answer is a plain number
  or fraction, compared as exact fractions, so 0.5, 1/2 and \\frac{1}{2} match.

Models are asked to end with a line "Final answer: X"; the extractors fall back
to the usual "The answer is (X)" and \\boxed{X} forms. Pure Python, so CI can
test it.
"""

from __future__ import annotations

import re
from fractions import Fraction

FINAL_LINE = re.compile(r"final answer\s*[:：]\s*(.+)", re.IGNORECASE)
ANSWER_IS = re.compile(r"answer is\s*:?\s*\**\(?([A-J])\)?(?![A-Za-z])")
LEADING_LETTER = re.compile(r"^\(?([A-J])\)?\s*(?:[:.)]|$)")
BRACKETED_LETTER = re.compile(r"\(([A-J])\)")
FRAC = re.compile(r"^(-?)\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}$")
SLASH = re.compile(r"^-?\d+/-?\d+$")
DECIMAL = re.compile(r"^-?(\d+\.?\d*|\.\d+)$")
NUMBER_TOKEN = re.compile(r"-?\\[dt]?frac\{-?\d+\}\{-?\d+\}|-?\d+/\d+|-?\d*\.?\d+")


def last_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...} or \\fbox{...}, braces balanced."""
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    i = text.index("{", start) + 1
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1] if depth == 0 else None


def parse_number(text: str) -> Fraction | None:
    """A plain number or fraction, or None for anything else."""
    s = text.strip().strip("$").strip()
    for noise in ("\\!", "\\,", " ", "\\left", "\\right", "^\\circ", "^{\\circ}", "\\%", "%"):
        s = s.replace(noise, "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")  # thousands separators only, never "1,2"
    s = s.rstrip(".")
    if m := FRAC.match(s):
        sign, num, den = m.groups()
        return None if int(den) == 0 else Fraction(int(num), int(den)) * (-1 if sign else 1)
    if SLASH.match(s):
        num, den = s.split("/")
        return None if int(den) == 0 else Fraction(int(num), int(den))
    if DECIMAL.match(s):
        return Fraction(s)
    return None


def gold_answer(kind: str, gold: str) -> str | Fraction | None:
    """The reference answer: an option letter, or a number (None if the
    reference is not a plain number, e.g. an expression or an interval)."""
    if kind == "choice":
        m = re.match(r"\s*\(?([A-J])\)?\s*:", gold)
        return m.group(1) if m else None
    boxed = last_boxed(gold)
    return parse_number(boxed) if boxed is not None else None


def extract_answer(kind: str, text: str) -> str | Fraction | None:
    finals = FINAL_LINE.findall(text)
    final = finals[-1].strip().strip("*").strip() if finals else None
    if kind == "choice":
        # Explicit forms only: a bare "I" in "I think B" is also an option letter.
        if final:
            for pattern in (LEADING_LETTER, BRACKETED_LETTER, ANSWER_IS):
                if m := pattern.search(final):
                    return m.group(1)
        matches = ANSWER_IS.findall(text)
        return matches[-1] if matches else None
    for candidate in (final, last_boxed(final or ""), last_boxed(text)):
        if candidate is None:
            continue
        if (value := parse_number(candidate)) is not None:
            return value
        tokens = NUMBER_TOKEN.findall(candidate)
        if len(tokens) == 1 and (value := parse_number(tokens[0])) is not None:
            return value
    return None


def is_correct(kind: str, response: str, gold: str) -> bool:
    expected = gold_answer(kind, gold)
    return expected is not None and extract_answer(kind, response) == expected
