"""Exact-match grading for the live benchmark: no LLM judge.

Two kinds of question, both with an answer that can be checked by code:
- "choice": multiple choice (MMLU-Pro), graded on the option letter.
- "number": a maths problem (MATH) whose reference answer is a plain number
  or fraction, compared as exact fractions, so 0.5, 1/2 and \\frac{1}{2} match.

Models are asked to end with "Final answer: X", but often do not: a reasoning
model may reply with the bare value, "**Answer: (B).**", or an option's text
instead of its letter. The extractors accept those forms when they are
unambiguous and never guess otherwise; an unreadable answer counts as wrong.
Pure Python, so CI can test it.
"""

from __future__ import annotations

import re
from fractions import Fraction

FINAL_LINE = re.compile(r"final answer\s*[:：]\s*(.+)", re.IGNORECASE)
# "The answer is (B)", "Answer: (B)", "the correct choice is **I: 0.71**". The
# letter must not start a sentence ("Answer: I think B" is not I).
ANSWER_LETTER = re.compile(r"(?i:answer|choice|option)(?:\s+is)?\s*[:\-]?\s*(?:\*\*)?\s*\(?([A-J])\)?"
                           r"(?![A-Za-z0-9])(?!\s+[a-z])")
ANSWER_IS_BRACKETED = re.compile(r"(?i:answer is)[^A-Za-z(\n]{0,12}\(([A-J])\)")
ANSWER_IS_VALUE = re.compile(r"(?i:answer is)\s*:?\s*(.+?)\s*(?:\.\s|\.?$)", re.MULTILINE)
LEADING_LETTER = re.compile(r"^\(?([A-J])\)?\s*(?:[:.)]|$)")
BRACKETED_LETTER = re.compile(r"\(([A-J])\)")
BARE_LETTER = re.compile(r"^\s*\**\(?([A-J])\)?\**\.?\s*$")
OPTION_LINE = re.compile(r"^\(([A-J])\):\s*(.+?)\s*$", re.MULTILINE)
FRAC = re.compile(r"^(-?)\\[dt]?frac(?:\{(-?\d+)\}|(\d))(?:\{(-?\d+)\}|(\d))$")
SLASH = re.compile(r"^-?\d+/-?\d+$")
DECIMAL = re.compile(r"^-?(\d+\.?\d*|\.\d+)$")
NUMBER_TOKEN = re.compile(r"-?\\[dt]?frac(?:\{-?\d+\}|\d)(?:\{-?\d+\}|\d)|-?\d+/\d+|-?\d*\.?\d+")
NOISE = ("\\[", "\\]", "\\(", "\\)", "\\!", "\\,", "\\left", "\\right", "^\\circ", "^{\\circ}",
         "°", "\\%", "%", "$")


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
    s = text.strip()
    for noise in NOISE:
        s = s.replace(noise, "")
    if re.search(r"\d\s+\d", s):
        return None  # "12 15" is two numbers, not 1215
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"_\{?\d+\}?$", "", s)  # base subscript: 2003_6
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")  # thousands separators only, never "1,2"
    s = s.rstrip(".")
    if m := FRAC.match(s):
        sign, num_b, num_d, den_b, den_d = m.groups()
        num, den = int(num_b or num_d), int(den_b or den_d)
        return None if den == 0 else Fraction(num, den) * (-1 if sign else 1)
    if SLASH.match(s):
        num, den = s.split("/")
        return None if int(den) == 0 else Fraction(int(num), int(den))
    if DECIMAL.match(s):
        return Fraction(s)
    return None


def single_number(text: str) -> Fraction | None:
    """The text as a number, or the only number in it; None if ambiguous."""
    if (value := parse_number(text)) is not None:
        return value
    tokens = NUMBER_TOKEN.findall(text)
    return parse_number(tokens[0]) if len(tokens) == 1 else None


def parse_options(prompt: str) -> dict[str, str]:
    """The "(A): text" option lines of an MMLU-Pro prompt."""
    return dict(OPTION_LINE.findall(prompt))


def match_option(candidate: str, options: dict[str, str]) -> str | None:
    """The one option whose text equals the candidate, as text or as a number."""
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().strip("*").strip().rstrip(".").strip().lower())
    value = parse_number(candidate)
    hits = {letter for letter, text in options.items()
            if norm(text) == norm(candidate) or (value is not None and parse_number(text) == value)}
    return hits.pop() if len(hits) == 1 else None


def gold_answer(kind: str, gold: str) -> str | Fraction | None:
    """The reference answer: an option letter, or a number (None if the
    reference is not a plain number, e.g. an expression or an interval)."""
    if kind == "choice":
        m = re.match(r"\s*\(?([A-J])\)?\s*:", gold)
        return m.group(1) if m else None
    boxed = last_boxed(gold)
    return parse_number(boxed) if boxed is not None else None


def extract_choice(text: str, final: str | None, options: dict[str, str] | None) -> str | None:
    if final:
        for pattern in (LEADING_LETTER, BRACKETED_LETTER, ANSWER_LETTER, ANSWER_IS_BRACKETED):
            if m := pattern.search(final):
                return m.group(1)
    if m := BARE_LETTER.match(text):
        return m.group(1)
    # The model's concluding statement: the last "answer is (X)"-style match.
    found = [(m.start(), m.group(1)) for p in (ANSWER_LETTER, ANSWER_IS_BRACKETED) for m in p.finditer(text)]
    if found:
        return max(found)[1]
    if options:  # the option's text instead of its letter: "The answer is 45°"
        values = ANSWER_IS_VALUE.findall(text)
        for candidate in (final, values[-1] if values else None, text if len(text) <= 80 else None):
            if candidate and (letter := match_option(candidate, options)):
                return letter
    return None


def extract_number(text: str, final: str | None) -> Fraction | None:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    for candidate in (final, last_boxed(final or ""), last_boxed(text), text,
                      lines[-1] if lines else None):
        if candidate and (value := single_number(candidate)) is not None:
            return value
    return None


def extract_answer(kind: str, text: str, options: dict[str, str] | None = None):
    """The model's answer: an option letter or a number, or None if unreadable."""
    finals = FINAL_LINE.findall(text)
    final = finals[-1].strip().strip("*").strip() if finals else None
    if kind == "choice":
        return extract_choice(text, final, options)
    return extract_number(text, final)


def is_correct(kind: str, response: str, gold: str, options: dict[str, str] | None = None) -> bool:
    expected = gold_answer(kind, gold)
    return expected is not None and extract_answer(kind, response, options) == expected
