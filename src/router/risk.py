"""Silent-failure risk rules.

Escalation catches answers that *look* broken: empty, refused, cut off,
hedging, missing requested code. It cannot catch an answer that looks fine
and is wrong. Live runs showed where that happens: exact arithmetic, counting
letters, checking primality, and exhaustive "list all X that..." questions.
A small model answers these fluently and confidently, so nothing downstream
can tell a wrong answer from a right one.

These rules flag such questions *before* routing so they go straight to the
large model. They are deliberately narrow and readable: each rule names one
question shape, and a match is reported by name so a misroute can be traced.

Provenance: the rules were written after seeing the adversarial examples in
the eval set (a01 arithmetic, a02 primality, a03 exhaustive list, a10 letter
counting), so they are tuned on that data. evals/exact_answer_eval.py
measures them on generated questions the rules were not written against,
where correctness is checked programmatically rather than by a judge model.
"""

from __future__ import annotations

import re

# Two numbers joined by an arithmetic operator. A bare hyphen only counts with
# spaces around it, so dates (2026-10-07) and ranges (3-5) don't match.
_ARITHMETIC = re.compile(
    r"\d[\d,.]*\s*(?:[+*/×÷^]|\s-\s|plus|minus|times|divided by|multiplied by|to the power of)"
    r"\s*\(?\s*-?\d",
    re.IGNORECASE,
)

# Character-level counting: "how many r's", "how many times does the letter",
# "count the vowels".
_COUNTING = re.compile(
    r"\bhow many\s+(?:[a-z]'?s\b|times\b|(?:letters?|characters?|vowels?|consonants?|syllables?)\b)"
    r"|\bcount (?:the )?(?:number of )?(?:letters|characters|vowels|consonants|occurrences)\b",
    re.IGNORECASE,
)

# Number-theory checks a fluent guess gets wrong: primality, factoring,
# divisibility, gcd/lcm.
_NUMBER_THEORY = re.compile(
    r"\bis\s+\d{2,}\s+(?:a\s+)?prime\b|\bprime factori[sz]ation\b|\bfactori[sz]e\s+\d"
    r"|\bdivisible by\b|\b(?:gcd|lcm|greatest common divisor|least common multiple)\b",
    re.IGNORECASE,
)

# Character-level string transforms.
_STRING_TRANSFORM = re.compile(
    r"\b(?:reverse|backwards?)\b.{0,40}\b(?:word|string|letters|characters|sentence)\b"
    r"|\b(?:word|string|letters|characters)\b.{0,40}\bbackwards\b",
    re.IGNORECASE,
)

# Open-world exhaustive lists: "List the countries that border...", "Name all
# the elements which...". A list drawn from text in the prompt ("List the verbs
# in this sentence") is closed-world and not flagged.
_EXHAUSTIVE_LIST = re.compile(
    r"\b(?:list|name|enumerate)\s+(?:all\s+|every\s+|each\s+)?(?:the\s+)?(?:\w+\s+){0,2}?"
    r"\w+s\s+(?:that|which|who|whose)\b",
    re.IGNORECASE,
)
_CLOSED_WORLD = re.compile(
    r"\b(?:in|from)\s+(?:this|the following|the above|the text|the passage|below)\b|:\s*\S",
    re.IGNORECASE,
)

RULES: dict[str, re.Pattern] = {
    "exact_arithmetic": _ARITHMETIC,
    "character_counting": _COUNTING,
    "number_theory": _NUMBER_THEORY,
    "string_transform": _STRING_TRANSFORM,
    "exhaustive_list": _EXHAUSTIVE_LIST,
}


def silent_failure_risks(query: str) -> list[str]:
    """Names of the rules this query matches; empty if none."""
    hits = [name for name, pattern in RULES.items()
            if name != "exhaustive_list" and pattern.search(query)]
    if _EXHAUSTIVE_LIST.search(query) and not _CLOSED_WORLD.search(query):
        hits.append("exhaustive_list")
    return hits
