"""Post-hoc quality guards on the small model's output.

Design note: these are *deterministic checks*,
not a second LLM call. Asking a model "was your answer good?" costs as much as
just using the big model, and self-assessment is unreliable. Cheap structural
signals catch most real failures for free.

Routing wrong is survivable if you can detect it. This is that detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import MIN_ANSWER_CHARS
from .types import CallResult

HEDGE_PATTERNS = [
    r"\bI (?:can(?:'|no)t|am unable to|don'?t (?:know|have))\b",
    r"\bas an AI\b",
    r"\bI'?m not sure\b",
    r"\binsufficient information\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str | None = None


def should_escalate(result: CallResult, query: str) -> EscalationDecision:
    """Decide whether the small model's answer warrants a large-model retry."""
    if result.error:
        return EscalationDecision(True, "api_error")

    if result.stop_reason == "refusal":
        return EscalationDecision(True, "refusal")

    text = result.text.strip()

    if len(text) < MIN_ANSWER_CHARS:
        return EscalationDecision(True, "too_short")

    if HEDGE_RE.search(text):
        return EscalationDecision(True, "hedged")

    # Truncation: hit the token ceiling mid-thought.
    if result.stop_reason == "max_tokens":
        return EscalationDecision(True, "truncated")

    # Code was asked for but no code came back.
    if re.search(r"\b(code|function|script|query|sql)\b", query, re.IGNORECASE):
        if "```" not in text and not re.search(r"\bSELECT\b|\bdef \b", text):
            return EscalationDecision(True, "expected_code_missing")

    return EscalationDecision(False)
