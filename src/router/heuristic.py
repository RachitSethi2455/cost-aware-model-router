"""Rule-based router — the day-1 baseline.

Exists so the classifier has something to beat. Keep it simple and legible:
its job is to be a fair, honest baseline, not to be clever. If you tune this
until it matches the classifier, you have destroyed your own headline result.
"""

from __future__ import annotations

from .features import QueryFeatures, extract


def score(features: QueryFeatures) -> float:
    """Return a pseudo-probability of 'this query needs the large model'."""
    s = 0.0
    # Length is a weak signal on its own — a rambling question can still be
    # trivial (see the adversarial_long_but_easy cases in the eval set).
    if features.n_words > 30:
        s += 0.25
    if features.n_words > 120:
        s += 0.15
    # Reasoning markers are the strongest single signal, so cap at 3 not 2.
    s += 0.22 * min(features.reasoning_marker_count, 3)
    s -= 0.25 * min(features.simple_marker_count, 2)
    if features.has_code:
        s += 0.20
    s += 0.10 * min(features.n_constraints, 3)
    return max(0.0, min(1.0, s))


def route(query: str, threshold: float = 0.5) -> tuple[str, float]:
    """Return (tier, confidence) where tier is 'small' or 'large'."""
    p = score(extract(query))
    return ("large" if p >= threshold else "small"), p
