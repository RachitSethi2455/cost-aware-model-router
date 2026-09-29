"""Query feature extraction.

Six cheap, interpretable features. Interpretability matters here: when the
router misroutes, you want to be able to explain *why*, and a
logistic regression over named features lets you read the coefficients.

Deliberately no embeddings — an embedding call would cost as much as the
routing decision saves.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Words that tend to signal multi-step reasoning rather than lookup.
REASONING_MARKERS = {
    "why", "explain", "compare", "analyze", "analyse", "evaluate", "derive",
    "prove", "design", "architect", "tradeoff", "trade-off", "implications",
    "critique", "optimize", "optimise", "refactor", "debug", "strategy",
    "step by step", "walk me through", "pros and cons",
}

# Words that tend to signal a lookup / format / transform task.
SIMPLE_MARKERS = {
    "what is", "who is", "when did", "define", "list", "translate",
    "convert", "summarize in", "capital of", "spell", "rewrite",
    "shorten", "extract", "classify", "yes or no",
}

CODE_HINT = re.compile(r"```|\bdef \b|\bclass \b|\bSELECT\b|\bimport \b|[{};]")
QUESTION_MARK = re.compile(r"\?")

FEATURE_NAMES = [
    "n_words",
    "n_sentences",
    "reasoning_marker_count",
    "simple_marker_count",
    "has_code",
    "n_constraints",
]


@dataclass
class QueryFeatures:
    n_words: int
    n_sentences: int
    reasoning_marker_count: int
    simple_marker_count: int
    has_code: int
    n_constraints: int

    def to_vector(self) -> list[float]:
        """Ordered vector matching FEATURE_NAMES."""
        d = asdict(self)
        return [float(d[name]) for name in FEATURE_NAMES]

    def to_dict(self) -> dict:
        return asdict(self)


def _count_markers(text: str, markers: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for m in markers if m in lowered)


def _count_constraints(text: str) -> int:
    """Rough proxy for 'how many things must the answer satisfy'.

    Counts enumerated requirements, bullets, and conjunctive 'and also' style
    additions. More constraints => harder to satisfy in one pass.
    """
    bullets = len(re.findall(r"^\s*[-*\d]+[.)]?\s+", text, flags=re.MULTILINE))
    conjunctions = len(re.findall(r"\b(and also|as well as|additionally|make sure|ensure that)\b",
                                  text, flags=re.IGNORECASE))
    return bullets + conjunctions


def extract(query: str) -> QueryFeatures:
    """Extract routing features from a raw user query."""
    words = query.split()
    sentences = [s for s in re.split(r"[.!?\n]+", query) if s.strip()]
    return QueryFeatures(
        n_words=len(words),
        n_sentences=len(sentences),
        reasoning_marker_count=_count_markers(query, REASONING_MARKERS),
        simple_marker_count=_count_markers(query, SIMPLE_MARKERS),
        has_code=1 if CODE_HINT.search(query) else 0,
        n_constraints=_count_constraints(query),
    )
