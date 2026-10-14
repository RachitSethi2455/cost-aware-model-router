"""Query feature extraction.

Cheap, interpretable features. Interpretability matters here: when the
router misroutes, you want to be able to explain *why*, and an additive
model over named features gives each one an exact contribution.

Two feature sets:
- v1 (`extract`, seven features): used by the heuristic and the first
  router, and kept unchanged so their documented results stay reproducible.
- v2 (`extract_v2`, nine features): used by the outcome router. The live
  test showed v1 reads formatting as difficulty: every line break counted as
  a sentence, so a multiple-choice question with its options on separate
  lines looked like a long multi-part task, and LaTeX braces counted as code.
  v2 counts sentences by punctuation only, skips option lines, ignores LaTeX
  when looking for code, and says what kind of question it is instead
  (multiple choice, maths).

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

# A request *for* code, as opposed to code pasted *into* the query (CODE_HINT).
# "Write a Python function to ..." has no code in it, so has_code alone
# routed it to the small model.
CODE_REQUEST = re.compile(
    r"\b(write|implement|build|create|refactor|fix|debug)\b[^.?!\n]{0,40}"
    r"\b(function|class|query|script|endpoint|program|code|api|cache|sql|python)\b",
    re.IGNORECASE,
)

FEATURE_NAMES = [
    "n_words",
    "n_sentences",
    "reasoning_marker_count",
    "simple_marker_count",
    "has_code",
    "n_constraints",
    "asks_for_code",
]


@dataclass
class QueryFeatures:
    n_words: int
    n_sentences: int
    reasoning_marker_count: int
    simple_marker_count: int
    has_code: int
    n_constraints: int
    asks_for_code: int

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
        asks_for_code=1 if CODE_REQUEST.search(query) else 0,
    )


# ── v2: formatting-proof features for the outcome router ────────────────────

FEATURE_NAMES_V2 = FEATURE_NAMES + ["is_multiple_choice", "has_math"]

# An answer option at the start of a line: "(A): text", "A) text", "A. text".
OPTION_LINE = re.compile(r"^\s*\(?([A-J])\)?\s*[:.)]\s+\S", re.MULTILINE)
# "(A)" written inline, for options run together on one line.
INLINE_OPTION = re.compile(r"\(([A-J])\)")
INLINE_OPTION_RUN = re.compile(r"\([A-J]\)[^()\n]*")
CODE_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
LATEX = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
# One line is enough for these; looser code-like lines need two.
STRONG_CODE = re.compile(r"^\s*(?:def|class) \w+|\bSELECT\b.+\bFROM\b|^\s*#include\b", re.MULTILINE)
CODE_LINE = re.compile(
    r"^\s*(?:import \w|from [\w.]+ import |public |private |function \w*\s*\(|const \w+ =|let \w+ =|"
    r"var \w+ =|for\s*\(|while\s*\(|if\s*\(.*\)\s*\{)|;\s*$|\{\s*$",
    re.MULTILINE)
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE)
MATH_HINT = re.compile(
    r"\$[^$\n]+\$|\\(?:frac|dfrac|sqrt|cdot|times|pi|sum|int|log|sin|cos|tan|binom|le|ge)\b"
    r"|\d\s*[\^*/=]\s*-?\d|\b\d+\s*[+\-]\s*\d+\s*=")
SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
CONJUNCTION = re.compile(r"\b(and also|as well as|additionally|make sure|ensure that)\b", re.IGNORECASE)


@dataclass
class QueryFeaturesV2(QueryFeatures):
    is_multiple_choice: int = 0
    has_math: int = 0

    def to_vector(self) -> list[float]:
        d = asdict(self)
        return [float(d[name]) for name in FEATURE_NAMES_V2]


def _prose(query: str) -> str:
    """The question's prose: no code blocks, LaTeX as a placeholder word (so
    "x = 1." keeps its sentence), and no answer options, on their own lines
    or run together inline."""
    text = LATEX.sub(" x ", CODE_FENCE.sub(" ", query))
    text = " ".join(line for line in text.splitlines() if not OPTION_LINE.match(line))
    if {"A", "B", "C"} <= set(INLINE_OPTION.findall(text)):
        text = INLINE_OPTION_RUN.sub(" ", text)
    return text


def extract_v2(query: str) -> QueryFeaturesV2:
    """Like extract(), but blind to line breaks and LaTeX, and naming the kind
    of question: the same question formatted differently gets the same
    features (tested in tests/test_router.py)."""
    no_latex = LATEX.sub(" ", query)
    letters = set(OPTION_LINE.findall(query)) | set(INLINE_OPTION.findall(query))
    sentences = [s for s in SENTENCE_END.split(_prose(query)) if re.search(r"\w", s)]
    return QueryFeaturesV2(
        n_words=len(query.split()),
        n_sentences=max(1, len(sentences)),
        reasoning_marker_count=_count_markers(query, REASONING_MARKERS),
        simple_marker_count=_count_markers(query, SIMPLE_MARKERS),
        has_code=1 if ("```" in query or STRONG_CODE.search(no_latex)
                       or len(CODE_LINE.findall(no_latex)) >= 2) else 0,
        n_constraints=len(BULLET.findall(query)) + len(CONJUNCTION.findall(query)),
        asks_for_code=1 if CODE_REQUEST.search(query) else 0,
        is_multiple_choice=1 if {"A", "B", "C"} <= letters else 0,
        has_math=1 if MATH_HINT.search(query) else 0,
    )


FEATURE_SETS = {"v1": (FEATURE_NAMES, extract), "v2": (FEATURE_NAMES_V2, extract_v2)}
