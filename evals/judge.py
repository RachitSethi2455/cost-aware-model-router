"""LLM-as-judge scoring.

Method: reference-based grading. The large model's answer is treated as the
reference, and each candidate answer is graded against it on a 0-2 scale.

Known limitation, documented in the README: treating the large model's output as ground truth
caps measured quality at 1.00 for the all-large arm by construction. It
measures *quality retention under routing*, not absolute correctness. A human-
labeled subset would be needed for the latter.
"""

from __future__ import annotations

import json
import re

from router.config import JUDGE
from router.llm import LLMClient

JUDGE_SYSTEM = """You are a strict grader comparing a candidate answer to a reference answer.

Score the candidate on a 0-2 scale:
2 = Matches the reference in correctness and completeness. Minor wording or style differences are fine.
1 = Partially correct. Right direction but missing a key requirement, or contains a non-fatal error.
0 = Wrong, off-topic, refuses, or omits something the question explicitly required.

Judge substance, not length or eloquence. A short correct answer scores 2.

Respond with ONLY a JSON object, no markdown fences, no preamble:
{"score": <0|1|2>, "reason": "<one short sentence>"}"""

JUDGE_TEMPLATE = """QUESTION:
{query}

REFERENCE ANSWER:
{reference}

CANDIDATE ANSWER:
{candidate}"""


def _parse(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(cleaned)
        score = int(obj.get("score", 0))
        return {"score": max(0, min(2, score)), "reason": str(obj.get("reason", ""))[:200]}
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fall back to the first standalone digit 0-2 in the response.
        match = re.search(r"\b([012])\b", cleaned)
        return {
            "score": int(match.group(1)) if match else 0,
            "reason": "unparseable_judge_output",
        }


def grade(client: LLMClient, query: str, reference: str, candidate: str) -> dict:
    """Return {'score': 0|1|2|None, 'reason': str, 'cost_usd': float}.

    score is None when the judge call itself failed.
    """
    if not candidate.strip():
        return {"score": 0, "reason": "empty_candidate", "cost_usd": 0.0}
    if candidate.strip() == reference.strip():
        return {"score": 2, "reason": "identical_to_reference", "cost_usd": 0.0}

    # Thinking judges (e.g. gemini-3.8-flash) spend hidden reasoning tokens
    # from this budget; at 200 the verdict came back empty.
    result = client.complete(
        JUDGE,
        JUDGE_TEMPLATE.format(query=query, reference=reference, candidate=candidate),
        system=JUDGE_SYSTEM,
        max_tokens=2048,
    )
    if result.error:
        # A failed judge call is not evidence the answer was bad; the
        # benchmark excludes these from the quality mean and reports them.
        return {"score": None, "reason": f"judge_error: {result.error[:120]}",
                "cost_usd": 0.0}
    parsed = _parse(result.text)
    parsed["cost_usd"] = result.cost_usd
    return parsed
