"""Shared data types with no third-party dependencies.

Deliberately separate from llm.py so that the deterministic logic
(escalation checks, feature extraction, routing) can be imported and tested
without the Anthropic SDK installed or an API key present. CI runs the full
offline suite on every push because of this split.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CallResult:
    """The outcome of a single model call, with full cost accounting."""

    text: str
    model_id: str
    tier: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool = False
    stop_reason: str | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)
