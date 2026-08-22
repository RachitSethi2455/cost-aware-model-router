"""Provider-agnostic client with caching and cost accounting.

Every call returns a CallResult carrying tokens, USD cost, and wall-clock
latency. Measuring at the call site rather than estimating later is what makes
the benchmark numbers defensible.

The vendor SDK lives behind router.providers, so switching from Anthropic to
OpenAI, Groq, Gemini, or a local Ollama model is a config change, not a code
change.
"""

from __future__ import annotations

import time

from .cache import ResponseCache
from .config import MAX_TOKENS, ModelSpec
from .providers import get_provider
from .types import CallResult


class LLMClient:
    def __init__(self, cache: ResponseCache | None = None):
        self.cache = cache or ResponseCache()

    def complete(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str | None = None,
        max_tokens: int = MAX_TOKENS,
    ) -> CallResult:
        # Cache key includes provider: the same model_id on two providers can
        # return different text, and conflating them would poison the results.
        cache_key = f"{spec.provider}||{system or ''}||{prompt}||{max_tokens}"
        hit = self.cache.get(spec.model_id, cache_key)
        if hit:
            return CallResult(**{**hit, "cached": True})

        provider = get_provider(spec.provider)

        start = time.perf_counter()
        text, in_tok, out_tok, stop_reason, error = provider.complete(
            spec.model_id, prompt, system, max_tokens
        )
        latency = time.perf_counter() - start

        result = CallResult(
            text=text,
            model_id=spec.model_id,
            tier=spec.name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=spec.cost(in_tok, out_tok),
            latency_s=latency,
            stop_reason=stop_reason,
            error=error,
            meta={"provider": spec.provider},
        )

        # Never cache failures — a transient rate limit would poison the run.
        if not error:
            payload = result.__dict__.copy()
            payload.pop("cached", None)
            self.cache.set(spec.model_id, cache_key, payload)
        return result
