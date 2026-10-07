"""Provider-agnostic client with caching and cost accounting.

Every call returns a CallResult carrying tokens, USD cost, and wall-clock
latency. Measuring at the call site rather than estimating later is what makes
the benchmark numbers defensible.

The vendor SDK lives behind router.providers, so switching from Anthropic to
OpenAI, Groq, Gemini, or a local Ollama model is a config change, not a code
change.
"""

from __future__ import annotations

import json
import time

from .cache import ResponseCache
from .config import MAX_TOKENS, ModelSpec
from .providers import get_provider
from .types import CallResult


class LLMClient:
    def __init__(self, cache: ResponseCache | None = None):
        self.cache = cache or ResponseCache()

    @staticmethod
    def _cache_key(
        spec: ModelSpec,
        prompt: str,
        system: str | None,
        max_tokens: int,
        history: list[dict] | None = None,
    ) -> str:
        # Includes provider: the same model_id on two providers can return
        # different text, and conflating them would poison the results.
        key = f"{spec.provider}||{system or ''}||{prompt}||{max_tokens}"
        # Single-turn keys keep their original form so existing caches stay valid.
        if history:
            key += "||" + json.dumps(history, sort_keys=True, ensure_ascii=False)
        return key

    def cached(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str | None = None,
        max_tokens: int = MAX_TOKENS,
        history: list[dict] | None = None,
    ) -> CallResult | None:
        """The cached result for this call, or None. Never calls the provider."""
        key = self._cache_key(spec, prompt, system, max_tokens, history)
        hit = self.cache.get(spec.model_id, key)
        return CallResult(**{**hit, "cached": True}) if hit else None

    def complete(
        self,
        spec: ModelSpec,
        prompt: str,
        system: str | None = None,
        max_tokens: int = MAX_TOKENS,
        history: list[dict] | None = None,
    ) -> CallResult:
        """history: earlier turns as [{"role": "user"|"assistant", "content": str}],
        sent before `prompt`, which is the latest user message."""
        cache_key = self._cache_key(spec, prompt, system, max_tokens, history)
        hit = self.cached(spec, prompt, system, max_tokens, history)
        if hit:
            return hit

        provider = get_provider(spec.provider)

        start = time.perf_counter()
        text, in_tok, out_tok, stop_reason, error = provider.complete(
            spec.model_id, prompt, system, max_tokens, history=history
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
