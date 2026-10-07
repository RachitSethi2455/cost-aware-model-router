"""Anthropic Messages API adapter."""

from __future__ import annotations

import os

from ..config import PROVIDER_MAX_RETRIES


class AnthropicProvider:
    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY"):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install anthropic") from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set. Add it to your .env file.")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=PROVIDER_MAX_RETRIES)

    def complete(self, model_id, prompt, system, max_tokens, history=None):
        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [*(history or []), {"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        try:
            resp = self.client.messages.create(**kwargs)
        except self._anthropic.APIError as exc:
            return "", 0, 0, None, f"{type(exc).__name__}: {exc}"

        # Safety classifiers can return stop_reason "refusal"; check before
        # reading content blocks.
        if resp.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            )

        return (
            text,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            resp.stop_reason,
            None,
        )
