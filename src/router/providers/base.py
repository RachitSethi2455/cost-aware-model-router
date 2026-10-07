"""Provider abstraction.

The router's value proposition — route cheap requests to a cheap model — is
provider-agnostic. Only the transport differs. This package isolates that
difference so the routing, escalation, and eval code never import a vendor SDK.

Two adapters cover the field:
  - AnthropicProvider    : Anthropic Messages API
  - OpenAICompatProvider : everything else (OpenAI, Groq, Gemini, DeepSeek,
                           Together, Fireworks, Mistral, xAI, OpenRouter,
                           Ollama, LM Studio) via /v1/chat/completions
"""

from __future__ import annotations

from typing import Protocol

from ..types import CallResult


class Provider(Protocol):
    """Anything that can turn a prompt into a CallResult with cost accounting."""

    def complete(
        self,
        model_id: str,
        prompt: str,
        system: str | None,
        max_tokens: int,
        history: list[dict] | None = None,
    ) -> tuple[str, int, int, str | None, str | None]:
        """Return (text, input_tokens, output_tokens, stop_reason, error).

        history holds earlier turns ({"role": "user"|"assistant", "content"})
        sent before `prompt`, the latest user message.

        Must not raise on API errors — return them in the error slot so the
        escalation layer can decide what to do.
        """
        ...


__all__ = ["Provider", "CallResult"]
