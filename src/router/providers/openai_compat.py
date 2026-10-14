"""OpenAI-compatible provider.

Works with any endpoint exposing /v1/chat/completions. Swap `base_url` and
`api_key_env`; nothing else changes.

    Provider     base_url                                          key env
    ---------    ----------------------------------------------    ------------------
    OpenAI       https://api.openai.com/v1                         OPENAI_API_KEY
    Groq         https://api.groq.com/openai/v1                    GROQ_API_KEY
    Gemini       https://generativelanguage.googleapis.com/v1beta/openai/
                                                                   GEMINI_API_KEY
    DeepSeek     https://api.deepseek.com/v1                       DEEPSEEK_API_KEY
    Together     https://api.together.xyz/v1                       TOGETHER_API_KEY
    Fireworks    https://api.fireworks.ai/inference/v1             FIREWORKS_API_KEY
    Mistral      https://api.mistral.ai/v1                         MISTRAL_API_KEY
    xAI          https://api.x.ai/v1                               XAI_API_KEY
    OpenRouter   https://openrouter.ai/api/v1                      OPENROUTER_API_KEY
    Ollama       http://localhost:11434/v1                         (any string)
    LM Studio    http://localhost:1234/v1                          (any string)

Install once: pip install openai
"""

from __future__ import annotations

import os

from ..config import PROVIDER_MAX_RETRIES


class OpenAICompatProvider:
    def __init__(self, base_url: str, api_key_env: str, extra_headers: dict | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install openai") from exc

        api_key = os.environ.get(api_key_env)
        if not api_key and base_url.startswith(("http://localhost", "http://127.0.0.1")):
            api_key = "local"
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set. Add it to your .env file."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=PROVIDER_MAX_RETRIES)
        self.extra_headers = extra_headers or {}

    @staticmethod
    def _messages(prompt, system, history) -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history or [])
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _stop_reason(finish: str | None) -> str | None:
        """Normalize finish_reason to the vocabulary escalation.py expects."""
        return {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}.get(finish, finish)

    @staticmethod
    def _tokens(usage, prompt: str, text: str) -> tuple[int, int]:
        # Some providers omit usage; fall back to a rough char/4 estimate so
        # cost accounting degrades gracefully instead of silently reporting $0.
        if not usage:
            return len(prompt) // 4, len(text) // 4
        get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)
        in_tok, out_tok = get("prompt_tokens") or 0, get("completion_tokens") or 0
        # Some providers (Gemini) report hidden reasoning tokens only in
        # total_tokens. They are billed as output, so count them; otherwise
        # a thinking model looks ~25x cheaper than it is.
        hidden = (get("total_tokens") or 0) - in_tok - out_tok
        return in_tok, out_tok + max(hidden, 0)

    def complete(self, model_id, prompt, system, max_tokens, history=None):
        try:
            resp = self.client.chat.completions.create(
                model=model_id,
                messages=self._messages(prompt, system, history),
                max_tokens=max_tokens,
                extra_headers=self.extra_headers or None,
            )
        except Exception as exc:  # noqa: BLE001 — surface, never crash the pipeline
            return "", 0, 0, None, f"{type(exc).__name__}: {exc}"

        choice = resp.choices[0]
        text = choice.message.content or ""
        in_tok, out_tok = self._tokens(resp.usage, prompt, text)
        return text, in_tok, out_tok, self._stop_reason(choice.finish_reason), None

    def stream(self, model_id, prompt, system, max_tokens, history=None):
        """Yield the answer's text as it arrives, then return the same tuple as
        complete(). Usage comes in the last chunk (include_usage), or in
        x_groq.usage on Groq."""
        parts, finish, usage = [], None, None
        try:
            chunks = self.client.chat.completions.create(
                model=model_id,
                messages=self._messages(prompt, system, history),
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=self.extra_headers or None,
            )
            for chunk in chunks:
                groq = (chunk.model_extra or {}).get("x_groq") or {}
                usage = chunk.usage or (groq.get("usage") if isinstance(groq, dict) else None) or usage
                for choice in chunk.choices:
                    if piece := choice.delta.content:
                        parts.append(piece)
                        yield piece
                    finish = choice.finish_reason or finish
        except Exception as exc:  # noqa: BLE001 — surface, never crash the pipeline
            return "".join(parts), 0, 0, None, f"{type(exc).__name__}: {exc}"
        text = "".join(parts)
        in_tok, out_tok = self._tokens(usage, prompt, text)
        return text, in_tok, out_tok, self._stop_reason(finish), None
