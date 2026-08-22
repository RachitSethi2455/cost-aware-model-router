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


class OpenAICompatProvider:
    def __init__(self, base_url: str, api_key_env: str, extra_headers: dict | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install openai") from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set. Add it to your .env file."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.extra_headers = extra_headers or {}

    def complete(self, model_id, prompt, system, max_tokens):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                extra_headers=self.extra_headers or None,
            )
        except Exception as exc:  # noqa: BLE001 — surface, never crash the pipeline
            return "", 0, 0, None, f"{type(exc).__name__}: {exc}"

        choice = resp.choices[0]
        text = choice.message.content or ""

        # Normalize finish_reason to the vocabulary escalation.py expects.
        finish = choice.finish_reason
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "refusal",
        }.get(finish, finish)

        usage = resp.usage
        # Some providers omit usage; fall back to a rough char/4 estimate so
        # cost accounting degrades gracefully instead of silently reporting $0.
        if usage:
            in_tok, out_tok = usage.prompt_tokens, usage.completion_tokens
        else:
            in_tok = len(prompt) // 4
            out_tok = len(text) // 4

        return text, in_tok, out_tok, stop_reason, None
