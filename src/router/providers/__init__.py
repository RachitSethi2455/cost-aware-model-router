"""Provider adapters and the registry that builds them from config."""

from __future__ import annotations

from .base import Provider

_CACHE: dict[str, Provider] = {}

# base_url + key env for every OpenAI-compatible vendor.
OPENAI_COMPAT_ENDPOINTS = {
    "openai":     ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "deepseek":   ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "together":   ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks":  ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "mistral":    ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "xai":        ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama":     ("http://localhost:11434/v1", "OLLAMA_API_KEY"),
    "lmstudio":   ("http://localhost:1234/v1", "LMSTUDIO_API_KEY"),
}


def get_provider(name: str) -> Provider:
    """Build (and memoize) a provider adapter by name."""
    if name in _CACHE:
        return _CACHE[name]

    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        provider = AnthropicProvider()
    elif name in OPENAI_COMPAT_ENDPOINTS:
        from .openai_compat import OpenAICompatProvider
        base_url, key_env = OPENAI_COMPAT_ENDPOINTS[name]
        provider = OpenAICompatProvider(base_url=base_url, api_key_env=key_env)
    else:
        raise ValueError(
            f"Unknown provider '{name}'. Known: anthropic, "
            + ", ".join(OPENAI_COMPAT_ENDPOINTS)
        )

    _CACHE[name] = provider
    return provider


__all__ = ["Provider", "get_provider", "OPENAI_COMPAT_ENDPOINTS"]
