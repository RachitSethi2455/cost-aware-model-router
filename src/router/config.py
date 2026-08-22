"""Central configuration: model tiers, pricing, and tunable thresholds.

Pricing verified against Anthropic's published rates (USD per million tokens).
Keep this file as the single source of truth so the benchmark and the API
never disagree about what a request cost.
"""

from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_PATH = DATA_DIR / "response_cache.sqlite"
MODEL_ARTIFACT = DATA_DIR / "router_clf.joblib"


@dataclass(frozen=True)
class ModelSpec:
    """A single model tier: which provider serves it and what it costs."""

    name: str
    model_id: str
    input_per_mtok: float   # USD per 1M input tokens
    output_per_mtok: float  # USD per 1M output tokens
    provider: str = "anthropic"

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Exact USD cost of one call."""
        return (
            input_tokens / 1_000_000 * self.input_per_mtok
            + output_tokens / 1_000_000 * self.output_per_mtok
        )


# ---- Tier presets ---------------------------------------------------------
# Pick one with PROVIDER_PRESET in .env, or define your own below.
#
# IMPORTANT: the per-MTok rates below are what every cost number in this
# project is computed from. Copy the exact current rates off your provider's
# pricing page before you run the benchmark. A wrong rate here silently
# invalidates your headline result.

PRESETS: dict[str, dict[str, ModelSpec]] = {
    "anthropic": {
        "small": ModelSpec("small", "claude-haiku-4-5-20251001", 1.00, 5.00, "anthropic"),
        "large": ModelSpec("large", "claude-opus-5", 5.00, 25.00, "anthropic"),
        "judge": ModelSpec("judge", "claude-sonnet-5", 2.00, 10.00, "anthropic"),
    },
    "openai": {
        "small": ModelSpec("small", "gpt-4o-mini", 0.15, 0.60, "openai"),
        "large": ModelSpec("large", "gpt-4o", 2.50, 10.00, "openai"),
        "judge": ModelSpec("judge", "gpt-4o-mini", 0.15, 0.60, "openai"),
    },
    "gemini": {
        "small": ModelSpec("small", "gemini-2.0-flash", 0.10, 0.40, "gemini"),
        "large": ModelSpec("large", "gemini-2.5-pro", 1.25, 10.00, "gemini"),
        "judge": ModelSpec("judge", "gemini-2.0-flash", 0.10, 0.40, "gemini"),
    },
    "groq": {
        "small": ModelSpec("small", "llama-3.1-8b-instant", 0.05, 0.08, "groq"),
        "large": ModelSpec("large", "llama-3.3-70b-versatile", 0.59, 0.79, "groq"),
        "judge": ModelSpec("judge", "llama-3.3-70b-versatile", 0.59, 0.79, "groq"),
    },
    "deepseek": {
        "small": ModelSpec("small", "deepseek-chat", 0.27, 1.10, "deepseek"),
        "large": ModelSpec("large", "deepseek-reasoner", 0.55, 2.19, "deepseek"),
        "judge": ModelSpec("judge", "deepseek-chat", 0.27, 1.10, "deepseek"),
    },
    "openrouter": {
        "small": ModelSpec("small", "meta-llama/llama-3.1-8b-instruct", 0.02, 0.05, "openrouter"),
        "large": ModelSpec("large", "anthropic/claude-sonnet-4.5", 3.00, 15.00, "openrouter"),
        "judge": ModelSpec("judge", "meta-llama/llama-3.3-70b-instruct", 0.12, 0.30, "openrouter"),
    },
    "local": {  # Ollama / LM Studio — free, so cost deltas come out as zero
        "small": ModelSpec("small", "llama3.2:3b", 0.0, 0.0, "ollama"),
        "large": ModelSpec("large", "llama3.1:8b", 0.0, 0.0, "ollama"),
        "judge": ModelSpec("judge", "llama3.1:8b", 0.0, 0.0, "ollama"),
    },
}

PROVIDER_PRESET = os.getenv("PROVIDER_PRESET", "anthropic")
if PROVIDER_PRESET not in PRESETS:
    raise ValueError(
        f"Unknown PROVIDER_PRESET '{PROVIDER_PRESET}'. Options: {list(PRESETS)}"
    )

_active = PRESETS[PROVIDER_PRESET]
SMALL = _active["small"]
LARGE = _active["large"]
JUDGE = _active["judge"]

TIERS = {"small": SMALL, "large": LARGE}

# ---- Tunables -------------------------------------------------------------
# Router decision threshold. P(complex) above this routes to LARGE.
ROUTE_THRESHOLD = float(os.getenv("ROUTE_THRESHOLD", "0.50"))

# Escalation guards applied to the SMALL model's output.
MIN_ANSWER_CHARS = int(os.getenv("MIN_ANSWER_CHARS", "20"))
MAX_ESCALATIONS = 1  # one retry on the large model, never a loop

# Generation settings, held constant across arms so the benchmark is fair.
MAX_TOKENS = 1024

