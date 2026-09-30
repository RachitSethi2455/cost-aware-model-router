"""Export real routed answers from the response cache for the demo page.

The public demo never calls a model, so it shows answers recorded from live
runs instead. Every field comes from the cache entry the router wrote at the
time: nothing here is typed in by hand.

Usage:
    python scripts/export_demo_examples.py            # after running the queries live
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router.cache import ResponseCache  # noqa: E402
from router.config import LARGE, MAX_TOKENS, PROVIDER_PRESET, SMALL  # noqa: E402

OUT = ROOT / "src" / "api" / "static" / "examples.json"

# (query, tier that served it in the live run)
EXAMPLES = [
    ("What is the capital of Japan?", "small"),
    ("Explain why the CAP theorem is often oversimplified, and compare it with PACELC.", "large"),
    ("How many r's are in strawberry?", "small"),
    ("Write a Python function to merge two sorted lists.", "small"),
]


def main() -> None:
    cache = ResponseCache()
    tiers = {"small": SMALL, "large": LARGE}
    out = []
    for query, tier in EXAMPLES:
        spec = tiers[tier]
        # Same key LLMClient.complete uses (no system prompt, default budget).
        hit = cache.get(spec.model_id, f"{spec.provider}||||{query}||{MAX_TOKENS}")
        if not hit:
            raise SystemExit(f"Not in cache: {query!r} on {spec.model_id}. Run it live first.")
        out.append({
            "query": query,
            "tier": tier,
            "provider": PROVIDER_PRESET,
            "model_id": hit["model_id"],
            "cost_usd": hit["cost_usd"],
            "latency_s": round(hit["latency_s"], 1),
            "output_tokens": hit["output_tokens"],
            "answer": hit["text"],
        })
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)} examples -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
