"""Render the cost/quality frontier chart for the README.

Usage:
    python scripts/make_chart.py results/benchmark_main.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "always_small": "All small (Haiku 4.5)",
    "always_large": "All large (Opus 5)",
    "heuristic": "Heuristic router",
    "classifier": "Classifier router",
}
COLORS = {
    "always_small": "#9aa0a6",
    "always_large": "#c5221f",
    "heuristic": "#1a73e8",
    "classifier": "#188038",
}


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_main.json")
    summaries = json.loads(path.read_text())

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s in summaries:
        arm = s["arm"]
        ax.scatter(
            s["cost_per_100_usd"], s["quality"],
            s=190, color=COLORS.get(arm, "#5f6368"),
            edgecolor="white", linewidth=1.5, zorder=3,
            label=LABELS.get(arm, arm),
        )
        ax.annotate(
            f"  {LABELS.get(arm, arm)}",
            (s["cost_per_100_usd"], s["quality"]),
            fontsize=9, va="center", color="#3c4043",
        )

    ax.set_xlabel("Cost per 100 requests (USD)", fontsize=11)
    ax.set_ylabel("Quality (judge score, normalized 0-1)", fontsize=11)
    ax.set_title("Cost / quality frontier across routing strategies", fontsize=13, pad=14)
    ax.grid(alpha=0.25, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.22, y=0.15)

    out = Path("results/cost_quality.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
