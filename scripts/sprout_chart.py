"""Render the SPROUT cost-quality curves for the README.

Reads results/sprout_replay_{pair}.json (written by evals/sprout_replay.py)
and draws PGR against the share of queries sent to the large model, one
panel per model pair, with the random router as the diagonal.

Usage:
    python scripts/sprout_chart.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {"llama": "Llama 3.1 8B → 70B", "gpt4o": "GPT-4o-mini → GPT-4o"}
LINES = {  # router: (legend label, colour, line style)
    "minilm_knn": ("MiniLM kNN (gain)", "#188038", "-"),
    "tfidf_ridge": ("TF-IDF ridge (gain)", "#1a73e8", "-"),
    "seven_features_gbm": ("7 features, boosted trees (gain)", "#e37400", "-"),
    "length": ("Prompt length only", "#9aa0a6", "--"),
    "label_router": ("First router (60 hand labels)", "#c5221f", "-."),
    "outcome_router": ("Shipped router, v2 features (outcomes, 6 pairs)", "#9334e6", "-"),
    "outcome_router_v1": ("Outcome router, v1 features", "#9334e6", ":"),
}


def main() -> None:
    available = [p for p in PAIRS if (ROOT / "results" / f"sprout_replay_{p}.json").exists()]
    fig, axes = plt.subplots(1, len(available), figsize=(6.2 * len(available), 4.8), squeeze=False)
    for ax, pair in zip(axes[0], available, strict=True):
        res = json.loads((ROOT / "results" / f"sprout_replay_{pair}.json").read_text())
        grid = [i / 100 for i in range(101)]
        ax.plot([0, 1], [0, 1], color="#5f6368", lw=1, label="Random (APGR 0.50)")
        for name, (label, colour, style) in LINES.items():
            row = res["routers"][name]
            ax.plot(grid, row["pgr_curve"], color=colour, ls=style, lw=1.8,
                    label=f"{label}, APGR {row['apgr']:.2f}")
        oracle = res["oracle"]
        ax.scatter([oracle["cpt80"]], [0.8], marker="*", s=140, color="#202124", zorder=4,
                   label=f"Oracle reaches 80% at {oracle['cpt80']:.0%}")
        ax.axhline(0.8, color="#dadce0", lw=0.8, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel("Share of queries sent to the large model (cost)")
        ax.set_ylabel("Performance gap recovered (PGR)")
        ax.set_title(f"{PAIRS[pair]}, {res['n_test']:,} held-out prompts", fontsize=11)
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.suptitle("SPROUT offline replay: cost-quality curves (higher and further left is better)",
                 fontsize=12)
    fig.tight_layout()
    out = ROOT / "results" / "sprout_pgr.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
