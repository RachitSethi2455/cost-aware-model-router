"""Four-arm benchmark: all-small / all-large / heuristic / classifier.

Every arm runs through the same RouterPipeline so the comparison is not
confounded by code-path differences.

Usage:
    python evals/run_benchmark.py                    # all arms
    python evals/run_benchmark.py --arms always_small classifier
    python evals/run_benchmark.py --no-escalation    # ablation
    python evals/run_benchmark.py --limit 10         # smoke test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge import grade  # noqa: E402
from router.config import LARGE, RESULTS_DIR  # noqa: E402
from router.llm import LLMClient  # noqa: E402
from router.pipeline import RouterPipeline  # noqa: E402

DATASET = Path(__file__).parent / "dataset.jsonl"
ARMS = ["always_small", "always_large", "heuristic", "classifier"]


def load_dataset(limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def build_references(client: LLMClient, rows: list[dict]) -> dict[str, str]:
    """Reference answers from the large model. Cached, so this costs once."""
    refs = {}
    for i, row in enumerate(rows, 1):
        print(f"  reference {i}/{len(rows)} [{row['id']}]", end="\r", flush=True)
        refs[row["id"]] = client.complete(LARGE, row["query"]).text
    print()
    return refs


def run_arm(
    arm: str,
    rows: list[dict],
    refs: dict[str, str],
    client: LLMClient,
    enable_escalation: bool,
) -> dict:
    pipe = RouterPipeline(mode=arm, client=client, enable_escalation=enable_escalation)
    records = []

    for i, row in enumerate(rows, 1):
        print(f"  [{arm}] {i}/{len(rows)} {row['id']}   ", end="\r", flush=True)
        resp = pipe.run(row["query"])
        verdict = grade(client, row["query"], refs[row["id"]], resp.answer)
        records.append({
            "id": row["id"],
            "category": row["category"],
            "prior": row["prior"],
            "tier_chosen": resp.tier_chosen,
            "tier_served": resp.tier_served,
            "escalated": resp.escalated,
            "escalation_reason": resp.escalation_reason,
            "confidence": round(resp.router_confidence, 4),
            "score": verdict["score"],
            "judge_reason": verdict["reason"],
            "cost_usd": resp.total_cost_usd,
            "latency_s": resp.total_latency_s,
            "n_calls": len(resp.calls),
        })
    print()

    n = len(records)
    latencies = sorted(r["latency_s"] for r in records)
    routed_small = sum(1 for r in records if r["tier_chosen"] == "small")
    correct_route = sum(
        1 for r in records if r["tier_chosen"] == r["prior"]
    )

    return {
        "arm": arm,
        "escalation_enabled": enable_escalation,
        "n": n,
        "quality": round(statistics.mean(r["score"] for r in records) / 2, 4),
        "cost_per_100_usd": round(sum(r["cost_usd"] for r in records) / n * 100, 4),
        "p50_latency_s": round(statistics.median(latencies), 3),
        "p95_latency_s": round(latencies[int(0.95 * (n - 1))], 3),
        "pct_routed_small": round(100 * routed_small / n, 1),
        "escalation_rate": round(100 * sum(r["escalated"] for r in records) / n, 1),
        "routing_accuracy_vs_prior": round(100 * correct_route / n, 1),
        "records": records,
    }


def print_table(summaries: list[dict]) -> None:
    header = f"{'Arm':<16}{'Quality':>9}{'$/100req':>11}{'p50 s':>8}{'%small':>9}{'%escal':>9}{'RouteAcc':>10}"
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['arm']:<16}{s['quality']:>9.3f}{s['cost_per_100_usd']:>11.3f}"
            f"{s['p50_latency_s']:>8.2f}{s['pct_routed_small']:>9.1f}"
            f"{s['escalation_rate']:>9.1f}{s['routing_accuracy_vs_prior']:>10.1f}"
        )
    print()

    base = next((s for s in summaries if s["arm"] == "always_large"), None)
    best = next((s for s in summaries if s["arm"] == "classifier"), None)
    if base and best and base["cost_per_100_usd"]:
        saving = 100 * (1 - best["cost_per_100_usd"] / base["cost_per_100_usd"])
        retention = 100 * best["quality"] / base["quality"] if base["quality"] else 0
        print(f"Headline: {saving:.0f}% cost reduction, {retention:.0f}% quality retention "
              f"vs always-large.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-escalation", action="store_true")
    ap.add_argument("--tag", default="main", help="suffix for the results filename")
    args = ap.parse_args()

    rows = load_dataset(args.limit)
    client = LLMClient()
    print(f"Dataset: {len(rows)} queries\nBuilding reference answers...")
    refs = build_references(client, rows)

    summaries = []
    for arm in args.arms:
        summaries.append(run_arm(arm, rows, refs, client, not args.no_escalation))

    print_table(summaries)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"benchmark_{args.tag}.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
