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
import os
import statistics
import sys
import time
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# RetryingClient below owns retries for the benchmark. With the SDK's own
# 2 retries on top, one failing query cost up to 12 requests (4 x 3).
# Must be set before router.config is imported.
os.environ.setdefault("PROVIDER_MAX_RETRIES", "0")

from judge import grade  # noqa: E402

from router.config import (  # noqa: E402
    ENABLE_CODE_TOOL,
    ENABLE_RISK_RULES,
    JUDGE,
    LARGE,
    PROVIDER_PRESET,
    RESULTS_DIR,
    SMALL,
)
from router.llm import LLMClient  # noqa: E402
from router.pipeline import RouterPipeline  # noqa: E402
from router.types import CallResult  # noqa: E402

DATASET = Path(__file__).parent / "dataset.jsonl"
ARMS = ["always_small", "always_large", "heuristic", "classifier"]
RETRY_DELAYS_S = (5, 15, 30)

# Daily quotas don't recover within a retry window (Gemini free tier:
# "GenerateRequestsPerDayPerProjectPerModel-FreeTier").
DAILY_QUOTA_MARKERS = ("PerDay", "per day", "daily")


class DailyQuotaExhausted(RuntimeError):
    pass


class RetryingClient(LLMClient):
    """Retries failed calls with backoff, for the benchmark only.

    Free tiers return transient 429/503s. Without retries, one "high demand"
    error would score as a wrong answer (or poison a reference) and the arms
    would differ by luck, not routing. The API path does not retry: there the
    escalation layer decides what a failure means.

    A daily quota is not transient, so it stops the run instead. Successful
    calls are cached, so re-running after the quota resets picks up where
    this run stopped.
    """

    def complete(self, *args, **kwargs) -> CallResult:
        result = super().complete(*args, **kwargs)
        for delay in RETRY_DELAYS_S:
            if not result.error:
                break
            if any(m in result.error for m in DAILY_QUOTA_MARKERS):
                raise DailyQuotaExhausted(
                    f"Daily quota exhausted for {result.model_id}. Completed calls "
                    "are cached; re-run the same command after the quota resets "
                    "to continue."
                )
            time.sleep(delay)
            result = super().complete(*args, **kwargs)
        return result


def load_dataset(limit: int | None = None) -> list[dict]:
    """Load the eval set; with --limit, interleave categories.

    The file is ordered by category (all 's' lookups first), so a plain
    head(10) would be ten easy questions and would never exercise routing.
    """
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if not limit:
        return rows
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["id"].rstrip("0123456789"), []).append(row)
    mixed = [r for batch in zip_longest(*groups.values()) for r in batch if r]
    return mixed[:limit]


def build_references(
    client: LLMClient, rows: list[dict], cached_only: bool = False
) -> dict[str, str]:
    """Reference answers from the large model. Cached, so this costs once.

    Rows whose reference could not be produced are dropped: grading against
    an empty reference would score every arm arbitrarily. With cached_only,
    rows without a cached reference are dropped without calling the model,
    so a run can finish when the large model's quota is gone.
    """
    refs = {}
    for i, row in enumerate(rows, 1):
        print(f"  reference {i}/{len(rows)} [{row['id']}]", end="\r", flush=True)
        if cached_only:
            result = client.cached(LARGE, row["query"])
            if result is None:
                print(f"\n  - skipping {row['id']}: no cached reference")
                continue
        else:
            result = client.complete(LARGE, row["query"])
        if result.error or not result.text.strip():
            print(f"\n  ! dropping {row['id']}: no reference ({result.error or 'empty'})")
            continue
        refs[row["id"]] = result.text
    print()
    return refs


def run_arm(
    arm: str,
    rows: list[dict],
    refs: dict[str, str],
    client: LLMClient,
    enable_escalation: bool,
    enable_risk_rules: bool = False,
    enable_code_tool: bool = False,
) -> dict:
    pipe = RouterPipeline(mode=arm, client=client, enable_escalation=enable_escalation,
                          enable_risk_rules=enable_risk_rules,
                          enable_code_tool=enable_code_tool)
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
            # Model latency as originally measured: the cache stores each
            # call's latency, while wall-clock time would read ~0 for every
            # arm that reuses cached answers.
            "latency_s": sum(c.latency_s for c in resp.calls),
            "n_calls": len(resp.calls),
            "risk_override": resp.risk_override,
            "tool": resp.tool,
            "failed": bool(resp.calls and resp.calls[-1].error),
        })
    print()

    n = len(records)
    graded = [r["score"] for r in records if r["score"] is not None]
    latencies = sorted(r["latency_s"] for r in records)
    routed_small = sum(1 for r in records if r["tier_chosen"] == "small")
    correct_route = sum(
        1 for r in records if r["tier_chosen"] == r["prior"]
    )

    return {
        "arm": arm,
        # Provenance, so any number quoted later can be traced to its models.
        "provider_preset": PROVIDER_PRESET,
        "models": {"small": SMALL.model_id, "large": LARGE.model_id, "judge": JUDGE.model_id},
        "query_ids": [r["id"] for r in records],
        "escalation_enabled": enable_escalation,
        "risk_rules_enabled": enable_risk_rules,
        "code_tool_enabled": enable_code_tool,
        "n": n,
        "quality": round(statistics.mean(graded) / 2, 4) if graded else None,
        "judge_errors": n - len(graded),
        "call_failures": sum(r["failed"] for r in records),
        "cost_per_100_usd": round(sum(r["cost_usd"] for r in records) / n * 100, 4),
        "p50_latency_s": round(statistics.median(latencies), 3),
        "p95_latency_s": round(latencies[int(0.95 * (n - 1))], 3),
        "pct_routed_small": round(100 * routed_small / n, 1),
        "escalation_rate": round(100 * sum(r["escalated"] for r in records) / n, 1),
        "routing_accuracy_vs_prior": round(100 * correct_route / n, 1),
        "records": records,
    }


def print_table(summaries: list[dict]) -> None:
    header = (f"{'Arm':<16}{'Quality':>9}{'$/100req':>11}{'p50 s':>8}{'%small':>9}"
              f"{'%escal':>9}{'RouteAcc':>10}{'Fail':>6}{'JudgeErr':>10}")
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        quality = s["quality"] if s["quality"] is not None else float("nan")
        print(
            f"{s['arm']:<16}{quality:>9.3f}{s['cost_per_100_usd']:>11.3f}"
            f"{s['p50_latency_s']:>8.2f}{s['pct_routed_small']:>9.1f}"
            f"{s['escalation_rate']:>9.1f}{s['routing_accuracy_vs_prior']:>10.1f}"
            f"{s['call_failures']:>6}{s['judge_errors']:>10}"
        )
    print()

    # Report every router against always-large, not just the classifier:
    # on a small sample the baseline can win, and that must be visible.
    base = next((s for s in summaries if s["arm"] == "always_large"), None)
    if not base or not base["cost_per_100_usd"] or not base["quality"]:
        return
    for s in summaries:
        if s["arm"] not in ("heuristic", "classifier") or s["quality"] is None:
            continue
        saving = 100 * (1 - s["cost_per_100_usd"] / base["cost_per_100_usd"])
        retention = 100 * s["quality"] / base["quality"]
        print(f"{s['arm']:<11} vs always-large: {saving:.0f}% cost reduction, "
              f"{retention:.0f}% quality retention (n={s['n']})")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-escalation", action="store_true")
    ap.add_argument("--risk-rules", action="store_true", default=ENABLE_RISK_RULES,
                    help="let the silent-failure risk rules override routing "
                         "(default: ENABLE_RISK_RULES, off)")
    ap.add_argument("--code-tool", action="store_true", default=ENABLE_CODE_TOOL,
                    help="answer exact-answer questions by computed expression "
                         "(default: ENABLE_CODE_TOOL)")
    ap.add_argument("--cached-only", action="store_true",
                    help="use only questions whose reference answer is already cached "
                         "(no new large-model reference calls)")
    ap.add_argument("--tag", default="main", help="suffix for the results filename")
    args = ap.parse_args()

    rows = load_dataset(args.limit)
    client = RetryingClient()
    print(f"Dataset: {len(rows)} queries ({', '.join(r['id'] for r in rows)})\n"
          "Building reference answers...")
    try:
        refs = build_references(client, rows, cached_only=args.cached_only)
        rows = [r for r in rows if r["id"] in refs]

        summaries = []
        for arm in args.arms:
            summaries.append(run_arm(arm, rows, refs, client, not args.no_escalation,
                                     args.risk_rules, args.code_tool))
    except DailyQuotaExhausted as exc:
        # No partial results file: a table built from some arms and not
        # others would invite exactly the comparison the benchmark prevents.
        print(f"\nStopped: {exc}")
        raise SystemExit(2) from None

    print_table(summaries)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"benchmark_{args.tag}.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
