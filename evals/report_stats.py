"""Uncertainty and standard routing metrics for the committed results.

Every headline number in the README should carry its sample size and a 95%
interval, and router results should be read against the two references the
routing literature uses (RouteLLM, arXiv 2406.18665):

- PGR, performance gap recovered: (q_router - q_small) / (q_large - q_small),
  the share of the small-to-large quality gap the router closes.
- A random router at matched cost: sending a random fraction c of queries to
  the large model scores q_small + c * (q_large - q_small) in expectation,
  i.e. PGR = c. A router is only useful if it beats that line.

Intervals for benchmark metrics come from a paired bootstrap over questions
(every arm resampled with the same question ids); proportions use the Wilson
interval. No API calls: this only reads results/*.json.

Usage:
    python evals/report_stats.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
BOOTSTRAP_RESAMPLES = 10_000
SEED = 0


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _arm_stats(arms: dict, ids: list[str]) -> dict:
    """Metrics for one (re)sample of question ids."""
    def quality(arm):
        return sum(arms[arm][i]["score"] for i in ids) / (2 * len(ids))

    def cost(arm):
        return sum(arms[arm][i]["cost_usd"] for i in ids) / len(ids) * 100

    def large_share(arm):
        return sum(arms[arm][i]["tier_chosen"] == "large" for i in ids) / len(ids)

    q_small, q_large = quality("always_small"), quality("always_large")
    gap = q_large - q_small
    out = {}
    for arm in ("heuristic", "classifier"):
        q, c = quality(arm), large_share(arm)
        q_random = q_small + c * gap
        out[arm] = {
            "quality": q,
            "retention": q / q_large if q_large else float("nan"),
            "cost_reduction": 1 - cost(arm) / cost("always_large"),
            "large_share": c,
            "random_quality": q_random,
            "gain_over_random": q - q_random,
            # Undefined when the small model never falls short in a resample.
            "pgr": (q - q_small) / gap if gap > 0 else None,
        }
    out["always_small"] = {"quality": q_small}
    return out


def _interval(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def benchmark_report(path: Path) -> str:
    summaries = json.loads(path.read_text(encoding="utf-8"))
    arms = {s["arm"]: {r["id"]: r for r in s["records"]} for s in summaries}
    ids = list(arms["always_small"])
    point = _arm_stats(arms, ids)

    rng = random.Random(SEED)
    draws = [_arm_stats(arms, [rng.choice(ids) for _ in ids]) for _ in range(BOOTSTRAP_RESAMPLES)]

    lines = [
        f"### {path.name} (n = {len(ids)} questions; preset {summaries[0].get('provider_preset')})",
        "",
        f"Always-small quality {point['always_small']['quality']:.3f}; always-large is the "
        "judge's reference, 1.000 by construction.",
        "",
        "| Router | Quality | Retention | Cost reduction | Large-call share | "
        "Random at same share | Gain over random | PGR (random = share) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in ("heuristic", "classifier"):
        p = point[arm]
        ci = {k: _interval([d[arm][k] for d in draws])
              for k in ("quality", "retention", "cost_reduction", "gain_over_random")}
        pgr_draws = [d[arm]["pgr"] for d in draws if d[arm]["pgr"] is not None]
        pgr_ci = _interval(pgr_draws) if pgr_draws else (float("nan"), float("nan"))
        undefined = 1 - len(pgr_draws) / BOOTSTRAP_RESAMPLES
        lines.append(
            f"| {arm} | {p['quality']:.3f} [{ci['quality'][0]:.3f}, {ci['quality'][1]:.3f}] "
            f"| {p['retention']:.0%} [{ci['retention'][0]:.0%}, {ci['retention'][1]:.0%}] "
            f"| {p['cost_reduction']:.0%} [{ci['cost_reduction'][0]:.0%}, "
            f"{ci['cost_reduction'][1]:.0%}] "
            f"| {p['large_share']:.0%} | {p['random_quality']:.3f} "
            f"| {p['gain_over_random']:+.3f} [{ci['gain_over_random'][0]:+.3f}, "
            f"{ci['gain_over_random'][1]:+.3f}] "
            f"| {p['pgr']:.2f} vs {p['large_share']:.2f} [{pgr_ci[0]:.2f}, {pgr_ci[1]:.2f}]"
            + (f" ({undefined:.0%} of resamples undefined)" if undefined > 0.005 else "")
            + " |"
        )
    return "\n".join(lines)


def exact_answer_report(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    qs = data["questions"]
    rows = []
    for label, key in (("Small model answering directly", "small"), ("Writing code", "code")):
        vals = [q.get(key) for q in qs if q.get(key) is not None]
        if vals:
            k, n = sum(vals), len(vals)
            lo, hi = wilson(k, n)
            rows.append(f"| {label} | {k}/{n} = {k / n:.0%} | [{lo:.0%}, {hi:.0%}] |")
    with_code = [q for q in qs if q.get("small") is not None and q.get("code") is not None]
    if with_code and "code_status" in with_code[0]:
        policy = [q["code"] if q["code_status"] == "answered" else q["small"] for q in with_code]
        k, n = sum(policy), len(policy)
        lo, hi = wilson(k, n)
        rows.append(f"| Router policy | {k}/{n} = {k / n:.0%} | [{lo:.0%}, {hi:.0%}] |")
    return "\n".join([f"### {path.name}", "", "| | Correct | 95% Wilson interval |",
                      "|---|---|---|", *rows])


def main() -> None:
    print("## Live benchmark\n")
    for name in ("benchmark_gemini_free16.json", "benchmark_gemini_free_n9.json"):
        if (RESULTS / name).exists():
            print(benchmark_report(RESULTS / name), "\n")
    print("## Exact-answer evaluation\n")
    for name in ("exact_answer_eval_seed19.json", "exact_answer_eval_seed17_llm_paraphrase.json"):
        if (RESULTS / name).exists():
            print(exact_answer_report(RESULTS / name), "\n")


if __name__ == "__main__":
    main()
