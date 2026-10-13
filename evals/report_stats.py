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
# Routed arms, in report order; a results file reports the ones it contains.
ROUTED_ARMS = ("heuristic", "classifier", "outcome")


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
    for arm in [a for a in ROUTED_ARMS if a in arms]:
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


def replay_outcome_arm(arms: dict) -> dict:
    """The outcome router's arm for a run made before it existed: each question
    takes the recorded always-small or always-large answer, whichever tier the
    router picks at the default threshold. No model is called."""
    import sys
    sys.path.insert(0, str(RESULTS.parent / "src"))
    from router.config import ROUTE_THRESHOLD
    from router.outcome import OutcomeRouter

    dataset = RESULTS.parent / "evals" / "dataset.jsonl"
    queries = {row["id"]: row["query"] for row in
               (json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line)}
    router, out = OutcomeRouter(), {}
    for qid in arms["always_small"]:
        tier = router.route(queries[qid], ROUTE_THRESHOLD)[0]
        out[qid] = {**arms[f"always_{tier}"][qid], "tier_chosen": tier}
    return out


def benchmark_report(path: Path) -> str:
    summaries = json.loads(path.read_text(encoding="utf-8"))
    arms = {s["arm"]: {r["id"]: r for r in s["records"]} for s in summaries}
    replayed = "outcome" not in arms
    if replayed:
        arms["outcome"] = replay_outcome_arm(arms)
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
    for arm in [a for a in ROUTED_ARMS if a in arms]:
        p = point[arm]
        ci = {k: _interval([d[arm][k] for d in draws])
              for k in ("quality", "retention", "cost_reduction", "gain_over_random")}
        pgr_draws = [d[arm]["pgr"] for d in draws if d[arm]["pgr"] is not None]
        pgr_ci = _interval(pgr_draws) if pgr_draws else (float("nan"), float("nan"))
        undefined = 1 - len(pgr_draws) / BOOTSTRAP_RESAMPLES
        label = f"{arm} (replayed on the recorded answers)" if arm == "outcome" and replayed else arm
        lines.append(
            f"| {label} | {p['quality']:.3f} [{ci['quality'][0]:.3f}, {ci['quality'][1]:.3f}] "
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


def eval_set_routing() -> str:
    """How many of the 60 hand-written questions each learned router sends to
    the large model at the default threshold."""
    import sys
    sys.path.insert(0, str(RESULTS.parent / "src"))
    from router.classifier import ClassifierRouter
    from router.config import ROUTE_THRESHOLD
    from router.outcome import OutcomeRouter

    dataset = RESULTS.parent / "evals" / "dataset.jsonl"
    queries = [json.loads(line)["query"] for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    counts = {name: sum(router.route(q, ROUTE_THRESHOLD)[0] == "large" for q in queries)
              for name, router in (("outcome", OutcomeRouter()), ("classifier", ClassifierRouter()))}
    return (f"At threshold {ROUTE_THRESHOLD}, the large model gets {counts['outcome']} of "
            f"{len(queries)} eval-set questions from the outcome router and "
            f"{counts['classifier']} from the first (classifier) router.")


def main() -> None:
    print("## Eval-set routing\n")
    print(eval_set_routing(), "\n")
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
