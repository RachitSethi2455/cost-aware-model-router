"""Live exact-answer benchmark: the shipped router on real calls, no LLM judge.

Questions: SPROUT's held-out test split, half MMLU-Pro (multiple choice) and
half MATH problems whose reference answer is a plain number or fraction, so
every answer is checked by code (evals/exact_match.py). The routers never
trained on these questions.

Protocol: both models answer every question once (responses are cached), then
every router is replayed on those answers at the default threshold: the
outcome router, the first (hand-label) router and the heuristic, against
always-small, always-large, random routing at the same share and the oracle.
As in the service, the router reads only the question; the answer-format
instruction goes in the system message, which routing never sees. Escalation
is applied as in the pipeline: a routed-small answer that fails the
deterministic checks is replaced by the large model's, paying for both.

Free-tier friendly: calls to each model are paced under a tokens-per-minute
budget, and the run stops when a daily limit, or its own daily token budget
(default 150K per model, leaving headroom for other tools on the same key), is
reached; completed calls stay in the cache, so re-running continues where it
stopped at no extra cost.

Usage (needs the provider's API key in the environment, e.g. GROQ_API_KEY):
    python evals/live_exact_benchmark.py --preset groq-mixed --limit 4   # smoke test
    python evals/live_exact_benchmark.py --preset groq-mixed             # 200 questions
    python evals/live_exact_benchmark.py --preset groq-mixed --cached-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import numpy as np
from exact_match import extract_answer, gold_answer, parse_options
from report_stats import wilson
from routing_metrics import paired_bootstrap, summarize

from router import heuristic
from router.classifier import ClassifierRouter
from router.config import PRESETS, ROUTE_THRESHOLD
from router.escalation import should_escalate
from router.features import extract
from router.llm import LLMClient
from router.outcome import OutcomeRouter

RESULTS = ROOT / "results"
QUESTIONS = ROOT / "evals" / "live_exact_questions.json"  # keys only, no question text
SYSTEM = ("Work through the question briefly, then end your reply with one final line of the "
          "form 'Final answer: X'. For a multiple-choice question X is the option letter only; "
          "otherwise X is the final value only.")
SOURCES = {"choice": "TIGER-Lab/MMLU-Pro", "number": "lighteval/MATH"}
ROUTERS = ("outcome", "classifier", "heuristic")
RESAMPLES = 2000
DAILY = re.compile(r"per day|\bTPD\b|\bRPD\b", re.IGNORECASE)
RETRY_IN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.IGNORECASE)


class DailyLimit(Exception):
    pass


def select_questions(n: int, seed: int = 0) -> list[dict]:
    """n questions, half per source, sampled with a fixed seed, then shuffled
    so a run stopped by a daily limit still covers both kinds.

    The default set's keys are frozen in QUESTIONS: which MATH problems are
    eligible depends on which reference answers the grader can read, so
    without the freeze an improvement to the grader would silently swap the
    questions out from under a run in progress."""
    import pyarrow.parquet as pq
    from sprout_replay import DATA, DOWNLOAD

    path = DATA / "test-00000-of-00001.parquet"
    if not path.exists():
        raise SystemExit(f"SPROUT test split missing from {DATA}. Download it:\n{DOWNLOAD}")
    cols = pq.read_table(path, columns=["key", "dataset", "prompt", "golden_answer"]).to_pydict()
    rows = {r["key"]: r for r in (dict(zip(cols, values, strict=True))
                                  for values in zip(*cols.values(), strict=True))}
    frozen = json.loads(QUESTIONS.read_text()) if QUESTIONS.exists() else None
    if frozen and (frozen["n"], frozen["seed"]) == (n, seed):
        chosen = [(q["key"], q["kind"]) for q in frozen["questions"]]
    else:
        rng = random.Random(seed)
        chosen = []
        for kind, prefix in SOURCES.items():
            pool = sorted(key for key, r in rows.items() if r["dataset"].startswith(prefix)
                          and gold_answer(kind, r["golden_answer"]) is not None)
            chosen += [(key, kind) for key in rng.sample(pool, n // 2)]
        rng.shuffle(chosen)
    return [{"key": key, "source": SOURCES[kind].split("/")[-1], "kind": kind,
             "prompt": rows[key]["prompt"], "gold": rows[key]["golden_answer"]} for key, kind in chosen]


class Pacer:
    """Keeps each model under a tokens-per-minute budget: after a call that
    used t tokens, the next call to that model waits 60 * t / tpm seconds."""

    def __init__(self, tpm: dict[str, int]):
        self.tpm, self.ready = tpm, {}

    def wait(self, tier: str) -> None:
        delay = self.ready.get(tier, 0.0) - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def used(self, tier: str, tokens: int) -> None:
        self.ready[tier] = time.monotonic() + 60 * tokens / self.tpm[tier]


class DailyBudget:
    """Caps this benchmark's tokens per model per day (local date), so a key
    shared with other tools keeps headroom: provider limits are per model and
    per account, so a second key would not help. Usage persists across runs."""

    def __init__(self, path: Path, limit: int):
        self.path, self.limit, self.today = path, limit, dt.date.today().isoformat()
        saved = json.loads(path.read_text()) if path.exists() else {}
        self.used = saved.get(self.today, {})

    def check(self, model_id: str) -> None:
        if self.used.get(model_id, 0) >= self.limit:
            raise DailyLimit(f"this benchmark's budget of {self.limit:,} tokens today for "
                             f"{model_id} is used up (--daily-token-budget)")

    def add(self, model_id: str, tokens: int) -> None:
        self.used[model_id] = self.used.get(model_id, 0) + tokens
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps({self.today: self.used}))


def ask(client: LLMClient, spec, prompt: str, pacer: Pacer, budget: DailyBudget,
        cached_only: bool, retries: int = 6):
    """One cached model call. Waits out per-minute limits; raises DailyLimit
    when a daily quota or budget is used up. Returns None in cached-only mode
    on a miss."""
    if (hit := client.cached(spec, prompt, SYSTEM)) is not None or cached_only:
        return hit
    for _ in range(retries):
        budget.check(spec.model_id)
        pacer.wait(spec.name)
        result = client.complete(spec, prompt, system=SYSTEM)
        pacer.used(spec.name, result.input_tokens + result.output_tokens)
        budget.add(spec.model_id, result.input_tokens + result.output_tokens)
        if not result.error or ("429" not in result.error and "RateLimit" not in result.error):
            return result  # an answer, or a real failure (never cached, retried next run)
        if DAILY.search(result.error):
            raise DailyLimit(result.error)
        m = RETRY_IN.search(result.error)
        time.sleep(int(m.group(1) or 0) * 60 + float(m.group(2)) + 1 if m else 30)
    return result


def record(q: dict, small, large, routers: dict) -> dict:
    gold = gold_answer(q["kind"], q["gold"])
    out = {"key": q["key"], "source": q["source"], "kind": q["kind"], "gold": str(gold),
           "n_words": extract(q["prompt"]).n_words,
           "scores": {name: router(q["prompt"]) for name, router in routers.items()}}
    for tier, r in (("small", small), ("large", large)):
        answer = extract_answer(q["kind"], r.text, parse_options(q["prompt"]))
        out[tier] = {"answer": None if answer is None else str(answer), "correct": gold is not None and answer == gold,
                     "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                     "cost_usd": r.cost_usd, "latency_s": r.latency_s, "stop_reason": r.stop_reason}
    decision = should_escalate(small, q["prompt"])
    out["small"]["escalate"], out["small"]["escalation_reason"] = decision.escalate, decision.reason
    return out


def serve(records: list[dict], to_large: np.ndarray) -> dict:
    """Per-question outcome of a routing policy, with pipeline escalation:
    a routed-small answer that fails the checks is retried on the large model."""
    small = {k: np.array([r["small"][k] for r in records], dtype=float)
             for k in ("correct", "cost_usd", "latency_s", "escalate")}
    large = {k: np.array([r["large"][k] for r in records], dtype=float)
             for k in ("correct", "cost_usd", "latency_s")}
    escalated = (~to_large) & (small["escalate"] > 0)
    used_large = to_large | escalated
    return {"correct": np.where(used_large, large["correct"], small["correct"]),
            "cost": np.where(to_large, large["cost_usd"],
                             small["cost_usd"] + np.where(escalated, large["cost_usd"], 0)),
            "latency": np.where(to_large, large["latency_s"],
                                small["latency_s"] + np.where(escalated, large["latency_s"], 0)),
            "used_large": used_large, "escalated": escalated}


def arm_stats(arm: dict, base: dict, idx: np.ndarray) -> dict:
    q_small, q_large = base["small"]["correct"][idx].mean(), base["large"]["correct"][idx].mean()
    q, share = arm["correct"][idx].mean(), arm["used_large"][idx].mean()
    random_q = q_small + share * (q_large - q_small)
    return {"accuracy": q, "large_share": share, "random_accuracy": random_q,
            "gain_over_random": q - random_q,
            "cost_reduction": 1 - arm["cost"][idx].sum() / base["large"]["cost"][idx].sum(),
            "pgr": (q - q_small) / (q_large - q_small) if q_large > q_small else float("nan")}


def interval(values) -> tuple[float, float]:
    values = np.sort(np.asarray(values))
    values = values[~np.isnan(values)]
    return float(values[int(0.025 * len(values))]), float(values[int(0.975 * len(values)) - 1])


def analyze(records: list[dict], threshold: float, seed: int = 0) -> dict:
    n = len(records)
    small_q = np.array([r["small"]["correct"] for r in records], dtype=float)
    large_q = np.array([r["large"]["correct"] for r in records], dtype=float)
    # Fixed-tier baselines are left alone, as in the pipeline: no escalation.
    base = {"small": {"correct": small_q, "used_large": np.zeros(n, bool), "escalated": np.zeros(n, bool),
                      "cost": np.array([r["small"]["cost_usd"] for r in records]),
                      "latency": np.array([r["small"]["latency_s"] for r in records])},
            "large": serve(records, np.ones(n, bool))}
    arms = {"always_small": base["small"], "always_large": base["large"]}
    for name in ROUTERS:
        arms[name] = serve(records, np.array([r["scores"][name] >= threshold for r in records]))

    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, n, n) for _ in range(RESAMPLES)]
    everything = np.arange(n)
    summary = {"n": n, "threshold": threshold, "arms": {}}
    for name, arm in arms.items():
        point = arm_stats(arm, base, everything)
        boot = [arm_stats(arm, base, idx) for idx in draws]
        k = int(arm["correct"].sum())
        summary["arms"][name] = {
            **point, "correct": k, "accuracy_wilson": wilson(k, n),
            "cost_usd": float(arm["cost"].sum()), "p50_latency_s": float(np.median(arm["latency"])),
            "escalations": int(arm["escalated"].sum()),
            **{f"{key}_ci": interval([b[key] for b in boot])
               for key in ("accuracy", "cost_reduction", "gain_over_random")}}

    # Routing curves (no escalation): how well each router ranks questions.
    summary["curves"] = {}
    if large_q.mean() > small_q.mean():
        scores = {name: np.array([r["scores"][name] for r in records]) for name in ROUTERS}
        scores["length"] = np.array([r["n_words"] for r in records], dtype=float)
        for name, s in scores.items():
            curve = summarize(s, small_q, large_q)
            boot = paired_bootstrap(s, small_q, large_q, resamples=RESAMPLES)
            summary["curves"][name] = {"apgr": curve["apgr"], "apgr_ci": boot["apgr_ci"],
                                       "cpt50": curve["cpt50"], "cpt80": curve["cpt80"]}
        diff = paired_bootstrap(scores["outcome"], small_q, large_q,
                                baseline_scores=scores["classifier"], resamples=RESAMPLES)
        summary["curves"]["outcome_minus_classifier_ci"] = diff["diff_ci"]
        oracle = summarize(large_q - small_q, small_q, large_q)
        summary["curves"]["oracle"] = {"apgr": oracle["apgr"], "cpt80": oracle["cpt80"]}
    return summary


def print_summary(summary: dict, records: list[dict]) -> None:
    n = summary["n"]
    by_kind = {k: [r for r in records if r["kind"] == k] for k in SOURCES}
    print(f"\n{n} questions (" + ", ".join(f"{len(v)} {k}" for k, v in by_kind.items())
          + f"), threshold {summary['threshold']}")
    for kind, rs in by_kind.items():
        if rs:
            print(f"  {kind:<7} small {np.mean([r['small']['correct'] for r in rs]):.0%}  "
                  f"large {np.mean([r['large']['correct'] for r in rs]):.0%}")
    print("\n| Arm | Accuracy [95% CI] | Large-model share | Cost reduction [95% CI] "
          "| Gain over random at same share [95% CI] | PGR | p50 latency | Escalations |")
    print("|---|---|---|---|---|---|---|---|")
    for name, a in summary["arms"].items():
        print(f"| {name} | {a['accuracy']:.1%} [{a['accuracy_ci'][0]:.1%}, {a['accuracy_ci'][1]:.1%}] "
              f"| {a['large_share']:.0%} | {a['cost_reduction']:.0%} [{a['cost_reduction_ci'][0]:.0%}, "
              f"{a['cost_reduction_ci'][1]:.0%}] | {a['gain_over_random']:+.3f} "
              f"[{a['gain_over_random_ci'][0]:+.3f}, {a['gain_over_random_ci'][1]:+.3f}] "
              f"| {a['pgr']:.2f} | {a['p50_latency_s']:.1f} s | {a['escalations']} |")
    curves = summary["curves"]
    if curves:
        print("\nRouting curves (no escalation): APGR [95% CI], random = 0.5")
        for name in (*ROUTERS, "length"):
            c = curves[name]
            print(f"  {name:<10} {c['apgr']:.3f} [{c['apgr_ci'][0]:.3f}, {c['apgr_ci'][1]:.3f}]  "
                  f"80% of the gap at {c['cpt80']:.0%} large calls")
        lo, hi = curves["outcome_minus_classifier_ci"]
        print(f"  outcome minus classifier: [{lo:+.3f}, {hi:+.3f}]; oracle APGR "
              f"{curves['oracle']['apgr']:.2f}, 80% of the gap at {curves['oracle']['cpt80']:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="groq-mixed", choices=sorted(PRESETS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    ap.add_argument("--tpm-small", type=int, default=6000, help="tokens per minute for the small model")
    ap.add_argument("--tpm-large", type=int, default=8000, help="tokens per minute for the large model")
    ap.add_argument("--daily-token-budget", type=int, default=150_000,
                    help="stop after this many tokens per model per day (Groq free tier: 200K)")
    ap.add_argument("--cached-only", action="store_true", help="re-score from the cache, no calls")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    specs = PRESETS[args.preset]
    questions = select_questions(args.n, args.seed)[:args.limit]
    client, pacer = LLMClient(), Pacer({"small": args.tpm_small, "large": args.tpm_large})
    budget = DailyBudget(ROOT / "data" / "live_benchmark_usage.json", args.daily_token_budget)
    outcome, classifier = OutcomeRouter(), ClassifierRouter()
    routers = {"outcome": lambda p: outcome.route(p, ROUTE_THRESHOLD)[1],
               "classifier": lambda p: classifier.route(p, ROUTE_THRESHOLD)[1],
               "heuristic": lambda p: heuristic.route(p, ROUTE_THRESHOLD)[1]}
    print(f"{args.preset}: small {specs['small'].model_id}, large {specs['large'].model_id}; "
          f"{len(questions)} questions")

    records, failures, stopped = [], {"small": 0, "large": 0}, None
    for i, q in enumerate(questions, 1):
        try:
            small = ask(client, specs["small"], q["prompt"], pacer, budget, args.cached_only)
            large = ask(client, specs["large"], q["prompt"], pacer, budget, args.cached_only)
        except DailyLimit as exc:
            stopped = str(exc)[:300]
            print(f"\nDaily limit reached after {len(records)} questions; completed calls are "
                  f"cached, so re-running continues from here.\n  {stopped}")
            break
        failed = [t for t, r in (("small", small), ("large", large)) if r is None or r.error]
        if failed:
            for tier in failed:
                failures[tier] += 1
            errors = [r.error for r in (small, large) if r is not None and r.error]
            if errors and sum(failures.values()) >= 3 and not records:
                raise SystemExit(f"The first calls failed; check the model ids and key:\n  {errors[0]}")
            continue
        records.append(record(q, small, large, routers))
        r = records[-1]
        mark = lambda ok: "right" if ok else "wrong"  # noqa: E731
        print(f"[{i}/{len(questions)}] {q['kind']:<6} small {mark(r['small']['correct'])}, "
              f"large {mark(r['large']['correct'])}"
              + (f", escalate ({r['small']['escalation_reason']})" if r["small"]["escalate"] else ""))

    if not records:
        raise SystemExit("No complete questions yet.")
    summary = analyze(records, ROUTE_THRESHOLD)
    print_summary(summary, records)
    complete = len(records) == len(questions)
    suffix = f"_limit{args.limit}" if args.limit else ""
    out = RESULTS / f"live_exact_{args.preset}{suffix}.json"
    out.write_text(json.dumps({
        "preset": args.preset, "models": {t: specs[t].model_id for t in ("small", "large")},
        "prices_per_mtok": {t: [specs[t].input_per_mtok, specs[t].output_per_mtok]
                            for t in ("small", "large")},
        "date": dt.date.today().isoformat(), "seed": args.seed, "system_prompt": SYSTEM,
        "questions": len(questions), "complete": complete, "stopped_by": stopped,
        "failed_calls": failures, "summary": summary, "records": records}, indent=1, default=float))
    print(f"\nWrote {out.relative_to(ROOT)}" + ("" if complete else " (partial)"))


if __name__ == "__main__":
    main()
