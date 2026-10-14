"""Does training on a pair's own outcomes fix routing for that pair?

The live test found that no router beat random routing on allam-2-7b ->
gpt-oss-120b: which kinds of question gain most from the large model depends
on the pair, and the routers had only other pairs' outcomes (SPROUT). This
experiment plays out the fix the service now supports (router/shadow.py):
collect the pair's own outcomes, train on them, and test once.

- Own traffic: 300 held-out SPROUT questions, disjoint from the live test set
  (evals/own_traffic_questions.json), both models' answers graded by exact
  match (live_exact_benchmark.py --question-set own). Gain = large right
  minus small right.
- Settings: the 300 are split 200 / 100 (fixed seed); the weight of an own
  row against a SPROUT row and the tree size are chosen by APGR on the 100.
- Final model: SPROUT's six pairs plus all 300 own rows at that weight.
- Test, once: the 200 live questions' cached answers, against the shipped
  router (no own data) and a router trained on the own rows alone.

Usage (after the collection is complete):
    python evals/own_traffic_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import numpy as np
from live_exact_benchmark import select_questions
from routing_metrics import paired_bootstrap, summarize
from sprout_replay import ADDITIVE_LEAVES, RESULTS, additive, load_models, tune
from train_outcome_router import MODELS, ROUTER_PAIRS, pooled

from router.features import FEATURE_NAMES_V2, extract_v2
from router.outcome import OutcomeRouter

OWN = RESULTS / "live_exact_groq-allam_own.json"
LIVE = RESULTS / "live_exact_groq-allam.json"
WEIGHTS = (1.0, 5.0, 20.0, 100.0)
SPLIT_SEED = 0
HALF = 50  # index of a 50% large-call share on routing_metrics.COST_GRID


def own_rows(path: Path) -> dict:
    saved = json.loads(path.read_text())
    if not saved["complete"]:
        raise SystemExit(f"{path.name} is partial ({len(saved['records'])} of {saved['questions']} "
                         "questions); finish the collection first: scripts/collect_own_traffic.ps1")
    recs = saved["records"]
    return {"x": np.array([[float(r["features_v2"][n]) for n in FEATURE_NAMES_V2] for r in recs]),
            "small": np.array([float(r["small"]["correct"]) for r in recs]),
            "large": np.array([float(r["large"]["correct"]) for r in recs])}


def take(rows: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in rows.items()}


def fit(x_sprout, y_sprout, own: dict, weight: float, val: dict):
    """Additive model on SPROUT plus own rows at `weight`; tree size by APGR on val."""
    x = np.vstack([x_sprout, own["x"]]) if weight else x_sprout
    y = np.concatenate([y_sprout, own["large"] - own["small"]]) if weight else y_sprout
    w = np.concatenate([np.ones(len(y_sprout)), np.full(len(own["x"]), weight)]) if weight else None
    return tune(additive, ADDITIVE_LEAVES, x, y, val["x"], val, sample_weight=w)


def main() -> None:
    own = own_rows(OWN)
    live_saved = json.loads(LIVE.read_text())
    prompts = {q["key"]: q["prompt"] for q in select_questions(200, 0, "live")}
    live = {"x": np.array([extract_v2(prompts[r["key"]]).to_vector() for r in live_saved["records"]]),
            "small": np.array([float(r["small"]["correct"]) for r in live_saved["records"]]),
            "large": np.array([float(r["large"]["correct"]) for r in live_saved["records"]])}
    keys = [r["key"] for r in live_saved["records"]]
    shipped = OutcomeRouter()
    live["shipped"] = np.array([shipped.predicted_gain(prompts[k]) for k in keys])

    print("Loading SPROUT (six pairs, v2 features) ...")
    train = load_models("train", MODELS)
    x_sprout = np.array([extract_v2(p).to_vector() for p in train["prompt"]])
    y_sprout = pooled(train, list(ROUTER_PAIRS))["gain"]

    order = np.random.default_rng(SPLIT_SEED).permutation(len(own["x"]))
    own_train, own_val = take(own, order[:200]), take(own, order[200:])
    print(f"Own traffic: {len(own['x'])} questions; the large model was right and the small one wrong on "
          f"{int(((own['large'] - own['small']) > 0).sum())}. Choosing the weight on {len(order[200:])}:")
    best = None
    for weight in WEIGHTS:
        model, leaves = fit(x_sprout, y_sprout, own_train, weight, own_val)
        value = summarize(model.predict(own_val["x"]), own_val["small"], own_val["large"])["apgr"]
        print(f"  weight {weight:>5g}: own-validation APGR {value:.3f} (leaves {leaves})")
        if best is None or value > best[0]:
            best = (value, weight)
    weight = best[1]

    final, _ = fit(x_sprout, y_sprout, own, weight, own_val)
    own_only, _ = tune(additive, ADDITIVE_LEAVES, own["x"], own["large"] - own["small"], own_val["x"], own_val)
    scores = {"shipped (SPROUT only)": live["shipped"],
              f"SPROUT + own traffic (weight {weight:g})": final.predict(live["x"]),
              "own traffic only": own_only.predict(live["x"]),
              "prompt length": live["x"][:, FEATURE_NAMES_V2.index("n_words")]}

    print(f"\nTest, once: the {len(keys)} live questions (allam-2-7b {live['small'].mean():.1%}, "
          f"gpt-oss-120b {live['large'].mean():.1%})")
    print("| Router | APGR [95% CI] | Minus shipped [95% CI] | Quality gap recovered at 50% large calls |")
    print("|---|---|---|---|")
    report = {"weight": weight, "own_questions": len(own["x"]), "routers": {}}
    for name, s in scores.items():
        curve = summarize(s, live["small"], live["large"])
        ci = paired_bootstrap(s, live["small"], live["large"], resamples=2000)["apgr_ci"]
        diff = paired_bootstrap(s, live["small"], live["large"], baseline_scores=scores["shipped (SPROUT only)"],
                                resamples=2000)["diff_ci"]
        report["routers"][name] = {"apgr": curve["apgr"], "apgr_ci": ci, "minus_shipped_ci": diff,
                                   "pgr_at_half": float(curve["pgr"][HALF])}
        print(f"| {name} | {curve['apgr']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {diff[0]:+.3f}, {diff[1]:+.3f} "
              f"| {curve['pgr'][HALF]:.0%} |")
    oracle = summarize(live["large"] - live["small"], live["small"], live["large"])
    report["oracle_apgr"] = oracle["apgr"]
    print(f"Random routing: APGR 0.500, 50% at 50% large calls. Oracle APGR {oracle['apgr']:.2f}.")
    out = RESULTS / "own_traffic_experiment.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
