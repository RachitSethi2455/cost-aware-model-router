"""Train the shipped router on recorded outcomes and export it as JSON.

The first router learned from 60 hand labels of how hard a question looks; on
SPROUT's recorded outcomes it did no better than prompt length. This trains
its replacement (src/router/outcome.py):

- Data: SPROUT's train split (30,968 prompts) with six small -> large model
  pairs, no model in two pairs. The service runs on a pair SPROUT does not
  have (Gemini), so a router trained on one pair's outcomes is the wrong
  target: trained on Llama 8B -> 70B alone, it lost to the hand-label router
  on GPT-4o-mini -> GPT-4o. The target is the gain (large grade minus small
  grade) averaged over the pairs.
- Model: boosted trees over the same seven features, constrained so each
  tree uses one feature. The model is then a sum of per-feature step
  functions, so every prediction has an exact per-feature explanation.
- Evaluation: leave one pair out. Train on the other five pairs, choose
  settings on the validation split, test on the held-out pair's test split.
  That pair plays the part of Gemini: a pair the router never saw.
- The shipped model is then trained on all six pairs.

The export (src/router/outcome_model.json) holds the base value, one step
function per feature and the validation percentiles of the predicted gain;
no dataset text. It is checked against the trained model on every SPROUT
prompt before it replaces the shipped file.

Usage (needs requirements-eval.txt and data/sprout; see evals/sprout_replay.py):
    python evals/train_outcome_router.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]

import numpy as np
from routing_metrics import paired_bootstrap, summarize
from sprout_replay import (
    ADDITIVE_LEAVES,
    MIN_SOURCE_GAP,
    RESULTS,
    ROOT,
    additive,
    load_models,
    tune,
)

from router.classifier import ClassifierRouter
from router.features import FEATURE_NAMES, extract
from router.outcome import MODEL_PATH, OutcomeRouter

ROUTER_PAIRS = {  # small -> large; no model appears in two pairs
    "Llama 3.1 8B -> 70B": ("wxai-llama-3-1-8b-instruct", "wxai-llama-3-1-70b-instruct"),
    "GPT-4o-mini -> GPT-4o": ("openai-gpt-4o-mini", "openai-gpt-4o"),
    "Granite 3 2B -> 8B": ("wxai-granite-3-2b-instruct-8k-max-tokens",
                           "wxai-granite-3-8b-instruct-8k-max-tokens"),
    "Llama 3.2 3B -> 3.3 70B": ("wxai-llama-3-2-3b-instruct", "wxai-llama-3-3-70b-instruct"),
    "Llama 3.2 1B -> 3 405B": ("wxai-llama-3-2-1b-instruct", "wxai-llama-3-405b-instruct"),
    "Mixtral 8x7B -> Claude 3.5 Sonnet": ("wxai-mixtral-8x7b-instruct-v01",
                                          "aws-claude-3-5-sonnet-v1"),
}
MODELS = sorted({m for pair in ROUTER_PAIRS.values() for m in pair})
SINGLE_PAIR = "Llama 3.1 8B -> 70B"
HALF = 50  # index of a 50% large-call share on routing_metrics.COST_GRID


def pooled(data: dict, pairs: list[str]) -> dict:
    """Small score, large score and gain, averaged over the given pairs."""
    small = np.mean([data[ROUTER_PAIRS[p][0]] for p in pairs], axis=0)
    large = np.mean([data[ROUTER_PAIRS[p][1]] for p in pairs], axis=0)
    return {"small": small, "large": large, "gain": large - small}


def fit(x: dict, train: dict, val: dict, pairs: list[str]):
    """Additive model on the pairs' average gain, settings chosen on validation."""
    return tune(additive, ADDITIVE_LEAVES, x["train"], pooled(train, pairs)["gain"], x["val"],
                pooled(val, pairs))


def export(model, x_train: np.ndarray) -> tuple[float, dict]:
    """Base value and per-feature step functions of an additive model.

    With no interactions, predict(x) = predict(0) + sum_i [predict(0 with x_i) -
    predict(0)], so each feature's function can be read off one column at a
    time. The features are whole numbers, and values above the training
    maximum fall in the last bin, so evaluating 0..max is exact.
    """
    zero = np.zeros((1, len(FEATURE_NAMES)))
    at_zero = float(model.predict(zero)[0])
    base, steps = at_zero, {}
    for i, name in enumerate(FEATURE_NAMES):
        top = int(x_train[:, i].max())
        grid = np.zeros((top + 1, len(FEATURE_NAMES)))
        grid[:, i] = np.arange(top + 1)
        f = model.predict(grid) - at_zero
        # Centre on the average training prompt, so the base is the mean
        # prediction and a contribution reads as "versus an average prompt".
        shift = float(f[x_train[:, i].astype(int)].mean())
        f -= shift
        base += shift
        starts = [0] + [v for v in range(1, top + 1) if f[v] != f[v - 1]]
        steps[name] = {"from": starts, "contribution": [float(f[v]) for v in starts]}
    return base, steps


def score_scale(gains: np.ndarray, points: int = 201) -> dict:
    """Validation percentiles of the predicted gain. Tied quantiles (many
    prompts share a feature vector) are merged at their average level."""
    levels = np.linspace(0, 1, points)
    quantiles = np.quantile(gains, levels)
    unique = np.unique(quantiles)
    return {"gain": unique.tolist(), "level": [float(levels[quantiles == u].mean()) for u in unique]}


def leave_one_pair_out(x, train, val, test, resamples: int) -> dict:
    label_router = ClassifierRouter()
    baselines = {"label_router": np.array([label_router.route(p)[1] for p in test["prompt"]]),
                 "length": x["test"][:, 0]}
    # One pair's outcomes alone, to show why the shipped model pools six.
    single, _ = fit(x, train, val, [SINGLE_PAIR])
    baselines["single_pair"] = single.predict(x["test"])
    print("\nLeave one pair out: trained on the other five pairs, tested on this one "
          f"({len(test['prompt'])} test prompts each).")
    print("| Held-out pair | Gap | Outcome router, pair unseen [95% CI] | Trained on this pair "
          "| Trained on Llama 8B->70B only | Label router | Length "
          "| Unseen minus label router [95% CI] | PGR at 50% large: unseen / label |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for held in ROUTER_PAIRS:
        others = [p for p in ROUTER_PAIRS if p != held]
        unseen_model, leaves = fit(x, train, val, others)
        own_model, _ = fit(x, train, val, [held])
        t = pooled(test, [held])
        scores = {"unseen": unseen_model.predict(x["test"]), "own": own_model.predict(x["test"]),
                  **baselines}
        s = {name: summarize(v, t["small"], t["large"]) for name, v in scores.items()}
        ci = paired_bootstrap(scores["unseen"], t["small"], t["large"], resamples=resamples)
        diff = paired_bootstrap(scores["unseen"], t["small"], t["large"],
                                baseline_scores=scores["label_router"], resamples=resamples)
        diff_len = paired_bootstrap(scores["unseen"], t["small"], t["large"],
                                    baseline_scores=scores["length"], resamples=resamples)
        rows[held] = {"gap": float(t["large"].mean() - t["small"].mean()), "leaves": leaves,
                      **{f"apgr_{k}": v["apgr"] for k, v in s.items()},
                      "apgr_unseen_ci": ci["apgr_ci"],
                      "unseen_minus_label_router_ci": diff["diff_ci"],
                      "unseen_minus_length_ci": diff_len["diff_ci"],
                      "pgr_at_half": {k: float(v["pgr"][HALF]) for k, v in s.items()}}
        r = rows[held]
        print(f"| {held} | {r['gap']:.3f} | {r['apgr_unseen']:.3f} [{ci['apgr_ci'][0]:.3f}, "
              f"{ci['apgr_ci'][1]:.3f}] | {r['apgr_own']:.3f} | {r['apgr_single_pair']:.3f} | {r['apgr_label_router']:.3f} "
              f"| {r['apgr_length']:.3f} | {r['apgr_unseen'] - r['apgr_label_router']:+.3f} "
              f"[{diff['diff_ci'][0]:+.3f}, {diff['diff_ci'][1]:+.3f}] "
              f"| {r['pgr_at_half']['unseen']:.0%} / {r['pgr_at_half']['label_router']:.0%} |")
        r["short"] = short_prompts(scores, t, x["test"], resamples)
    unseen = np.array([r["apgr_unseen"] for r in rows.values()])
    label = np.array([r["apgr_label_router"] for r in rows.values()])
    length = np.array([r["apgr_length"] for r in rows.values()])
    print(f"\nMean APGR over held-out pairs: outcome router {unseen.mean():.3f}, label router "
          f"{label.mean():.3f}, length {length.mean():.3f}. Outcome router ahead of the label "
          f"router on {(unseen > label).sum()} of {len(rows)} pairs, of length on "
          f"{(unseen > length).sum()} of {len(rows)}.")

    n_short = int(is_short(x["test"]).sum())
    print(f"\nShort prompts only ({n_short} test prompts with at most {SHORT_WORDS} words and "
          f"{SHORT_SENTENCES} sentences, like the service's questions), pair unseen:")
    print("| Held-out pair | Gap | Outcome router | Label router | Length | Outcome minus label [95% CI] |")
    print("|---|---|---|---|---|---|")
    for held, r in rows.items():
        s = r["short"]
        if s is None:
            print(f"| {held} | | skipped: gap under {MIN_SOURCE_GAP} | | | |")
            continue
        print(f"| {held} | {s['gap']:.3f} | {s['apgr_unseen']:.3f} | {s['apgr_label_router']:.3f} "
              f"| {s['apgr_length']:.3f} | {s['apgr_unseen'] - s['apgr_label_router']:+.3f} "
              f"[{s['unseen_minus_label_router_ci'][0]:+.3f}, {s['unseen_minus_label_router_ci'][1]:+.3f}] |")
    return rows


SHORT_WORDS, SHORT_SENTENCES = 60, 3  # the 60 hand-written questions: median 18 words, 2 sentences


def is_short(x: np.ndarray) -> np.ndarray:
    return (x[:, 0] <= SHORT_WORDS) & (x[:, 1] <= SHORT_SENTENCES)


def short_prompts(scores: dict, t: dict, x_test: np.ndarray, resamples: int) -> dict | None:
    """The same comparison on short prompts only, or None if the gap is too small."""
    mask = is_short(x_test)
    small, large = t["small"][mask], t["large"][mask]
    if large.mean() - small.mean() < MIN_SOURCE_GAP:
        return None
    diff = paired_bootstrap(scores["unseen"][mask], small, large,
                            baseline_scores=scores["label_router"][mask], resamples=resamples)
    return {"n": int(mask.sum()), "gap": float(large.mean() - small.mean()),
            **{f"apgr_{k}": summarize(scores[k][mask], small, large)["apgr"]
               for k in ("unseen", "label_router", "length")},
            "unseen_minus_label_router_ci": diff["diff_ci"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=1000)
    args = ap.parse_args()

    train, val, test = (load_models(s, MODELS) for s in ("train", "validation", "test"))
    x = {name: np.array([extract(p).to_vector() for p in d["prompt"]])
         for name, d in (("train", train), ("val", val), ("test", test))}
    print(f"SPROUT: {len(train['prompt'])} train, {len(val['prompt'])} validation, "
          f"{len(test['prompt'])} test prompts; {len(ROUTER_PAIRS)} model pairs")
    lopo = leave_one_pair_out(x, train, val, test, args.resamples)

    pairs = list(ROUTER_PAIRS)
    model, leaves = fit(x, train, val, pairs)
    val_apgr = summarize(model.predict(x["val"]), *(pooled(val, pairs)[k] for k in ("small", "large")))["apgr"]
    print(f"\nShipped model: all six pairs, max_leaf_nodes={leaves} chosen on validation "
          f"(APGR on the pairs' average {val_apgr:.3f}), {model.n_iter_} trees (early stopping, at most 200)")

    base, steps = export(model, x["train"])
    spec = {
        "description": "Additive router: predicted gain = base + sum of per-feature step "
                       "functions. Written by evals/train_outcome_router.py; see "
                       "src/router/outcome.py.",
        "trained_on": {"dataset": "CARROT-LLM-Routing/SPROUT, train split",
                       "prompts": len(train["prompt"]), "pairs": ROUTER_PAIRS,
                       "target": "large grade - small grade, averaged over the pairs"},
        "settings": {"max_leaf_nodes": leaves, "trees": int(model.n_iter_),
                     "learning_rate": 0.05, "interaction_cst": "no_interactions"},
        "validation_apgr": round(val_apgr, 4),
        "base": base,
        "features": steps,
        "score_scale": score_scale(model.predict(x["val"])),
    }

    # The export must reproduce the trained model on every prompt before it
    # replaces the shipped file.
    candidate = MODEL_PATH.with_suffix(".candidate.json")
    candidate.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    router = OutcomeRouter(candidate)
    for name, d in (("train", train), ("val", val), ("test", test)):
        exported = np.array([router.predicted_gain(p) for p in d["prompt"]])
        worst = float(np.abs(exported - model.predict(x[name])).max())
        if worst > 1e-9:
            candidate.unlink()
            raise SystemExit(f"Export differs from the trained model on {name} by {worst}")
    candidate.replace(MODEL_PATH)
    print(f"Export reproduces the trained model on all "
          f"{len(train['prompt']) + len(val['prompt']) + len(test['prompt'])} prompts; "
          f"wrote {MODEL_PATH.relative_to(ROOT)} ({MODEL_PATH.stat().st_size // 1024} KB)")

    router = OutcomeRouter()
    shipped = np.array([router.route(p)[1] for p in test["prompt"]])
    print(f"Default threshold 0.5 sends {(shipped >= 0.5).mean():.0%} of SPROUT test prompts "
          "to the large model")
    report = {"pairs": {k: list(v) for k, v in ROUTER_PAIRS.items()},
              "leave_one_pair_out": lopo,
              "shipped": {"settings": spec["settings"], "validation_apgr": val_apgr,
                          "large_share_at_default_threshold": float((shipped >= 0.5).mean())}}
    out = RESULTS / "outcome_router_training.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
