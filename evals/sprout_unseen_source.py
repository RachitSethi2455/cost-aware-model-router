"""Do SPROUT-trained routers carry over to a kind of prompt they never saw?

sprout_replay.py tests on held-out prompts from the same sources as the
training data, which favours anything that learns topic words. This script
holds out a whole source instead: train on the train split of every other
source, choose settings on the validation prompts of those other sources, and
test on the held-out source's test prompts. It is the SPROUT version of the
README's "unseen category" test, where on 60 hand-labelled queries the seven
hand-made features beat bag-of-words (73% vs 60%).

All learned routers are trained on the gain (large score minus small score),
which beat the strict label on validation for every model.

Usage (after sprout_replay.py, whose cached embeddings and results it reuses):
    python evals/sprout_unseen_source.py --pair llama
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]

import numpy as np
from routing_metrics import paired_bootstrap, summarize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sprout_replay import (
    KNN_K,
    MAIN_SOURCES,
    MIN_SOURCE_GAP,
    PAIRS,
    RESULTS,
    RIDGE_ALPHA,
    ROOT,
    embeddings,
    knn_scores,
    load_split,
    predict,
    ridge,
    tune,
)

from router.classifier import ClassifierRouter
from router.features import extract

ROUTERS = ("tfidf_ridge", "minilm_knn", "seven_features_gbm", "seven_features_ridge", "length",
           "shipped_router")


def subset(data: dict, mask: np.ndarray) -> dict:
    return {k: (v[mask] if isinstance(v, np.ndarray) else [x for x, m in zip(v, mask, strict=True) if m])
            for k, v in data.items()}


def from_source(data: dict, prefix: str) -> np.ndarray:
    return np.array([d.startswith(prefix) for d in data["dataset"]])


def route_unseen(train, val, test, feats, emb) -> dict:
    """Train every router on (train, val) and score the test prompts."""
    scores = {}
    model, _ = tune(ridge(True), RIDGE_ALPHA, feats["train"], train["gain"], feats["val"], val)
    scores["seven_features_ridge"] = predict(model, feats["test"])
    model, _ = tune(lambda leaves: HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=leaves, random_state=0),
        (7, 15, 31), feats["train"], train["gain"], feats["val"], val)
    scores["seven_features_gbm"] = predict(model, feats["test"])
    scores["length"] = feats["test"][:, 0]
    shipped = ClassifierRouter()
    scores["shipped_router"] = np.array([shipped.route(p)[1] for p in test["prompt"]])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                                 sublinear_tf=True)
    words = {"train": vectorizer.fit_transform(train["prompt"]),
             "val": vectorizer.transform(val["prompt"]), "test": vectorizer.transform(test["prompt"])}
    model, _ = tune(ridge(False), RIDGE_ALPHA, words["train"], train["gain"], words["val"], val)
    scores["tfidf_ridge"] = predict(model, words["test"])

    k = max(KNN_K, key=lambda k: summarize(knn_scores(emb["train"], train["gain"], emb["val"], k),
                                           val["small"], val["large"])["apgr"])
    scores["minilm_knn"] = knn_scores(emb["train"], train["gain"], emb["test"], k)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=sorted(PAIRS), default="llama")
    ap.add_argument("--resamples", type=int, default=1000)
    args = ap.parse_args()

    splits = {"train": load_split("train", args.pair), "val": load_split("validation", args.pair),
              "test": load_split("test", args.pair)}
    feats = {s: np.array([extract(p).to_vector() for p in d["prompt"]]) for s, d in splits.items()}
    emb = {s: embeddings(name, splits[s]["prompt"])
           for s, name in (("train", "train"), ("val", "validation"), ("test", "test"))}
    seen_path = RESULTS / f"sprout_replay_{args.pair}.json"
    seen = json.loads(seen_path.read_text())["by_source"] if seen_path.exists() else {}

    results = {"pair": dict(zip(("small", "large"), PAIRS[args.pair], strict=True)), "sources": {}}
    for prefix, label in MAIN_SOURCES.items():
        masks = {"train": ~from_source(splits["train"], prefix),
                 "val": ~from_source(splits["val"], prefix),
                 "test": from_source(splits["test"], prefix)}
        data = {s: subset(splits[s], m) for s, m in masks.items()}
        small, large = data["test"]["small"], data["test"]["large"]
        if large.mean() - small.mean() < MIN_SOURCE_GAP:
            print(f"\nHeld out: {label}: skipped, small {small.mean():.3f} vs large "
                  f"{large.mean():.3f} is too small a gap to rank routers")
            continue
        scores = route_unseen(data["train"], data["val"], data["test"],
                              {s: feats[s][m] for s, m in masks.items()},
                              {s: emb[s][m] for s, m in masks.items()})

        print(f"\nHeld out: {label} (test n={len(small)}, small {small.mean():.3f}, large "
              f"{large.mean():.3f}; trained on {len(data['train']['gain'])} prompts from other sources)")
        print("| Router | APGR, source unseen [95% CI] | APGR, source seen in training |")
        print("|---|---|---|")
        rows = {}
        for name in ROUTERS:
            apgr = summarize(scores[name], small, large)["apgr"]
            ci = paired_bootstrap(scores[name], small, large, resamples=args.resamples)["apgr_ci"]
            in_dist = seen.get(label, {}).get(name)
            rows[name] = {"apgr": apgr, "apgr_ci": ci, "apgr_seen": in_dist}
            shown = f"{in_dist:.3f}" if in_dist is not None else "–"
            print(f"| {name} | {apgr:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {shown} |")
        diff = paired_bootstrap(scores["seven_features_gbm"], small, large,
                                baseline_scores=scores["tfidf_ridge"], resamples=args.resamples)
        point = rows["seven_features_gbm"]["apgr"] - rows["tfidf_ridge"]["apgr"]
        print(f"seven_features_gbm - tfidf_ridge: {point:+.3f} "
              f"[{diff['diff_ci'][0]:+.3f}, {diff['diff_ci'][1]:+.3f}]")
        results["sources"][label] = {"n_test": len(small), "quality_small": float(small.mean()),
                                     "quality_large": float(large.mean()), "routers": rows,
                                     "gbm_minus_tfidf": {"diff": point, "ci": diff["diff_ci"]}}

    out = RESULTS / f"sprout_unseen_source_{args.pair}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
