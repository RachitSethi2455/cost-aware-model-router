"""Offline router evaluation on SPROUT: thousands of graded prompts, $0.

SPROUT (CARROT-LLM-Routing/SPROUT on Hugging Face) holds about 44k prompts,
each answered by 13 models and graded 0-1 by a judge against a gold answer.
Replaying routers on those recorded outcomes measures them on ~6,600 held-out
prompts without a single API call. The dataset card states no licence, so the
data is used for evaluation only and is not redistributed here.

Protocol (the one published routers are judged by):
- Pair: a small and a large model from the same family.
- Outcome targets, not "looks hard" priors. Each learned router is trained
  two ways: on the label "the large model scored strictly higher", and on the
  gain itself (large score minus small score). The label treats a 0.1 judge
  wobble the same as a wrong-to-right fix; the gain is what the cost-quality
  curve actually rewards.
- Routers are trained on SPROUT's train split, settings are chosen on
  validation, and the test split is only used for reporting.
- Metrics (evals/routing_metrics.py): the RouteLLM cost-quality measures
  APGR, CPT(50%), CPT(80%), with paired bootstrap intervals and the random
  (APGR 0.5) and oracle references, plus AUROC for the strict label. Cost is
  the share of queries sent to the large model.

Routers compared:
- seven_features_*:   this project's 7 features (ridge, gradient boosting, logistic)
- outcome_router:     the shipped router (src/router/outcome_model.json): the seven
                      features, additive, trained on six pairs' outcomes including this one
- label_router:       the first router, trained on 60 hand labels, unchanged
- heuristic, length:  the hand-written rules; prompt length alone
- tfidf_*:            TF-IDF + ridge / logistic regression
- minilm_*:           MiniLM sentence embeddings + kNN / ridge / logistic regression
- features_plus_minilm: both feature sets concatenated, ridge

Usage (the script prints the download commands if the data is missing):
    pip install -r requirements-eval.txt
    python evals/sprout_replay.py --pair llama     # Llama 3.1 8B -> 70B
    python evals/sprout_replay.py --pair gpt4o     # GPT-4o-mini -> GPT-4o
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import numpy as np
import pyarrow.parquet as pq
from routing_metrics import auroc, paired_bootstrap, random_apgr, summarize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from router import heuristic
from router.classifier import ClassifierRouter
from router.features import extract
from router.outcome import OutcomeRouter

DATA = ROOT / "data" / "sprout"
RESULTS = ROOT / "results"
SPLITS = {
    "train": ["train-00000-of-00003", "train-00001-of-00003", "train-00002-of-00003"],
    "validation": ["validation-00000-of-00001"],
    "test": ["test-00000-of-00001"],
}
PAIRS = {
    "llama": ("wxai-llama-3-1-8b-instruct", "wxai-llama-3-1-70b-instruct"),
    "gpt4o": ("openai-gpt-4o-mini", "openai-gpt-4o"),
}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CHARS = 1000  # about 250 tokens; MiniLM was trained on 256
MAIN_SOURCES = {"openhermes/teknium": "OpenHermes (chat)", "TIGER-Lab/MMLU-Pro": "MMLU-Pro",
                "lighteval/MATH": "MATH"}
# PGR divides by the small-to-large quality gap; on a source where the two
# models are within a few points of each other it is mostly noise.
MIN_SOURCE_GAP = 0.05
DOWNLOAD = r"""mkdir -p data/sprout
for f in train-00000-of-00003 train-00001-of-00003 train-00002-of-00003 \
         validation-00000-of-00001 test-00000-of-00001; do
  curl -L -C - -o data/sprout/$f.parquet \
    https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT/resolve/main/data/$f.parquet
done"""
TABLE = [  # (router, training target), in print order
    ("minilm_knn", "gain"), ("minilm_ridge", "gain"), ("features_plus_minilm", "gain"),
    ("tfidf_ridge", "gain"), ("seven_features_gbm", "gain"), ("seven_features_ridge", "gain"),
    ("minilm_knn_label", "label"), ("minilm_lr", "label"), ("tfidf_lr", "label"),
    ("seven_features_lr", "label"),
    ("outcome_router", "gain, 6 pairs"), ("length", "none"), ("heuristic", "none"),
    ("label_router", "60 hand labels"),
]
# Paired comparisons on the same resampled queries: (router, baseline).
COMPARISONS = [("seven_features_ridge", "minilm_knn"), ("seven_features_ridge", "length"),
               ("tfidf_ridge", "minilm_knn"), ("minilm_knn", "minilm_knn_label"),
               ("seven_features_ridge", "seven_features_lr")]


def load_models(split: str, models: list[str]) -> dict:
    """Prompts, sources and the given models' scores for one split, reading
    only the needed nested fields (the full files also hold every response).
    Prompts missing any of these scores are dropped."""
    missing = [name for name in SPLITS[split] if not (DATA / f"{name}.parquet").exists()]
    if missing:
        raise SystemExit(f"SPROUT files missing from {DATA}. Download them (about 550 MB):\n{DOWNLOAD}")
    columns = ["key", "dataset", "prompt"] + [f"{m}.score" for m in models]
    out = {"key": [], "dataset": [], "prompt": [], **{m: [] for m in models}}
    for name in SPLITS[split]:
        table = pq.read_table(DATA / f"{name}.parquet", columns=columns)
        # Nested fields come back named "score", so go by position.
        for key, col in zip(out, table.columns, strict=True):
            out[key] += col.to_pylist()
    data = {k: np.asarray(v, dtype=float) if k in models else v for k, v in out.items()}
    keep = ~np.any([np.isnan(data[m]) for m in models], axis=0)
    return {k: (v[keep] if isinstance(v, np.ndarray) else [x for x, m in zip(v, keep, strict=True) if m])
            for k, v in data.items()}


def load_split(split: str, pair: str) -> dict:
    """One split for one model pair, with the gain and the strict label."""
    small, large = PAIRS[pair]
    data = load_models(split, [small, large])
    data["small"], data["large"] = data.pop(small), data.pop(large)
    data["gain"] = data["large"] - data["small"]
    data["label"] = (data["gain"] > 0).astype(int)
    return data


def embeddings(split: str, prompts: list[str]) -> np.ndarray:
    """MiniLM embeddings, cached per split (they don't depend on the pair)."""
    cache = DATA / f"minilm_{split}.npy"
    if cache.exists():
        cached = np.load(cache)
        if len(cached) == len(prompts):
            return cached
    from fastembed import TextEmbedding
    model = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(ROOT / "data" / "fastembed_cache"))
    start = time.perf_counter()
    # Batches are padded to their longest prompt, and a few SPROUT prompts are
    # whole RAG documents, so unsorted batches ran at the 512-token maximum
    # (25+ CPU-minutes for 6.6k prompts). Sorting by length and keeping the
    # first EMBED_CHARS characters, where the question is, fixes that.
    texts = [p[:EMBED_CHARS] for p in prompts]
    order = np.argsort([len(t) for t in texts], kind="stable")
    sorted_vecs = np.asarray(list(model.embed([texts[i] for i in order], batch_size=64)),
                             dtype=np.float32)
    vectors = np.empty_like(sorted_vecs)
    vectors[order] = sorted_vecs
    print(f"  embedded {len(prompts)} {split} prompts in {time.perf_counter() - start:.0f}s")
    np.save(cache, vectors)
    return vectors


def knn_scores(train_vecs, train_targets, query_vecs, k: int, chunk: int = 32) -> np.ndarray:
    """Mean target (gain, or large-won label) over the k most similar training prompts."""
    train = train_vecs / np.linalg.norm(train_vecs, axis=1, keepdims=True)
    query = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    out = np.empty(len(query))
    for start in range(0, len(query), chunk):
        sims = query[start:start + chunk] @ train.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        out[start:start + chunk] = train_targets[top].mean(axis=1)
    return out


def predict(model, x) -> np.ndarray:
    return model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)


def tune(build, grid, x_train, y_train, x_val, val):
    """Fit one model per setting; keep the one with the best validation APGR."""
    best = None
    for setting in grid:
        model = build(setting).fit(x_train, y_train)
        value = summarize(predict(model, x_val), val["small"], val["large"])["apgr"]
        if best is None or value > best[0]:
            best = (value, setting, model)
    return best[2], best[1]


def logistic(scale: bool):
    steps = [StandardScaler()] if scale else []
    return lambda c: make_pipeline(*steps, LogisticRegression(C=c, max_iter=2000,
                                                              class_weight="balanced"))


def ridge(scale: bool):
    steps = [StandardScaler()] if scale else []
    return lambda alpha: make_pipeline(*steps, Ridge(alpha=alpha))


def additive(leaves: int) -> HistGradientBoostingRegressor:
    """Boosted trees where each tree uses one feature, so the model is a sum of
    per-feature step functions: the shipped router (evals/train_outcome_router.py)."""
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=leaves,
                                         interaction_cst="no_interactions", random_state=0)


LOGISTIC_C = (0.01, 0.1, 1.0, 10.0)
RIDGE_ALPHA = (0.1, 1.0, 10.0, 100.0, 1000.0)
KNN_K = (10, 25, 50, 100, 200)
ADDITIVE_LEAVES = (3, 7, 15, 31)


def train_routers(train, val, test, feats, emb) -> tuple[dict, dict]:
    scores, notes = {}, {}

    def fit(name, build, grid, x, target, label):
        model, setting = tune(build, grid, x["train"], train[target], x["val"], val)
        scores[name] = predict(model, x["test"])
        notes[name] = f"{label}={setting}"

    fit("seven_features_ridge", ridge(True), RIDGE_ALPHA, feats, "gain", "alpha")
    fit("seven_features_lr", logistic(True), LOGISTIC_C, feats, "label", "C")
    # Gradient boosting gives the seven features a fair non-linear chance.
    fit("seven_features_gbm",
        lambda leaves: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                     max_leaf_nodes=leaves, random_state=0),
        (7, 15, 31), feats, "gain", "leaves")

    label_router, outcome_router = ClassifierRouter(), OutcomeRouter()
    scores["label_router"] = np.array([label_router.route(p)[1] for p in test["prompt"]])
    scores["outcome_router"] = np.array([outcome_router.route(p)[1] for p in test["prompt"]])
    scores["heuristic"] = np.array([heuristic.score(extract(p)) for p in test["prompt"]])
    scores["length"] = feats["test"][:, 0]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                                 sublinear_tf=True)
    words = {"train": vectorizer.fit_transform(train["prompt"]),
             "val": vectorizer.transform(val["prompt"]),
             "test": vectorizer.transform(test["prompt"])}
    fit("tfidf_ridge", ridge(False), RIDGE_ALPHA, words, "gain", "alpha")
    fit("tfidf_lr", logistic(False), LOGISTIC_C, words, "label", "C")
    del words, vectorizer  # hundreds of MB; the kNN step below needs the room

    fit("minilm_ridge", ridge(False), RIDGE_ALPHA, emb, "gain", "alpha")
    fit("minilm_lr", logistic(False), LOGISTIC_C, emb, "label", "C")
    for name, target in (("minilm_knn", "gain"), ("minilm_knn_label", "label")):
        best = max(KNN_K, key=lambda k, t=target: summarize(
            knn_scores(emb["train"], train[t], emb["val"], k), val["small"], val["large"])["apgr"])
        scores[name] = knn_scores(emb["train"], train[target], emb["test"], best)
        notes[name] = f"k={best}"

    scaler = StandardScaler().fit(feats["train"])
    both = {s: np.hstack([scaler.transform(feats[s]), emb[s]]) for s in ("train", "val", "test")}
    fit("features_plus_minilm", ridge(False), RIDGE_ALPHA, both, "gain", "alpha")
    return scores, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=sorted(PAIRS), default="llama")
    ap.add_argument("--resamples", type=int, default=1000)
    args = ap.parse_args()
    small_model, large_model = PAIRS[args.pair]

    print(f"Loading SPROUT, pair {small_model} -> {large_model}")
    train, val, test = (load_split(s, args.pair) for s in ("train", "validation", "test"))
    for name, d in (("train", train), ("validation", val), ("test", test)):
        print(f"  {name}: {len(d['label'])} prompts, large better on {d['label'].mean():.1%} "
              f"(by 0.5 or more on {(d['gain'] >= 0.5).mean():.1%}), "
              f"small better on {(d['gain'] < 0).mean():.1%}")

    print("Features")
    feats = {name: np.array([extract(p).to_vector() for p in d["prompt"]])
             for name, d in (("train", train), ("val", val), ("test", test))}
    emb = {"train": embeddings("train", train["prompt"]),
           "val": embeddings("validation", val["prompt"]),
           "test": embeddings("test", test["prompt"])}

    print("Training routers (settings chosen on validation)")
    scores, notes = train_routers(train, val, test, feats, emb)

    small, large = test["small"], test["large"]
    n = len(small)
    print(f"\nTest split: {n} prompts. Always-small quality {small.mean():.3f}, "
          f"always-large {large.mean():.3f}, random-router APGR {random_apgr(n):.3f}")

    results = {"pair": {"small": small_model, "large": large_model}, "n_test": n,
               "quality_small": float(small.mean()), "quality_large": float(large.mean()),
               "large_better_share": float(test["label"].mean()),
               "random_apgr": random_apgr(n), "routers": {}}
    oracle = summarize(test["gain"], small, large)
    results["oracle"] = {k: oracle[k] for k in ("apgr", "cpt50", "cpt80")}

    print("\n| Router | Trained on | APGR [95% CI] | APGR − random [95% CI] | CPT(50%) | CPT(80%) "
          "| AUROC | Setting |")
    print("|---|---|---|---|---|---|---|---|")
    for name, target in TABLE:
        s = summarize(scores[name], small, large)
        boot = paired_bootstrap(scores[name], small, large, resamples=args.resamples)
        row = {"target": target, "auroc": auroc(test["label"], scores[name]), "apgr": s["apgr"],
               "apgr_ci": boot["apgr_ci"], "apgr_minus_random_ci": boot["diff_ci"],
               "cpt50": s["cpt50"], "cpt80": s["cpt80"], "setting": notes.get(name, ""),
               "pgr_curve": [round(float(x), 4) for x in s["pgr"]]}
        results["routers"][name] = row
        print(f"| {name} | {target} | {row['apgr']:.3f} [{row['apgr_ci'][0]:.3f}, "
              f"{row['apgr_ci'][1]:.3f}] | {row['apgr'] - random_apgr(n):+.3f} "
              f"[{row['apgr_minus_random_ci'][0]:+.3f}, {row['apgr_minus_random_ci'][1]:+.3f}] "
              f"| {row['cpt50']:.0%} | {row['cpt80']:.0%} | {row['auroc']:.3f} | {row['setting']} |")
    print(f"| oracle | | {oracle['apgr']:.3f} | | {oracle['cpt50']:.0%} | {oracle['cpt80']:.0%} "
          f"| | |")

    print("\nPaired APGR differences, 95% CI:")
    results["comparisons"] = {}
    for name, base in COMPARISONS:
        diff = paired_bootstrap(scores[name], small, large, baseline_scores=scores[base],
                                resamples=args.resamples)
        point = results["routers"][name]["apgr"] - results["routers"][base]["apgr"]
        results["comparisons"][f"{name} - {base}"] = {"diff": point, "ci": diff["diff_ci"]}
        print(f"  {name} - {base}: {point:+.3f} [{diff['diff_ci'][0]:+.3f}, "
              f"{diff['diff_ci'][1]:+.3f}]")

    print("\nAPGR by source (test split):")
    results["by_source"] = {}
    shown = ("minilm_knn", "tfidf_ridge", "seven_features_ridge", "seven_features_gbm", "length")
    for prefix, label in MAIN_SOURCES.items():
        mask = np.array([d.startswith(prefix) for d in test["dataset"]])
        gap = large[mask].mean() - small[mask].mean()
        if mask.sum() < 50 or gap < MIN_SOURCE_GAP:
            print(f"  {label:<18} n={mask.sum():<5} skipped: quality gap {gap:+.3f} is too small")
            continue
        row = {name: summarize(scores[name][mask], small[mask], large[mask])["apgr"]
               for name in shown}
        results["by_source"][label] = {"n": int(mask.sum()), **row}
        print(f"  {label:<18} n={mask.sum():<5} " +
              "  ".join(f"{k} {v:.3f}" for k, v in row.items()))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"sprout_replay_{args.pair}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
