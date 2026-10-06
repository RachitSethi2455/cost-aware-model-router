"""Offline router analysis: baselines, ranking quality, threshold trade-off.

No API calls. Every model is scored on the same 20 reshuffled stratified
5-fold splits, using out-of-fold predictions only, so the comparison is fair.

Questions it answers:
  - Do the 7 hand-made features beat a plain bag-of-words text classifier,
    both on random splits and on a question category never seen in training
    (leave-one-category-out: does the model learn difficulty or topic words)?
  - How well does the router *rank* queries (AUC), independent of threshold?
  - What does moving the threshold trade: cheaper routing vs hard queries
    sent to the small model (which then rely on escalation)?

Usage:
    python evals/analyze_router.py
    python evals/analyze_router.py --markdown     # README-ready tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from router import heuristic  # noqa: E402
from router.classifier import build_pipeline  # noqa: E402
from router.features import extract  # noqa: E402

DATASET = Path(__file__).parent / "dataset.jsonl"
N_SPLITS, N_REPEATS = 5, 20
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def load() -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    return ([r["query"] for r in rows],
            np.array([int(r["prior"] == "large") for r in rows]),
            np.array([r["category"] for r in rows]))


def tfidf_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def out_of_fold(queries: list[str], y: np.ndarray) -> dict[str, list[np.ndarray]]:
    """P(needs large) for every query, one array per repeat, per model."""
    X7 = np.array([extract(q).to_vector() for q in queries])
    Xtext = np.array(queries, dtype=object)
    heur = np.array([heuristic.score(extract(q)) for q in queries])

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=0)
    oof = {name: [] for name in ("majority", "heuristic", "router", "tfidf")}
    current = {name: np.zeros(len(y)) for name in oof}

    for i, (tr, te) in enumerate(cv.split(X7, y)):
        current["majority"][te] = y[tr].mean()  # constant: the training prevalence
        current["heuristic"][te] = heur[te]     # no training
        current["router"][te] = build_pipeline().fit(X7[tr], y[tr]).predict_proba(X7[te])[:, 1]
        current["tfidf"][te] = tfidf_pipeline().fit(Xtext[tr], y[tr]).predict_proba(Xtext[te])[:, 1]
        if (i + 1) % N_SPLITS == 0:  # one full pass over the data
            for name in oof:
                oof[name].append(current[name].copy())
    return oof


def leave_one_category_out(queries: list[str], y: np.ndarray, cats: np.ndarray) -> dict[str, dict]:
    """Train on all other categories, test on one never seen. Correct counts per model.

    Random splits put near-duplicate phrasings in train and test; this asks
    whether a model learned *difficulty* or just the topic words of each
    category.
    """
    X7 = np.array([extract(q).to_vector() for q in queries])
    Xtext = np.array(queries, dtype=object)
    heur = np.array([heuristic.score(extract(q)) >= 0.5 for q in queries])
    result = {name: {} for name in ("majority", "heuristic", "router", "tfidf")}
    for c in sorted(set(cats)):
        te, tr = cats == c, cats != c
        result["majority"][c] = int(((y[tr].mean() >= 0.5) == y[te]).sum())
        result["heuristic"][c] = int((heur[te] == y[te]).sum())
        result["router"][c] = int((build_pipeline().fit(X7[tr], y[tr]).predict(X7[te]) == y[te]).sum())
        result["tfidf"][c] = int((tfidf_pipeline().fit(Xtext[tr], y[tr]).predict(Xtext[te]) == y[te]).sum())
    return result


def summarize(y: np.ndarray, repeats: list[np.ndarray], probabilistic: bool = True) -> dict:
    acc = [((p >= 0.5).astype(int) == y).mean() for p in repeats]
    auc = [roc_auc_score(y, p) for p in repeats]
    brier = [brier_score_loss(y, p) for p in repeats]
    out = {k: (float(np.mean(v)), float(np.std(v))) for k, v in
           (("accuracy", acc), ("auc", auc), ("brier", brier))}
    if not probabilistic:
        out["brier"] = None  # rule scores are not probabilities
    return out


def threshold_table(y: np.ndarray, repeats: list[np.ndarray]) -> list[dict]:
    hard, easy = y == 1, y == 0
    rows = []
    for t in THRESHOLDS:
        to_large = [p >= t for p in repeats]
        rows.append({
            "threshold": t,
            "pct_small": 100 * np.mean([(~d).mean() for d in to_large]),
            # Hard queries sent to the small model: these cost an escalation
            # round trip if the checks catch the failure, a weak answer if not.
            "hard_to_small": 100 * np.mean([(~d[hard]).mean() for d in to_large]),
            # Easy queries sent to the large model: correct answer, wasted money.
            "easy_to_large": 100 * np.mean([d[easy].mean() for d in to_large]),
        })
    return rows


LABELS = {
    "majority": "Majority class",
    "heuristic": "Heuristic rules",
    "tfidf": "TF-IDF + logistic regression (bag of words)",
    "router": "**Router: 7 features + logistic regression**",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    queries, y, cats = load()
    oof = out_of_fold(queries, y)
    loco = leave_one_category_out(queries, y, cats)
    print(f"{len(y)} queries ({y.sum()} need large). Random splits: {N_REPEATS}x repeated "
          f"{N_SPLITS}-fold CV, out-of-fold predictions, ± = spread across repeats. "
          f"Unseen category: leave-one-category-out over {len(set(cats))} categories.\n")

    print("| Model | Accuracy, random splits | Accuracy, unseen category | AUC | Brier (lower is better) |")
    print("|---|---|---|---|---|")
    for name in ("majority", "heuristic", "tfidf", "router"):
        if name == "majority":
            # Its only "ranking" is the training prevalence, which shifts
            # slightly between folds; that is noise, not signal.
            auc, brier = "0.500 (no ranking)", f"{summarize(y, oof[name])['brier'][0]:.3f}"
            acc = f"{100 * summarize(y, oof[name])['accuracy'][0]:.1f}%"
        else:
            s = summarize(y, oof[name], probabilistic=name != "heuristic")
            acc = f"{100 * s['accuracy'][0]:.1f}% ± {100 * s['accuracy'][1]:.1f}"
            auc = f"{s['auc'][0]:.3f} ± {s['auc'][1]:.3f}"
            brier = "n/a (scores, not probabilities)" if s["brier"] is None else \
                f"{s['brier'][0]:.3f} ± {s['brier'][1]:.3f}"
        unseen = f"{100 * sum(loco[name].values()) / len(y):.1f}%"
        print(f"| {LABELS[name]} | {acc} | {unseen} | {auc} | {brier} |")

    if not args.markdown:
        print("\nUnseen-category correct counts (router vs tf-idf):")
        for c in sorted(set(cats)):
            print(f"  {c:<28}{(cats == c).sum():>3}  router {loco['router'][c]:>2}  "
                  f"tfidf {loco['tfidf'][c]:>2}")

    print("\nRouter threshold trade-off (mean over repeats):\n")
    print("| Threshold | Routed to small | Hard queries sent to small | Easy queries sent to large |")
    print("|---|---|---|---|")
    for r in threshold_table(y, oof["router"]):
        bold = "**" if r["threshold"] == 0.5 else ""
        print(f"| {bold}{r['threshold']:.1f}{bold} | {r['pct_small']:.0f}% | "
              f"{r['hard_to_small']:.0f}% | {r['easy_to_large']:.0f}% |")
    if not args.markdown:
        print("\nHard-to-small misroutes rely on escalation; easy-to-large ones only cost money.")


if __name__ == "__main__":
    main()
