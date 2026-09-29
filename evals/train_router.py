"""Train the logistic-regression router and report honest CV metrics.

Labels come from the dataset's `prior` field by default. If you have already
run the benchmark, pass --from-benchmark to relabel using observed outcomes:
a query is labeled 'small suffices' only if the small model actually scored
full marks on it. That is the more defensible label source, because it reflects
observed outcomes rather than hand-assigned priors.

Usage:
    python evals/train_router.py
    python evals/train_router.py --from-benchmark results/benchmark_main.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from router.classifier import coefficients, save, train  # noqa: E402
from router.config import MODEL_ARTIFACT  # noqa: E402
from router.features import extract  # noqa: E402

DATASET = Path(__file__).parent / "dataset.jsonl"


def load_rows() -> list[dict]:
    return [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]


def labels_from_prior(rows: list[dict]) -> list[int]:
    return [1 if r["prior"] == "large" else 0 for r in rows]


def labels_from_benchmark(rows: list[dict], path: Path) -> list[int]:
    """A query needs the large model iff the small model did NOT score 2/2."""
    summaries = json.loads(path.read_text())
    small_arm = next(s for s in summaries if s["arm"] == "always_small")
    by_id = {r["id"]: r["score"] for r in small_arm["records"]}
    missing = [r["id"] for r in rows if r["id"] not in by_id]
    if missing:
        raise SystemExit(f"Benchmark missing {len(missing)} ids, e.g. {missing[:3]}")
    return [0 if by_id[r["id"]] == 2 else 1 for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-benchmark", type=Path, default=None)
    args = ap.parse_args()

    rows = load_rows()
    queries = [r["query"] for r in rows]
    if args.from_benchmark:
        labels = labels_from_benchmark(rows, args.from_benchmark)
        source = f"benchmark outcomes ({args.from_benchmark.name})"
    else:
        labels = labels_from_prior(rows)
        source = "dataset priors"

    n_large = sum(labels)
    print(f"Label source: {source}")
    print(f"Examples: {len(labels)}  (needs-large: {n_large}, small-suffices: {len(labels) - n_large})")

    X = np.array([extract(q).to_vector() for q in queries])
    y = np.array(labels)

    from router.classifier import build_pipeline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="accuracy")
    f1 = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="f1")
    print(f"5-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"5-fold CV F1:       {f1.mean():.3f} (+/- {f1.std():.3f})")
    print("NOTE: report the CV number, not train accuracy. With ~60 examples "
          "the std is wide; say so rather than quoting a point estimate.")

    pipe = train(queries, labels)
    save(pipe)
    print(f"\nSaved -> {MODEL_ARTIFACT}")
    print("\nFeature coefficients (positive = pushes toward large model):")
    for name, coef in sorted(coefficients(pipe).items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<28}{coef:>8.3f}")


if __name__ == "__main__":
    main()
