"""Learned router — logistic regression over the six extracted features.

Trained on the eval set's ground-truth labels (see evals/train_router.py).
Falls back to the heuristic when no trained artifact is present, so the API
works on a clean checkout before you've trained anything.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import heuristic
from .config import MODEL_ARTIFACT
from .features import FEATURE_NAMES, extract


def build_pipeline() -> Pipeline:
    """Scaler + logistic regression.

    Scaling matters: n_words spans 3-300 while has_code is 0/1, and an
    unscaled LR would let word count dominate the coefficients.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train(queries: list[str], labels: list[int]) -> Pipeline:
    """labels: 1 = needs large model, 0 = small model suffices."""
    X = np.array([extract(q).to_vector() for q in queries])
    y = np.array(labels)
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe


def save(pipe: Pipeline, path: Path = MODEL_ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: Path = MODEL_ARTIFACT) -> Pipeline | None:
    if not path.exists():
        return None
    return joblib.load(path)


def coefficients(pipe: Pipeline) -> dict[str, float]:
    """Readable feature weights — useful for the README and for interviews."""
    clf = pipe.named_steps["clf"]
    return dict(zip(FEATURE_NAMES, clf.coef_[0].round(3).tolist()))


class ClassifierRouter:
    """Wraps the trained pipeline with a heuristic fallback."""

    def __init__(self, path: Path = MODEL_ARTIFACT):
        self.pipe = load(path)
        self.trained = self.pipe is not None

    def route(self, query: str, threshold: float = 0.5) -> tuple[str, float]:
        if not self.trained:
            return heuristic.route(query, threshold)
        X = np.array([extract(query).to_vector()])
        p = float(self.pipe.predict_proba(X)[0][1])
        return ("large" if p >= threshold else "small"), p
