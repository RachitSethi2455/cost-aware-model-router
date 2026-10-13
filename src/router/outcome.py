"""Outcome-trained router: an additive model over the seven features.

The first router learned from 60 hand labels of how hard a question *looks*.
On recorded outcomes it did no better than prompt length (README, "Offline
evaluation on SPROUT"). This one learns from what happened: for 30,968 SPROUT
prompts answered by Llama 3.1 8B and 70B, how much higher the 70B answer was
graded (0-1 scale). It predicts that gain from the same seven features.

The model is additive: predicted gain = base + one step function per feature,
so every prediction is an exact sum of per-feature contributions, as with the
logistic regression it replaces. evals/train_outcome_router.py fits it and
writes the step functions to outcome_model.json; serving reads that file with
plain Python, so there is no pickle, no scikit-learn at request time, and
nothing from the dataset's text in the repository.

The routing score is the percentile of a prompt's predicted gain among
SPROUT's validation prompts (0 = the large model helps least, 1 = most).
Threshold t therefore sends roughly the top (1 - t) share of that traffic to
the large model: 0.5 means about half.
"""

from __future__ import annotations

import json
import sys
from bisect import bisect_right
from pathlib import Path

from . import heuristic
from .features import FEATURE_NAMES, extract

MODEL_PATH = Path(__file__).with_name("outcome_model.json")


class OutcomeRouter:
    def __init__(self, path: Path = MODEL_PATH):
        self.trained = path.exists()
        if not self.trained:
            print(f"[router] warning: {path.name} not found; using the heuristic router.",
                  file=sys.stderr)
            return
        spec = json.loads(path.read_text(encoding="utf-8"))
        if list(spec["features"]) != FEATURE_NAMES:
            raise ValueError(f"{path.name} was trained on other features; "
                             "re-run evals/train_outcome_router.py")
        self.base = spec["base"]
        self.steps = {name: (f["from"], f["contribution"]) for name, f in spec["features"].items()}
        self.scale_gain = spec["score_scale"]["gain"]
        self.scale_level = spec["score_scale"]["level"]

    def _terms(self, query: str) -> list[tuple[str, float, float]]:
        """(feature, value, contribution to the predicted gain) per feature."""
        values = extract(query).to_vector()
        terms = []
        for name, value in zip(FEATURE_NAMES, values, strict=True):
            starts, contributions = self.steps[name]
            terms.append((name, value, contributions[bisect_right(starts, value) - 1]))
        return terms

    def predicted_gain(self, query: str) -> float:
        return self.base + sum(t for _, _, t in self._terms(query))

    def score(self, gain: float) -> float:
        """Percentile of `gain` among validation prompts, interpolated."""
        xs, ys = self.scale_gain, self.scale_level
        i = bisect_right(xs, gain)
        if i == 0:
            return ys[0]
        if i == len(xs):
            return ys[-1]
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
        return y0 + (y1 - y0) * (gain - x0) / (x1 - x0)

    def route(self, query: str, threshold: float = 0.5) -> tuple[str, float]:
        if not self.trained:
            return heuristic.route(query, threshold)
        s = self.score(self.predicted_gain(query))
        return ("large" if s >= threshold else "small"), s

    def contributions(self, query: str) -> dict | None:
        """Per-feature push on the predicted gain, in grade points (0-1 scale).

        intercept + sum(contributions) = predicted_gain exactly. The intercept
        is the average prediction, so each contribution says how far this
        feature moves the prompt from an average one. None on the heuristic.
        """
        if not self.trained:
            return None
        terms = self._terms(query)
        return {
            "unit": "gain",
            "intercept": self.base,
            "predicted_gain": self.base + sum(t for _, _, t in terms),
            "features": [{"name": n, "value": v, "contribution": t} for n, v, t in terms],
        }
