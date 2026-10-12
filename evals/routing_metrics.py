"""Standard routing metrics, as defined by RouteLLM (arXiv 2406.18665).

A router gives every query a score; a higher score means "send this to the
large model". Sweeping the share c of queries sent to the large model (the
highest-scoring c) traces a cost-quality curve:

    quality(c) = mean over queries of (large score if routed large, else small score)
    PGR(c)     = (quality(c) - q_small) / (q_large - q_small)   performance gap recovered
    APGR       = area under PGR(c) for c in [0, 1]
    CPT(x)     = smallest c with PGR(c) >= x                     call-performance threshold

References: a random router recovers PGR(c) = c on average, so its APGR is
0.5. The oracle sends exactly the queries where the large model beats the
small one, cheapest first. PGR can exceed 1 when the small model is sometimes
better than the large one, because always-large pays for those queries too.

numpy only, so it is importable in CI without the evaluation extras.
"""

from __future__ import annotations

import numpy as np

COST_GRID = np.linspace(0.0, 1.0, 101)


def quality_curve(router_scores, small, large, grid=COST_GRID, seed: int = 0) -> np.ndarray:
    """Mean quality at each large-call share in `grid`.

    Ties in router_scores are broken at random (fixed seed) so a router that
    gives many queries the same score is not credited with a lucky order.
    """
    scores = np.asarray(router_scores, dtype=float)
    small = np.asarray(small, dtype=float)
    large = np.asarray(large, dtype=float)
    n = len(scores)
    jitter = np.random.default_rng(seed).random(n) * 1e-9
    order = np.lexsort((jitter, -scores))  # primary: score descending
    gain = (large - small)[order]
    cumulative = np.concatenate([[0.0], np.cumsum(gain)])
    # floor, not round: at share c the router may use at most c * n large
    # calls, so it is never credited with fewer calls than it makes.
    k = np.floor(np.asarray(grid) * n + 1e-9).astype(int)
    return (small.sum() + cumulative[k]) / n


def pgr_curve(quality, small, large) -> np.ndarray:
    q_small, q_large = float(np.mean(small)), float(np.mean(large))
    gap = q_large - q_small
    if gap <= 0:
        raise ValueError("the large model is not better than the small one on this data")
    return (np.asarray(quality) - q_small) / gap


def random_apgr(n: int, grid=COST_GRID) -> float:
    """Expected APGR of random routing on n queries: 0.5 - 1/(2n) once the
    call count is floored to whole queries, i.e. 0.5 for any real test set."""
    grid = np.asarray(grid)
    return apgr(np.floor(grid * n + 1e-9) / n, grid)


def apgr(pgr, grid=COST_GRID) -> float:
    """Area under the PGR curve over large-call share in [0, 1]."""
    pgr, grid = np.asarray(pgr), np.asarray(grid)
    return float(np.sum((pgr[1:] + pgr[:-1]) / 2 * np.diff(grid)))


def cpt(pgr, target: float, grid=COST_GRID) -> float:
    """Smallest large-call share reaching PGR >= target (nan if never)."""
    reached = np.nonzero(np.asarray(pgr) >= target)[0]
    return float(np.asarray(grid)[reached[0]]) if len(reached) else float("nan")


def auroc(labels, scores) -> float:
    """Area under the ROC curve (Mann-Whitney), with tied scores counted half."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):  # average ranks over ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def summarize(router_scores, small, large, grid=COST_GRID, seed: int = 0) -> dict:
    """APGR, CPT(50%), CPT(80%) and the PGR curve for one router."""
    pgr = pgr_curve(quality_curve(router_scores, small, large, grid, seed), small, large)
    return {"apgr": apgr(pgr, grid), "cpt50": cpt(pgr, 0.5, grid), "cpt80": cpt(pgr, 0.8, grid),
            "pgr": pgr}


def paired_bootstrap(router_scores, small, large, baseline_scores=None, resamples: int = 1000,
                     seed: int = 0) -> dict:
    """95% intervals for APGR, and for APGR minus a baseline's APGR on the same
    resampled queries (random routing if baseline_scores is None: APGR 0.5)."""
    rng = np.random.default_rng(seed)
    router_scores, small, large = (np.asarray(x, dtype=float)
                                   for x in (router_scores, small, large))
    base = None if baseline_scores is None else np.asarray(baseline_scores, dtype=float)
    n = len(router_scores)
    values, diffs = [], []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        s, lg = small[idx], large[idx]
        if lg.mean() <= s.mean():
            continue  # PGR undefined in this resample
        value = summarize(router_scores[idx], s, lg)["apgr"]
        reference = random_apgr(n) if base is None else summarize(base[idx], s, lg)["apgr"]
        values.append(value)
        diffs.append(value - reference)

    def interval(xs):
        xs = np.sort(xs)
        return float(xs[int(0.025 * len(xs))]), float(xs[int(0.975 * len(xs)) - 1])

    return {"apgr_ci": interval(values), "diff_ci": interval(diffs), "resamples": len(values)}
