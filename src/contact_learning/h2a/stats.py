"""Paired statistics for H2-001A.

Every student seed is evaluated on the SAME held-out episodes, so episodes and
student seeds are crossed. Paired differences are arranged as a matrix
D[student_seed, episode] and resampled with a crossed bootstrap (episodes and
seeds independently, with replacement). Episode-only and seed-only intervals
are reported too, to separate environment variation from training variation.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def paired_matrix(df: pd.DataFrame, a: str, b: str, metric: str) -> np.ndarray:
    """Matrix of metric(a) - metric(b), rows = student seeds, cols = episodes present for both."""
    w = df[df.condition.isin([a, b])].pivot_table(index=["student_seed", "seed"], columns="condition",
                                                   values=metric, aggfunc="first")
    w = w.dropna()
    d = (w[a].astype(float) - w[b].astype(float)).unstack("seed")
    d = d.dropna(axis=1)
    return d.to_numpy()


def crossed_bootstrap(D: np.ndarray, n_boot: int = 4000, seed: int = 0, alpha: float = 0.05) -> Dict:
    S, E = D.shape
    rng = np.random.default_rng(seed)
    est = float(D.mean())
    bs, be, bc = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, S, S)
        ei = rng.integers(0, E, E)
        bc[i] = D[np.ix_(si, ei)].mean()
        be[i] = D[:, ei].mean()
        bs[i] = D[si].mean()
    q = lambda x: (float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2)))
    p = float(min(1.0, 2 * min((bc <= 0).mean(), (bc >= 0).mean())))
    if np.all(D == 0):
        p = 1.0
    return {"est": est, "ci": q(bc), "ci_episodes_only": q(be), "ci_seeds_only": q(bs), "p_boot": p,
            "n_seeds": S, "n_episodes": E, "per_seed": D.mean(1).tolist()}


def mean_ci(df: pd.DataFrame, cond: str, metric: str, n_boot: int = 4000, seed: int = 0) -> Dict:
    w = df[df.condition == cond].pivot_table(index="student_seed", columns="seed", values=metric, aggfunc="first")
    w = w.dropna(axis=1)
    return crossed_bootstrap(w.to_numpy(dtype=float), n_boot, seed)


def verdict(res: Dict, delta: float, p_adj: float) -> str:
    lo, hi = res["ci"]
    if lo > 0 and res["est"] >= delta and p_adj < 0.05:
        return "positive (>= delta)"
    if lo > 0 and p_adj < 0.05:
        return "positive but below delta"
    if hi < 0 and res["est"] <= -delta and p_adj < 0.05:
        return "negative (<= -delta)"
    if hi < 0 and p_adj < 0.05:
        return "negative but above -delta"
    if lo > -delta and hi < delta:
        return "equivalent within delta"
    return "inconclusive"
