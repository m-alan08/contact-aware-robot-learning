"""Uncertainty estimates. The unit of analysis is always a trial (one episode, one
correction), never a timestep."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, stat=np.mean):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = stat(x[idx], axis=1)
    return float(stat(x)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def wilcoxon_p(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 6 or np.allclose(d, 0):
        return float("nan")
    return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return np.nan, np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact McNemar test for paired binary outcomes (b, c = discordant counts)."""
    n = b + c
    if n == 0:
        return float("nan")
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    order = np.argsort(p[ok])
    pv = p[ok][order]
    m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv)
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out.tolist()
