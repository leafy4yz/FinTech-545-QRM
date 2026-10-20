"""
qrmlib.conditional -- conditional distributions of a multivariate normal (Week 2).

Partition X = [X1, X2] with X2 = a observed:

    mu_bar    = mu_1 + Sigma_12 Sigma_22^{-1} (a - mu_2)
    Sigma_bar = Sigma_11 - Sigma_12 Sigma_22^{-1} Sigma_21      (no `a` in it)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


def conditional_normal(mu, sigma, known: Sequence[int], a):
    """
    General partitioned-normal conditional.

    mu, sigma : mean vector and covariance matrix of the full vector
    known     : indices of the observed block (X2)
    a         : observed values for those indices
    Returns (mu_bar, sigma_bar) for the remaining (unknown) indices, in order.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    known = list(known)
    unknown = [i for i in range(len(mu)) if i not in known]
    s11 = sigma[np.ix_(unknown, unknown)]
    s12 = sigma[np.ix_(unknown, known)]
    s22 = sigma[np.ix_(known, known)]
    k = s12 @ np.linalg.inv(s22)
    mu_bar = mu[unknown] + k @ (np.asarray(a, dtype=float) - mu[known])
    sigma_bar = s11 - k @ s12.T
    return mu_bar, sigma_bar


@dataclass
class ConditionalLine:
    """E[y | x] = alpha + beta x, with constant conditional variance."""
    alpha: float
    beta: float
    cond_var: float
    mu_x: float
    sd_x: float

    @property
    def cond_std(self) -> float:
        return float(np.sqrt(self.cond_var))

    def mean(self, x):
        return self.alpha + self.beta * np.asarray(x, dtype=float)

    def band(self, x, z: float = 1.96):
        """(lower, upper) prediction band mean +/- z * cond_std at each x."""
        m = self.mean(x)
        return m - z * self.cond_std, m + z * self.cond_std


def conditional_line(df: pd.DataFrame, x: str, y: str) -> ConditionalLine:
    """
    Bivariate case from the sample covariance (n - 1):
    beta = Cov(x, y) / Var(x),  alpha = mean(y) - beta mean(x),
    Var(y | x) = Var(y) - Cov(x, y)^2 / Var(x).
    """
    s = df[[x, y]].cov()
    sxx, syy, sxy = s.loc[x, x], s.loc[y, y], s.loc[x, y]
    mx, my = df[x].mean(), df[y].mean()
    beta = sxy / sxx
    return ConditionalLine(float(my - beta * mx), float(beta),
                           float(syy - sxy ** 2 / sxx), float(mx), float(np.sqrt(sxx)))


def band_inside(line: ConditionalLine, x, y, z: float = 1.96) -> np.ndarray:
    """Boolean array: is each observation inside its own +/- z band?"""
    lo, hi = line.band(x, z)
    y = np.asarray(y, dtype=float)
    return (y >= lo) & (y <= hi)


def coverage_by_bucket(line: ConditionalLine, x, inside,
                       edges=(1.0, 2.0)) -> pd.DataFrame:
    """
    Coverage split on |z| = |x - mean(x)| / sd(x):  |z| < 1, 1 <= |z| < 2, |z| >= 2
    (for the default edges).  Returns n and coverage per bucket.
    """
    z = np.abs(np.asarray(x, dtype=float) - line.mu_x) / line.sd_x
    inside = np.asarray(inside)
    lo_edges = (0.0,) + tuple(edges)
    hi_edges = tuple(edges) + (np.inf,)
    rows = {}
    for lo, hi in zip(lo_edges, hi_edges):
        if lo == 0.0:
            name, mask = f"|z1| < {hi:g}", z < hi
        elif np.isinf(hi):
            name, mask = f"|z1| >= {lo:g}", z >= lo
        else:
            name, mask = f"{lo:g} <= |z1| < {hi:g}", (z >= lo) & (z < hi)
        n_b = int(mask.sum())
        rows[name] = {"n": n_b, "coverage": inside[mask].mean() if n_b > 0 else np.nan}
    return pd.DataFrame(rows).T
