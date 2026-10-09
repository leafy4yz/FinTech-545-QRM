"""
qrmlib.portfolio -- building a covariance matrix from correlations and
standard deviations, and the portfolio variance formula from Week 2.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def full_history_std(df: pd.DataFrame) -> pd.Series:
    """
    Sample standard deviation (n - 1) of each column using EVERY observed value
    of that column (pandas skips NaN).  This is deliberate: the rank/consistency
    problem lives in the correlations, so each series keeps its full history for
    its own variance (Week 3, Section 4, item 7).
    """
    return df.std(ddof=1)


def cov_from_corr(corr, sd) -> np.ndarray:
    """Sigma = D C D with D = diag(sd): entry (i, j) = sd_i * corr_ij * sd_j."""
    d = np.diag(np.asarray(sd, dtype=float))
    return d @ np.asarray(corr, dtype=float) @ d


def portfolio_variance(weights, cov) -> float:
    """Variance of a portfolio with position vector w:  w' Sigma w."""
    w = np.asarray(weights, dtype=float)
    return float(w @ np.asarray(cov, dtype=float) @ w)


def tracking_weights(columns: Sequence[str], basket: Mapping[str, float],
                     index: str = "IDX") -> np.ndarray:
    """
    Position vector (in the order of ``columns``) for "long one unit of the
    index, short the weighted basket":  +1 on ``index``, -basket[name] on each
    basket member, 0 elsewhere.
    """
    w = np.zeros(len(columns))
    for k, name in enumerate(columns):
        if name == index:
            w[k] = 1.0
        elif name in basket:
            w[k] = -float(basket[name])
    return w


def scenario_pnl(returns: pd.DataFrame, holdings: Mapping[str, float]) -> pd.Series:
    """
    Dollar P&L of a set of positions in every scenario (row) of `returns`:
        P&L_s = sum_i holdings_i * r_{s,i}
    `holdings` maps column name -> dollars invested (negative = short).
    Assets not named in `holdings` are held at zero.
    """
    w = pd.Series(holdings, dtype=float)
    missing = set(w.index) - set(returns.columns)
    if missing:
        raise KeyError(f"no return column for {sorted(missing)}")
    pnl = returns[w.index].to_numpy(dtype=float) @ w.to_numpy()
    return pd.Series(pnl, index=returns.index, name="P&L")
