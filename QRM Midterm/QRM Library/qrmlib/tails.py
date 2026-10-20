"""
qrmlib.tails -- counting tail events across scenarios and checking whether a
risk measure is subadditive (Weeks 4 and 5).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def tail_counts(returns: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Number of scenarios in which each column is BELOW `threshold`
    (e.g. threshold=-0.20 counts 'loses more than 20%'), plus:
        'any' : at least one column below the threshold
        'all' : every column below the threshold at once
    """
    below = returns.to_numpy(dtype=float) < threshold
    out = pd.Series(below.sum(axis=0), index=returns.columns, dtype=int)
    out["any"] = int(below.any(axis=1).sum())
    out["all"] = int(below.all(axis=1).sum())
    return out


def subadditivity(measure: Callable[[np.ndarray], float], x, y) -> dict:
    """
    Check q(X + Y) <= q(X) + q(Y) for a risk measure q reported as a positive loss.

    `measure` is any function of a P&L sample, e.g.
        lambda pnl: var_historical(pnl, alpha=0.05)
    Returns the standalone values, their sum, the combined value, the
    diversification benefit (sum - combined; negative means combining ADDED
    risk) and whether subadditivity holds.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    qx, qy, qxy = measure(x), measure(y), measure(x + y)
    return {"q(X)": qx, "q(Y)": qy, "q(X)+q(Y)": qx + qy, "q(X+Y)": qxy,
            "benefit": qx + qy - qxy, "subadditive": bool(qxy <= qx + qy + 1e-9)}
