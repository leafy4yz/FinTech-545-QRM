"""Port of library/ewCov.jl"""
import numpy as np


def exp_weights(m, lam):
    """Exponential weights of length m (oldest observation first, row 0),
    normalized to sum to 1. Port of Julia's expW(m, lambda)."""
    i = np.arange(1, m + 1)
    w = (1.0 - lam) * lam ** (m - i)
    w = w / w.sum()
    return w


def ew_covar(x, lam):
    """Exponentially weighted covariance matrix of x (rows = time, oldest
    first; columns = variables). Port of Julia's ewCovar(x, lambda)."""
    x = np.asarray(x, dtype=float)
    m, n = x.shape
    w = exp_weights(m, lam)
    xm = np.sqrt(w)[:, None] * (x - w @ x)
    return xm.T @ xm
