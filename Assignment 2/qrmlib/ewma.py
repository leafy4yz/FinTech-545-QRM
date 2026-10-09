"""
qrmlib.ewma -- exponentially weighted variance / covariance (Week 3).

Convention (same as the class code): observations are in time order, oldest
first.  The most recent observation gets raw weight (1 - lambda), the one before
it (1 - lambda) * lambda, and so on; the raw weights are then normalised to sum
to one because the sample is finite.
"""
from __future__ import annotations

import numpy as np


def ew_weights(n: int, lam: float) -> np.ndarray:
    """
    Normalised exponential weights for n observations, OLDEST FIRST.
    w[-1] (today) is the largest; w[t] proportional to (1-lam) * lam^(n-1-t).
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must be in (0, 1)")
    age = np.arange(n - 1, -1, -1)              # age 0 = most recent
    w = (1.0 - lam) * lam ** age
    return w / w.sum()


def ew_covariance(x, lam: float) -> np.ndarray:
    """
    Exponentially weighted covariance matrix of the columns of x (rows = time,
    oldest first).  Deviations are taken from the (unweighted) sample mean,
    as in the class implementation:  Sigma = sum_t w_t (x_t - m)(x_t - m)'.
    A 1-D x returns a 1x1 matrix.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    w = ew_weights(x.shape[0], lam)
    xm = x - x.mean(axis=0)
    return (w[:, None] * xm).T @ xm


def ew_variance(x, lam: float) -> float:
    """Exponentially weighted variance of a single series."""
    return float(ew_covariance(np.asarray(x, dtype=float).ravel(), lam)[0, 0])


def ew_std(x, lam: float) -> float:
    """Exponentially weighted standard deviation of a single series."""
    return float(np.sqrt(ew_variance(x, lam)))


def effective_sample_size(lam: float, n: int | None = None) -> float:
    """
    Kish effective sample size  n_eff = 1 / sum(w_t^2).

    With an infinite history this is (1 + lam) / (1 - lam).  Pass ``n`` to use
    the actual finite, normalised weights instead.
    """
    if n is None:
        return (1.0 + lam) / (1.0 - lam)
    w = ew_weights(n, lam)
    return float(1.0 / np.sum(w ** 2))


def half_life(lam: float) -> float:
    """Number of days for a weight to fall by half:  ln(0.5) / ln(lambda)."""
    return float(np.log(0.5) / np.log(lam))


def weight_on_recent(n: int, lam: float, k: int) -> float:
    """Fraction of the total (normalised) weight placed on the most recent k days."""
    return float(ew_weights(n, lam)[-k:].sum()) if k > 0 else 0.0


def ew_std_error(sigma: float, n_eff: float) -> float:
    """Approximate standard error of an EW volatility estimate: sigma / sqrt(2 n_eff)."""
    return float(sigma / np.sqrt(2.0 * n_eff))

