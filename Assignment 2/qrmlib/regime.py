"""
qrmlib.regime -- locating a volatility break and the kurtosis a mixture of
normal regimes produces.
"""
from __future__ import annotations

import numpy as np


def variance_breakpoint(x, min_segment: int = 10):
    """
    Single change point in variance by maximum likelihood.

    For every split k (first segment x[:k], second x[k:]), each segment is
    treated as zero-mean normal with its own variance; the profile
    log-likelihood is  -0.5 * (k ln s1^2 + (n-k) ln s2^2)  up to a constant.
    Returns (k_best, loglik_by_k) where k_best is the index of the first
    observation in the new regime.  Used only to CHECK the eyeballed regime
    from the plot.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    c = np.cumsum(x ** 2)
    ll = np.full(n, -np.inf)
    for k in range(min_segment, n - min_segment + 1):
        v1 = c[k - 1] / k
        v2 = (c[-1] - c[k - 1]) / (n - k)
        ll[k] = -0.5 * (k * np.log(v1) + (n - k) * np.log(v2))
    return int(np.argmax(ll)), ll


def regime_stds(x, k_recent: int, ddof: int = 1) -> tuple[float, float]:
    """Standard deviation of the older part x[:-k] and the recent part x[-k:]."""
    x = np.asarray(x, dtype=float)
    return float(x[:-k_recent].std(ddof=ddof)), float(x[-k_recent:].std(ddof=ddof))


def normal_mixture_moments(weights, sigmas) -> dict:
    """
    Std dev and excess kurtosis of a mixture of ZERO-MEAN normals.

        Var  = sum p_i s_i^2
        E X^4 = 3 sum p_i s_i^4
        excess kurtosis = 3 * sum p_i s_i^4 / (sum p_i s_i^2)^2 - 3

    Any spread in the s_i makes this strictly positive, even though every
    component is normal (Jensen's inequality on s^2).
    """
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    s2 = np.asarray(sigmas, dtype=float) ** 2
    m2 = float(np.sum(p * s2))
    m4 = float(3.0 * np.sum(p * s2 ** 2))
    return {"std": np.sqrt(m2), "excess kurtosis": m4 / m2 ** 2 - 3.0}
