"""
qrmlib.criteria -- information criteria for choosing between fitted models (Week 2).

k = number of estimated parameters, n = sample size, loglik = maximised log likelihood.
Lower is better for all three.
"""
from __future__ import annotations

import numpy as np


def aic(loglik: float, k: int) -> float:
    """AIC = -2 LL + 2k."""
    return float(-2.0 * loglik + 2.0 * k)


def aicc(loglik: float, k: int, n: int) -> float:
    """AICc = -2 LL + 2k + 2k(k+1)/(n-k-1).  Lower is better."""
    return float(-2.0 * loglik + 2.0 * k + 2.0 * k * (k + 1) / (n - k - 1))


def bic(loglik: float, k: int, n: int) -> float:
    """BIC = -2 LL + k ln n.  Lower is better."""
    return float(-2.0 * loglik + k * np.log(n))
