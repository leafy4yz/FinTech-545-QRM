"""
qrmlib.regression -- y = alpha + beta x + e fitted three ways (Week 2, Assignment 1):

    ols_se          closed-form least squares with classical standard errors
    mle_regression_normal  maximum likelihood, normal errors   (k = 3: alpha, beta, sigma)
    mle_regression_t       maximum likelihood, Student t errors (k = 4: alpha, beta, scale, nu)

plus AICc for choosing between the error models (OLS without standard errors, for
the factor model, is qrmlib.factor.ols).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .criteria import aicc


@dataclass
class OLSResult:
    alpha: float
    beta: float
    se_alpha: float
    se_beta: float
    s: float            # residual std dev, s^2 = SSE / (n - 2)
    resid: np.ndarray


def ols_se(x, y) -> OLSResult:
    """
    OLS of y on x with an intercept, by the normal equations.

    s^2 = SSE / (n - 2);  SE(beta) = sqrt(s^2 / sum (x - xbar)^2);
    SE(alpha) = sqrt(s^2 (1/n + xbar^2 / sum (x - xbar)^2)).
    These are the same numbers statsmodels reports (bse, mse_resid).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    X = np.column_stack([np.ones(n), x])
    # beta_hat = (X'X)^{-1} X'y, computed through the pseudo-inverse exactly as
    # statsmodels does, so the digits match statsmodels.OLS bit for bit
    pinv = np.linalg.pinv(X)
    alpha, beta = pinv @ y
    e = y - X @ np.array([alpha, beta])
    s2 = e @ e / (n - 2)
    # Var(beta_hat) = s^2 (X'X)^{-1};  the diagonal gives
    # SE(beta) = sqrt(s^2 / sum (x - xbar)^2) and SE(alpha) = sqrt(s^2 (1/n + xbar^2 / Sxx))
    se_alpha, se_beta = np.sqrt(np.diag(s2 * (pinv @ pinv.T)))
    return OLSResult(float(alpha), float(beta), float(se_alpha), float(se_beta),
                     float(np.sqrt(s2)), e)


@dataclass
class MLERegression:
    kind: str              # "normal" or "t"
    alpha: float
    beta: float
    scale: float           # sigma for normal, t scale s for t
    nu: Optional[float]
    loglik: float
    n: int

    @property
    def k(self) -> int:
        return 3 if self.kind == "normal" else 4

    @property
    def aicc(self) -> float:
        return aicc(self.loglik, self.k, self.n)

    @property
    def error_dist(self):
        """Fitted error distribution (mean 0) as a frozen scipy distribution."""
        if self.kind == "normal":
            return stats.norm(loc=0, scale=self.scale)
        return stats.t(df=self.nu, loc=0, scale=self.scale)


def normal_negll(params, x, y) -> float:
    """Negative log likelihood of y = a + b x + e, e ~ N(0, s^2)."""
    a, b, s = params
    e = y - a - b * x
    n = e.shape[0]
    ll = -n / 2 * np.log(2 * np.pi * s ** 2) - (e @ e) / (2 * s ** 2)
    return -ll


def t_negll(params, x, y) -> float:
    """Negative log likelihood of y = a + b x + e, e ~ s * t_nu."""
    a, b, s, nu = params
    e = y - a - b * x
    # location-scale t: pdf(e) = (1/s) t.pdf(e/s, nu)
    return -np.sum(stats.t.logpdf(e / s, df=nu) - np.log(s))


def mle_regression_normal(x, y, start=None) -> MLERegression:
    """MLE with normal errors.  Starts from OLS unless `start` = (a, b, s) is given."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if start is None:
        o = ols_se(x, y)
        start = [o.alpha, o.beta, o.s]
    res = minimize(normal_negll, x0=list(start), args=(x, y),
                   bounds=[(None, None), (None, None), (1e-8, None)])
    a, b, s = res.x
    return MLERegression("normal", float(a), float(b), float(s), None, float(-res.fun), x.shape[0])


def mle_regression_t(x, y, start=None, nu0: float = 4.0) -> MLERegression:
    """MLE with Student t errors.  Starts from OLS with nu = nu0 unless `start` = (a, b, s, nu)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if start is None:
        o = ols_se(x, y)
        start = [o.alpha, o.beta, o.s, nu0]
    res = minimize(t_negll, x0=list(start), args=(x, y),
                   bounds=[(None, None), (None, None), (1e-8, None), (1e-3, None)])
    a, b, s, nu = res.x
    return MLERegression("t", float(a), float(b), float(s), float(nu), float(-res.fun), x.shape[0])
