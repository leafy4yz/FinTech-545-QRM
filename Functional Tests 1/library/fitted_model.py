"""Port of library/fitted_model.jl -- only the pieces test_setup.jl uses:
FittedModel, aicc, fit_normal, fit_general_t, fit_regression_t,
fit_nig_moments, fit_NIG_mle.

fit_general_johnsonsu*, fit_weibull, fit_skewnormal, and the FastNIGCDF
helper were not ported: test_setup.jl never calls them (the NIG fits only
need mu/alpha/beta/delta, which don't require the fast-CDF machinery).

Where Julia used JuMP/Ipopt for the T-distribution MLEs, this port uses
scipy.optimize with the same starting values and constraints (s >= 1e-6,
nu >= 2.0001) that the Julia code uses -- a different optimizer, but the
same well-behaved concave-ish objective, so it should land on the same
optimum to numerical precision. fit_NIG_mle calls scipy.stats.norminvgauss
.fit() directly, exactly like the Julia code does through PyCall, so that
one should match Julia's output very closely.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import optimize, stats


@dataclass
class FittedModel:
    beta: Optional[np.ndarray]          # regression coefficients, or None
    error_model: object                 # a scipy.stats frozen distribution (has .logpdf/.cdf/.ppf)
    eval: Callable                      # eval(u) -> quantile(s), or eval(x, u) for the regression fit
    errors: np.ndarray
    u: np.ndarray
    n_dist_params: int                  # size(Distributions.params(d),1) in the Julia code
    params: dict = field(default_factory=dict)  # convenience: named distribution parameters


def aicc(fitted_model: FittedModel, data):
    """Corrected AIC. Port of Julia's aicc(m::FittedModel, data)."""
    data = np.asarray(data, dtype=float)
    d = fitted_model.error_model
    ll = np.sum(d.logpdf(data))
    n = len(data)
    k = fitted_model.n_dist_params
    if fitted_model.beta is not None:
        k += len(fitted_model.beta)
    return -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def fit_normal(x):
    """Port of Julia's fit_normal(x)."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std(ddof=1)  # Julia's Statistics.std default: corrected (n-1)

    error_model = stats.norm(loc=m, scale=s)
    errors = x - m
    u = error_model.cdf(x)

    def eval_fn(u):
        return error_model.ppf(u)

    return FittedModel(None, error_model, eval_fn, errors, u,
                        n_dist_params=2, params={"mu": m, "sigma": s})


def fit_general_t(x):
    """MLE fit of a location-scale Student's t. Port of Julia's
    fit_general_t(x) (JuMP/Ipopt -> scipy.optimize.minimize)."""
    x = np.asarray(x, dtype=float)

    start_m = x.mean()
    kurt = stats.kurtosis(x, fisher=True, bias=True)  # StatsBase.kurtosis default: biased/population excess kurtosis
    start_nu = 6.0 / kurt + 4.0
    start_s = np.sqrt(x.var(ddof=1) * (start_nu - 2.0) / start_nu)

    def neg_ll(theta):
        mu, s, nu = theta
        return -np.sum(stats.t.logpdf(x, df=nu, loc=mu, scale=s))

    x0 = [start_m, max(start_s, 1e-3), max(start_nu, 2.0001)]
    bounds = [(None, None), (1e-6, None), (2.0001, None)]

    res = optimize.minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"Optimization Failed - {res.message}")

    mu, s, nu = res.x
    error_model = stats.t(df=nu, loc=mu, scale=s)

    errors = x - mu
    u = error_model.cdf(x)

    def eval_fn(u):
        return error_model.ppf(u)

    return FittedModel(None, error_model, eval_fn, errors, u,
                        n_dist_params=3, params={"mu": mu, "sigma": s, "nu": nu})


def fit_regression_t(y, x):
    """Regression with Student's-t errors (location fixed at 0), MLE.
    Port of Julia's fit_regression_t(y, x)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n = x.shape[0]
    big_x = np.hstack([np.ones((n, 1)), x])
    n_b = big_x.shape[1]

    b_start, *_ = np.linalg.lstsq(big_x, y, rcond=None)
    e = y - big_x @ b_start
    kurt = stats.kurtosis(e, fisher=True, bias=True)
    start_nu = 6.0 / kurt + 4.0
    start_s = np.sqrt(e.var(ddof=1) * (start_nu - 2.0) / start_nu)

    def neg_ll(theta):
        s = theta[0]
        nu = theta[1]
        beta = theta[2:]
        resid = y - big_x @ beta
        return -np.sum(stats.t.logpdf(resid, df=nu, loc=0.0, scale=s))

    x0 = np.concatenate([[max(start_s, 1e-3)], [max(start_nu, 2.0001)], b_start])
    bounds = [(1e-6, None), (2.0001, None)] + [(None, None)] * n_b

    res = optimize.minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"Optimization Failed - {res.message}")

    s = res.x[0]
    nu = res.x[1]
    beta = res.x[2:]

    error_model = stats.t(df=nu, loc=0.0, scale=s)  # m is constrained to 0, as in the Julia code

    def eval_fn(x_new, u):
        x_new = np.asarray(x_new, dtype=float)
        if x_new.ndim == 1:
            x_new = x_new.reshape(-1, 1)
        n2 = x_new.shape[0]
        big_x_new = np.hstack([np.ones((n2, 1)), x_new])
        return big_x_new @ beta + error_model.ppf(u)

    errors = y - eval_fn(x, np.full(n, 0.5))
    u = error_model.cdf(errors)

    return FittedModel(beta, error_model, eval_fn, errors, u,
                        n_dist_params=3, params={"mu": 0.0, "sigma": s, "nu": nu})


def _nig_fitted_model(mu, alpha, beta, delta, x):
    """Shared tail of fit_nig_moments / fit_NIG_mle. scipy.stats.norminvgauss
    is parameterized as (a, b, loc, scale) with a = alpha*delta, b = beta*delta,
    loc = mu, scale = delta -- same conversion the Julia code's comments spell out
    for comparing against scipy's own fit()."""
    x = np.asarray(x, dtype=float)
    a = alpha * delta
    b = beta * delta
    error_model = stats.norminvgauss(a, b, loc=mu, scale=delta)

    errors = x - error_model.mean()
    u = error_model.cdf(x)

    def eval_fn(u):
        return error_model.ppf(u)

    return FittedModel(None, error_model, eval_fn, errors, u,
                        n_dist_params=4,
                        params={"mu": mu, "alpha": alpha, "beta": beta, "delta": delta})


def fit_nig_moments(x):
    """Closed-form method-of-moments fit of a Normal Inverse Gaussian.
    Port of Julia's fit_nig_moments(x)."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    v = x.var(ddof=1)  # Julia's Statistics.var default: corrected (n-1)
    s = stats.skew(x, bias=True)             # StatsBase.skewness default: biased/population
    k = stats.kurtosis(x, fisher=True, bias=True)  # StatsBase.kurtosis default: biased/population excess

    if not (k > 0):
        raise ValueError(f"NIG method of moments needs positive excess kurtosis, got {k}")
    t = s ** 2 / k
    if not (t < 3 / 5):
        raise ValueError("Sample is outside the NIG region: excess kurtosis must exceed (5/3)*skew^2")

    rho2 = t / (3 - 4 * t)
    rho = np.sign(s) * np.sqrt(rho2)

    d_ = 3 * (1 + 4 * rho2) / k  # delta*gamma
    alpha = np.sqrt(d_ / (v * (1 - rho2) ** 2))
    beta = rho * alpha
    gamma = alpha * np.sqrt(1 - rho2)
    delta = d_ / gamma
    mu = m - delta * beta / gamma

    return _nig_fitted_model(mu, alpha, beta, delta, x)


def fit_NIG_mle(x):
    """MLE fit of a Normal Inverse Gaussian via scipy.stats.norminvgauss.fit
    -- the same routine the Julia code calls through PyCall, so this should
    match the Julia output closely. Port of Julia's fit_NIG_mle(x)."""
    x = np.asarray(x, dtype=float)
    a, b, mu, delta = stats.norminvgauss.fit(x)
    alpha = a / delta
    beta = b / delta
    return _nig_fitted_model(mu, alpha, beta, delta, x)
