"""
qrmlib.var -- one-period Value at Risk (Week 4) and Expected Shortfall (Week 5).

Convention: VaR is reported as a POSITIVE number for a loss,
    VaR_alpha = -quantile_alpha(return distribution) * position value.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .fitting import TFit


def var_normal(sigma: float, alpha: float = 0.05, mu: float = 0.0, value: float = 1.0) -> float:
    """Normal VaR:  -(mu + sigma * z_alpha) * value."""
    return float(-(mu + sigma * stats.norm.ppf(alpha)) * value)


def var_t(fit: TFit, alpha: float = 0.05, value: float = 1.0) -> float:
    """Student t VaR from a fitted location-scale t:  -(mu + sigma * t_nu^{-1}(alpha)) * value."""
    return float(-fit.ppf(alpha) * value)


def var_historical(returns, alpha: float = 0.05, value: float = 1.0,
                   method: str = "class") -> float:
    """
    Historical-simulation VaR from a sample of returns.

    method="class"   : sort ascending and take the floor(alpha*n)-th smallest
                       return (1-based), exactly as in the class code
                       (``a = floor(.05*n); VaR = -sorted[a]``).
    method="quantile": numpy's linearly interpolated alpha-quantile.
    The two differ only by interpolation between neighbouring order statistics.
    """
    r = np.sort(np.asarray(returns, dtype=float))
    if method == "class":
        a = max(int(np.floor(alpha * r.size)), 1)
        q = r[a - 1]
    elif method == "quantile":
        q = np.quantile(r, alpha)
    else:
        raise ValueError("method must be 'class' or 'quantile'")
    return float(-q * value)


def var_noise_normal(sigma: float, n_eff: float, alpha: float = 0.05, value: float = 1.0) -> float:
    """
    One standard error of a normal VaR that comes from the volatility estimate
    alone:  |z_alpha| * value * sigma / sqrt(2 n_eff).
    """
    return float(-stats.norm.ppf(alpha) * value * sigma / np.sqrt(2.0 * n_eff))


# ---------------------------------------------------------------------------
# Expected Shortfall (Week 5).  Same sign convention as VaR: a positive number
# is a loss.  ES = -E[X | X <= -VaR].
# ---------------------------------------------------------------------------
def es_historical(returns, alpha: float = 0.05, value: float = 1.0,
                  method: str = "class") -> float:
    """
    Historical-simulation Expected Shortfall: minus the mean of the tail.

    method="class"   : the a = floor(alpha*n) smallest values, i.e. every
                       observation up to and including the one var_historical
                       (method="class") reports.  This pairs ES with VaR exactly
                       as the class code does (``ES = -mean(sorted[1:a])``).
    method="quantile": mean of every value <= numpy's interpolated alpha-quantile.

    Pass dollar P&L with value=1, or returns with value = position size.
    """
    r = np.sort(np.asarray(returns, dtype=float))
    if method == "class":
        a = max(int(np.floor(alpha * r.size)), 1)
        tail = r[:a]
    elif method == "quantile":
        tail = r[r <= np.quantile(r, alpha)]
    else:
        raise ValueError("method must be 'class' or 'quantile'")
    return float(-tail.mean() * value)


def es_normal(sigma: float, alpha: float = 0.05, mu: float = 0.0, value: float = 1.0) -> float:
    """Normal ES (Week 5, Section 1.1):  (-mu + sigma * phi(z_alpha) / alpha) * value."""
    return float((-mu + sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha) * value)


def es_t(fit: TFit, alpha: float = 0.05, value: float = 1.0) -> float:
    """
    Student t ES from a fitted location-scale t (closed form, needs nu > 1):

        q  = t_nu^{-1}(alpha)
        ES = -mu + sigma * (nu + q^2) / (nu - 1) * f_nu(q) / alpha

    sigma is the t SCALE (fit.sigma), not the standard deviation.
    """
    if fit.nu <= 1:
        raise ValueError("ES of a t distribution needs nu > 1")
    q = stats.t.ppf(alpha, fit.nu)
    es_std = (fit.nu + q * q) / (fit.nu - 1.0) * stats.t.pdf(q, fit.nu) / alpha
    return float((-fit.mu + fit.sigma * es_std) * value)


def var_es_table(pnl, alpha: float = 0.05, method: str = "class",
                 total: str | None = None):
    """
    Historical-style VaR and ES for every column of a scenario table
    (historical returns x positions, or simulated P&L).

    pnl   : DataFrame, rows = scenarios, columns = positions ($ P&L or returns)
    total : if given (e.g. "Portfolio"), adds the row-sum column first
    Returns a DataFrame indexed by position with columns VaR, ES.
    """
    import pandas as pd

    df = pd.DataFrame(pnl).copy()
    if total is not None:
        df[total] = df.sum(axis=1)
    return pd.DataFrame({c: {"VaR": var_historical(df[c], alpha, 1.0, method),
                             "ES": es_historical(df[c], alpha, 1.0, method)}
                         for c in df.columns}).T


def risk_table(x, alphas=(0.05, 0.01), value: float = 1.0, demean: bool = False,
               method: str = "class", normal_sd: str = "sample", from_mean: bool = False):
    """
    One series, three models: VaR and ES under a fitted normal, a fitted
    location-scale t (MLE) and historical simulation, at each alpha.

    value     : position size, e.g. 1_000_000 for $ figures (default 1 = return units)
    demean    : remove the sample mean before fitting / historical calc
    normal_sd : "sample" -> sigma = std with n - 1 (class fit_normal, matches the
                course tests 8.1 / 8.4); "mle" -> std with n
    from_mean : False -> absolute VaR/ES (loss measured from 0, the default)
                True  -> "diff from mean": VaR/ES measured from each model's mean
                (absolute + mean), the second reporting convention in Week 4
    Rows = Normal, Student t, Historical; columns = "VaR 5%", "ES 5%", "VaR 1%", "ES 1%".
    """
    import pandas as pd
    from .fitting import fit_normal, fit_t

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if demean:
        x = x - x.mean()
    mu, sd_mle, _ = fit_normal(x)
    if normal_sd == "sample":
        sd = float(x.std(ddof=1))
    elif normal_sd == "mle":
        sd = sd_mle
    else:
        raise ValueError("normal_sd must be 'sample' or 'mle'")
    tf = fit_t(x)
    shift = {"Normal": mu, "Student t": tf.mu, "Historical": float(x.mean())} if from_mean \
        else {"Normal": 0.0, "Student t": 0.0, "Historical": 0.0}
    out = {"Normal": {}, "Student t": {}, "Historical": {}}
    for a in alphas:
        lab = f"{100 * a:g}%"
        vals = {
            "Normal": (var_normal(sd, a, mu), es_normal(sd, a, mu)),
            "Student t": (var_t(tf, a), es_t(tf, a)),
            "Historical": (var_historical(x, a, 1.0, method), es_historical(x, a, 1.0, method)),
        }
        for k, (v, e) in vals.items():
            out[k][f"VaR {lab}"] = (v + shift[k]) * value
            out[k][f"ES {lab}"] = (e + shift[k]) * value
    table = pd.DataFrame(out).T
    table.attrs["t_fit"] = tf
    table.attrs["normal_fit"] = (mu, sd)
    return table
