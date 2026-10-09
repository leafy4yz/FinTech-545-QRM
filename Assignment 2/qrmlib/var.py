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
