"""
qrmlib.factor -- single-index (market) factor model:  r_i = alpha_i + beta_i r_M + e_i
(Weeks 2 and 5).

Fit each stock by OLS on the market, then rebuild the stocks' covariance as

    Sigma = beta beta' var(r_M) + Sigma_e

where Sigma_e is the residual covariance -- either the full matrix or only its
diagonal (the "independent residuals" shortcut).  The same pieces drive a
Monte Carlo simulation of the stock returns.

Variance convention: var(r_M) and Sigma_e both use n - 1.  With OLS residuals
(mean zero and uncorrelated with the market by construction) the full-residual
Sigma above is then EXACTLY the sample covariance of the stock returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class OLSFit:
    """y = alpha + beta x + e, by ordinary least squares."""
    alpha: float
    beta: float
    resid: np.ndarray
    r2: float

    @property
    def n(self) -> int:
        return self.resid.size

    @property
    def resid_sd(self) -> float:
        """Residual standard deviation, n - 1 (matches the covariance convention)."""
        return float(np.std(self.resid, ddof=1))

    @property
    def resid_se(self) -> float:
        """Regression standard error, n - 2 (two estimated coefficients)."""
        return float(np.sqrt(np.sum(self.resid ** 2) / (self.n - 2)))


def ols(y, x) -> OLSFit:
    """Simple regression of y on x with an intercept."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - a - b * x
    r2 = 1.0 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    return OLSFit(float(a), float(b), e, float(r2))


@dataclass
class FactorModel:
    """Fitted single-index model for several stocks."""
    stocks: list
    alpha: pd.Series
    beta: pd.Series
    market_var: float
    resid: pd.DataFrame
    fits: dict

    def resid_cov(self, diagonal: bool = False) -> pd.DataFrame:
        """Residual covariance (n - 1); diagonal=True zeroes the off-diagonal."""
        c = self.resid.cov(ddof=1)
        return pd.DataFrame(np.diag(np.diag(c)), index=c.index, columns=c.columns) if diagonal else c

    def resid_corr(self) -> pd.DataFrame:
        return self.resid.corr()

    def implied_cov(self, diagonal: bool = False) -> pd.DataFrame:
        """Sigma = beta beta' var(r_M) + Sigma_e."""
        b = self.beta.to_numpy()
        sig = np.outer(b, b) * self.market_var + self.resid_cov(diagonal).to_numpy()
        return pd.DataFrame(sig, index=self.stocks, columns=self.stocks)

    def simulate(self, n: int, diagonal: bool = False, rng=None,
                 zero_mean: bool = True) -> pd.DataFrame:
        """
        Draw n days: r_M ~ N(0, var_M), e ~ N(0, Sigma_e), r_i = alpha_i + beta_i r_M + e_i.
        zero_mean=True drops the alphas (and the market mean is zero), i.e. the
        "assume zero expected returns" convention.  The market and the errors
        are drawn independently of each other, as the model assumes.
        """
        g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        m = g.normal(0.0, np.sqrt(self.market_var), size=n)
        e = g.multivariate_normal(np.zeros(len(self.stocks)),
                                  self.resid_cov(diagonal).to_numpy(), size=n)
        a = np.zeros(len(self.stocks)) if zero_mean else self.alpha.to_numpy()
        r = a + np.outer(m, self.beta.to_numpy()) + e
        return pd.DataFrame(r, columns=self.stocks)


def fit_factor_model(returns: pd.DataFrame, market: str = "MKT",
                     stocks: Sequence[str] | None = None) -> FactorModel:
    """Regress every stock column on the market column."""
    stocks = list(stocks) if stocks is not None else [c for c in returns if c != market]
    x = returns[market].to_numpy(float)
    fits = {s: ols(returns[s], x) for s in stocks}
    return FactorModel(
        stocks=stocks,
        alpha=pd.Series({s: f.alpha for s, f in fits.items()}),
        beta=pd.Series({s: f.beta for s, f in fits.items()}),
        market_var=float(np.var(x, ddof=1)),
        resid=pd.DataFrame({s: f.resid for s, f in fits.items()}, index=returns.index),
        fits=fits,
    )

