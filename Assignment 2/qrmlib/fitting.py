"""
qrmlib.fitting -- maximum likelihood fits of univariate distributions (Week 1/4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass
class TFit:
    """Location-scale Student t:  X = mu + sigma * T_nu."""
    mu: float
    sigma: float
    nu: float
    loglik: float
    n: int

    @property
    def std(self) -> float:
        """Implied standard deviation sigma * sqrt(nu / (nu - 2)) (inf if nu <= 2)."""
        return self.sigma * np.sqrt(self.nu / (self.nu - 2.0)) if self.nu > 2 else np.inf

    @property
    def excess_kurtosis(self) -> float:
        """Implied excess kurtosis 6 / (nu - 4) (inf if nu <= 4)."""
        return 6.0 / (self.nu - 4.0) if self.nu > 4 else np.inf

    def aicc(self, k: int = 3) -> float:
        return 2 * k - 2 * self.loglik + 2 * k * (k + 1) / (self.n - k - 1)

    def ppf(self, q):
        return self.mu + self.sigma * stats.t.ppf(q, self.nu)


def fit_normal(x) -> tuple[float, float, float]:
    """Normal MLE: (mu, sigma, loglik).  sigma uses n (the MLE), not n-1."""
    x = np.asarray(x, dtype=float)
    mu, sigma = x.mean(), x.std(ddof=0)
    return float(mu), float(sigma), float(stats.norm.logpdf(x, mu, sigma).sum())


def fit_t(x, nu_bounds=(2.0001, 500.0)) -> TFit:
    """
    Fit a location-scale t by maximum likelihood over (mu, sigma, nu).

    Starting values come from the method of moments: nu0 = 6/excess_kurt + 4
    (clipped), sigma0 = sd * sqrt((nu0 - 2)/nu0).  sigma is optimised on the log
    scale so it stays positive; nu is bounded below just above 2 so the variance
    exists (same lower bound idea as the class code).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    k = stats.kurtosis(x, fisher=True)
    nu0 = float(np.clip(6.0 / k + 4.0 if k > 0 else 30.0, 3.0, 100.0))
    s0 = x.std(ddof=1) * np.sqrt((nu0 - 2.0) / nu0)

    def nll(p):
        mu, log_s, nu = p
        return -stats.t.logpdf(x, nu, loc=mu, scale=np.exp(log_s)).sum()

    res = optimize.minimize(
        nll, x0=[x.mean(), np.log(s0), nu0], method="L-BFGS-B",
        bounds=[(None, None), (None, None), nu_bounds],
    )
    if not res.success:
        # fall back to a derivative-free search from the same start
        res = optimize.minimize(nll, x0=res.x, method="Nelder-Mead",
                                options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
    mu, log_s, nu = res.x
    return TFit(mu=float(mu), sigma=float(np.exp(log_s)), nu=float(nu),
                loglik=float(-res.fun), n=n)
