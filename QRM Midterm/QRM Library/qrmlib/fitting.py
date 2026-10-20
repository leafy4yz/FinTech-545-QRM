"""
qrmlib.fitting -- maximum likelihood fits of univariate distributions (Week 1/4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
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

    The nu direction of the likelihood is very flat, so the first fit is
    checked against a profile likelihood over a grid of nu and replaced only if
    the profile finds a strictly higher log likelihood.
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
    best_x, best_f = res.x, res.fun

    # Safety net: the likelihood is very flat in nu and L-BFGS-B can stop well
    # short of the optimum (e.g. nu = 12 when nu = 4 is clearly better).
    # Profile it: on a grid of nu (even in 1/nu), fit mu and log sigma with nu
    # held fixed, then polish all three from the best grid point.  The first
    # fit is only replaced if this finds a strictly higher log likelihood.
    def nll_fixed(p2, nu):
        return -stats.t.logpdf(x, nu, loc=p2[0], scale=np.exp(p2[1])).sum()

    lo = max(nu_bounds[0], 2.05)
    grid = 1.0 / np.linspace(1.0 / min(nu_bounds[1], 200.0), 1.0 / lo, 25)
    prof = []
    for nu_g in grid:
        start2 = [np.median(x), np.log(x.std(ddof=1) * np.sqrt((nu_g - 2.0) / nu_g))]
        r2 = optimize.minimize(nll_fixed, start2, args=(nu_g,), method="L-BFGS-B")
        prof.append((r2.fun, nu_g, r2.x))
    f_g, nu_g, x_g = min(prof, key=lambda t: t[0])
    r3 = optimize.minimize(nll, x0=[x_g[0], x_g[1], nu_g], method="L-BFGS-B",
                           bounds=[(None, None), (None, None), nu_bounds])
    cand = (r3.x, r3.fun) if r3.fun <= f_g else (np.array([x_g[0], x_g[1], nu_g]), f_g)
    r4 = optimize.minimize(nll, x0=cand[0], method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-11})
    if r4.fun < cand[1] and nu_bounds[0] <= r4.x[2] <= nu_bounds[1]:
        cand = (r4.x, r4.fun)
    if cand[1] < best_f - 1e-7:
        best_x, best_f = cand
    mu, log_s, nu = best_x
    return TFit(mu=float(mu), sigma=float(np.exp(log_s)), nu=float(nu),
                loglik=float(-best_f), n=n)


def compare_normal_t(x) -> pd.DataFrame:
    """
    Fit a normal and a t by MLE and put them side by side:
    mu, sigma (t: scale), nu, implied std, log likelihood, k, AIC, AICc, BIC.
    The model with the lower AICc / BIC is the better fit.
    """
    from .criteria import aic, aicc, bic

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    mu, sd, ll_n = fit_normal(x)
    tf = fit_t(x)
    rows = {
        "Normal": {"mu": mu, "sigma": sd, "nu": np.nan, "std": sd, "loglik": ll_n, "k": 2},
        "Student t": {"mu": tf.mu, "sigma": tf.sigma, "nu": tf.nu, "std": tf.std,
                      "loglik": tf.loglik, "k": 3},
    }
    for r in rows.values():
        r["AIC"] = aic(r["loglik"], r["k"])
        r["AICc"] = aicc(r["loglik"], r["k"], n)
        r["BIC"] = bic(r["loglik"], r["k"], n)
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Normal inverse Gaussian (Week 5)
# ---------------------------------------------------------------------------
@dataclass
class NIGFit:
    """
    Normal inverse Gaussian, classical parameters (mu, alpha, beta, delta),
    |beta| < alpha, delta > 0.  scipy.stats.norminvgauss(a, b, loc, scale) with
    a = alpha * delta, b = beta * delta, loc = mu, scale = delta.
    """
    mu: float
    alpha: float
    beta: float
    delta: float
    loglik: float
    n: int

    @property
    def dist(self):
        """Frozen scipy distribution."""
        return stats.norminvgauss(self.alpha * self.delta, self.beta * self.delta,
                                  loc=self.mu, scale=self.delta)

    def ppf(self, q, n_grid: int = 200_001, width: float = 40.0):
        """
        Inverse CDF by interpolation (scipy's norminvgauss.ppf is far too slow
        for simulation).  CDF = cumulative trapezoid of the pdf on a fine grid
        of mean +/- width sd, normalised, then inverted with np.interp.
        """
        d = self.dist
        m, s = float(d.mean()), float(d.std())
        x = np.linspace(m - width * s, m + width * s, n_grid)
        pdf = d.pdf(x)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))])
        cdf /= cdf[-1]
        keep = np.concatenate([[True], np.diff(cdf) > 0])
        return np.interp(np.asarray(q, dtype=float), cdf[keep], x[keep])

    def cdf(self, x):
        return self.dist.cdf(x)

    def aicc(self, k: int = 4) -> float:
        return 2 * k - 2 * self.loglik + 2 * k * (k + 1) / (self.n - k - 1)


def fit_nig(x) -> NIGFit:
    """
    NIG by maximum likelihood.  Unconstrained parameters:
        a = exp(la) (= alpha*delta), b = a * tanh(tb) (so |b| < a),
        loc = mu, scale = exp(ls) (= delta).
    Start (symmetric case, beta = 0): excess kurtosis = 3/a, variance = delta^2/a.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    k = stats.kurtosis(x, fisher=True)
    a0 = float(np.clip(3.0 / k if k > 0 else 50.0, 0.5, 100.0))

    def unpack(p):
        a = np.exp(p[0])
        return a, a * np.tanh(p[1]), p[2], np.exp(p[3])

    def nll(p):
        a, b, loc, sc = unpack(p)
        v = stats.norminvgauss.logpdf(x, a, b, loc=loc, scale=sc).sum()
        return -v if np.isfinite(v) else 1e12

    best = None
    for a_start in (a0, 1.0, 5.0, 20.0):
        p0 = [np.log(a_start), 0.0, np.median(x), np.log(x.std(ddof=1) * np.sqrt(a_start))]
        r = optimize.minimize(nll, p0, method="Nelder-Mead",
                              options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12})
        r = optimize.minimize(nll, r.x, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    a, b, loc, sc = unpack(best.x)
    return NIGFit(mu=float(loc), alpha=float(a / sc), beta=float(b / sc), delta=float(sc),
                  loglik=float(-best.fun), n=n)


def fit_normal_t_nig(x) -> pd.DataFrame:
    """
    Normal, Student t and NIG fits of one series, side by side.
    Normal: mu = mean, sigma = sample sd (n - 1, class convention); loglik at those values.
    t: location mu, scale sigma, nu (MLE).  NIG: mu, alpha, beta, delta (MLE).
    """
    from .criteria import aicc

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    ll_n = float(stats.norm.logpdf(x, mu, sd).sum())
    tf = fit_t(x)
    nf = fit_nig(x)
    rows = {
        "Normal": {"mu": mu, "sigma": sd, "loglik": ll_n, "AICc": aicc(ll_n, 2, n)},
        "Student t": {"mu": tf.mu, "sigma": tf.sigma, "nu": tf.nu,
                      "loglik": tf.loglik, "AICc": aicc(tf.loglik, 3, n)},
        "NIG": {"mu": nf.mu, "alpha": nf.alpha, "beta": nf.beta, "delta": nf.delta,
                "loglik": nf.loglik, "AICc": aicc(nf.loglik, 4, n)},
    }
    cols = ["mu", "sigma", "nu", "alpha", "beta", "delta", "loglik", "AICc"]
    return pd.DataFrame(rows).T.reindex(columns=cols)
