"""
qrmlib.copula -- Gaussian and Student t copulas (Week 5).

Workflow this module supports
-----------------------------
1. Fit each margin (normal or t, chosen by AICc)          -> fit_margin
2. Push each series through its fitted CDF to get U       -> MarginFit.cdf
3. Estimate R from Kendall's tau:  R_ij = sin(pi/2 tau)   -> kendall_corr
4. Fit the copula on U (Gaussian: nothing left to fit;
   t: nu by maximum likelihood with R held fixed)         -> fit_gaussian_copula, fit_t_copula
5. Simulate uniforms from the copula and map them back
   through the margins' inverse CDFs                      -> CopulaFit.simulate, MarginFit.ppf

Plus the diagnostics used to judge whether the joint tails are more connected
than the middle: rank uniforms, joint-tail counts and tail dependence.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .correlation import is_psd
from .criteria import aicc, bic
from .fitting import fit_normal, fit_t
from .psd_repair import rebonato_jackel


# ---------------------------------------------------------------------------
# Margins
# ---------------------------------------------------------------------------
@dataclass
class MarginFit:
    """A fitted margin: normal (nu=None) or location-scale t."""
    kind: str                      # "normal" or "t"
    mu: float
    sigma: float
    nu: Optional[float]
    loglik: float
    n: int

    @property
    def k(self) -> int:
        return 2 if self.kind == "normal" else 3

    @property
    def aicc(self) -> float:
        return aicc(self.loglik, self.k, self.n)

    @property
    def dist(self):
        if self.kind == "normal":
            return stats.norm(loc=self.mu, scale=self.sigma)
        return stats.t(df=self.nu, loc=self.mu, scale=self.sigma)

    def cdf(self, x):
        return self.dist.cdf(x)

    def ppf(self, u):
        return self.dist.ppf(u)


def fit_margin(x) -> tuple[MarginFit, MarginFit, MarginFit]:
    """
    Fit a normal and a t by maximum likelihood (qrmlib.fitting) and pick the
    one with the lower AICc.  Returns (chosen, normal_fit, t_fit).
    """
    x = np.asarray(x, dtype=float)
    mu, sd, ll = fit_normal(x)
    nfit = MarginFit("normal", mu, sd, None, ll, x.size)
    tf = fit_t(x)
    tfit = MarginFit("t", tf.mu, tf.sigma, tf.nu, tf.loglik, x.size)
    best = nfit if nfit.aicc <= tfit.aicc else tfit
    return best, nfit, tfit


def to_uniforms(df: pd.DataFrame, margins: dict, eps: float = 1e-12) -> pd.DataFrame:
    """U_i = F_i(X_i) through each column's fitted margin (clipped away from 0 and 1)."""
    u = {c: np.clip(margins[c].cdf(df[c].to_numpy(float)), eps, 1 - eps) for c in df}
    return pd.DataFrame(u, index=df.index)


def from_uniforms(u, margins: dict, columns) -> pd.DataFrame:
    """X_i = F_i^{-1}(U_i): map simulated copula uniforms back to returns."""
    u = np.asarray(u, dtype=float)
    return pd.DataFrame({c: margins[c].ppf(u[:, i]) for i, c in enumerate(columns)})


# ---------------------------------------------------------------------------
# Rank diagnostics
# ---------------------------------------------------------------------------
def pseudo_uniforms(df: pd.DataFrame) -> pd.DataFrame:
    """Ranks scaled into (0, 1):  rank / (n + 1).  No margin model needed."""
    return df.rank() / (len(df) + 1.0)


def joint_tail_counts(u, q: float = 0.025, columns=None) -> pd.DataFrame:
    """
    For every pair, the number of rows in which BOTH sit in their own worst q
    (u <= q, 'lower') and in which both sit in their own best q (u >= 1-q, 'upper').
    """
    if isinstance(u, pd.DataFrame):
        columns = list(u.columns) if columns is None else columns
        u = u.to_numpy(float)
    u = np.asarray(u, dtype=float)
    columns = columns or [f"X{i + 1}" for i in range(u.shape[1])]
    rows = {}
    for i, j in combinations(range(u.shape[1]), 2):
        rows[f"{columns[i]}-{columns[j]}"] = {
            "lower": int(np.sum((u[:, i] <= q) & (u[:, j] <= q))),
            "upper": int(np.sum((u[:, i] >= 1 - q) & (u[:, j] >= 1 - q))),
        }
    return pd.DataFrame(rows).T


def expected_joint_tail_independent(n: int, q: float = 0.025) -> float:
    """Under independence P(U1 <= q, U2 <= q) = q^2, so n q^2 days per pair per side."""
    return float(n * q * q)


# ---------------------------------------------------------------------------
# Correlation from Kendall's tau
# ---------------------------------------------------------------------------
def kendall_corr(df) -> np.ndarray:
    """
    R_ij = sin(pi/2 * tau_ij).  The relation holds for every elliptical copula
    (Gaussian and t alike) and is invariant to the margins.  Repaired with
    Rebonato-Jackel if the pairwise result is not PSD.
    """
    x = np.asarray(df, dtype=float)
    d = x.shape[1]
    r = np.eye(d)
    for i, j in combinations(range(d), 2):
        tau = stats.kendalltau(x[:, i], x[:, j])[0]
        r[i, j] = r[j, i] = np.sin(np.pi / 2.0 * tau)
    return r if is_psd(r) else rebonato_jackel(r)


# ---------------------------------------------------------------------------
# Copula densities, fits and simulation
# ---------------------------------------------------------------------------
def gaussian_copula_logpdf(u, r) -> np.ndarray:
    """log c(u) = log phi_R(z) - sum_i log phi(z_i),  z = Phi^{-1}(u).  One value per row."""
    z = stats.norm.ppf(np.asarray(u, dtype=float))
    joint = stats.multivariate_normal(mean=np.zeros(len(r)), cov=r).logpdf(z)
    return np.atleast_1d(joint - stats.norm.logpdf(z).sum(axis=1))


def t_copula_logpdf(u, r, nu: float) -> np.ndarray:
    """log c(u) = log t_{R,nu}(x) - sum_i log t_nu(x_i),  x = t_nu^{-1}(u).  One value per row."""
    x = stats.t.ppf(np.asarray(u, dtype=float), nu)
    joint = stats.multivariate_t(loc=np.zeros(len(r)), shape=r, df=nu).logpdf(x)
    return np.atleast_1d(joint - stats.t.logpdf(x, nu).sum(axis=1))


@dataclass
class CopulaFit:
    kind: str                      # "gaussian" or "t"
    r: np.ndarray
    nu: Optional[float]
    loglik: float
    n: int

    @property
    def k(self) -> int:
        """Free parameters: d(d-1)/2 correlations, plus nu for the t."""
        d = len(self.r)
        return d * (d - 1) // 2 + (0 if self.kind == "gaussian" else 1)

    @property
    def aicc(self) -> float:
        return aicc(self.loglik, self.k, self.n)

    @property
    def bic(self) -> float:
        return bic(self.loglik, self.k, self.n)

    def logpdf(self, u) -> np.ndarray:
        if self.kind == "gaussian":
            return gaussian_copula_logpdf(u, self.r)
        return t_copula_logpdf(u, self.r, self.nu)

    def simulate(self, n: int, rng=None) -> np.ndarray:
        if self.kind == "gaussian":
            return simulate_gaussian_copula(self.r, n, rng)
        return simulate_t_copula(self.r, self.nu, n, rng)


def fit_gaussian_copula(u, r) -> CopulaFit:
    """With R fixed from Kendall's tau there is nothing left to optimise."""
    u = np.asarray(u, dtype=float)
    return CopulaFit("gaussian", r, None, float(gaussian_copula_logpdf(u, r).sum()), len(u))


def fit_t_copula(u, r, nu_bounds=(2.01, 200.0)) -> CopulaFit:
    """R held at the Kendall estimate; nu chosen by maximum likelihood."""
    u = np.asarray(u, dtype=float)
    res = optimize.minimize_scalar(lambda nu: -t_copula_logpdf(u, r, nu).sum(),
                                   bounds=nu_bounds, method="bounded",
                                   options={"xatol": 1e-5})
    return CopulaFit("t", r, float(res.x), float(-res.fun), len(u))


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def simulate_gaussian_copula(r, n: int, rng=None) -> np.ndarray:
    """U = Phi(Z),  Z ~ N(0, R)."""
    z = _rng(rng).multivariate_normal(np.zeros(len(r)), r, size=n)
    return stats.norm.cdf(z)


def simulate_t_copula(r, nu: float, n: int, rng=None) -> np.ndarray:
    """U = t_nu(Z / sqrt(W/nu)),  Z ~ N(0, R),  W ~ chi2(nu).  One W per row."""
    g = _rng(rng)
    z = g.multivariate_normal(np.zeros(len(r)), r, size=n)
    w = g.chisquare(nu, size=n) / nu
    return stats.t.cdf(z / np.sqrt(w)[:, None], nu)


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------
def t_tail_dependence(rho: float, nu: float) -> float:
    """lambda = 2 t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ).  Same in both tails."""
    return float(2.0 * stats.t.cdf(-np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)), nu + 1.0))


def gaussian_tail_dependence(rho: float) -> float:
    """Zero for every |rho| < 1: extremes become independent in the limit."""
    return 0.0 if abs(rho) < 1 else 1.0


# ---------------------------------------------------------------------------
# Convenience: fit every margin, Spearman R (Week 5, 4.1)
# ---------------------------------------------------------------------------
def fit_margins(df: pd.DataFrame, family="aicc", normal_sd: str = "sample") -> dict:
    """
    Fit a margin to every column.  family:
      "aicc"   -> normal or t, whichever has the lower AICc (fit_margin's choice)
      "normal" -> normal for every column
      "t"      -> t for every column
      dict     -> {column: one of the above}
    normal_sd: "sample" (n - 1, the class fit_normal; default) or "mle" (n).
    Applies to margins that end up normal.  The AICc choice itself is always
    made on the MLE fits.
    Returns {column: MarginFit}.
    """
    fams = family if isinstance(family, dict) else {c: family for c in df.columns}
    out = {}
    for c in df.columns:
        x = df[c].to_numpy(float)
        best, nfit, tfit = fit_margin(x)
        m = {"aicc": best, "normal": nfit, "t": tfit}[fams[c]]
        if m.kind == "normal" and normal_sd == "sample":
            sd = float(np.std(x, ddof=1))
            m = MarginFit("normal", m.mu, sd, None,
                          float(stats.norm.logpdf(x, m.mu, sd).sum()), x.size)
        elif normal_sd not in ("sample", "mle"):
            raise ValueError("normal_sd must be 'sample' or 'mle'")
        out[c] = m
    return out


def margin_table(margins: dict) -> pd.DataFrame:
    """One row per column: kind, mu, sigma (t: scale), nu, loglik, AICc."""
    return pd.DataFrame({c: {"kind": m.kind, "mu": m.mu, "sigma": m.sigma,
                             "nu": m.nu if m.nu is not None else np.nan,
                             "loglik": m.loglik, "AICc": m.aicc}
                         for c, m in margins.items()}).T


def spearman_corr(u) -> np.ndarray:
    """
    Spearman rank correlation of the columns of u (data or uniforms -- ranks are
    the same either way).  The class choice for a Gaussian copula.  Repaired
    with Rebonato-Jackel if not PSD (only possible with missing data).
    """
    r = pd.DataFrame(np.asarray(u, dtype=float)).corr(method="spearman").to_numpy()
    return r if is_psd(r) else rebonato_jackel(r)


# ---------------------------------------------------------------------------
# Margins chosen per column (normal / t / NIG) and Gaussian-vs-t comparison
# ---------------------------------------------------------------------------
def fit_chosen_margins(df: pd.DataFrame, choice: dict, normal_sd: str = "sample") -> dict:
    """
    Fit the margin named in `choice` for every column:  "normal", "t" or "nig".
    Normal uses the n - 1 sd by default (class convention).  Every returned
    object has .cdf and .ppf, so it works with to_uniforms / from_uniforms.
    """
    from .fitting import fit_nig

    out = {}
    for c in df.columns:
        x = df[c].to_numpy(float)
        kind = choice[c].lower()
        if kind == "normal":
            mu = float(x.mean())
            sd = float(np.std(x, ddof=1 if normal_sd == "sample" else 0))
            out[c] = MarginFit("normal", mu, sd, None,
                               float(stats.norm.logpdf(x, mu, sd).sum()), x.size)
        elif kind == "t":
            tf = fit_t(x)
            out[c] = MarginFit("t", tf.mu, tf.sigma, tf.nu, tf.loglik, x.size)
        elif kind == "nig":
            out[c] = fit_nig(x)
        else:
            raise ValueError(f"unknown margin '{choice[c]}' for {c}")
    return out


def copula_corr(u, method: str) -> np.ndarray:
    """Copula correlation from uniforms: 'spearman' or 'kendall'."""
    if method == "spearman":
        return spearman_corr(u)
    if method == "kendall":
        return kendall_corr(u)
    raise ValueError("method must be 'spearman' or 'kendall'")


def compare_copulas(u, gaussian_corr: str = "kendall", t_corr: str = "kendall"):
    """
    Fit a Gaussian copula (R from gaussian_corr) and a t copula (R from t_corr,
    nu by MLE) to the same uniforms.  Default: the same Kendall R for both, so
    the likelihoods differ only through the copula family (t nu at its upper
    bound = the t collapses to the Gaussian).  Returns (table, {"Gaussian": fit, "t": fit});
    table has nu, loglik, k, AICc, BIC.  Lower AICc / BIC = better fit.
    """
    u = np.asarray(u, dtype=float)
    g = fit_gaussian_copula(u, copula_corr(u, gaussian_corr))
    t = fit_t_copula(u, copula_corr(u, t_corr))
    fits = {"Gaussian": g, "t": t}
    table = pd.DataFrame({k: {"nu": f.nu if f.nu is not None else np.nan,
                              "loglik": f.loglik, "k": f.k,
                              "AICc": f.aicc, "BIC": f.bic}
                          for k, f in fits.items()}).T
    return table, fits
