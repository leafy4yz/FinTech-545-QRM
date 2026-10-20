"""
qrmlib.simulation -- multivariate simulation (Week 3 Sections 1.2 and 7, Week 5 Section 5.2).

    X = (L Z + mu)',  Sigma = L L',  Z ~ N(0, I)

L comes from the PSD-safe Cholesky (qrmlib.cholesky.chol_psd) or from PCA,
B = S Lambda^{1/2}, optionally keeping only enough components to explain a
target share of the variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .cholesky import chol_psd


def variance_explained(cov) -> pd.DataFrame:
    """Eigenvalues (largest first), share of total variance and cumulative share."""
    vals = np.linalg.eigvalsh(np.asarray(cov, dtype=float))[::-1]
    pos = np.clip(vals, 0.0, None)
    pct = pos / pos.sum()
    return pd.DataFrame({"eigenvalue": vals, "pct": pct, "cum_pct": np.cumsum(pct)},
                        index=pd.RangeIndex(1, len(vals) + 1, name="component"))


def pca_root(cov, pct_explained: float = 1.0, tol: float = 1e-8) -> np.ndarray:
    """
    B = S sqrt(Lambda) from the positive eigenvalues, largest first, keeping the
    fewest components whose cumulative share reaches pct_explained.
    Shape (n, m) with m <= n, and B B' ~= Sigma.
    """
    a = np.asarray(cov, dtype=float)
    vals, vecs = np.linalg.eigh(a)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    keep = vals > tol * max(vals[0], 1e-300)
    vals, vecs = vals[keep], vecs[:, keep]
    if pct_explained < 1.0:
        cum = np.cumsum(vals) / vals.sum()
        m = int(np.searchsorted(cum, pct_explained - 1e-12) + 1)
        vals, vecs = vals[:m], vecs[:, :m]
    return vecs * np.sqrt(vals)


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def simulate_normal(cov, n_sim: int, mean=None, method: str = "cholesky",
                    pct_explained: float = 1.0, rng=None):
    """
    n_sim draws from N(mean, cov), mean defaults to 0.

    method="cholesky" : L = chol_psd(cov)   (works for PSD, raises if not PSD)
    method="pca"      : L = pca_root(cov, pct_explained)
    rng               : seed or np.random.Generator
    Returns a DataFrame with cov's labels if cov is a DataFrame, else ndarray (n_sim, n).
    """
    a = np.asarray(cov, dtype=float)
    if method == "cholesky":
        root = chol_psd(a)
    elif method == "pca":
        root = pca_root(a, pct_explained)
    else:
        raise ValueError("method must be 'cholesky' or 'pca'")
    z = _rng(rng).standard_normal((root.shape[1], n_sim))
    x = (root @ z).T
    if mean is not None:
        x = x + np.asarray(mean, dtype=float).reshape(1, -1)
    return pd.DataFrame(x, columns=cov.columns) if hasattr(cov, "columns") else x


def simulate_mvt(scale, nu: float, n_sim: int, mean=None, rng=None):
    """
    Multivariate t by the mixture  X = mu + sqrt(W) L Z,  W = nu / chi2_nu,
    one W per row (Week 5, 5.2).  `scale` is the SCALE matrix Sigma, so
    cov(X) = nu / (nu - 2) * Sigma.
    """
    a = np.asarray(scale, dtype=float)
    g = _rng(rng)
    z = g.standard_normal((a.shape[0], n_sim))
    w = nu / g.chisquare(nu, n_sim)
    x = (chol_psd(a) @ z).T * np.sqrt(w)[:, None]
    if mean is not None:
        x = x + np.asarray(mean, dtype=float).reshape(1, -1)
    return pd.DataFrame(x, columns=scale.columns) if hasattr(scale, "columns") else x


def mvt_logpdf(x, scale, nu: float, mean=None) -> np.ndarray:
    """Log density of the multivariate t at each row of x."""
    a = np.asarray(scale, dtype=float)
    loc = np.zeros(a.shape[0]) if mean is None else np.asarray(mean, dtype=float)
    return np.atleast_1d(stats.multivariate_t(loc=loc, shape=a, df=nu).logpdf(np.asarray(x, dtype=float)))
