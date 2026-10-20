"""
qrmlib.ewma -- exponentially weighted variance / covariance (Week 3).

Convention (same as the class code): observations are in time order, oldest
first, and deviations are taken from the EW-weighted mean (mean="simple" gives
the plain sample mean that qrmlib <= 0.6 and Assignment 2 used).  The most
recent observation gets raw weight (1 - lambda), the one before
it (1 - lambda) * lambda, and so on; the raw weights are then normalised to sum
to one because the sample is finite.
"""
from __future__ import annotations

import numpy as np


def ew_weights(n: int, lam: float) -> np.ndarray:
    """
    Normalised exponential weights for n observations, OLDEST FIRST.
    w[-1] (today) is the largest; w[t] proportional to (1-lam) * lam^(n-1-t).
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must be in (0, 1)")
    age = np.arange(n - 1, -1, -1)              # age 0 = most recent
    w = (1.0 - lam) * lam ** age
    return w / w.sum()


def ew_covariance(x, lam: float, mean: str = "weighted") -> np.ndarray:
    """
    Exponentially weighted covariance matrix of the columns of x (rows = time,
    oldest first):  Sigma = sum_t w_t (x_t - m)(x_t - m)'.

    mean="weighted" (default): m = sum_t w_t x_t, the EW mean -- exactly the
                     class ewCovar (matches course tests 2.1-2.3).
    mean="simple"  : m = plain sample mean (what qrmlib <= 0.6 used; Assignment 2).
    mean="none"    : data already de-meaned, m = 0.
    For de-meaned data the choice barely matters.  A 1-D x returns a 1x1 matrix.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    w = ew_weights(x.shape[0], lam)
    if mean == "weighted":
        xm = x - w @ x
    elif mean == "simple":
        xm = x - x.mean(axis=0)
    elif mean == "none":
        xm = x
    else:
        raise ValueError("mean must be 'weighted', 'simple' or 'none'")
    return (w[:, None] * xm).T @ xm


def ew_variance(x, lam: float, mean: str = "weighted") -> float:
    """Exponentially weighted variance of a single series (see ew_covariance for `mean`)."""
    return float(ew_covariance(np.asarray(x, dtype=float).ravel(), lam, mean)[0, 0])


def ew_std(x, lam: float, mean: str = "weighted") -> float:
    """Exponentially weighted standard deviation of a single series."""
    return float(np.sqrt(ew_variance(x, lam, mean)))


def effective_sample_size(lam: float, n: int | None = None) -> float:
    """
    Kish effective sample size  n_eff = 1 / sum(w_t^2).

    With an infinite history this is (1 + lam) / (1 - lam).  Pass ``n`` to use
    the actual finite, normalised weights instead.
    """
    if n is None:
        return (1.0 + lam) / (1.0 - lam)
    w = ew_weights(n, lam)
    return float(1.0 / np.sum(w ** 2))


def half_life(lam: float) -> float:
    """Number of days for a weight to fall by half:  ln(0.5) / ln(lambda)."""
    return float(np.log(0.5) / np.log(lam))


def weight_on_recent(n: int, lam: float, k: int) -> float:
    """Fraction of the total (normalised) weight placed on the most recent k days."""
    return float(ew_weights(n, lam)[-k:].sum()) if k > 0 else 0.0


def ew_std_error(sigma: float, n_eff: float) -> float:
    """Approximate standard error of an EW volatility estimate: sigma / sqrt(2 n_eff)."""
    return float(sigma / np.sqrt(2.0 * n_eff))



# ---------------------------------------------------------------------------
# EW correlation, and a covariance built from two different lambdas
#   Sigma = D C D,  C = EW correlation (slow lambda),  D = diag(EW std) (fast lambda)
# Correlations move slowly -> HIGHER lambda (e.g. 0.97) for C.
# Variances update faster  -> LOWER lambda (e.g. 0.94) for D.
# ---------------------------------------------------------------------------
def _labelled(mat, x):
    if hasattr(x, "columns"):
        import pandas as pd
        return pd.DataFrame(mat, index=x.columns, columns=x.columns)
    return mat


def ew_cov_df(x, lam: float, mean: str = "weighted"):
    """ew_covariance that keeps DataFrame row/column labels."""
    return _labelled(ew_covariance(x, lam, mean), x)


def ew_correlation(x, lam: float, mean: str = "weighted"):
    """EW correlation matrix: the EW covariance rescaled to a unit diagonal."""
    c = ew_covariance(x, lam, mean)
    sd = np.sqrt(np.diag(c))
    corr = c / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return _labelled(corr, x)


def ew_std_vector(x, lam: float, mean: str = "weighted"):
    """EW standard deviation of every column (Series if x is a DataFrame)."""
    sd = np.sqrt(np.diag(ew_covariance(x, lam, mean)))
    if hasattr(x, "columns"):
        import pandas as pd
        return pd.Series(sd, index=x.columns, name=f"ew_std({lam})")
    return sd


def ew_mixed_covariance(x, lam_corr: float = 0.97, lam_var: float = 0.94,
                        mean: str = "weighted"):
    """
    Covariance = D C D with C = ew_correlation(x, lam_corr) and
    D = diag(ew_std_vector(x, lam_var)).  Returns (cov, corr, std).
    (Course test 2.3 is lam_corr=0.94, lam_var=0.97.)
    """
    corr = ew_correlation(x, lam_corr, mean)
    sd = ew_std_vector(x, lam_var, mean)
    s = np.asarray(sd)
    cov = np.asarray(corr) * np.outer(s, s)
    return _labelled(cov, x), corr, sd
