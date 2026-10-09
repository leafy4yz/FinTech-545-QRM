"""
qrmlib.correlation -- covariance / correlation estimators for data with gaps,
plus small helpers for inspecting the eigenvalues of the result.

Two ways of handling missing values are provided:

* complete case : drop every day on which *any* series is missing, then use the
  ordinary estimator on what is left.  One common sample => the result is
  guaranteed positive semi-definite (PSD).
* pairwise      : estimate each (i, j) entry from the days on which *both* i
  and j are observed.  Uses more data, but every entry comes from a different
  sample, so the matrix is NOT guaranteed to be PSD.

All estimators use the sample (n - 1) denominator, matching ``np.cov`` and
``DataFrame.cov``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Complete case
# --------------------------------------------------------------------------
def cov_complete(df: pd.DataFrame) -> pd.DataFrame:
    """Covariance matrix from the rows with no missing values."""
    return df.dropna().cov()


def corr_complete(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix from the rows with no missing values."""
    return df.dropna().corr()


# --------------------------------------------------------------------------
# Pairwise
# --------------------------------------------------------------------------
def _pairwise(df: pd.DataFrame, kind: str, min_obs: int) -> pd.DataFrame:
    """Shared loop for pairwise covariance ('cov') and correlation ('corr')."""
    x = df.to_numpy(dtype=float)
    m = x.shape[1]
    out = np.full((m, m), np.nan)
    observed = ~np.isnan(x)

    for i in range(m):
        for j in range(i + 1):
            # Rows on which BOTH series are observed.  For i == j this is
            # simply every observed day of series i (its own variance).
            ok = observed[:, i] & observed[:, j]
            if ok.sum() < min_obs:
                raise ValueError(
                    f"only {int(ok.sum())} jointly observed days for "
                    f"({df.columns[i]}, {df.columns[j]}); need >= {min_obs}"
                )
            xi, xj = x[ok, i], x[ok, j]
            if kind == "cov":
                val = np.cov(xi, xj, ddof=1)[0, 1]
            else:
                val = 1.0 if i == j else np.corrcoef(xi, xj)[0, 1]
            out[i, j] = out[j, i] = val   # fill both triangles: symmetric

    return pd.DataFrame(out, index=df.columns, columns=df.columns)


def cov_pairwise(df: pd.DataFrame, min_obs: int = 2) -> pd.DataFrame:
    """Pairwise-complete covariance matrix (may not be PSD)."""
    return _pairwise(df, "cov", min_obs)


def corr_pairwise(df: pd.DataFrame, min_obs: int = 2) -> pd.DataFrame:
    """Pairwise-complete correlation matrix (may not be PSD)."""
    return _pairwise(df, "corr", min_obs)


# --------------------------------------------------------------------------
# Eigenvalue helpers
# --------------------------------------------------------------------------
def eigenvalues(mat) -> np.ndarray:
    """Eigenvalues of a symmetric matrix, ascending (smallest first)."""
    return np.linalg.eigvalsh(np.asarray(mat, dtype=float))


def min_eigenvalue(mat) -> float:
    """Smallest eigenvalue of a symmetric matrix."""
    return float(eigenvalues(mat)[0])


def is_psd(mat, tol: float = 1e-10) -> bool:
    """
    True if the matrix is PSD up to floating point error, i.e. its smallest
    eigenvalue is not below -tol.  (A tiny negative eigenvalue like -1e-16 is
    arithmetic; -0.01 is a statement that the entries are inconsistent.)
    """
    return min_eigenvalue(mat) >= -tol
