"""Port of library/missing_cov.jl"""
import numpy as np


def _cov(x):
    """Column-wise sample covariance, matching Julia's Statistics.cov(X)
    (columns are variables, denominator n-1)."""
    x = np.asarray(x, dtype=float)
    if x.shape[0] < 2:
        n = x.shape[1]
        return np.full((n, n), np.nan)
    return np.cov(x, rowvar=False)


def _cor(x):
    """Column-wise correlation matrix, matching Julia's Statistics.cor(X)."""
    x = np.asarray(x, dtype=float)
    if x.shape[0] < 2:
        n = x.shape[1]
        return np.full((n, n), np.nan)
    return np.corrcoef(x, rowvar=False)


def missing_cov(x, skip_miss=True, fun=_cov):
    """Calculate covariance/correlation (or any 2-arg-returning `fun`) on
    data that may contain missing values (np.nan).

    x : (n, m) array-like, np.nan marks a missing cell.
    skip_miss : if True, drop every row that has a missing value anywhere,
        then call `fun` once on what's left.
        If False, compute `fun` pairwise -- for each (i, j) column pair, use
        only the rows where neither column i nor column j is missing.
    fun : a function taking a 2-D array and returning its covariance-like
        matrix (defaults to _cov). Pass `_cor` for a correlation version.
    """
    x = np.asarray(x, dtype=float)
    n, m = x.shape
    miss_mask = np.isnan(x)

    if not miss_mask.any():
        return fun(x)

    if skip_miss:
        # keep rows with no missing values anywhere
        rows = ~miss_mask.any(axis=1)
        return fun(x[rows, :])
    else:
        # pairwise -- for each cell (i, j), use rows valid in both columns
        out = np.full((m, m), np.nan)
        for i in range(m):
            for j in range(i + 1):
                rows = ~(miss_mask[:, i] | miss_mask[:, j])
                sub = x[np.ix_(rows, [i, j])]
                c = fun(sub)
                out[i, j] = c[0, 1]
                out[j, i] = out[i, j]
        return out
