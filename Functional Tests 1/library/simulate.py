"""Port of the simulation half of library/simulate.jl: simulateNormal and
simulate_pca.

NOTE: these draw random numbers with NumPy's Generator, which does not
reproduce Julia's Random/Distributions RNG stream. Given the same
covariance-matrix input, the *algorithm* here is a faithful, step-by-step
port -- but the actual simulated draws (and therefore sample statistics
like cov(simulate_normal(...))) will differ numerically from the Julia
run. See library/__init__.py for more detail.
"""
import numpy as np

from .psd import chol_psd, near_psd


def simulate_normal(n_samples, cov, mean=None, seed=1234, fix_method=near_psd):
    """Port of Julia's simulateNormal(N, cov; mean, seed, fixMethod).

    Returns an (n_samples, n) array, one simulated draw per row.
    """
    cov = np.asarray(cov, dtype=float)
    n, m = cov.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")

    _mean = np.zeros(n)
    if mean is not None and len(mean) > 0:
        mean = np.asarray(mean, dtype=float)
        if len(mean) != n:
            raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")
        _mean = mean.copy()

    try:
        l = np.linalg.cholesky(cov)  # numpy returns the lower-triangular root, like Julia's cholesky(cov).L
    except np.linalg.LinAlgError:
        # not positive definite -- try treating it as PSD, then fall back to a PSD fix
        try:
            l = chol_psd(cov)
        except Exception:
            l = chol_psd(fix_method(cov))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n_samples))
    out = (l @ z).T
    out = out + _mean
    return out


def simulate_pca(a, n_sim, pct_exp=1.0, mean=None, seed=1234):
    """Port of Julia's simulate_pca(a, nsim; pctExp, mean, seed).

    Returns an (n_sim, n) array, one simulated draw per row.
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]

    _mean = np.zeros(n)
    if mean is not None and len(mean) > 0:
        _mean = np.asarray(mean, dtype=float).copy()

    vals, vecs = np.linalg.eigh(a)  # ascending
    # Julia's eigen() also returns ascending order; flip to descending like the Julia code does
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    tv = vals.sum()

    posv = np.where(vals >= 1e-8)[0]
    if pct_exp < 1:
        nval = 0
        pct = 0.0
        for idx in range(len(posv)):
            pct += vals[idx] / tv
            nval += 1
            if pct >= pct_exp:
                break
        if nval < len(posv):
            posv = posv[:nval]

    vals = vals[posv]
    vecs = vecs[:, posv]

    b = vecs @ np.diag(np.sqrt(vals))

    rng = np.random.default_rng(seed)
    m = len(vals)
    r = rng.standard_normal((m, n_sim))

    out = (b @ r).T
    out = out + _mean
    return out
