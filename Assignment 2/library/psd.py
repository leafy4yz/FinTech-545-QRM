"""Port of the non-simulation half of library/simulate.jl:
near_psd, chol_psd!, and the Higham nearest-PSD helpers."""
import numpy as np


def near_psd(a, epsilon=0.0):
    """Rebonato & Jackel 'near_psd' fix -- port of Julia's near_psd(a)."""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    out = a.copy()

    diag_a = np.diag(out)
    inv_sd = None
    # if this looks like a covariance matrix (diag != 1), work in correlation space
    if np.sum(np.isclose(diag_a, 1.0)) != n:
        inv_sd = np.diag(1.0 / np.sqrt(diag_a))
        out = inv_sd @ out @ inv_sd

    vals, vecs = np.linalg.eigh(out)  # ascending, like Julia's eigen() on a Symmetric matrix
    vals = np.maximum(vals, epsilon)
    t_vec = 1.0 / ((vecs * vecs) @ vals)
    t_mat = np.diag(np.sqrt(t_vec))
    l_mat = np.diag(np.sqrt(vals))
    b = t_mat @ vecs @ l_mat
    out = b @ b.T

    if inv_sd is not None:
        inv_sd2 = np.diag(1.0 / np.diag(inv_sd))
        out = inv_sd2 @ out @ inv_sd2
    return out


def chol_psd(a, epsilon=-1e-8):
    """Cholesky factorization that tolerates a PSD (not strictly PD) input,
    zeroing out numerically-negative-but-tiny pivots. Port of Julia's
    chol_psd!(root, a, epsilon) -- returns the lower-triangular root instead
    of writing into a pre-allocated output array."""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        s = 0.0
        if j > 0:
            s = root[j, :j] @ root[j, :j]

        temp = a[j, j] - s
        if 0 >= temp >= epsilon:
            temp = 0.0
        if temp < 0:
            # Julia's sqrt(::Float64) raises DomainError on a negative input,
            # which is what lets simulate_normal's try/except fall through to
            # a PSD fix. np.sqrt would instead silently return nan, so raise
            # explicitly to keep that control flow intact.
            raise ValueError(f"chol_psd: negative pivot ({temp}) at column {j}; matrix is not PSD")
        root[j, j] = np.sqrt(temp)

        if root[j, j] == 0.0:
            root[j, j + 1:] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = root[i, :j] @ root[j, :j]
                root[i, j] = (a[i, j] - s) * ir
    return root


def _get_a_plus(a):
    vals, vecs = np.linalg.eigh(a)
    vals = np.diag(np.maximum(vals, 0.0))
    return vecs @ vals @ vecs.T


def _get_ps(a, w):
    w05 = np.sqrt(w)
    iw = np.linalg.inv(w05)
    return iw @ _get_a_plus(w05 @ a @ w05) @ iw


def _get_pu(a, w):
    ret = a.copy()
    np.fill_diagonal(ret, 1.0)
    return ret


def _wgt_norm(a, w):
    w05 = np.sqrt(w)
    w05 = w05 @ a @ w05
    return np.sum(w05 * w05)


def higham_nearest_psd(pc, w=None, epsilon=1e-9, max_iter=100, tol=1e-9, verbose=True):
    """Higham (2002) nearest-correlation-matrix algorithm. Port of Julia's
    higham_nearestPSD(pc, W, epsilon, maxIter, tol)."""
    pc = np.asarray(pc, dtype=float)
    n = pc.shape[0]
    if w is None:
        w = np.diag(np.full(n, 1.0))

    delta_s = 0.0
    inv_sd = None
    yk = pc.copy()

    diag_yk = np.diag(yk)
    if np.sum(np.isclose(diag_yk, 1.0)) != n:
        inv_sd = np.diag(1.0 / np.sqrt(diag_yk))
        yk = inv_sd @ yk @ inv_sd

    yo = yk.copy()
    norml = np.inf
    i = 1

    while i <= max_iter:
        rk = yk - delta_s
        xk = _get_ps(rk, w)
        delta_s = xk - rk
        yk = _get_pu(xk, w)
        norm = _wgt_norm(yk - yo, w)
        min_eig_val = np.min(np.real(np.linalg.eigvals(yk)))

        if abs(norm - norml) < tol and min_eig_val > -epsilon:
            break
        norml = norm
        i += 1

    if verbose:
        if i < max_iter:
            print(f"Higham Converged in {i} iterations.")
        else:
            print(f"Higham Convergence failed after {i - 1} iterations")

    if inv_sd is not None:
        inv_sd2 = np.diag(1.0 / np.diag(inv_sd))
        yk = inv_sd2 @ yk @ inv_sd2
    return yk
