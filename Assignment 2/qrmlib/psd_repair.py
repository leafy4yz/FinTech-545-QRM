"""
qrmlib.psd_repair -- turning a non-PSD correlation matrix into a PSD one
(Week 3, Section 6).

Two repairs are provided:

* rebonato_jackel          : one-shot spectral fix (clip eigenvalues, rescale
                             rows to restore the unit diagonal).  Fast, but not
                             guaranteed to be the *closest* PSD matrix.
* higham_nearest_correlation: iterative Dykstra alternating projections that
                             converge to the NEAREST correlation matrix in the
                             Frobenius norm (Higham 2002, Algorithm 3.3, with
                             the weight matrix W = I).

Both expect a symmetric matrix with a unit diagonal (a correlation matrix).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def frobenius_distance(a, b) -> float:
    """Frobenius norm ||a - b||_F = sqrt(sum of squared entry differences)."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), "fro"))


def _check_correlation(c: np.ndarray) -> None:
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(c, c.T):
        raise ValueError("matrix must be symmetric")
    if not np.allclose(np.diag(c), 1.0):
        raise ValueError("matrix must have a unit diagonal (a correlation matrix)")


# --------------------------------------------------------------------------
# Rebonato & Jackel
# --------------------------------------------------------------------------
def rebonato_jackel(c, epsilon: float = 0.0) -> np.ndarray:
    """
    Rebonato & Jackel (1999) spectral repair of a correlation matrix.

    1. C = S diag(lambda) S'                      (eigendecomposition)
    2. lambda'_i = max(lambda_i, epsilon)         (remove negative eigenvalues)
    3. t_i = 1 / sum_j S_ij^2 lambda'_j           (row scale that restores the
                                                   unit diagonal)
    4. B = sqrt(T) S sqrt(Lambda'),   C_hat = B B'
    """
    c = np.asarray(c, dtype=float)
    _check_correlation(c)

    vals, vecs = np.linalg.eigh(c)                 # ascending eigenvalues
    vals = np.maximum(vals, epsilon)               # step 2
    t = 1.0 / ((vecs * vecs) @ vals)               # step 3: S_ij^2 elementwise, then times lambda'
    b = np.diag(np.sqrt(t)) @ vecs @ np.diag(np.sqrt(vals))   # step 4
    return b @ b.T


# --------------------------------------------------------------------------
# Higham
# --------------------------------------------------------------------------
@dataclass
class HighamResult:
    """Repaired matrix plus diagnostics on how the iteration ended."""
    matrix: np.ndarray
    iterations: int
    converged: bool
    final_gamma: float          # ||Y_k - C||_F at the last iteration


def _proj_psd(a: np.ndarray) -> np.ndarray:
    """P_S: nearest PSD matrix (W = I) -- clip negative eigenvalues to 0."""
    vals, vecs = np.linalg.eigh(a)
    return (vecs * np.maximum(vals, 0.0)) @ vecs.T


def _proj_unit_diag(a: np.ndarray) -> np.ndarray:
    """P_U: nearest unit-diagonal matrix (W = I) -- set the diagonal to 1."""
    out = a.copy()
    np.fill_diagonal(out, 1.0)
    return out


def higham_nearest_correlation(c, max_iter: int = 10_000, tol: float = 1e-12,
                               eig_tol: float = 1e-12) -> HighamResult:
    """
    Higham (2002) nearest correlation matrix, W = I.

    Alternates the two projections with Dykstra's correction:

        R_k = Y_{k-1} - dS_{k-1}
        X_k = P_S(R_k)            (PSD, but diagonal no longer 1)
        dS_k = X_k - R_k          (Dykstra correction)
        Y_k = P_U(X_k)            (unit diagonal, may lose PSD slightly)
        gamma_k = ||Y_k - C||_F

    Stops when gamma has stopped changing (|gamma_k - gamma_{k-1}| < tol) AND
    the smallest eigenvalue of Y_k is >= -eig_tol.  The second condition is
    deliberate: convergence of the norm is not the same as a usable result
    (Week 3, 6.3) -- we check the eigenvalue, not just the norm.
    """
    c = np.asarray(c, dtype=float)
    _check_correlation(c)

    y = c.copy()
    delta_s = np.zeros_like(c)
    gamma_prev = np.inf
    gamma = np.inf
    converged = False
    k = 0

    for k in range(1, max_iter + 1):
        r = y - delta_s
        x = _proj_psd(r)
        delta_s = x - r
        y = _proj_unit_diag(x)
        gamma = frobenius_distance(y, c)
        min_eig = np.linalg.eigvalsh(y)[0]
        if abs(gamma - gamma_prev) < tol and min_eig >= -eig_tol:
            converged = True
            break
        gamma_prev = gamma

    return HighamResult(matrix=y, iterations=k, converged=converged, final_gamma=gamma)
