"""
qrmlib.cholesky -- Cholesky factorization written out by hand (Week 3, 2.1).

L is lower triangular with A = L L'.  Working down the columns, every entry of
L depends only on entries already computed:

    L_jj = sqrt( A_jj - sum_{k<j} L_jk^2 )
    L_ij = ( A_ij - sum_{k<j} L_ik L_jk ) / L_jj          (i > j)

If A is positive definite every L_jj is real and positive.  If A is only PSD
some L_jj is 0 (division fails); if A is not PSD some "pivot" under the square
root is negative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


class CholeskyError(np.linalg.LinAlgError):
    """Raised when a pivot is not strictly positive (chol_pd) or is too negative (chol_psd)."""

    def __init__(self, column: int, pivot: float):
        self.column = column      # 0-based column at which the factorization broke
        self.pivot = pivot        # value that was about to be square-rooted
        super().__init__(f"non-positive pivot {pivot:.6g} at column {column + 1}")


def chol_pd(a) -> np.ndarray:
    """
    Cholesky root of a positive DEFINITE matrix.  Raises CholeskyError at the
    first column whose pivot is <= 0.
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    root = np.zeros((n, n))
    for j in range(n):
        # Sum of squares of the row-j entries already placed (empty for j = 0).
        pivot = a[j, j] - root[j, :j] @ root[j, :j]
        if pivot <= 0.0:
            raise CholeskyError(j, pivot)
        root[j, j] = np.sqrt(pivot)
        # Fill the rest of column j below the diagonal.
        for i in range(j + 1, n):
            root[i, j] = (a[i, j] - root[i, :j] @ root[j, :j]) / root[j, j]
    return root


def chol_psd(a, tol: float = 1e-8) -> np.ndarray:
    """
    Cholesky root of a positive SEMI-definite matrix.

    A pivot in [-tol, 0] is treated as floating point error and set to 0; the
    column is then left at 0 (it carries no independent random number).  A
    pivot below -tol means the matrix is genuinely not PSD -> CholeskyError.
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    root = np.zeros((n, n))
    for j in range(n):
        pivot = a[j, j] - root[j, :j] @ root[j, :j]
        if -tol <= pivot <= 0.0:
            pivot = 0.0
        if pivot < 0.0:
            raise CholeskyError(j, pivot)
        root[j, j] = np.sqrt(pivot)
        if root[j, j] != 0.0:
            for i in range(j + 1, n):
                root[i, j] = (a[i, j] - root[i, :j] @ root[j, :j]) / root[j, j]
    return root


@dataclass
class CholeskyResult:
    """Outcome of ``attempt_cholesky`` -- what happened, in a printable form."""
    success: bool
    root: Optional[np.ndarray]
    column: Optional[int]       # 0-based failing column, None on success
    pivot: Optional[float]      # failing pivot, None on success
    message: str


def attempt_cholesky(a, labels: Optional[Sequence[str]] = None) -> CholeskyResult:
    """
    Try the strict (PD) factorization and report the outcome instead of raising.
    ``labels`` (e.g. the column names) are used to name the failing series.
    """
    try:
        root = chol_pd(a)
    except CholeskyError as err:
        name = labels[err.column] if labels is not None else f"column {err.column + 1}"
        return CholeskyResult(
            False, None, err.column, err.pivot,
            f"FAILED at {name}: pivot A_jj - sum(L_jk^2) = {err.pivot:.6g} <= 0, "
            f"so its square root is not real",
        )
    err_norm = np.max(np.abs(root @ root.T - np.asarray(a, dtype=float)))
    return CholeskyResult(True, root, None, None,
                          f"succeeded; max|LL' - A| = {err_norm:.2e}")
