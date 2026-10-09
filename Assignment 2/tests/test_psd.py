"""Tests for qrmlib.cholesky, qrmlib.psd_repair, qrmlib.portfolio and the
correlation estimators.  Run from the Assignment 2 folder:  python3 -m pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `import qrmlib` works without installing

from qrmlib import (CholeskyError, chol_pd, chol_psd, attempt_cholesky,
                    rebonato_jackel, higham_nearest_correlation, frobenius_distance,
                    min_eigenvalue, is_psd, corr_pairwise, corr_complete,
                    joint_obs_counts, n_complete,
                    cov_from_corr, portfolio_variance, tracking_weights)

# Worked example from the Week 3 notes (Section 2.1.1): L is known by hand.
A = np.array([[4, 2, -2, 4], [2, 10, 5, -1], [-2, 5, 21, 0], [4, -1, 0, 15]], float)
L_EXPECTED = np.array([[2, 0, 0, 0], [1, 3, 0, 0], [-1, 2, 4, 0], [2, -1, 1, 3]], float)

# A 3x3 "inconsistent" correlation matrix: A~B and B~C strongly positive,
# A~C strongly negative.  Not PSD (min eigenvalue < 0).
BAD = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])


def test_chol_pd_matches_hand_example():
    assert np.allclose(chol_pd(A), L_EXPECTED)
    assert np.allclose(chol_pd(A), np.linalg.cholesky(A))


def test_chol_pd_raises_on_non_psd():
    with pytest.raises(CholeskyError):
        chol_pd(BAD)


def test_chol_psd_handles_rank_deficient():
    s = np.full((3, 3), 1.0)            # rank 1, PSD
    root = chol_psd(s)
    assert np.allclose(root @ root.T, s)


def test_attempt_cholesky_reports_failure():
    res = attempt_cholesky(BAD, labels=["x", "y", "z"])
    assert not res.success and res.root is None
    assert "FAILED" in res.message


def test_rebonato_jackel_gives_psd_unit_diagonal():
    out = rebonato_jackel(BAD)
    assert min_eigenvalue(out) >= -1e-10
    assert np.allclose(np.diag(out), 1.0)


def test_higham_gives_psd_unit_diagonal_and_beats_rj():
    res = higham_nearest_correlation(BAD)
    assert res.converged
    assert min_eigenvalue(res.matrix) >= -1e-10
    assert np.allclose(np.diag(res.matrix), 1.0)
    # Higham is the NEAREST correlation matrix, so it can't be farther than R&J.
    assert frobenius_distance(res.matrix, BAD) <= frobenius_distance(rebonato_jackel(BAD), BAD) + 1e-9


def test_repairs_leave_valid_matrix_unchanged():
    good = np.array([[1, .3], [.3, 1]])
    assert np.allclose(rebonato_jackel(good), good)
    assert np.allclose(higham_nearest_correlation(good).matrix, good)


def test_pairwise_vs_complete_on_data_with_gaps():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("XYZ"))
    full = corr_pairwise(df)
    assert np.allclose(full, df.corr())             # no gaps -> same as ordinary corr
    df.iloc[:20, 2] = np.nan                         # Z missing for first 20 days
    assert n_complete(df) == 40
    assert joint_obs_counts(df).loc["X", "Z"] == 40
    pw = corr_pairwise(df)
    assert np.isclose(pw.loc["X", "Y"], df[["X", "Y"]].corr().loc["X", "Y"])   # XY uses all 60 days
    assert np.isclose(corr_complete(df).loc["X", "Z"], df.dropna().corr().loc["X", "Z"])
    assert is_psd(corr_complete(df))


def test_portfolio_variance_and_weights():
    cols = ["A", "B", "IDX"]
    w = tracking_weights(cols, {"A": 0.6, "B": 0.4})
    assert np.allclose(w, [-0.6, -0.4, 1.0])
    cov = cov_from_corr(np.eye(3), [1.0, 2.0, 3.0])   # uncorrelated -> sum w_i^2 sd_i^2
    assert np.isclose(portfolio_variance(w, cov), 0.36 * 1 + 0.16 * 4 + 9)
