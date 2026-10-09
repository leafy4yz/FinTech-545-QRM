"""Checks for the Problem 5 additions to qrmlib (run: python3 -m pytest tests)."""
import numpy as np
import pandas as pd

from qrmlib import ols, fit_factor_model


def _toy(n=5_000, seed=0):
    g = np.random.default_rng(seed)
    m = g.normal(0, 0.01, n)
    e = g.multivariate_normal([0, 0], [[1e-4, 6e-5], [6e-5, 1e-4]], n)
    return pd.DataFrame({"MKT": m,
                         "A": 0.001 + 1.2 * m + e[:, 0],
                         "B": -0.002 + 0.8 * m + e[:, 1]})


def test_ols_recovers_coefficients():
    r = _toy()
    f = ols(r["A"], r["MKT"])
    assert abs(f.beta - 1.2) < 0.05 and abs(f.alpha - 0.001) < 5e-4
    assert abs(f.resid.mean()) < 1e-12                       # intercept => mean-zero residuals
    assert abs(np.corrcoef(f.resid, r["MKT"])[0, 1]) < 1e-10  # orthogonal to the regressor
    assert f.resid_se > f.resid_sd                            # n-2 vs n-1


def test_full_residual_model_reproduces_sample_covariance_exactly():
    r = _toy()
    m = fit_factor_model(r, "MKT")
    np.testing.assert_allclose(m.implied_cov(), r[["A", "B"]].cov(), rtol=1e-10)
    assert abs(m.resid_corr().loc["A", "B"] - 0.6) < 0.05


def test_diagonal_shortcut_only_changes_off_diagonal():
    m = fit_factor_model(_toy(), "MKT")
    full, diag = m.implied_cov(), m.implied_cov(diagonal=True)
    np.testing.assert_allclose(np.diag(full), np.diag(diag))
    assert full.loc["A", "B"] > diag.loc["A", "B"]           # positive residual correlation lost


def test_simulation_matches_implied_covariance():
    m = fit_factor_model(_toy(), "MKT")
    for diagonal in (False, True):
        sim = m.simulate(400_000, diagonal=diagonal, rng=1)
        np.testing.assert_allclose(sim.cov(), m.implied_cov(diagonal), atol=3e-6)
        assert np.all(np.abs(sim.mean()) < 1e-4)              # zero expected returns
