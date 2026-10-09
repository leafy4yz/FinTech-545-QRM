"""Checks for the Problem 4 additions to qrmlib (run: python3 -m pytest tests)."""
import numpy as np
import pandas as pd
from scipy import stats

from qrmlib import (aicc, bic, fit_margin, pseudo_uniforms, joint_tail_counts,
                    expected_joint_tail_independent, kendall_corr,
                    gaussian_copula_logpdf, t_copula_logpdf, fit_t_copula,
                    simulate_gaussian_copula, simulate_t_copula,
                    t_tail_dependence, gaussian_tail_dependence, extreme_influence)

R = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])


def test_information_criteria():
    assert aicc(-10.0, 2, 100) == 20.0 + 4.0 + 12.0 / 97.0
    assert abs(bic(-10.0, 2, 100) - (20.0 + 2 * np.log(100))) < 1e-12


def test_independence_copula_density_is_one():
    u = np.random.default_rng(0).random((50, 3))
    np.testing.assert_allclose(gaussian_copula_logpdf(u, np.eye(3)), 0.0, atol=1e-10)
    np.testing.assert_allclose(t_copula_logpdf(u, np.eye(3), 1e7), 0.0, atol=1e-4)


def test_kendall_recovers_r_for_both_copulas():
    for u in (simulate_gaussian_copula(R, 20_000, 1), simulate_t_copula(R, 4.0, 20_000, 2)):
        np.testing.assert_allclose(kendall_corr(u), R, atol=0.03)


def test_t_copula_fit_recovers_nu():
    u = simulate_t_copula(R, 5.0, 5_000, 3)
    fit = fit_t_copula(u, kendall_corr(u))
    assert 3.5 < fit.nu < 7.5
    assert fit.k == 4


def test_simulated_margins_are_uniform():
    u = simulate_t_copula(R, 4.0, 50_000, 4)
    for j in range(3):
        assert stats.kstest(u[:, j], "uniform").pvalue > 0.001


def test_joint_tail_counts_and_independence():
    u = pd.DataFrame({"A": [0.01, 0.02, 0.99, 0.5], "B": [0.02, 0.9, 0.98, 0.01]})
    c = joint_tail_counts(u, 0.025)
    assert (c.loc["A-B", "lower"], c.loc["A-B", "upper"]) == (1, 1)
    assert expected_joint_tail_independent(1000, 0.025) == 0.625
    big = pd.DataFrame(np.random.default_rng(5).normal(size=(200_000, 2)), columns=["A", "B"])
    lower = joint_tail_counts(pseudo_uniforms(big), 0.025).loc["A-B", "lower"]
    assert abs(lower - 125) < 40           # 200,000 * 0.025^2 = 125


def test_tail_dependence():
    # lambda rises with rho, falls with nu, and is zero for the Gaussian
    assert t_tail_dependence(0.7, 4) > t_tail_dependence(0.3, 4)
    assert t_tail_dependence(0.5, 3) > t_tail_dependence(0.5, 30)
    assert abs(t_tail_dependence(0.5, 4) - 2 * stats.t.cdf(-np.sqrt(5 / 3), 5)) < 1e-12
    assert gaussian_tail_dependence(0.99) == 0.0
    # simulation check: P(U2 < q | U1 < q) at small q approaches lambda
    u = simulate_t_copula(np.array([[1, .5], [.5, 1]]), 3.0, 2_000_000, 6)
    q = 0.002
    cond = np.mean(u[u[:, 0] < q, 1] < q)
    assert abs(cond - t_tail_dependence(0.5, 3.0)) < 0.06


def test_fit_margin_picks_t_for_fat_tails_and_normal_for_normal():
    rng = np.random.default_rng(7)
    assert fit_margin(rng.standard_t(3, 2000))[0].kind == "t"
    assert fit_margin(rng.normal(size=2000))[0].kind == "normal"


def test_extreme_influence():
    x = np.r_[np.random.default_rng(8).normal(size=999), 25.0]
    tab = extreme_influence(x)
    assert tab.attrs["index"] == 999
    assert tab.loc["excess kurtosis", "all"] > 10 * abs(tab.loc["excess kurtosis", "drop extreme"])
