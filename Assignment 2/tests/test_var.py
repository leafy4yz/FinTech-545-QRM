"""Checks for the Problem 2 additions to qrmlib (run: python3 -m pytest tests)."""
import numpy as np
from scipy import stats

from qrmlib import (return_calculate, demean, ew_weights, ew_variance,
                    effective_sample_size, half_life, weight_on_recent,
                    fit_t, var_normal, var_t, var_historical,
                    variance_breakpoint, normal_mixture_moments)


def test_returns_discrete_and_log():
    p = np.array([100.0, 110.0, 99.0])
    np.testing.assert_allclose(return_calculate(p), [0.1, -0.1])
    np.testing.assert_allclose(return_calculate(p, "LOG"), np.log([1.1, 0.9]))
    assert abs(demean([1.0, 2.0, 3.0]).mean()) < 1e-15


def test_ew_weights_sum_and_order():
    w = ew_weights(100, 0.94)
    assert abs(w.sum() - 1) < 1e-12
    assert w[-1] > w[0]                                   # most recent is largest
    assert abs(w[-2] / w[-1] - 0.94) < 1e-12


def test_ew_variance_tends_to_equal_weight():
    x = np.random.default_rng(0).normal(size=300)
    assert abs(ew_variance(x, 0.999999) - x.var(ddof=0)) < 1e-4


def test_neff_and_half_life():
    assert abs(effective_sample_size(0.94) - 1.94 / 0.06) < 1e-12
    assert abs(effective_sample_size(0.94, 5000) - 1.94 / 0.06) < 1e-6
    assert abs(0.97 ** half_life(0.97) - 0.5) < 1e-12
    assert abs(weight_on_recent(10_000, 0.9, 10) - (1 - 0.9 ** 10)) < 1e-9


def test_t_fit_recovers_parameters():
    x = 0.01 * stats.t.rvs(5, size=20_000, random_state=1) + 0.001
    f = fit_t(x)
    assert abs(f.mu - 0.001) < 3e-4
    assert abs(f.sigma - 0.01) < 5e-4
    assert 4.0 < f.nu < 6.5
    assert abs(var_t(f, 0.05) + f.ppf(0.05)) < 1e-15


def test_var_formulas():
    assert abs(var_normal(0.02, 0.05, value=1e6) - 1.6448536 * 0.02 * 1e6) < 1.0
    r = np.arange(1, 101) / 100.0 - 0.5                   # -0.49 ... 0.50
    assert abs(var_historical(r, 0.05) - 0.45) < 1e-12    # 5th smallest


def test_breakpoint_and_mixture():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 0.01, 400), rng.normal(0, 0.03, 50)]
    k, _ = variance_breakpoint(x)
    assert abs(k - 400) <= 5
    assert abs(normal_mixture_moments([1, 0], [1, 2])["excess kurtosis"]) < 1e-12
    assert normal_mixture_moments([0.9, 0.1], [1, 2])["excess kurtosis"] > 0

