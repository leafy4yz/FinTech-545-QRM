"""Checks for the Problem 3 additions to qrmlib (run: python3 -m pytest tests)."""
import numpy as np
import pandas as pd
from scipy import stats

from qrmlib import (var_historical, es_historical, es_normal, scenario_pnl,
                    tail_counts, subadditivity)


def test_es_historical_class_convention():
    x = np.arange(1.0, 101.0)                 # 1..100; floor(.05*100) = 5 smallest
    assert var_historical(x, 0.05) == -5.0
    assert es_historical(x, 0.05) == -3.0     # -(1+2+3+4+5)/5
    assert es_historical(-x, 0.05) >= var_historical(-x, 0.05)   # ES >= VaR for losses


def test_es_normal_matches_simulation():
    z = np.random.default_rng(0).normal(0.01, 0.02, 400_000)
    assert abs(es_historical(z, 0.05, method="quantile") - es_normal(0.02, 0.05, mu=0.01)) < 2e-4
    # Week 5: at 5% the normal ES multiplier is 2.063
    assert abs(es_normal(1.0, 0.05) - 2.0627) < 1e-3


def test_scenario_pnl_and_tail_counts():
    r = pd.DataFrame({"A": [0.1, -0.5, 0.1, -0.6], "B": [-0.3, 0.1, 0.1, -0.7]})
    np.testing.assert_allclose(scenario_pnl(r, {"A": 2.0, "B": 1.0}), [-0.1, -0.9, 0.3, -1.9])
    c = tail_counts(r, -0.2)
    assert (c["A"], c["B"], c["any"], c["all"]) == (2, 2, 3, 1)


def test_var_fails_subadditivity_on_two_rare_defaults():
    # Week 4 bond example: each bond defaults with 4% probability, independently.
    rng = np.random.default_rng(1)
    n = 200_000
    a = np.where(rng.random(n) < 0.04, -100.0, 10.0)
    b = np.where(rng.random(n) < 0.04, -100.0, 10.0)
    var_res = subadditivity(lambda p: var_historical(p, 0.05), a, b)
    es_res = subadditivity(lambda p: es_historical(p, 0.05), a, b)
    assert not var_res["subadditive"]          # VaR breaks
    assert es_res["subadditive"]               # ES holds
