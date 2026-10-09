import numpy as np
from scipy import stats

from qrmlib.moments import four_moments, jarque_bera


def test_jarque_bera_matches_reported_moments():
    x = np.random.default_rng(1).standard_t(6, 500)
    m = four_moments(x)
    jb, p = jarque_bera(x)
    expected = len(x) / 6 * (m["skewness"] ** 2 + m["excess kurtosis"] ** 2 / 4)
    assert np.isclose(jb, expected)
    assert np.isclose(p, stats.chi2.sf(expected, 2))


def test_jarque_bera_bias_true_matches_scipy():
    x = np.random.default_rng(2).standard_t(5, 300)
    res = stats.jarque_bera(x)
    jb, p = jarque_bera(x, bias=True)
    assert np.isclose(jb, res.statistic) and np.isclose(p, res.pvalue)
