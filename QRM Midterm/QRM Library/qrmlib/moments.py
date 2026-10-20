"""
qrmlib.moments -- the first four sample moments (Week 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def four_moments(x, bias: bool = False) -> pd.Series:
    """
    Mean, variance, skewness and EXCESS kurtosis of a sample.

    bias=False (default) uses the sample-adjusted estimators: variance with
    n-1 and the usual small-sample corrections for skewness and kurtosis.
    bias=True gives the plain population-moment formulas (variance with n).
    With n = 500 the two differ only in the third or fourth significant digit.
    """
    x = np.asarray(x, dtype=float).squeeze()       # one-column DataFrame -> 1-D
    return pd.Series({
        # float() so newer scipy (0-d array results) still formats cleanly
        "mean": float(x.mean()),
        "variance": float(x.var(ddof=0 if bias else 1)),
        "skewness": float(stats.skew(x, bias=bias)),
        "excess kurtosis": float(stats.kurtosis(x, fisher=True, bias=bias)),
    })


def skew_kurt_std_errors(n: int) -> tuple[float, float]:
    """
    Large-sample standard errors of skewness and excess kurtosis under
    normality: sqrt(6/n) and sqrt(24/n).  Used to judge whether a sample
    moment is 'far' from what a single normal would produce.
    """
    return float(np.sqrt(6.0 / n)), float(np.sqrt(24.0 / n))


def jarque_bera(x, bias: bool = False) -> tuple[float, float]:
    """
    Jarque-Bera normality test: JB = n/6 * (S^2 + K^2/4), K = EXCESS kurtosis,
    p-value from chi-square(2).

    Built on four_moments() with the same `bias` setting, so the statistic
    matches the skewness and kurtosis reported alongside it.  bias=True
    reproduces scipy.stats.jarque_bera (population moments).
    """
    m = four_moments(x, bias=bias)
    n = len(np.asarray(x))
    s, k = float(m["skewness"]), float(m["excess kurtosis"])
    jb = n / 6.0 * (s ** 2 + k ** 2 / 4.0)
    return float(jb), float(stats.chi2.sf(jb, df=2))


def extreme_influence(x, bias: bool = False) -> pd.DataFrame:
    """
    How much the single most extreme observation (largest |x - mean|) drives
    the four moments: one column with every observation, one without it.
    The position and value of the dropped observation are in .attrs.
    """
    x = np.asarray(x, dtype=float)
    idx = int(np.argmax(np.abs(x - x.mean())))
    out = pd.DataFrame({"all": four_moments(x, bias),
                        "drop extreme": four_moments(np.delete(x, idx), bias)})
    out.attrs.update(index=idx, value=float(x[idx]))
    return out


def first4_moments(sample):
    """
    Assignment 1 version, by hand: (mean, variance, skewness, RAW kurtosis).

    variance divides by n - 1; skewness and kurtosis use n-normalised central
    moments.  Raw kurtosis reads 3.0 for a normal -- subtract 3 for excess.
    (four_moments() above reports EXCESS kurtosis with small-sample corrections.)
    """
    sample = np.asarray(sample, dtype=float)
    n = sample.shape[0]
    mu_hat = np.sum(sample) / n
    d = sample - mu_hat
    cm2 = d @ d / n
    sigma2_hat = d @ d / (n - 1)
    skew_hat = np.sum(d ** 3) / n / np.sqrt(cm2 * cm2 * cm2)
    kurt_hat = np.sum(d ** 4) / n / cm2 ** 2
    return mu_hat, sigma2_hat, skew_hat, kurt_hat


def quantile_exceedance(sample, dist, q: float = 0.01):
    """
    Count observations strictly below the q-quantile of a fitted (frozen scipy)
    distribution vs the count expected if the data followed it.
    Returns (threshold, n_below, n_expected).
    """
    sample = np.asarray(sample, dtype=float)
    threshold = dist.ppf(q)
    return float(threshold), int(np.sum(sample < threshold)), q * sample.shape[0]
