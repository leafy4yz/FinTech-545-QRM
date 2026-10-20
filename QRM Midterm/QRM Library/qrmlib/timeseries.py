"""
qrmlib.timeseries -- ACF/PACF plot and AR/MA order selection by AICc (Week 2).

Needs statsmodels (and matplotlib for the plot); imported inside the functions so the
rest of the library works without them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def acf_band(n: int, z: float = 1.96) -> float:
    """Large-sample significance band for sample autocorrelations: z / sqrt(n)."""
    return z / np.sqrt(n)


def plot_ts(y, img_name: str = "series.png", lags: int = 10, title=None):
    """Series on top, ACF and PACF below, with +/- 1.96/sqrt(n) dashed bands. Saves img_name."""
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import acf, pacf

    n = len(y)
    l = np.arange(1, lags + 1)
    acf_vals = acf(y, nlags=lags, fft=False)[1:]
    pacf_vals = pacf(y, nlags=lags)[1:]
    band = acf_band(n)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(np.arange(1, n + 1), y)
    if title is not None:
        ax0.set_title(title)

    for pos, vals, name in ((gs[1, 0], acf_vals, "ACF"), (gs[1, 1], pacf_vals, "PACF")):
        ax = fig.add_subplot(pos)
        ax.bar(l, vals)
        ax.axhline(band, color="red", linestyle="--", linewidth=1)
        ax.axhline(-band, color="red", linestyle="--", linewidth=1)
        ax.set_title(name)

    fig.savefig(img_name)
    return fig


def fit_ar_ma(x, max_p: int = 3, max_q: int = 3, trend: str = "c") -> dict:
    """
    Fit AR(1..max_p) and MA(1..max_q) with SARIMAX (constant included by default).
    Returns {"AR(1)": result, ..., "MA(3)": result}.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    results = {}
    for p in range(1, max_p + 1):
        results[f"AR({p})"] = SARIMAX(x, order=(p, 0, 0), trend=trend).fit(disp=False)
    for q in range(1, max_q + 1):
        results[f"MA({q})"] = SARIMAX(x, order=(0, 0, q), trend=trend).fit(disp=False)
    return results


def aicc_table(results: dict) -> pd.Series:
    """AICc of each fitted model (lower is better)."""
    return pd.Series({name: fit.aicc for name, fit in results.items()}, name="AICc")


def best_model(results: dict) -> str:
    """Name of the model with the lowest AICc."""
    return min(results, key=lambda k: results[k].aicc)
