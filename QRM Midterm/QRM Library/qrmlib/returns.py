"""
qrmlib.returns -- turning a price series into returns (Week 1 / Week 4).

Python port of the class ``return_calculate`` helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def return_calculate(prices, method: str = "DISCRETE", date_column: str | None = None):
    """
    Returns from consecutive prices.

    method = "DISCRETE" (arithmetic):  r_t = P_t / P_{t-1} - 1
    method = "LOG":                    r_t = ln(P_t / P_{t-1})

    ``prices`` may be a 1-D array/Series or a DataFrame.  For a DataFrame, an
    optional ``date_column`` is carried along (dropping its first row) and is not
    treated as a price.  The output is one row shorter than the input.
    """
    method = method.upper()
    if method not in ("DISCRETE", "LOG"):
        raise ValueError("method must be 'DISCRETE' or 'LOG'")

    if isinstance(prices, pd.DataFrame):
        dates = None
        px = prices
        if date_column is not None:
            dates = prices[date_column].iloc[1:].reset_index(drop=True)
            px = prices.drop(columns=[date_column])
        ratio = px.iloc[1:].to_numpy(float) / px.iloc[:-1].to_numpy(float)
        out = pd.DataFrame(ratio - 1.0 if method == "DISCRETE" else np.log(ratio),
                           columns=px.columns)
        if dates is not None:
            out.insert(0, date_column, dates)
        return out

    p = np.asarray(prices, dtype=float)
    ratio = p[1:] / p[:-1]
    return ratio - 1.0 if method == "DISCRETE" else np.log(ratio)


def demean(x):
    """Remove the sample mean (column-wise for 2-D input)."""
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=0)
