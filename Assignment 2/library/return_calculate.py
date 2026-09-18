"""Port of library/return_calculate.jl (return_accumulate.jl was not
ported: test_setup.jl never calls return_accumulate)."""
import numpy as np
import pandas as pd


def return_calculate(prices: pd.DataFrame, method="DISCRETE", date_column="date"):
    """Port of Julia's return_calculate(prices; method, dateColumn).

    prices: DataFrame with one date column (name given by date_column) and
        one or more price columns.
    method: "DISCRETE" (default) for simple returns p[t+1]/p[t] - 1,
        or "LOG" for log returns.
    """
    all_cols = list(prices.columns)
    vars_ = [c for c in all_cols if c != date_column]
    if len(vars_) == len(all_cols):
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {all_cols}")

    p = prices[vars_].to_numpy(dtype=float)
    n, m = p.shape

    p2 = p[1:, :] / p[:-1, :]

    method_u = method.upper()
    if method_u == "DISCRETE":
        p2 = p2 - 1.0
    elif method_u == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f'method: {method} must be in ("LOG","DISCRETE")')

    dates = prices[date_column].iloc[1:n].reset_index(drop=True)
    out = pd.DataFrame({date_column: dates})
    for i, v in enumerate(vars_):
        out[v] = p2[:, i]
    return out
