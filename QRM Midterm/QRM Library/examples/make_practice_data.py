"""
Writes three small practice CSVs (made-up data) so you can try every recipe in
USAGE.md before the exam:

    practice_returns.csv   one fat-tailed return series, column "x"
    practice_series.csv    three return series with a volatility shift, columns A, B, C
    practice_prices.csv    250 days of prices for four stocks, with a Date column

Run:  python3 make_practice_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
g = np.random.default_rng(545)

# 1. one fat-tailed series
pd.DataFrame({"x": 0.0005 + 0.012 * g.standard_t(4, 1000) / np.sqrt(2)}).to_csv(
    HERE / "practice_returns.csv", index=False)

# 2. three series whose volatility jumps for the last 60 days
corr = np.array([[1, .6, .3], [.6, 1, .4], [.3, .4, 1]])
z = g.multivariate_normal(np.zeros(3), corr, 500)
vol = np.where(np.arange(500)[:, None] < 440, 0.01, 0.025)
pd.DataFrame(z * vol, columns=list("ABC")).to_csv(HERE / "practice_series.csv", index=False)

# 3. prices for four stocks (t-distributed shocks, correlated)
corr4 = np.array([[1, .5, .4, .3], [.5, 1, .45, .35], [.4, .45, 1, .5], [.3, .35, .5, 1]])
z = g.multivariate_normal(np.zeros(4), corr4, 250) / np.sqrt(g.chisquare(5, (250, 1)) / 5)
r = 0.012 * z * np.sqrt(3 / 5)
start = np.array([45.0, 120.0, 310.0, 82.0])
px = start * np.cumprod(1 + r, axis=0)
dates = pd.bdate_range("2025-01-02", periods=250).strftime("%Y-%m-%d")
out = pd.DataFrame(px, columns=["AAA", "BBB", "CCC", "DDD"])
out.insert(0, "Date", dates)
out.to_csv(HERE / "practice_prices.csv", index=False)
print("wrote practice_returns.csv, practice_series.csv, practice_prices.csv")
