#FINTECH 545 Midterm Problem 2
#Oct 2026
#Ivy Zhu

# Library: qrmlib (QRM Library/, v0.7.0)
# Data: problem2.csv -- daily prices, columns Date, SPY, XYZ (501 rows, 2024-01-02 on)
# Position: 100 shares of XYZ.  Model: r_XYZ = alpha + beta * r_SPY + eps

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
try:
    import qrmlib as q
except ImportError:                                    # not pip-installed: use the copy in this folder
    sys.path.insert(0, str(HERE / "QRM Library"))
    import qrmlib as q


# 1. Load data ------------------------------------------------------------------
def load_data():
    # Date kept as a column (returns.py: return_calculate(..., date_column="Date"))
    return pd.read_csv(HERE / "problem2.csv", parse_dates=["Date"])

#Part B
# 2. Moments --------------------------------------------------------------------
# returns.py: return_calculate; moments.py: four_moments (n-1 variance, excess kurtosis)
def part_moments(returns):
    table = pd.DataFrame({col: q.four_moments(returns[col]) for col in ["XYZ", "SPY"]})
    print(f"First four moments of arithmetic returns (n = {len(returns)})")
    print(table.to_string(float_format="%.6e"))
    return table


# 3. Returns and Regression ---------------------------------------------------------------------
# factor.py: ols  (r_XYZ = alpha + beta * r_SPY + eps)
SHARES = pd.Series({"XYZ": 100.0, "SPY": 0.0})     # position in shares


def part_a(returns):
    fit = q.ols(returns["XYZ"], returns["SPY"])
    table = pd.Series({
        "alpha": fit.alpha,
        "beta": fit.beta,
        "residual sd (n-1)": fit.resid_sd,
        "residual se (n-2)": fit.resid_se,
        "R^2": fit.r2,
    })
    print(f"Part a: OLS regression of r_XYZ on r_SPY (arithmetic returns, n = {fit.n})")
    print(table.to_string(float_format="%.6f"))
    return fit

#Part B
# 4. Delta normal VaR / ES -----------------------------------------------------
# portfolio_risk.py: position_values, delta_normal_risk  (zero mean, Sigma = sample cov, n-1)
def part_delta_normal(prices, returns, alphas=(0.05, 0.01)):
    cols = list(SHARES.index)
    cov = returns[cols].cov(ddof=1)
    values = q.position_values(SHARES, prices[cols].iloc[-1])
    total = values.sum()
    print("\nSample covariance matrix of arithmetic returns")
    print(cov.to_string(float_format="%.6e"))
    rows = {}
    for a in alphas:
        dn = q.delta_normal_risk(values, cov, alpha=a)
        rows[f"{100 * (1 - a):.0f}%"] = {
            "VaR (%)": 100 * dn.loc["Portfolio", "VaR"] / total,
            "ES (%)": 100 * dn.loc["Portfolio", "ES"] / total,
        }
    table = pd.DataFrame(rows).T
    print("Delta normal VaR and ES of the portfolio")
    print(table.to_string(float_format="%.4f"))
    return table


# 5. Historical ES -------------------------------------------------------------
# portfolio_risk.py: position_values, historical_risk  (class method: mean of floor(alpha*n) worst)
def part_historical_es(prices, returns, alphas=(0.05, 0.01)):
    cols = list(SHARES.index)
    values = q.position_values(SHARES, prices[cols].iloc[-1])
    total = values.sum()
    rows = {}
    for a in alphas:
        h = q.historical_risk(returns[cols], values, alpha=a)
        rows[f"{100 * (1 - a):.0f}%"] = {"ES (%)": 100 * h.loc["Portfolio", "ES"] / total}
    table = pd.DataFrame(rows).T
    print("\nHistorical ES of the portfolio")
    print(table.to_string(float_format="%.4f"))
    return table


def main():
    prices = load_data()
    returns = q.return_calculate(prices, method="DISCRETE", date_column="Date")
    part_moments(returns)
    fit = part_a(returns)
    part_delta_normal(prices, returns)
    part_historical_es(prices, returns)


if __name__ == "__main__":
    main()
