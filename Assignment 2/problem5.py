#FINTECH 545 Assignment 2 Problem 5
#Friday, October 9th, 2026
#Ivy Zhu
#
# Model Based Simulation and Residual Correlation.
# Run from the "Assignment 2" folder:   python3 problem5.py
# Prints every number used in the write-up.
#
# Data: problem5.csv holds 501 daily prices for a market index (MKT) and two
# stocks (A, B), giving 500 arithmetic returns.  Expected returns are taken to
# be zero, so the simulations drop the fitted alphas and the market mean.
#
# Portfolios (dollars per stock):
#   P1: long $1M A, long $1M B
#   P2: long $1M A, short $1M B

from pathlib import Path

import numpy as np
import pandas as pd

# Library modules (each is a file in qrmlib/):
from qrmlib import returns, factor, portfolio, var

pd.set_option("display.precision", 6)
pd.set_option("display.width", 140)

HERE = Path(__file__).resolve().parent
ALPHA = 0.05                  # assignment default VaR level
N_SIM = 100_000               # part (c)
SEED = 545                    # fixed so every run prints the same numbers
MARKET, STOCKS = "MKT", ["A", "B"]
PORTFOLIOS = {
    "P1 (long A, long B)":  {"A": 1_000_000, "B": 1_000_000},
    "P2 (long A, short B)": {"A": 1_000_000, "B": -1_000_000},
}
FULL, DIAG = "full residual cov", "independent residuals"

# Conventions (stated once, used everywhere):
#  * VaR is a POSITIVE number for a loss, one day, at alpha = 5%.
#  * Simulated VaR = minus the floor(alpha*n)-th smallest simulated P&L
#    (class code, same as Problems 3 and 4): the 5,000th of 100,000.
#  * Normal VaR from a covariance Sigma:  VaR = -z_alpha * sqrt(w' Sigma w), mean 0.
#  * Variances and covariances use n - 1 throughout.

# ===========================================================================
# Step 1: load prices and compute arithmetic returns
# ===========================================================================
def load():
    prices = pd.read_csv(HERE / "problem5.csv")
    r = returns.return_calculate(prices, "DISCRETE", date_column="Day")   # qrmlib/returns.py
    r = r.set_index("Day")[[MARKET] + STOCKS]
    print(f"{len(prices)} prices -> {len(r)} arithmetic returns; columns {list(r.columns)}")
    return r

def weights(holdings):
    """Position vector in STOCKS order."""
    return np.array([holdings.get(s, 0.0) for s in STOCKS], dtype=float)

# ===========================================================================
# Step 2: part (a) -- the prediction, written as the Week 2 variance formula
# ===========================================================================
def part_a():
    print("\n=== (a) Prediction from the portfolio variance formula ===")
    print("  r_i = a_i + b_i r_M + e_i  =>  Var(w'r) = (w'b)^2 var_M + wA^2 var(eA) + wB^2 var(eB)"
          "\n                                            + 2 wA wB cov(eA, eB)")
    print("  The shortcut sets cov(eA, eB) = 0.  A shared industry makes cov(eA, eB) > 0, so:")
    for name, h in PORTFOLIOS.items():
        sign = np.sign(h["A"] * h["B"])
        effect = "drops a POSITIVE term -> UNDERSTATES VaR" if sign > 0 else \
                 "drops a NEGATIVE term -> OVERSTATES VaR"
        print(f"  {name}: wA wB {'>' if sign > 0 else '<'} 0, shortcut {effect}")

# ===========================================================================
# Step 3: part (b) -- the two regressions
# ===========================================================================
def part_b(r):
    model = factor.fit_factor_model(r, MARKET, STOCKS)              # qrmlib/factor.py
    tab = pd.DataFrame({s: {"alpha": f.alpha, "beta": f.beta,
                            "resid sd (n-1)": f.resid_sd,
                            "resid sd (n-2, reg. s.e.)": f.resid_se,
                            "R^2": f.r2}
                        for s, f in model.fits.items()}).T
    print("\n=== (b) OLS of each stock on the market ===")
    print(tab.to_string(float_format="{:.6g}".format))
    rho = model.resid_corr().loc["A", "B"]
    print(f"  corr(eA, eB) = {rho:.4f}")
    print(f"  corr(A, B) raw returns = {r['A'].corr(r['B']):.4f};  "
          f"market sd = {np.sqrt(model.market_var):.6f}")
    return model

# ===========================================================================
# Step 4: part (c) -- simulate with the full and the diagonal residual covariance
# ===========================================================================
def part_c(model):
    rng = np.random.default_rng(SEED)
    rows = {}
    for label, diagonal in [(FULL, False), (DIAG, True)]:
        sim = model.simulate(N_SIM, diagonal=diagonal, rng=rng)    # factor.py
        rows[label] = {name: var.var_historical(                   # qrmlib/var.py
                           portfolio.scenario_pnl(sim, h), ALPHA)  # qrmlib/portfolio.py
                       for name, h in PORTFOLIOS.items()}
    tab = pd.DataFrame(rows)
    print(f"\n=== (c) Simulated {ALPHA:.0%} VaR, {N_SIM:,} days, $ (positive = loss) ===")
    print(tab.to_string(float_format="{:,.0f}".format))
    print("  Residual covariance used for the 'full' run:")
    print(model.resid_cov().to_string(float_format="{:.4e}".format))
    return tab

# ===========================================================================
# Step 5: part (d) -- delta normal VaR straight from the sample covariance
# ===========================================================================
def normal_var_from_cov(cov, holdings):
    sd = np.sqrt(portfolio.portfolio_variance(weights(holdings), cov))   # portfolio.py
    return var.var_normal(sd, ALPHA)                                      # var.py

def part_d(r):
    cov = r[STOCKS].cov(ddof=1)
    tab = pd.Series({name: normal_var_from_cov(cov, h) for name, h in PORTFOLIOS.items()},
                    name="delta normal VaR")
    print(f"\n=== (d) Delta normal {ALPHA:.0%} VaR from the sample covariance of A and B ===")
    print(cov.to_string(float_format="{:.4e}".format))
    print(tab.to_string(float_format="{:,.0f}".format))
    return cov, tab

# ===========================================================================
# Print Output
# ===========================================================================
def main():
    r = load()                                       # step 1
    part_a()                                         # step 2
    model = part_b(r)                                # step 3
    sim_tab = part_c(model)                          # step 4
    part_d(r)                                        # step 5

if __name__ == "__main__":
    main()
