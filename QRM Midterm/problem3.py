#FINTECH 545 Midterm Problem 3
#Oct 2026
#Ivy Zhu

# Library: qrmlib (QRM Library/, v0.7.0)
# Data: problem3.csv          -- daily prices, columns Date, S1..S10 (1001 rows, 2022-01-03 on)
#       problem3_holdings.csv -- Stock, Book (A: S1-S5, B: S6-S10), Holding (shares)

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
    prices = pd.read_csv(HERE / "problem3.csv", parse_dates=["Date"])
    holdings = pd.read_csv(HERE / "problem3_holdings.csv").set_index("Stock")
    return prices, holdings


# 2. Part a ---------------------------------------------------------------------
# returns.py: return_calculate; fitting.py: fit_normal_t_nig (normal, t, NIG per stock)
def part_a(returns, stocks):
    fits = {s: q.fit_normal_t_nig(returns[s]) for s in stocks}
    table = pd.concat(fits, names=["Stock", "Model"])
    with pd.option_context("display.width", 200):
        print("Part a: fitted parameters, arithmetic returns")
        for model in ["Normal", "Student t", "NIG"]:
            sub = table.xs(model, level="Model").dropna(axis=1, how="all")
            print(f"\n{model}")
            print(sub.to_string(float_format="%.6f"))
    return table


# 3. Part b ---------------------------------------------------------------------
# copula.py: fit_chosen_margins, to_uniforms, compare_copulas
#   both copulas use the same R = sin(pi/2 tau) (Kendall); t copula nu by MLE
CHOICE = {"S1": "normal", "S2": "t", "S3": "nig", "S4": "normal", "S5": "t",
          "S6": "nig", "S7": "normal", "S8": "t", "S9": "nig", "S10": "t"}   # lowest AICc, part a


def part_b(returns, stocks):
    rets = returns[stocks]
    margins = q.fit_chosen_margins(rets, CHOICE)
    u = q.to_uniforms(rets, margins)
    table, fits = q.compare_copulas(u, gaussian_corr="kendall", t_corr="kendall")
    print("\nPart b: copula fits on U = F_i(r_i), margins from part a")
    print(table.to_string(float_format="%.4f"))
    print(f"Better fit (lower AICc): {table['AICc'].idxmin()} copula")
    return margins, u, fits


# 4. Part c ---------------------------------------------------------------------
# copula.py: CopulaFit.simulate, from_uniforms; portfolio_risk.py: pnl_from_returns, group_pnl
# var.py: var_es_table (absolute VaR / ES of simulated $ P&L, alpha = 5%)
N_SIM = 5_000
SEED = 545


def part_c(cop, margins, values, books, stocks, alpha=0.05):
    usim = cop["Gaussian"].simulate(N_SIM, rng=SEED)
    sims = q.from_uniforms(usim, margins, stocks)
    pnl = q.group_pnl(q.pnl_from_returns(sims, values), books)
    table = q.var_es_table(pnl, alpha=alpha)
    table.columns = ["VaR ($)", "ES ($)"]
    print(f"\nPart c: Gaussian copula, {N_SIM:,} draws, 95% absolute VaR and ES ($)")
    print(table.to_string(float_format="{:,.2f}".format))
    return table


def main():
    prices, holdings = load_data()
    shares = holdings["Holding"]                       # Series indexed S1..S10
    books = holdings["Book"]                           # Series indexed S1..S10 -> "A" / "B"
    stocks = list(shares.index)
    returns = q.return_calculate(prices, method="DISCRETE", date_column="Date")
    values = q.position_values(shares, prices[stocks].iloc[-1])   # $ per stock at last price
    fits = part_a(returns, stocks)
    margins, u, cop = part_b(returns, stocks)
    part_c(cop, margins, values, books, stocks)


if __name__ == "__main__":
    main()
