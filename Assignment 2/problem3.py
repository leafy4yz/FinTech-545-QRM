#FINTECH 545 Assignment 2 Problem 3
#Friday, October 9th, 2026
#Ivy Zhu
#
# When Diversification Raises VaR.
# Run from the "Assignment 2" folder:   python3 problem3.py
# Prints every number used in the write-up and saves problem3_pnl.png.
#
# Data: problem3.csv holds 10,000 simulated ONE-YEAR returns for two bonds,
# A and B (bought at 90, pay 100 unless they default).  The columns are already
# returns, so no price-to-return step is needed.  Every scenario is equally
# likely, so "historical" VaR/ES here means: read the quantile straight off
# the 10,000 scenario P&Ls.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # write the figure to a file, no window needed
import matplotlib.pyplot as plt

# Library modules (each is a file in qrmlib/):
from qrmlib import moments, portfolio, var, tails

pd.set_option("display.precision", 6)
pd.set_option("display.width", 140)

HERE = Path(__file__).resolve().parent
ALPHAS = [0.05, 0.01]          # 5% is the assignment default; part (e) asks for 1%
LOSS_THRESHOLD = -0.20         # part (b): "loses more than 20%"  <=>  return < -0.20

# The positions we are asked about, as {bond: dollars invested}.
POSITIONS = {
    "$1M A":       {"A": 1_000_000},
    "$1M B":       {"B": 1_000_000},
    "$2M A":       {"A": 2_000_000},               # the concentrated choice
    "$1M A + $1M B": {"A": 1_000_000, "B": 1_000_000},  # the diversified choice
}
CONCENTRATED, DIVERSIFIED = "$2M A", "$1M A + $1M B"

# Conventions (stated once, used everywhere):
#  * VaR and ES are POSITIVE numbers for a loss (assignment instructions).
#  * Historical VaR = minus the floor(alpha*n)-th smallest P&L (class code);
#    with n = 10,000 that is the 500th smallest at 5% and the 100th at 1%.
#  * Historical ES  = minus the mean of those floor(alpha*n) smallest P&Ls, so
#    ES always averages the tail that ends exactly at the VaR point.

# ===========================================================================
# Step 1: load the scenarios and build the P&L of every position
# ===========================================================================
def load():
    df = pd.read_csv(HERE / "problem3.csv").set_index("Scenario")
    rets = df[["A", "B"]]
    pnl = {name: portfolio.scenario_pnl(rets, h)                # qrmlib/portfolio.py
           for name, h in POSITIONS.items()}
    print(f"{len(rets)} scenarios loaded; columns {list(rets.columns)}")
    print(f"corr(A, B) = {rets['A'].corr(rets['B']):.4f}")
    return rets, pd.DataFrame(pnl)

# ===========================================================================
# Step 2: plot the P&L distribution of the two choices (Predict)
# ===========================================================================
def plot_pnl(pnl, path=HERE / "problem3_pnl.png"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(pnl[[CONCENTRATED, DIVERSIFIED]].min().min(),
                       pnl[[CONCENTRATED, DIVERSIFIED]].max().max(), 120)
    for name, colour in [(CONCENTRATED, "tab:blue"), (DIVERSIFIED, "tab:orange")]:
        label = name.replace("$", r"\$")       # a bare "$" would start matplotlib math text
        # Left: ordinary counts -- dominated by the no-default hump on the right.
        ax[0].hist(pnl[name] / 1e3, bins=bins / 1e3, alpha=0.55, color=colour, label=label)
        # Right: the same histogram on a log scale, so the default clusters
        # (a few hundred scenarios out of 10,000) are visible at all.
        ax[1].hist(pnl[name] / 1e3, bins=bins / 1e3, alpha=0.55, color=colour, label=label, log=True)
        # Mark where each choice's 5% historical VaR sits (the P&L at the 500th worst scenario).
        q05 = -var.var_historical(pnl[name], 0.05)               # qrmlib/var.py
        ax[1].axvline(q05 / 1e3, color=colour, ls="--", lw=1.2,
                      label=f"5% quantile, {label}: {q05 / 1e3:,.0f}k")
    for a in ax:
        a.set_xlabel("one-year P&L ($ thousands)")
        a.legend(loc="upper left")
    ax[0].set_ylabel("scenarios")
    ax[0].set_title("P&L of the two choices")
    ax[1].set_title("same, log scale (shows the default tail)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {path.name}")

# ===========================================================================
# Step 3: part (a) -- first four moments of each bond
# ===========================================================================
def part_a(rets, pnl):
    print("\n=== (a) First four moments of each bond's one-year return ===")
    tab = pd.DataFrame({c: moments.four_moments(rets[c]) for c in rets})   # moments.py
    tab.loc["std dev"] = np.sqrt(tab.loc["variance"])
    print(tab.to_string(float_format="{:.6g}".format))
    # Not asked for, but useful for the "what does this mean for a normal VaR"
    # discussion: the same moments for the two P&L choices.
    print("\nSame moments for the two choices' dollar P&L:")
    tab_p = pd.DataFrame({c: moments.four_moments(pnl[c]) for c in [CONCENTRATED, DIVERSIFIED]})
    tab_p.loc["std dev"] = np.sqrt(tab_p.loc["variance"])
    print(tab_p.to_string(float_format="{:.6g}".format))
    return tab

# ===========================================================================
# Step 4: part (b) -- count the large-loss scenarios
# ===========================================================================
def part_b(rets):
    n = len(rets)
    c = tails.tail_counts(rets, LOSS_THRESHOLD)                  # qrmlib/tails.py
    print(f"\n=== (b) Scenarios with a loss of more than {-LOSS_THRESHOLD:.0%} ===")
    labels = {"A": "A loses > 20%", "B": "B loses > 20%",
              "any": "at least one loses > 20%", "all": "both lose > 20%"}
    for k, lab in labels.items():
        print(f"  {lab:<26} {c[k]:>5}   ({c[k] / n:.2%} of scenarios)")
    # The comparison that drives the whole problem: how many scenarios sit in
    # the tail that a 5% (or 1%) VaR reads from.
    for a in ALPHAS:
        print(f"  alpha = {a:.0%}: the VaR reads the {int(np.floor(a * n))}th worst scenario")
    print("  Each bond alone defaults in fewer than 5% of scenarios, but at least one of the "
          "two defaults in more than 5%.")
    return c

# ===========================================================================
# Step 5: parts (c), (d), (e) -- historical VaR/ES and normal VaR
# ===========================================================================
def risk_table(pnl, alpha):
    """Historical VaR and ES, and normal VaR/ES from each P&L's sample mean and sd."""
    rows = {}
    for name in POSITIONS:
        x = pnl[name]
        mu, sd = x.mean(), x.std(ddof=1)
        rows[name] = {
            "hist VaR": var.var_historical(x, alpha),            # qrmlib/var.py
            "hist ES":  var.es_historical(x, alpha),             # var.py
            # Normal uses the sample MEAN as well as the sd (part d says so):
            # VaR = -(mu + sd * z_alpha).  The bonds earn ~8% on average, so
            # mu is far from zero and matters here.
            "normal VaR": var.var_normal(sd, alpha, mu=mu),      # var.py
            "normal ES":  var.es_normal(sd, alpha, mu=mu),       # var.py
            "mean P&L": mu,
            "sd P&L": sd,
        }
    return pd.DataFrame(rows).T

def parts_c_d_e(pnl):
    out = {}
    fmt = "{:,.0f}".format
    for alpha in ALPHAS:
        tab = risk_table(pnl, alpha)
        out[alpha] = tab
        part = "(c) + (d)" if alpha == 0.05 else "(e) [and normal for comparison]"
        print(f"\n=== {part} Risk at alpha = {alpha:.0%}, one year, $ (positive = loss) ===")
        print(tab.to_string(float_format=fmt))
        print("  Negative VaR = the alpha-quantile is still a PROFIT.")
        # Also show the interpolated-quantile version so the convention choice is visible.
        alt = {n: var.var_historical(pnl[n], alpha, method="quantile") for n in POSITIONS}
        print("  hist VaR with numpy's interpolated quantile instead: "
              + ", ".join(f"{n}: {fmt(v)}" for n, v in alt.items()))
    return out

# ===========================================================================
# Step 6: parts (f), (g), (h) -- subadditivity and which choice each measure favours
# ===========================================================================
def part_f_h(pnl):
    x, y = pnl["$1M A"].to_numpy(), pnl["$1M B"].to_numpy()
    measures = {}
    for alpha in ALPHAS:
        measures[f"hist VaR {alpha:.0%}"] = lambda p, a=alpha: var.var_historical(p, a)
        measures[f"hist ES {alpha:.0%}"] = lambda p, a=alpha: var.es_historical(p, a)
        measures[f"normal VaR {alpha:.0%}"] = (
            lambda p, a=alpha: var.var_normal(np.std(p, ddof=1), a, mu=np.mean(p)))
    rows = {name: tails.subadditivity(q, x, y) for name, q in measures.items()}   # tails.py
    tab = pd.DataFrame(rows).T
    print("\n=== (f)/(h) Subadditivity: q($1M A + $1M B) <= q($1M A) + q($1M B)? ===")
    print("  benefit = q(A) + q(B) - q(A+B); negative means combining ADDED risk")
    print(tab.to_string(formatters={c: "{:,.0f}".format for c in
                                    ["q(X)", "q(Y)", "q(X)+q(Y)", "q(X+Y)", "benefit"]}))
    return tab

def part_g(tables):
    print("\n=== (g)/(h) Which choice does each measure prefer (lower risk)? ===")
    for alpha, tab in tables.items():
        for col in ["hist VaR", "hist ES", "normal VaR"]:
            c, d = tab.loc[CONCENTRATED, col], tab.loc[DIVERSIFIED, col]
            pick = "diversified" if d < c else "concentrated"
            print(f"  {col:<10} at {alpha:.0%}:  concentrated {c:>12,.0f}   "
                  f"diversified {d:>12,.0f}   -> favours {pick}")
    # Why the normal always favours diversification: its VaR is linear in sd,
    # and sd(A+B) <= sd(A) + sd(B) whenever corr < 1.
    t = tables[0.05]
    print(f"  sd: concentrated {t.loc[CONCENTRATED, 'sd P&L']:,.0f}  vs diversified "
          f"{t.loc[DIVERSIFIED, 'sd P&L']:,.0f}  (same mean P&L to within "
          f"{abs(t.loc[CONCENTRATED, 'mean P&L'] - t.loc[DIVERSIFIED, 'mean P&L']):,.0f})")

# ===========================================================================
# Print Output
# ===========================================================================
def main():
    rets, pnl = load()                     # step 1
    plot_pnl(pnl)                          # step 2
    part_a(rets, pnl)                      # step 3
    part_b(rets)                           # step 4
    tables = parts_c_d_e(pnl)              # step 5
    part_f_h(pnl)                          # step 6
    part_g(tables)

if __name__ == "__main__":
    main()
