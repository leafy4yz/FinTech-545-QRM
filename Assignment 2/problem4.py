#FINTECH 545 Assignment 2 Problem 4
#Friday, October 9th, 2026
#Ivy Zhu
#
# Gaussian or t Copula.
# Run from the "Assignment 2" folder:   python3 problem4.py
# Prints every number used in the write-up and saves problem4_ranks.png.
#
# Data: problem4.csv holds 1,000 days of returns for three assets, X1-X3.
# We hold $1,000,000 of each, so the portfolio P&L on a day is
#     P&L = 1,000,000 * (r1 + r2 + r3).

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # write the figure to a file, no window needed
import matplotlib.pyplot as plt

# Library modules (each is a file in qrmlib/):
from qrmlib import moments, portfolio, var, copula

pd.set_option("display.precision", 6)
pd.set_option("display.width", 140)

HERE = Path(__file__).resolve().parent
HOLDING = 1_000_000
ALPHAS = [0.05, 0.01]
TAIL_Q = 0.025            # part (b): each series' own worst / best 2.5%
OUTER = 0.05              # part (i): "outer 5% on either side"
N_SIM = 100_000           # part (e)
SEED = 545                # fixed so every run prints the same numbers

# Conventions (stated once, used everywhere):
#  * VaR and ES are POSITIVE numbers for a loss.
#  * VaR = minus the floor(alpha*n)-th smallest P&L, ES = minus the mean of the
#    floor(alpha*n) smallest P&Ls (class code, same as Problem 3).  Applied to
#    the 1,000 historical days and to each set of 100,000 simulated days.
#  * R comes from Kendall's tau, R_ij = sin(pi/2 tau_ij), and is held FIXED
#    for both copulas.  The Gaussian then has 3 free parameters (the
#    correlations) and the t has 4 (correlations + nu).

# ===========================================================================
# Step 1: load the returns and the portfolio P&L
# ===========================================================================
def load():
    df = pd.read_csv(HERE / "problem4.csv")
    rets = df[[c for c in df.columns if c.startswith("X")]].astype(float)
    holdings = {c: HOLDING for c in rets.columns}
    pnl = portfolio.scenario_pnl(rets, holdings)                    # qrmlib/portfolio.py
    print(f"{len(rets)} days loaded; columns {list(rets.columns)}; ${HOLDING:,} in each")
    return rets, holdings, pnl

# ===========================================================================
# Step 2: part (a) -- four moments and the influence of the most extreme day
# ===========================================================================
def part_a(rets):
    print("\n=== (a) First four moments of each series ===")
    tab = pd.DataFrame({c: moments.four_moments(rets[c]) for c in rets})   # qrmlib/moments.py
    tab.loc["std dev"] = np.sqrt(tab.loc["variance"])
    print(tab.to_string(float_format="{:.6g}".format))
    se_s, se_k = moments.skew_kurt_std_errors(len(rets))
    print(f"  Normal-sample standard errors: skew {se_s:.3f}, excess kurtosis {se_k:.3f}")
    # A t with nu > 4 has excess kurtosis 6/(nu-4); invert it for a rough nu.
    for c in rets:
        k = tab.loc["excess kurtosis", c]
        guess = f"t with nu ~ {6 / k + 4:.1f}" if k > 2 * se_k else "normal"
        print(f"  {c}: kurtosis suggests {guess}")

    print("\n  Dropping the single most extreme day of each series:")
    for c in rets:
        inf = moments.extreme_influence(rets[c])                     # moments.py
        print(f"  {c}: day {inf.attrs['index']} (row {inf.attrs['index'] + 2} of the csv), "
              f"return {inf.attrs['value']:+.5f}")
        print(inf.loc[["skewness", "excess kurtosis"]].to_string(float_format="{:.4f}".format))
    return tab

# ===========================================================================
# Step 3: part (b) -- rank uniforms, pair plots and joint-tail counts
# ===========================================================================
def plot_ranks(u, path=HERE / "problem4_ranks.png"):
    pairs = list(combinations(u.columns, 2))
    fig, ax = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5))
    for a, (i, j) in zip(ax, pairs):
        a.scatter(u[i], u[j], s=5, alpha=0.5)
        for lo in (0.0, 1 - TAIL_Q):                     # shade the joint 2.5% corners
            a.add_patch(plt.Rectangle((lo, lo), TAIL_Q, TAIL_Q, color="red", alpha=0.3))
        a.set_xlabel(f"rank({i}) / (n+1)")
        a.set_ylabel(f"rank({j}) / (n+1)")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_aspect("equal")
        a.set_title(f"{i} vs {j}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {path.name}")

def part_b(rets):
    u = copula.pseudo_uniforms(rets)                                # qrmlib/copula.py
    plot_ranks(u)
    counts = copula.joint_tail_counts(u, TAIL_Q)                    # copula.py
    expected = copula.expected_joint_tail_independent(len(rets), TAIL_Q)
    print(f"\n=== (b) Days both series sit in their own worst / best {TAIL_Q:.1%} ===")
    print(counts.to_string())
    print(f"  Expected under independence: {len(rets)} x {TAIL_Q}^2 = {expected:.3f} "
          "days per pair per side")
    print("  Rank correlations (how connected the middle is):")
    print(rets.corr(method="spearman").to_string(float_format="{:.3f}".format))
    return u, counts

# ===========================================================================
# Step 4: part (c) -- normal vs t for each margin by AICc
# ===========================================================================
def part_c(rets):
    margins, rows = {}, {}
    for c in rets:
        best, nfit, tfit = copula.fit_margin(rets[c])               # copula.py (uses fitting.py)
        margins[c] = best
        rows[c] = {"AICc normal": nfit.aicc, "AICc t": tfit.aicc,
                   "AICc(n) - AICc(t)": nfit.aicc - tfit.aicc,
                   "t mu": tfit.mu, "t sigma": tfit.sigma, "t nu": tfit.nu,
                   "normal sd": nfit.sigma, "chosen": best.kind}
    tab = pd.DataFrame(rows).T
    print("\n=== (c) Margin fits (lower AICc wins) ===")
    print(tab.to_string(float_format="{:.6g}".format))
    return margins

# ===========================================================================
# Step 5: part (d) -- uniforms through the margins, R from tau, fit both copulas
# ===========================================================================
def part_d(rets, margins):
    u = copula.to_uniforms(rets, margins)                           # copula.py
    r = copula.kendall_corr(rets)                                   # copula.py
    g = copula.fit_gaussian_copula(u.to_numpy(), r)
    t = copula.fit_t_copula(u.to_numpy(), r)
    print("\n=== (d) R from Kendall's tau:  R_ij = sin(pi/2 * tau_ij) ===")
    print(pd.DataFrame(r, index=rets.columns, columns=rets.columns)
          .to_string(float_format="{:.4f}".format))
    tab = pd.DataFrame({
        "Gaussian": {"log likelihood": g.loglik, "nu hat": np.inf, "k": g.k,
                     "AICc": g.aicc, "BIC": g.bic},
        "t":        {"log likelihood": t.loglik, "nu hat": t.nu, "k": t.k,
                     "AICc": t.aicc, "BIC": t.bic},
    }).T
    print("\n=== (d) Copula fits ===")
    print(tab.to_string(float_format="{:.4f}".format))
    return u, r, g, t

# ===========================================================================
# Step 6: part (e) -- simulate 100,000 days from each copula; VaR and ES
# ===========================================================================
def part_e(rets, holdings, pnl_hist, margins, g, t):
    rng = np.random.default_rng(SEED)
    sims = {"Gaussian": g.simulate(N_SIM, rng), "t": t.simulate(N_SIM, rng)}
    pnl = {}
    for name, usim in sims.items():
        x = copula.from_uniforms(usim, margins, rets.columns)       # through the fitted margins
        pnl[name] = portfolio.scenario_pnl(x, holdings)             # portfolio.py
    pnl["Historical"] = pnl_hist

    rows = {}
    for name, p in pnl.items():
        rows[name] = {}
        for a in ALPHAS:
            rows[name][f"VaR {a:.0%}"] = var.var_historical(p, a)   # qrmlib/var.py
            rows[name][f"ES {a:.0%}"] = var.es_historical(p, a)     # var.py
    tab = pd.DataFrame(rows).T
    print(f"\n=== (e) Portfolio VaR / ES, one day, $ (positive = loss); "
          f"{N_SIM:,} simulated days per copula ===")
    print(tab.to_string(float_format="{:,.0f}".format))
    return sims, tab

# ===========================================================================
# Step 7: part (f) -- joint-tail days each copula implies over 1,000 days
# ===========================================================================
def part_f(rets, sims):
    scale = len(rets) / N_SIM
    implied = {name: copula.joint_tail_counts(u, TAIL_Q, list(rets.columns)) * scale
               for name, u in sims.items()}                         # copula.py
    print(f"\n=== (f) Implied joint-tail days per {len(rets):,} days ===")
    for name, tab in implied.items():
        print(f"  {name} copula:")
        print(tab.to_string(float_format="{:.2f}".format))
    return implied

# ===========================================================================
# Step 8: part (g) -- which copula wins, and do the tail counts agree?
# ===========================================================================
def part_g(g, t, counts, implied):
    print("\n=== (g) Which copula wins? ===")
    print(f"  log likelihood: t - Gaussian = {t.loglik - g.loglik:.3f}")
    print(f"  AICc: Gaussian - t = {g.aicc - t.aicc:.3f}   (positive favours t)")
    print(f"  BIC:  Gaussian - t = {g.bic - t.bic:.3f}")
    print(f"  Akaike weight of the t: {1 / (1 + np.exp(-(g.aicc - t.aicc) / 2)):.4f}")
    cmp = pd.concat({"data": counts.astype(float),
                     "Gaussian": implied["Gaussian"], "t": implied["t"]}, axis=1)
    cmp.loc["total"] = cmp.sum()
    print("\n  Joint-tail days: data vs implied")
    print(cmp.to_string(float_format="{:.2f}".format))
    return cmp

# ===========================================================================
# Step 9: part (h) -- which risk number moves when the copula changes?
# ===========================================================================
def part_h(risk):
    change = risk.loc["t"] - risk.loc["Gaussian"]
    tab = pd.DataFrame({"Gaussian": risk.loc["Gaussian"], "t": risk.loc["t"],
                        "t - Gaussian ($)": change,
                        "relative": change / risk.loc["Gaussian"]})
    print("\n=== (h) Moving from the Gaussian to the t copula ===")
    print(tab.to_string(formatters={"Gaussian": "{:,.0f}".format, "t": "{:,.0f}".format,
                                    "t - Gaussian ($)": "{:+,.0f}".format,
                                    "relative": "{:+.2%}".format}))
    print(f"  Largest move: {tab['relative'].abs().idxmax()}")
    return tab

# ===========================================================================
# Step 10: part (i) -- where does the log-likelihood gain come from?
# ===========================================================================
def part_i(u, g, t):
    uu = u.to_numpy()
    diff = t.logpdf(uu) - g.logpdf(uu)                              # one value per day
    middle = np.all((uu > OUTER) & (uu < 1 - OUTER), axis=1)
    total, mid = diff.sum(), diff[middle].sum()
    print("\n=== (i) Daily contributions to l_t - l_Gauss ===")
    print(f"  total                                   {total:9.3f}")
    print(f"  days with NO series in its outer {OUTER:.0%} ({middle.sum()} days): "
          f"{mid:9.3f}  ({mid / total:.1%} of total)")
    print(f"  days with at least one in its outer {OUTER:.0%} ({(~middle).sum()} days): "
          f"{total - mid:9.3f}  ({(total - mid) / total:.1%} of total)")
    print(f"  mean per middle day {diff[middle].mean():+.4f}, "
          f"per outer day {diff[~middle].mean():+.4f}")
    print(f"  middle days where the t fits better: {np.mean(diff[middle] > 0):.1%}")
    return diff, middle

# ===========================================================================
# Step 11: part (j) -- tail dependence of the most correlated pair
# ===========================================================================
def part_j(rets, r, t):
    cols = list(rets.columns)
    i, j = max(combinations(range(len(cols)), 2), key=lambda p: r[p[0], p[1]])
    rho = r[i, j]
    lam_t = copula.t_tail_dependence(rho, t.nu)                     # copula.py
    print(f"\n=== (j) Tail dependence, most correlated pair {cols[i]}-{cols[j]} "
          f"(rho = {rho:.4f}) ===")
    print(f"  t copula (nu = {t.nu:.3f}):  lambda = {lam_t:.4f}  (same in both tails)")
    print(f"  Gaussian copula:          lambda = {copula.gaussian_tail_dependence(rho):.4f}")
    return lam_t

# ===========================================================================
# Print Output
# ===========================================================================
def main():
    rets, holdings, pnl = load()                     # step 1
    part_a(rets)                                     # step 2
    _, counts = part_b(rets)                         # step 3
    margins = part_c(rets)                           # step 4
    u, r, g, t = part_d(rets, margins)               # step 5
    sims, risk = part_e(rets, holdings, pnl, margins, g, t)   # step 6
    implied = part_f(rets, sims)                     # step 7
    part_g(g, t, counts, implied)                    # step 8
    part_h(risk)                                     # step 9
    part_i(u, g, t)                                  # step 10
    part_j(rets, r, t)                               # step 11

if __name__ == "__main__":
    main()
