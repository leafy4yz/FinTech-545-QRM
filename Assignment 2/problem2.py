#FINTECH 545 Assignment 2 Problem 2
#Friday, October 9th, 2026
#Ivy Zhu
#
# A Volatility Estimate After the Regime Changed.
# Run from the "Assignment 2" folder:   python3 problem2.py
# Prints every number used in the write-up and saves problem2_returns.png.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # write the figure to a file, no window needed
import matplotlib.pyplot as plt

# Library modules (each is a file in qrmlib/):
from qrmlib import returns, ewma, moments, fitting, var, regime

pd.set_option("display.precision", 6)
pd.set_option("display.width", 120)

HERE = Path(__file__).resolve().parent
PV = 1_000_000.0          # position value, $
ALPHA = 0.05              # VaR level (assignment default: 5%, reported as a positive loss)
LAMBDAS = [0.97, 0.94]

# Number of recent days that look different on the plot (part b).
# Read off problem2_returns.png: the last ~40 returns are visibly wider than
# everything before them.  regime.variance_breakpoint() is printed below as an
# independent check on this eyeballed number.
N_RECENT = 40

# ===========================================================================
# Step 1: load prices, arithmetic returns, remove the sample mean
# ===========================================================================
def load_returns():
    prices = pd.read_csv(HERE / "problem2.csv")["Price"]
    r = returns.return_calculate(prices, method="DISCRETE")   # qrmlib/returns.py
    print(f"{len(prices)} prices -> {len(r)} arithmetic returns; "
          f"sample mean removed = {r.mean():.6e}")
    return returns.demean(r)                                   # qrmlib/returns.py

# ===========================================================================
# Step 2: plot the returns (Predict)
# ===========================================================================
def plot_returns(r, n_recent, path=HERE / "problem2_returns.png"):
    n = len(r)
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 2]})

    # Top: the demeaned returns with the recent window shaded.
    ax[0].plot(np.arange(1, n + 1), r, lw=0.8, color="black")
    ax[0].axvspan(n - n_recent + 0.5, n + 0.5, color="red", alpha=0.15,
                  label=f"last {n_recent} days")
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_title("problem2: demeaned daily arithmetic returns")
    ax[0].set_ylabel("return")
    ax[0].legend(loc="upper left")

    # Bottom: how much weight each lambda puts on each day (for part b).
    for lam in LAMBDAS:
        ax[1].plot(np.arange(1, n + 1), ewma.ew_weights(n, lam), label=f"lambda = {lam}")
    ax[1].axvspan(n - n_recent + 0.5, n + 0.5, color="red", alpha=0.15)
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_title("normalised exponential weights")
    ax[1].set_xlabel("day")
    ax[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {path.name}")

# ===========================================================================
# Step 3: part (b) -- effective sample size, half life, weight on recent days
# ===========================================================================
def part_b(r):
    n = len(r)
    k_break, _ = regime.variance_breakpoint(r)                 # qrmlib/regime.py
    print(f"\nEyeballed recent regime: last {N_RECENT} days")
    print(f"Check -- max-likelihood variance break: new regime starts at return "
          f"{k_break + 1} of {n}, i.e. the last {n - k_break} days")

    rows = []
    for lam in LAMBDAS:
        rows.append({
            "lambda": lam,
            # infinite-history formula (1+lam)/(1-lam) ...
            "n_eff (1+l)/(1-l)": ewma.effective_sample_size(lam),        # ewma.py
            # ... and the exact value for our finite, renormalised 500 weights
            "n_eff (500 obs)": ewma.effective_sample_size(lam, n),       # ewma.py
            "half life (days)": ewma.half_life(lam),                      # ewma.py
            f"weight on last {N_RECENT}": ewma.weight_on_recent(n, lam, N_RECENT),
            "equal-weight share": N_RECENT / n,
        })
    tab = pd.DataFrame(rows).set_index("lambda")
    print("\n=== (b) Exponential weighting ===")
    print(tab.to_string(float_format="{:.4f}".format))
    return tab

# ===========================================================================
# Step 4: part (c) -- first four moments of the full sample
# ===========================================================================
def part_c(r):
    m = moments.four_moments(r)                                 # qrmlib/moments.py
    se_skew, se_kurt = moments.skew_kurt_std_errors(len(r))     # moments.py
    jb, jb_p = moments.jarque_bera(r)                           # moments.py
    print("\n=== (c) First four moments (full sample) ===")
    print(m.to_string(float_format="{:.6g}".format))
    print(f"std dev                {np.sqrt(m['variance']):.6f}")
    print(f"Under one normal: s.e.(skew) ~ {se_skew:.3f}, s.e.(excess kurt) ~ {se_kurt:.3f}")
    print(f"  skew z-score   = {m['skewness'] / se_skew:6.2f}")
    print(f"  kurt z-score   = {m['excess kurtosis'] / se_kurt:6.2f}")
    print(f"Jarque-Bera = {jb:.2f}, p-value = {jb_p:.3g}")
    return m

# ===========================================================================
# Step 5: part (d) -- the five VaR estimates in dollars
# ===========================================================================
def part_d(r):
    sd_eq = r.std(ddof=1)                                       # equally weighted, n-1
    sd_ew = {lam: ewma.ew_std(r, lam) for lam in LAMBDAS}       # qrmlib/ewma.py
    tfit = fitting.fit_t(r)                                     # qrmlib/fitting.py

    # Returns are demeaned, so the normal models use mu = 0.  The t keeps its
    # fitted location (tiny, see below) because that is what the MLE returned.
    est = {
        "1. Normal, equal weight":   (sd_eq,           var.var_normal(sd_eq, ALPHA, value=PV)),
        "2. Normal, EW lambda=0.97": (sd_ew[0.97],     var.var_normal(sd_ew[0.97], ALPHA, value=PV)),
        "3. Normal, EW lambda=0.94": (sd_ew[0.94],     var.var_normal(sd_ew[0.94], ALPHA, value=PV)),
        "4. Student t (MLE)":        (tfit.std,        var.var_t(tfit, ALPHA, value=PV)),
        # Historical: floor(0.05*500) = 25th smallest return, the class convention.
        "5. Historical simulation":  (sd_eq,           var.var_historical(r, ALPHA, value=PV)),
    }
    tab = pd.DataFrame(est, index=["vol used", "VaR ($)"]).T
    print(f"\n=== (d) One-day {ALPHA:.0%} VaR on ${PV:,.0f} ===")
    print(tab.to_string(formatters={"vol used": "{:.6f}".format, "VaR ($)": "{:,.0f}".format}))
    print("(historical row: 'vol used' is just the sample sd, for reference)")
    print(f"Historical with interpolated quantile instead: "
          f"${var.var_historical(r, ALPHA, PV, method='quantile'):,.0f}")

    print("\nRanked smallest to largest:")
    for name, v in tab["VaR ($)"].sort_values().items():
        print(f"  {name:<28} ${v:,.0f}")

    print("\nFitted t parameters (X = mu + sigma * T_nu):")
    print(f"  mu    = {tfit.mu:.6e}")
    print(f"  sigma = {tfit.sigma:.6f}")
    print(f"  nu    = {tfit.nu:.4f}")
    print(f"  implied sd = {tfit.std:.6f},  implied excess kurtosis = {tfit.excess_kurtosis:.4f}")
    _, _, ll_n = fitting.fit_normal(r)                          # fitting.py
    print(f"  log-lik t = {tfit.loglik:.2f}  vs normal = {ll_n:.2f}")
    return sd_eq, sd_ew, tfit, tab

# ===========================================================================
# Step 6: part (f) -- regime standard deviations and the mixture kurtosis
# ===========================================================================
def part_f(r, tfit):
    n = len(r)
    s_old, s_new = regime.regime_stds(r, N_RECENT)              # qrmlib/regime.py
    p_new = N_RECENT / n
    mix = regime.normal_mixture_moments([1 - p_new, p_new],     # regime.py
                                        [s_old, s_new])
    print("\n=== (f) Inside each regime ===")
    print(f"  first {n - N_RECENT} days : sd = {s_old:.6f}")
    print(f"  last  {N_RECENT} days  : sd = {s_new:.6f}   (ratio {s_new / s_old:.2f}x)")
    for label, seg in [("old", r[:-N_RECENT]), ("new", r[-N_RECENT:])]:
        mm = moments.four_moments(seg)
        print(f"  excess kurtosis inside {label} regime: {mm['excess kurtosis']:.3f}")
    print(f"Mixture of two normals ({1 - p_new:.0%} / {p_new:.0%}) implies:")
    print(f"  sd = {mix['std']:.6f},  excess kurtosis = {mix['excess kurtosis']:.3f}")
    print(f"Fitted t sigma = {tfit.sigma:.6f} (close to old-regime sd {s_old:.6f}), "
          f"nu = {tfit.nu:.2f}")
    print("Normal VaR at each regime's own sd:")
    print(f"  old regime: ${var.var_normal(s_old, ALPHA, value=PV):,.0f}")
    print(f"  new regime: ${var.var_normal(s_new, ALPHA, value=PV):,.0f}")
    return s_old, s_new

# ===========================================================================
# Step 7: part (g) -- is the gap between the two EW VaRs bigger than the noise?
# ===========================================================================
def part_g(r, sd_ew):
    n = len(r)
    var97 = var.var_normal(sd_ew[0.97], ALPHA, value=PV)
    var94 = var.var_normal(sd_ew[0.94], ALPHA, value=PV)
    gap = var94 - var97
    print("\n=== (g) EW VaR gap vs estimation noise sigma/sqrt(2 n_eff) ===")
    for lam in LAMBDAS:
        neff = ewma.effective_sample_size(lam, n)               # qrmlib/ewma.py
        se_sd = ewma.ew_std_error(sd_ew[lam], neff)             # ewma.py
        se_var = var.var_noise_normal(sd_ew[lam], neff, ALPHA, PV)   # qrmlib/var.py
        print(f"  lambda={lam}: sd={sd_ew[lam]:.6f}, n_eff={neff:.2f}, "
              f"s.e.(sd)={se_sd:.6f}, s.e.(VaR)=${se_var:,.0f}")
    se97 = var.var_noise_normal(sd_ew[0.97], ewma.effective_sample_size(0.97, n), ALPHA, PV)
    se94 = var.var_noise_normal(sd_ew[0.94], ewma.effective_sample_size(0.94, n), ALPHA, PV)
    print(f"  VaR(0.94) - VaR(0.97) = ${gap:,.0f}")
    print(f"  gap / s.e.(0.97)  = {gap / se97:.2f}")
    print(f"  gap / s.e.(0.94)  = {gap / se94:.2f}")
    # Joint noise of the two estimates combined in quadrature, sqrt(se1^2 + se2^2).
    # This treats the two estimates as independent; they share the same returns,
    # so it is an approximation.
    print(f"  gap / sqrt(se97^2 + se94^2) = {gap / np.hypot(se97, se94):.2f}  (assumes independence)")

# ===========================================================================
# Print Output
# ===========================================================================
def main():
    r = load_returns()                     # step 1
    plot_returns(r, N_RECENT)              # step 2
    part_b(r)                              # step 3
    part_c(r)                              # step 4
    sd_eq, sd_ew, tfit, _ = part_d(r)      # step 5
    part_f(r, tfit)                        # step 6
    part_g(r, sd_ew)                       # step 7

if __name__ == "__main__":
    main()
