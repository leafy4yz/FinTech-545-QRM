"""
Every recipe from USAGE.md, run end to end on the practice data.

    python3 make_practice_data.py     # once
    python3 cookbook.py

Each section prints what it computes so you can see what the functions return.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import qrmlib as q

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:,.6g}")
HERE = Path(__file__).resolve().parent


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------- 1. one series
section("1. One return series: moments, normal vs t, VaR / ES")
x = pd.read_csv(HERE / "practice_returns.csv")["x"]

print(q.four_moments(x))                      # mean, variance, skewness, EXCESS kurtosis
print(q.compare_normal_t(x))                  # fitted params + loglik, AIC, AICc, BIC
print(q.risk_table(x, alphas=(0.05, 0.01)))   # VaR/ES: normal, t, historical (return units)
print(q.risk_table(x, alphas=(0.05, 0.01), value=1_000_000))   # same in $ on a $1M position

tf = q.fit_t(x)                               # the pieces one at a time
print("t fit:", tf.mu, tf.sigma, tf.nu, "| VaR 1%", q.var_t(tf, 0.01), "| ES 1%", q.es_t(tf, 0.01))

# ---------------------------------------------------------------- 2. EW estimates
section("2. Exponentially weighted correlation, variance, and the combined covariance")
df = pd.read_csv(HERE / "practice_series.csv")

print("n_eff", q.effective_sample_size(0.97), q.effective_sample_size(0.94),
      "| half life", q.half_life(0.97), q.half_life(0.94))
corr97 = q.ew_correlation(df, 0.97)           # correlation with lambda = 0.97
var94 = q.ew_std_vector(df, 0.94) ** 2        # variances with lambda = 0.94
cov, corr, sd = q.ew_mixed_covariance(df, lam_corr=0.97, lam_var=0.94)
print(corr97, var94, cov, sep="\n\n")

# ---------------------------------------------------------------- 3. non-PSD matrix
section("3. Repairing a non-PSD covariance matrix")
c = pd.DataFrame([[1.0, 0.9, 0.7], [0.9, 1.0, -0.4], [0.7, -0.4, 1.0]],
                 index=list("ABC"), columns=list("ABC"))
sd3 = np.array([0.02, 0.03, 0.025])
bad = c * np.outer(sd3, sd3)                  # a covariance with inconsistent correlations

higham = q.near_psd_cov(bad, "higham")        # repair the correlation, keep the variances
rj = q.near_psd_cov(bad, "rj")
print(q.eigen_compare(original=bad, higham=higham, rebonato_jackel=rj))
print("distance from original: Higham", q.frobenius_distance(higham, bad),
      "| RJ", q.frobenius_distance(rj, bad))

# ---------------------------------------------------------------- 4. portfolio
section("4. Portfolio from prices: shares, returns, fits, $ VaR / ES")
prices = pd.read_csv(HERE / "practice_prices.csv")
stocks = prices.drop(columns="Date")
last = stocks.iloc[-1]

shares = q.shares_for_weights(last, 100_000)            # equal weights on $100,000
values = q.position_values(shares, last)
print(pd.DataFrame({"price": last, "shares": shares, "value": values}))

rets = q.return_calculate(prices, "DISCRETE", date_column="Date")
print(rets.head(), rets.tail(), sep="\n")
r = rets.drop(columns="Date")
r = r - r.mean()                                        # remove the mean

margins_t = q.fit_margins(r, "t")
margins_n = q.fit_margins(r, "normal")
print(q.margin_table(margins_t), q.margin_table(margins_n), sep="\n")

gc, info = q.copula_risk(r, values, alpha=0.01, family="t", copula="gaussian", rng=545)
hist = q.historical_risk(r, values, alpha=0.01)
dn = q.delta_normal_risk(values, r.cov(), alpha=0.01)
print(q.compare_methods({"Gaussian copula (t margins)": gc, "Historical": hist,
                         "Delta normal": dn}))
print("copula correlation R:\n", info["R"])

# ---------------------------------------------------------------- 5. simulation
section("5. Simulation and PCA")
S = r.cov()
print(q.variance_explained(S))
sims = q.simulate_normal(S, 10_000, method="pca", pct_explained=0.9, rng=1)
print("simulated vs input std:\n", pd.DataFrame({"sim": sims.std(), "input": np.sqrt(np.diag(S))}))
