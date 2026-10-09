#FINTECH 545 Assignment 2 Problem 1
#Friday, October 9th, 2026
#Ivy Zhu

from pathlib import Path

import numpy as np
import pandas as pd

# The five library modules (each is a file in qrmlib/):
from qrmlib import missing, correlation, cholesky, psd_repair, portfolio

# Show matrices with 4 decimals; eigenvalues in plain (not scientific) notation.
pd.set_option("display.precision", 4)
pd.set_option("display.width", 120)
np.set_printoptions(precision=6, suppress=True)

# Weights used to build the index from the four stocks (given in the assignment).
BASKET = {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10}

# ===========================================================================
# Step 1: load the data
# ===========================================================================
def load_data():
    """Read problem1.csv (path is relative to this file, so it runs anywhere).
    A blank cell = the asset did not trade that day = NaN."""
    path = Path(__file__).resolve().parent / "problem1.csv"
    return pd.read_csv(path)[["A", "B", "C", "D", "IDX"]]

# ===========================================================================
# Step 2: predict step -- how many days is each pair jointly observed?
# ===========================================================================
def predict_table(df):
    print("=== Joint observation counts (days both series trade) ===")
    print(missing.joint_obs_counts(df))                        # qrmlib/missing.py
    print("All five trade:", missing.n_complete(df),           # qrmlib/missing.py
          "of", len(df), "days\n")

# ===========================================================================
# Step 3: correlation matrices, eigenvalues, Cholesky attempts
# ===========================================================================
def part_d(df):
    # Complete case: only the days on which all five trade -> one common sample.
    # Pairwise: each entry uses every day on which that PAIR trades -> a
    # different sample for each entry.
    corr_cc = correlation.corr_complete(df)                    # qrmlib/correlation.py
    corr_pw = correlation.corr_pairwise(df)                    # qrmlib/correlation.py

    print("=== Complete-case correlation matrix ===")
    print(corr_cc)
    print("Eigenvalues (ascending):", correlation.eigenvalues(corr_cc))   # correlation.py

    print("\n=== Pairwise correlation matrix ===")
    print(corr_pw)
    print("Eigenvalues (ascending):", correlation.eigenvalues(corr_pw))   # correlation.py

    # Cholesky attempt.  attempt_cholesky uses the STRICT (positive-definite)
    # algorithm chol_pd, so a bad pivot shows up as a reported failure rather
    # than being silently patched to zero.
    print("\n=== Cholesky attempts ===")
    for name, mat in [("complete case", corr_cc), ("pairwise", corr_pw)]:
        res = cholesky.attempt_cholesky(mat.to_numpy(),        # qrmlib/cholesky.py
                                        labels=list(mat.columns))
        print(f"{name:>14}: {res.message}")

    return corr_cc, corr_pw

# ===========================================================================
# Step 4: covariance from pairwise correlations + tracking variance
# ===========================================================================
def part_e(df, corr_cc, corr_pw):
    # Each series' std dev uses ALL of its own observations (not just overlap
    # days); only the correlations carry the pairwise-consistency problem.
    sd = portfolio.full_history_std(df)                        # qrmlib/portfolio.py
    print("\n=== Full-history standard deviations ===")
    print(sd)

    # Position vector over (A, B, C, D, IDX): long 1 IDX, short the basket
    #   w = (-0.4, -0.3, -0.2, -0.1, +1)
    w = portfolio.tracking_weights(list(df.columns), BASKET,   # qrmlib/portfolio.py
                                   index="IDX")
    print("\nTracking portfolio weights:",
          {c: float(x) for c, x in zip(df.columns, w)})

    # Sigma = D C D, then variance = w' Sigma w
    cov_pw = portfolio.cov_from_corr(corr_pw.to_numpy(), sd.to_numpy())   # portfolio.py
    var_pw = portfolio.portfolio_variance(w, cov_pw)                      # portfolio.py

    # The part (e) covariance matrix itself (pairwise correlations x full-history
    # std devs).  Entry (i, j) = sd_i * corr_ij * sd_j; the diagonal is each
    # series' full-history variance.
    print("\n=== Covariance matrix: pairwise correlations ===")
    print(pd.DataFrame(cov_pw, index=df.columns, columns=df.columns)
            .to_string(float_format="{:.4e}".format))
    print("Eigenvalues (ascending):", correlation.eigenvalues(cov_pw))    # correlation.py
    print(f"\n Tracking portfolio variance, pairwise correlations: {var_pw:.4e}")
    if var_pw < 0:
        print("    -> NEGATIVE: a variance cannot be negative, so this Sigma is not a "
              "valid covariance matrix.")
    else:
        print(f"    -> daily std dev = {np.sqrt(var_pw):.6f}")

    # Reference only (not asked in (e)): complete-case correlations, which are
    # guaranteed PSD.  Useful for part (i).
    cov_cc = portfolio.cov_from_corr(corr_cc.to_numpy(), sd.to_numpy())   # portfolio.py
    var_cc = portfolio.portfolio_variance(w, cov_cc)                      # portfolio.py
    print(f"    (reference) complete-case correlations:               {var_cc:.4e}")

    return sd, w, var_pw

# ===========================================================================
# Step 5: repair the pairwise matrix two ways
# ===========================================================================
def part_f(corr_pw, sd, w, var_pw):
    c_pw = corr_pw.to_numpy()

    repairs = {}
    repairs["Rebonato-Jackel"] = psd_repair.rebonato_jackel(c_pw)       # qrmlib/psd_repair.py

    hig = psd_repair.higham_nearest_correlation(c_pw)                   # qrmlib/psd_repair.py
    repairs["Higham"] = hig.matrix
    print(f"\nHigham: {hig.iterations} iterations, converged={hig.converged}")

    # One table row per matrix: smallest eigenvalue, distance from the pairwise
    # matrix, and the tracking variance recomputed with that matrix.
    rows = [{
        "repair": "none (pairwise)",
        "min eigenvalue": correlation.min_eigenvalue(c_pw),             # correlation.py
        "Frobenius distance from pairwise": 0.0,
        "tracking variance": var_pw,
    }]
    for name, mat in repairs.items():
        cov = portfolio.cov_from_corr(mat, sd.to_numpy())               # portfolio.py
        rows.append({
            "repair": name,
            "min eigenvalue": correlation.min_eigenvalue(mat),          # correlation.py
            "Frobenius distance from pairwise": psd_repair.frobenius_distance(mat, c_pw),
            "tracking variance": portfolio.portfolio_variance(w, cov),  # portfolio.py
        })

    print("\n=== Repairs of the pairwise correlation matrix ===")
    print(pd.DataFrame(rows).set_index("repair").to_string(float_format="{:.6e}".format))

    return repairs

# ===========================================================================
# Step 6: support output for the Reconcile questions (h) and (i)
# ===========================================================================
def support_h_i(df, corr_cc, corr_pw, repairs):
    cols = list(df.columns)
    c_pw = corr_pw.to_numpy()
    joint = missing.joint_obs_counts(df)                                # qrmlib/missing.py

    # (h) Which entries did Higham move the most?  Sorted by absolute change,
    #     alongside the number of days behind each entry.
    changes = []
    for i in range(len(cols)):
        for j in range(i):
            changes.append({
                "pair": f"{cols[j]}-{cols[i]}",
                "pairwise": c_pw[i, j],
                "Higham": repairs["Higham"][i, j],
                "change": repairs["Higham"][i, j] - c_pw[i, j],
                "days observed": int(joint.iloc[i, j]),
            })
    chg = pd.DataFrame(changes)
    chg = chg.reindex(chg["change"].abs().sort_values(ascending=False).index)
    print("\n=== Higham changes by entry, largest first ===")
    print(chg.to_string(index=False))

    # (i) Size of the repair vs size of the estimator gap.
    dist = psd_repair.frobenius_distance                                # qrmlib/psd_repair.py
    print("\n=== Estimator choice vs repair choice (Frobenius distances) ===")
    print(f"complete case vs pairwise              : {dist(corr_cc.to_numpy(), c_pw):.4f}")
    print(f"pairwise vs Higham repair              : {dist(repairs['Higham'], c_pw):.4f}")
    print(f"pairwise vs Rebonato-Jackel repair     : {dist(repairs['Rebonato-Jackel'], c_pw):.4f}")
    print(f"Higham vs Rebonato-Jackel              : "
          f"{dist(repairs['Higham'], repairs['Rebonato-Jackel']):.4f}")

# ===========================================================================
# Print Output
# ===========================================================================
def main():
    df = load_data()                                       # step 1
    predict_table(df)                                      # step 2
    corr_cc, corr_pw = part_d(df)                          # step 3
    sd, w, var_pw = part_e(df, corr_cc, corr_pw)           # step 4
    repairs = part_f(corr_pw, sd, w, var_pw)               # step 5
    support_h_i(df, corr_cc, corr_pw, repairs)             # step 6

if __name__ == "__main__":
    main()
