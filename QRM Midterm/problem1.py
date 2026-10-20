#FINTECH 545 Midterm Problem 1
#Oct 2026
#Ivy Zhu

# Library: qrmlib (QRM Library/, v0.7.0)
# Data: problem1.csv -- 50 rows, columns x1..x5, with blank (missing) cells

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
    # blank cells -> NaN (missing.py treats NaN as "not observed")
    return pd.read_csv(HERE / "problem1.csv")


# 2. Part a ---------------------------------------------------------------------
# correlation.py: corr_pairwise
def part_a(df):
    corr = q.corr_pairwise(df)
    print("Part a: pairwise correlation matrix")
    print(corr.round(4).to_string())
    return corr


# 3. Part b ---------------------------------------------------------------------
# portfolio.py: full_history_var
def part_b(df):
    var = q.full_history_var(df)
    print("\nPart b: variance of each series (all observed values)")
    print(pd.DataFrame({"n_obs": df.notna().sum(), "variance": var}).to_string(float_format="%.6e"))
    return var


# 4. Part c ---------------------------------------------------------------------
# correlation.py: eigenvalues (ascending)
def part_c(corr):
    eig = q.eigenvalues(corr)
    print("\nPart c: eigenvalues of the pairwise correlation matrix (ascending)")
    print(pd.Series(eig, index=[f"lambda_{i + 1}" for i in range(len(eig))]).to_string(float_format="%.6f"))
    return eig


# 5. Part d ---------------------------------------------------------------------
# psd_repair.py: higham_nearest_correlation; correlation.py: eigenvalues
def part_d(corr):
    res = q.higham_nearest_correlation(corr)
    c_psd = pd.DataFrame(res.matrix, index=corr.index, columns=corr.columns)
    eig = q.eigenvalues(c_psd)
    print("\nPart d: Higham nearest PSD correlation matrix")
    print(c_psd.round(4).to_string())
    print(f"iterations = {res.iterations}, converged = {res.converged}, "
          f"||C_psd - C||_F = {res.final_gamma:.6f}")
    print("Eigenvalues (ascending)")
    print(pd.Series(eig, index=[f"lambda_{i + 1}" for i in range(len(eig))]).to_string(float_format="%.6f"))
    return c_psd, eig


# 6. Part e ---------------------------------------------------------------------
# portfolio.py: cov_from_corr  (Sigma = D C_psd D, D = diag(sqrt(var)))
def part_e(c_psd, var):
    sd = np.sqrt(var)
    cov = pd.DataFrame(q.cov_from_corr(c_psd, sd), index=c_psd.index, columns=c_psd.columns)
    print("\nPart e: covariance matrix (Higham correlation, full-sample variances)")
    print(cov.to_string(float_format="%.6e"))
    print("Eigenvalues (ascending)")
    eig = q.eigenvalues(cov)
    print(pd.Series(eig, index=[f"lambda_{i + 1}" for i in range(len(eig))]).to_string(float_format="%.6e"))
    return cov


def main():
    df = load_data()
    corr = part_a(df)
    var = part_b(df)
    part_c(corr)
    c_psd, _ = part_d(corr)
    part_e(c_psd, var)


if __name__ == "__main__":
    main()
