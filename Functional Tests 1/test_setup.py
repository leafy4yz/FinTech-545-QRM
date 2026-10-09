#QRM Functional Tests 1.1-7.6
#Sept 19th, 2026
#Ivy Zhu
"""
This script does not generate any input data. It expects every input CSV
named in TEST_MANIFEST (test1.csv, test2.csv, test5_1/2/3.csv, test6.csv,
test7_1/2/3/5.csv, ...) to available and just loads them.

Every function here is deterministic given its input (missing_cov, ewCovar,
near_psd, higham, chol_psd, return_calculate, simulate_normal/simulate_pca,
and every fit_* in Test 7), so given the same input CSVs the Julia script
used, these should reproduce Julia's output to floating point precision.
The one partial exception is Tests 5.1-5.5: they're the sample covariance
of a fresh 100,000-row random simulation, so even with an identical input
covariance matrix, NumPy's RNG (used inside simulate_normal/simulate_pca)
draws a different sample than Julia's would -- the result should be close,
not bit-identical.

Test 7.6 uses scipy.stats.norminvgauss.fit(), exactly like the Julia script
does through PyCall, so that step is a "native" Python computation, not a
port, and should be the closest match to the Julia output of anything here.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.missing_cov import missing_cov, _cov, _cor
from library.ew_cov import ew_covar
from library.psd import near_psd, chol_psd, higham_nearest_psd
from library.simulate import simulate_normal, simulate_pca
from library.return_calculate import return_calculate
from library.fitted_model import (
    fit_normal, fit_general_t, fit_regression_t, fit_nig_moments, fit_NIG_mle, aicc,
)
import numpy as np

# =============================================================================
# TEST MANIFEST -- one place to rename a test's input/output file(s)
# =============================================================================
# Transcribed directly from Tests.xlsx (Test / Description / Input /
# Output columns)
#
# Rows are kept for relevant tests in Tests.xlsx
TEST_MANIFEST = {
    "1.1": {"description": "Covariance Missing data, skip missing rows", "inputs": ("test1.csv",), "outputs": ("testout_1.1.csv",), "implemented": True},
    "1.2": {"description": "Correlation Missing data, skip missing rows", "inputs": ("test1.csv",), "outputs": ("testout_1.2.csv",), "implemented": True},
    "1.3": {"description": "Covariance Missing data, Pairwise", "inputs": ("test1.csv",), "outputs": ("testout_1.3.csv",), "implemented": True},
    "1.4": {"description": "Correlation Missing data, pairwise", "inputs": ("test1.csv",), "outputs": ("testout_1.4.csv",), "implemented": True},

    "2.1": {"description": "EW Covariance, lambda=0.97", "inputs": ("test2.csv",), "outputs": ("testout_2.1.csv",), "implemented": True},
    "2.2": {"description": "EW Correlation, lambda=0.94", "inputs": ("test2.csv",), "outputs": ("testout_2.2.csv",), "implemented": True},
    "2.3": {"description": "Covariance with EW Variance (l=0.97), EW Correlation (l=0.94)", "inputs": ("test2.csv",), "outputs": ("testout_2.3.csv",), "implemented": True},

    "3.1": {"description": "near_psd covariance", "inputs": ("testout_1.3.csv",), "outputs": ("testout_3.1.csv",), "implemented": True},
    "3.2": {"description": "near_psd correlation", "inputs": ("testout_1.4.csv",), "outputs": ("testout_3.2.csv",), "implemented": True},
    "3.3": {"description": "Higham covariance", "inputs": ("testout_1.3.csv",), "outputs": ("testout_3.3.csv",), "implemented": True},
    "3.4": {"description": "Higham correlation", "inputs": ("testout_1.4.csv",), "outputs": ("testout_3.4.csv",), "implemented": True},

    "4.1": {"description": "chol_psd", "inputs": ("testout_3.1.csv",), "outputs": ("testout_4.1.csv",), "implemented": True},

    "5.1": {"description": "Normal Simulation PD Input 0 mean - 100,000 simulations, compare input vs output covariance", "inputs": ("test5_1.csv",), "outputs": ("testout_5.1.csv",), "implemented": True},
    "5.2": {"description": "Normal Simulation PSD Input 0 mean - 100,000 simulations, compare input vs output covariance", "inputs": ("test5_2.csv",), "outputs": ("testout_5.2.csv",), "implemented": True},
    "5.3": {"description": "Normal Simulation nonPSD Input, 0 mean, near_psd fix - 100,000 simulations, compare input vs output covariance", "inputs": ("test5_3.csv",), "outputs": ("testout_5.3.csv",), "implemented": True},
    "5.4": {"description": "Normal Simulation PSD Input, 0 mean, higham fix - 100,000 simulations, compare input vs output covariance", "inputs": ("test5_3.csv",), "outputs": ("testout_5.4.csv",), "implemented": True},
    "5.5": {"description": "PCA Simulation, 99% explained, 0 mean - 100,000 simulations compare input vs output covariance", "inputs": ("test5_2.csv",), "outputs": ("testout_5.5.csv",), "implemented": True},

    "6.1": {"description": "calculate arithmetic returns", "inputs": ("test6.csv",), "outputs": ("testout_6.1.csv",), "implemented": True},
    "6.2": {"description": "calculate log returns", "inputs": ("test6.csv",), "outputs": ("testout_6.2.csv",), "implemented": True},

    "7.1": {"description": "Fit Normal Distribution", "inputs": ("test7_1.csv",), "outputs": ("testout_7.1.csv",), "implemented": True},
    "7.2": {"description": "Fit T Distribution", "inputs": ("test7_2.csv",), "outputs": ("testout_7.2.csv",), "implemented": True},
    "7.3": {"description": "T Regression", "inputs": ("test7_3.csv",), "outputs": ("testout_7.3.csv",), "implemented": True},
    "7.4": {"description": "Calculate AICC on fitted T", "inputs": ("test7_2.csv",), "outputs": ("testout_7.4.csv",), "implemented": True},
    "7.5": {"description": "Fit a Normal Inverse Gaussian by the method of moments. Returns mu, alpha, beta, delta.", "inputs": ("test7_5.csv",), "outputs": ("testout7_5.csv",), "implemented": True},
    "7.6": {"description": "Fit the same NIG by maximum likelihood. Returns mu, alpha, beta, delta.", "inputs": ("test7_5.csv",), "outputs": ("testout7_6.csv",), "implemented": True},

}


def infile(test_id, i=0):
    """The i-th input filename for a test, per TEST_MANIFEST."""
    return TEST_MANIFEST[test_id]["inputs"][i]


def outfile(test_id, i=0):
    """The i-th output filename for a test, per TEST_MANIFEST."""
    return TEST_MANIFEST[test_id]["outputs"][i]


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _path(name):
    return os.path.join(DATA_DIR, name)


def load_csv(name):
    """Load data/<name>. Raises FileNotFoundError (via pandas) if it isn't
    there -- this script never generates its own input data, everything
    named in TEST_MANIFEST is expected to already exist in data/."""
    return pd.read_csv(_path(name))


def write_csv(array, name, columns=None):
    """Write a 1-D or 2-D array out as data/<name>, generic x1..xN columns
    unless `columns` is given."""
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = columns or [f"x{i + 1}" for i in range(arr.shape[1])]
    pd.DataFrame(arr, columns=cols).to_csv(_path(name), index=False, na_rep="")


# ---------------------------------------------------------------------------
# Test 1 - missing covariance calculations
# ---------------------------------------------------------------------------

def test1():
    x = load_csv(infile("1.1")).to_numpy(dtype=float)

    # 1.1 Skip Missing rows - Covariance
    cout = missing_cov(x, skip_miss=True, fun=_cov)
    write_csv(cout, outfile("1.1"))

    # 1.2 Skip Missing rows - Correlation
    cout = missing_cov(x, skip_miss=True, fun=_cor)
    write_csv(cout, outfile("1.2"))

    # 1.3 Pairwise - Covariance
    cout = missing_cov(x, skip_miss=False, fun=_cov)
    write_csv(cout, outfile("1.3"))

    # 1.4 Pairwise - Correlation
    cout = missing_cov(x, skip_miss=False, fun=_cor)
    write_csv(cout, outfile("1.4"))


# ---------------------------------------------------------------------------
# Test 2 - EW Covariance
# ---------------------------------------------------------------------------

def test2():
    x = load_csv(infile("2.1")).to_numpy(dtype=float)

    # 2.1 EW Covariance lambda=0.97
    cout = ew_covar(x, 0.97)
    write_csv(cout, outfile("2.1"))

    # 2.2 EW Correlation lambda=0.94
    cout = ew_covar(x, 0.94)
    sd = 1.0 / np.sqrt(np.diag(cout))
    cout = np.diag(sd) @ cout @ np.diag(sd)
    write_csv(cout, outfile("2.2"))

    # 2.3 EW Cov w/ EW Var(lambda=0.94) EW Correlation(lambda=0.97)
    cout = ew_covar(x, 0.97)
    sd1 = np.sqrt(np.diag(cout))
    cout = ew_covar(x, 0.94)
    sd = 1.0 / np.sqrt(np.diag(cout))
    cout = np.diag(sd1) @ np.diag(sd) @ cout @ np.diag(sd) @ np.diag(sd1)
    write_csv(cout, outfile("2.3"))


# ---------------------------------------------------------------------------
# Test 3 - non-psd matrices
# ---------------------------------------------------------------------------

def test3():
    # 3.1 near_psd covariance
    cin = load_csv(infile("3.1")).to_numpy(dtype=float)
    cout = near_psd(cin)
    write_csv(cout, outfile("3.1"))

    # 3.2 near_psd Correlation
    cin = load_csv(infile("3.2")).to_numpy(dtype=float)
    cout = near_psd(cin)
    write_csv(cout, outfile("3.2"))

    # 3.3 Higham covariance
    cin = load_csv(infile("3.3")).to_numpy(dtype=float)
    cout = higham_nearest_psd(cin)
    write_csv(cout, outfile("3.3"))

    # 3.4 Higham Correlation
    cin = load_csv(infile("3.4")).to_numpy(dtype=float)
    cout = higham_nearest_psd(cin)
    write_csv(cout, outfile("3.4"))


# ---------------------------------------------------------------------------
# Test 4 - cholesky factorization
# ---------------------------------------------------------------------------

def test4():
    cin = load_csv(infile("4.1")).to_numpy(dtype=float)
    cout = chol_psd(cin)
    write_csv(cout, outfile("4.1"))


# ---------------------------------------------------------------------------
# Test 5 - Normal Simulation
# ---------------------------------------------------------------------------

def test5():
    # 5.1 PD Input
    cin = load_csv(infile("5.1")).to_numpy(dtype=float)
    sim = simulate_normal(100000, cin)
    cout = np.cov(sim, rowvar=False)
    write_csv(cout, outfile("5.1"))

    # 5.2 PSD Input
    cin = load_csv(infile("5.2")).to_numpy(dtype=float)
    sim = simulate_normal(100000, cin)
    cout = np.cov(sim, rowvar=False)
    write_csv(cout, outfile("5.2"))

    # 5.3 nonPSD Input, near_psd fix
    cin = load_csv(infile("5.3")).to_numpy(dtype=float)
    sim = simulate_normal(100000, cin, fix_method=near_psd)
    cout = np.cov(sim, rowvar=False)
    write_csv(cout, outfile("5.3"))

    # 5.4 nonPSD Input, Higham fix
    cin = load_csv(infile("5.4")).to_numpy(dtype=float)
    sim = simulate_normal(100000, cin, fix_method=higham_nearest_psd)
    cout = np.cov(sim, rowvar=False)
    write_csv(cout, outfile("5.4"))

    # 5.5 PSD Input - PCA Simulation
    cin = load_csv(infile("5.5")).to_numpy(dtype=float)
    sim = simulate_pca(cin, 100000, pct_exp=0.99)
    cout = np.cov(sim, rowvar=False)
    write_csv(cout, outfile("5.5"))


# ---------------------------------------------------------------------------
# Test 6 - returns
# ---------------------------------------------------------------------------

def test6():
    prices = load_csv(infile("6.1"))

    # 6.1 Arithmetic returns
    rout = return_calculate(prices, date_column="Date")
    rout.to_csv(_path(outfile("6.1")), index=False)

    # 6.2 Log returns
    rout = return_calculate(prices, method="LOG", date_column="Date")
    rout.to_csv(_path(outfile("6.2")), index=False)


# ---------------------------------------------------------------------------
# Test 7 - distribution fitting
# ---------------------------------------------------------------------------

def test7():
    # 7.1 Fit Normal Distribution
    cin = load_csv(infile("7.1")).to_numpy(dtype=float)
    fd = fit_normal(cin[:, 0])
    pd.DataFrame({"mu": [fd.params["mu"]], "sigma": [fd.params["sigma"]]}).to_csv(
        _path(outfile("7.1")), index=False)

    # 7.2 Fit TDist
    cin = load_csv(infile("7.2")).to_numpy(dtype=float)
    fd = fit_general_t(cin[:, 0])
    pd.DataFrame({
        "mu": [fd.params["mu"]], "sigma": [fd.params["sigma"]], "nu": [fd.params["nu"]],
    }).to_csv(_path(outfile("7.2")), index=False)

    # 7.3 Fit T Regression
    cin = load_csv(infile("7.3"))
    x_cols = [c for c in cin.columns if c != "y"]
    fd = fit_regression_t(cin["y"].to_numpy(dtype=float), cin[x_cols].to_numpy(dtype=float))
    pd.DataFrame({
        "mu": [fd.params["mu"]], "sigma": [fd.params["sigma"]], "nu": [fd.params["nu"]],
        "Alpha": [fd.beta[0]], "B1": [fd.beta[1]], "B2": [fd.beta[2]], "B3": [fd.beta[3]],
    }).to_csv(_path(outfile("7.3")), index=False)

    # 7.4 AICC on fitted T
    cin = load_csv(infile("7.4")).to_numpy(dtype=float)
    fd = fit_general_t(cin[:, 0])
    fd_aicc = aicc(fd, cin[:, 0])
    pd.DataFrame({"AICC": [fd_aicc]}).to_csv(_path(outfile("7.4")), index=False)

    # 7.5 Fit a NIG by the method of moments
    cin = load_csv(infile("7.5")).to_numpy(dtype=float)
    fd = fit_nig_moments(cin[:, 0])
    pd.DataFrame({
        "mu": [fd.params["mu"]], "alpha": [fd.params["alpha"]],
        "beta": [fd.params["beta"]], "delta": [fd.params["delta"]],
    }).to_csv(_path(outfile("7.5")), index=False)

    # 7.6 Fit the same NIG by maximum likelihood
    cin = load_csv(infile("7.6")).to_numpy(dtype=float)
    fd = fit_NIG_mle(cin[:, 0])
    pd.DataFrame({
        "mu": [fd.params["mu"]], "alpha": [fd.params["alpha"]],
        "beta": [fd.params["beta"]], "delta": [fd.params["delta"]],
    }).to_csv(_path(outfile("7.6")), index=False)


def main():
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    print("Done. Output written to", DATA_DIR)


if __name__ == "__main__":
    main()
