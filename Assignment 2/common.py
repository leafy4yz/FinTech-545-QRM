#QRM Functional Tests 1.1-7.6 -- shared setup
#Sept 19th, 2026
#Ivy Zhu
"""
Shared machinery for test1.py .. test7.py: the TEST_MANIFEST (the one place
to rename a test's input/output file), and the load_csv/write_csv helpers
built on it. Each testN.py imports what it needs from here -- this file has
no tests of its own.

None of the testN.py files generate their own input data. Each one expects
its input CSV(s), as named in TEST_MANIFEST, to already be available, and
just loads them.

"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
