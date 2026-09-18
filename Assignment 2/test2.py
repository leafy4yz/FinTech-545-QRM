"""Test 2 - EW Covariance (Tests 2.1-2.3). Run directly: `python test2.py`.
Input: test2.csv."""
import numpy as np

from common import infile, outfile, load_csv, write_csv
from library.ew_cov import ew_covar


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


if __name__ == "__main__":
    test2()
