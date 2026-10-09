"""Test 7 - distribution fitting (Tests 7.1-7.6). Run directly:
`python test7.py`. Inputs: test7_1.csv, test7_2.csv, test7_3.csv, test7_5.csv.

Test 7.6 uses scipy.stats.norminvgauss.fit(), exactly like the Julia script
does through PyCall, so that step is a "native" Python computation, not a
port, and should be the closest match to the Julia output of anything here."""
import pandas as pd

from common import infile, outfile, load_csv, _path
from library.fitted_model import (
    fit_normal, fit_general_t, fit_regression_t, fit_nig_moments, fit_NIG_mle, aicc,
)


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


if __name__ == "__main__":
    test7()
