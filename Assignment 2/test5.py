"""Test 5 - Normal Simulation (Tests 5.1-5.5). Run directly: `python test5.py`.
Inputs: test5_1.csv, test5_2.csv, test5_3.csv.

Note: these are the one partial exception to bit-for-bit matching -- each
output is the sample covariance of a fresh 100,000-row random draw, and
NumPy's RNG doesn't reproduce Julia's stream, so results should be close
to Julia's but not bit-identical. See common.py's docstring."""
import numpy as np

from common import infile, outfile, load_csv, write_csv
from library.psd import near_psd, higham_nearest_psd
from library.simulate import simulate_normal, simulate_pca


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


if __name__ == "__main__":
    test5()
