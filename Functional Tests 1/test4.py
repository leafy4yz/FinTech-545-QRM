"""Test 4 - cholesky factorization (Test 4.1). Run directly: `python test4.py`.
Input: testout_3.1.csv -- run test1.py then test3.py first."""
from common import infile, outfile, load_csv, write_csv
from library.psd import chol_psd


def test4():
    cin = load_csv(infile("4.1")).to_numpy(dtype=float)
    cout = chol_psd(cin)
    write_csv(cout, outfile("4.1"))


if __name__ == "__main__":
    test4()
