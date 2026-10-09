"""Test 3 - non-psd matrices (Tests 3.1-3.4). Run directly: `python test3.py`.
Inputs: testout_1.3.csv, testout_1.4.csv -- run test1.py first."""
from common import infile, outfile, load_csv, write_csv
from library.psd import near_psd, higham_nearest_psd


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


if __name__ == "__main__":
    test3()
