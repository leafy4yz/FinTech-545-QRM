"""Test 1 - missing covariance calculations (Tests 1.1-1.4). Run directly:
`python test1.py`. Input: test1.csv."""
from common import infile, outfile, load_csv, write_csv
from library.missing_cov import missing_cov, _cov, _cor


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

#Entry point: runs when this script is executed directly
if __name__ == "__main__":
    test1()
