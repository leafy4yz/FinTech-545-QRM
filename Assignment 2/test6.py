"""Test 6 - returns (Tests 6.1-6.2). Run directly: `python test6.py`.
Input: test6.csv."""
from common import infile, outfile, load_csv, _path
from library.return_calculate import return_calculate


def test6():
    prices = load_csv(infile("6.1"))

    # 6.1 Arithmetic returns
    rout = return_calculate(prices, date_column="Date")
    rout.to_csv(_path(outfile("6.1")), index=False)

    # 6.2 Log returns
    rout = return_calculate(prices, method="LOG", date_column="Date")
    rout.to_csv(_path(outfile("6.2")), index=False)


if __name__ == "__main__":
    test6()
