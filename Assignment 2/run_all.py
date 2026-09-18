"""Convenience runner: calls test1() through test7() in order, since later
tests depend on earlier ones' output (e.g. test3 needs testout_1.3.csv,
which test1 produces -- see each testN.py's docstring / TEST_MANIFEST's
"inputs"). Run directly: `python run_all.py`. Equivalent to running
test1.py .. test7.py individually, in numeric order."""
from common import DATA_DIR
from test1 import test1
from test2 import test2
from test3 import test3
from test4 import test4
from test5 import test5
from test6 import test6
from test7 import test7


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
