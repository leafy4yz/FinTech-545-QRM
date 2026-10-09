# QRM Functional Tests 1.1-7.6 -- Ivy Zhu

Python port of the course's Julia `test_setup.jl` / `library/*.jl`, covering
Tests 1.1 through 7.6.

## Contents

- `library/` -- ported functions (`missing_cov`, `ew_covar`, `near_psd`,
  `chol_psd`, `higham_nearest_psd`, `simulate_normal`, `simulate_pca`,
  `return_calculate`, `fit_normal`, `fit_general_t`, `fit_regression_t`,
  `fit_nig_moments`, `fit_NIG_mle`, `aicc`)
- `common.py` -- shared setup: the `TEST_MANIFEST` (input/output file names
  for every test), plus `load_csv()` / `write_csv()` helpers
- `test1.py` .. `test7.py` -- one file per test, each runnable on its own
- `run_all.py` -- runs `test1()` through `test7()` in order
- `data/` -- input CSVs and this script's output CSVs

## Requirements

```
pip install numpy scipy pandas
```

## Running

To run everything:

```
python3 run_all.py
```

To run a single test (e.g. Test 3):

```
python3 test3.py
```

Tests aren't fully independent -- Test 3 needs Test 1's output
(`testout_1.3.csv`, `testout_1.4.csv`), and Test 4 needs Test 3's output
(`testout_3.1.csv`). The first time through, run them in numeric order (or
just use `run_all.py`, which does that automatically). Every output CSV is
written to `data/`, named per `TEST_MANIFEST` in `common.py`.

## Notes

- Every function is deterministic given its input (same algorithm, same
  formulas).
- Regarding Tests 5.1-5.5: each is the sample covariance of a
  fresh 100,000-row random simulation. NumPy's RNG doesn't reproduce
  Julia's RNG stream, so those results should be close to Julia's, not
  bit-identical.
