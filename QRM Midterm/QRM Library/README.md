# qrmlib: FinTech 545 QRM library

Ivy Zhu · version 0.7.0 · October 2026

One library with everything from the course so far:

- **Assignment 1:** moments, OLS / MLE regression, Pearson vs Spearman, conditional normals, AR/MA.
- **Assignment 2:** missing data, Cholesky, PSD repair, EWMA, VaR/ES, copulas, factor models.
- **Weeks 3–5 tools added for the midterm:** t ES, EW correlation and two-λ covariance, PSD repair of
  covariance matrices, PCA / Cholesky simulation, copula Monte Carlo and portfolio $ VaR/ES.

It is a copy of the `qrmlib` from `Assignment 2/` with new modules added, checked against the course's
own test cases (the reference outputs in `Tests 1/`). The Assignment folders keep their own libraries,
so they still run exactly as submitted.

### Changes from the Assignment 2 version

Two existing functions changed while matching the course reference outputs:

1. **`ew_covariance` now subtracts the EW-weighted mean**, like the class `ewCovar`, which reproduces
   course tests 2.1–2.3 exactly. The old behaviour is `mean="simple"`.
2. **`fit_t` no longer stops early.** The t likelihood is very flat in ν, and L-BFGS-B could quit far from
   the optimum. On the course's test 9.1 data it stopped at ν = 12.1 when ν = 4.8 has a clearly higher
   likelihood. `fit_t` now checks a profile likelihood over ν and keeps whichever fit is better, which
   fixes the copula ES in test 9.1. A fit that was already optimal comes out the same.

Run with this library, Assignment 2 Problems 1, 3 and 5 print identical output. Problems 2 and 4 move
slightly: the EWMA VaRs, the t fits (now with higher log likelihoods) and the copula figures. Your
submitted folder is unaffected.

**How to install and call it: see [`USAGE.md`](USAGE.md).** Short version:

```bash
cd ~/Downloads/FinTech-545-QRM/"QRM Library"
python3 -m pip install -e .
```
```python
import qrmlib as q
q.risk_table(x, alphas=(0.05, 0.01))
```

## Modules

| Module | Contents | New in 0.7.0 |
|---|---|---|
| `moments.py` | `four_moments` (excess kurtosis), `jarque_bera`, `skew_kurt_std_errors`, `extreme_influence` | `first4_moments` (raw kurtosis, A1), `quantile_exceedance` |
| `fitting.py` | `fit_normal`, `fit_t` (MLE), `TFit` | `compare_normal_t` |
| `criteria.py` | | `aic`, `aicc`, `bic` |
| `regression.py` | | `ols_se`, `mle_regression_normal`, `mle_regression_t` (A1) |
| `correlation.py` | complete-case / pairwise cov & corr, eigenvalues, `is_psd` | `corr_matrices`, `largest_gap`, `cov_to_corr`, `to_latex_grid` |
| `conditional.py` | | `conditional_normal`, `conditional_line`, `band_inside`, `coverage_by_bucket` (A1) |
| `timeseries.py` | | `plot_ts`, `fit_ar_ma`, `aicc_table`, `best_model` (A1, needs statsmodels) |
| `missing.py` | joint-observation counts, complete rows | |
| `cholesky.py` | `chol_pd`, `chol_psd`, `attempt_cholesky` | |
| `psd_repair.py` | `rebonato_jackel`, `higham_nearest_correlation`, `frobenius_distance` | `near_psd_cov`, `eigen_compare` |
| `ewma.py` | `ew_weights`, `ew_covariance`, `ew_variance`, n_eff, half life | `ew_cov_df`, `ew_correlation`, `ew_std_vector`, `ew_mixed_covariance` |
| `simulation.py` | | `simulate_normal` (Cholesky / PCA), `pca_root`, `variance_explained`, `simulate_mvt`, `mvt_logpdf` |
| `returns.py` | `return_calculate`, `demean` | |
| `var.py` | normal / t / historical VaR, historical & normal ES | `es_t`, `var_es_table`, `risk_table` |
| `portfolio.py` | `cov_from_corr`, `portfolio_variance`, `scenario_pnl` | |
| `portfolio_risk.py` | | `shares_for_weights`, `position_values`, `historical_risk`, `delta_normal_risk`, `normal_mc_risk`, `copula_risk`, `compare_methods` |
| `copula.py` | margins, Kendall R, Gaussian / t copulas, joint tails, tail dependence | `fit_margins`, `margin_table`, `spearman_corr` |
| `tails.py`, `regime.py`, `factor.py` | as in Assignment 2 | |

## Conventions

- VaR and ES are **positive numbers for a loss**. α is the tail probability (0.05 = 5%).
- Historical / simulated VaR = −(⌊αn⌋-th smallest), ES = −mean of the ⌊αn⌋ smallest (class code).
- Returns are arithmetic unless you pass `"LOG"`. $ P&L of a stock position = value × return.
- **Normal VaR / ES and normal copula margins use σ with n − 1** (the class `fit_normal`); pass
  `normal_sd="mle"` for n. `fit_normal` itself returns the MLE (n), and sample covariances use n − 1.
  A t's `sigma` is its **scale**, not its std.
- VaR / ES are **absolute** (loss from 0) by default. `risk_table(..., from_mean=True)` gives the
  "diff from mean" version the course tests also report.
- EW weights: oldest row first, normalised. Deviations are from the **EW-weighted mean** (class
  `ewCovar`); `mean="simple"` gives the plain mean.
- Simulations take `rng=` (a seed or a numpy Generator) so results are reproducible.

## Tests

```bash
python3 -m pip install pytest statsmodels     # once
python3 -m pytest tests
```

There are 70 checks:

- `test_course_reference.py`: the course test cases, against the reference CSVs in
  `tests/course_data/`. Covers missing data (1.x), EW (2.x), near_psd / Higham (3.x), chol_psd (4.1),
  PCA simulation (5.5), returns (6.1), fits (7.1, 7.2), VaR / ES (8.1–8.6) and copula VaR / ES (9.1).
- `test_midterm.py`: the t ES against numerical integration, EW, PSD repair of covariances,
  simulation, and the portfolio methods.
- `test_a1_functions.py`: the Assignment 1 tools.
- The 35 Assignment 2 tests, unchanged.

## Folder layout

```
QRM Library/
├── README.md            this file
├── USAGE.md             how to install, import and call everything (start here)
├── pyproject.toml       package metadata (pip install -e .)
├── qrmlib/              the library
├── tests/               pytest checks (course_data/ = course test inputs and reference outputs)
└── examples/
    ├── make_practice_data.py   writes practice CSVs (made-up data)
    └── cookbook.py             every USAGE.md recipe, run end to end
```
