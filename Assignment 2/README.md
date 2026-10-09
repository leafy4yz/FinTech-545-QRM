# FinTech 545 — Assignment 2

Ivy Zhu · Friday, October 9th, 2026

Code for every number, table and figure in the written responses. Each problem is one
script (`problem1.py` … `problem5.py`) that prints its results part by part. All the
reusable calculations live in a small library, `qrmlib/`, which the scripts import.

## Requirements

- Python 3.9 or newer
- numpy, pandas, scipy, matplotlib (pytest only for the optional tests)

```bash
cd "Assignment 2"
python3 -m pip install numpy pandas scipy matplotlib pytest
```

No install of `qrmlib` is needed: the scripts import it from this folder. (If you prefer,
`python3 -m pip install -e .` installs it from `pyproject.toml`.)

## Reproducing the results

Run each script from the `Assignment 2` folder. Each takes a few seconds, prints every
number used in the write-up, and labels its output by part letter.

```bash
python3 problem1.py
python3 problem2.py
python3 problem3.py
python3 problem4.py
python3 problem5.py
```

To keep the output for reference, redirect it, e.g. `python3 problem4.py > problem4_output.txt`.

| Script | Write-up section | Data | Figure saved |
|---|---|---|---|
| `problem1.py` | 1. Correlations from Mismatched Histories | `problem1.csv` | — |
| `problem2.py` | 2. A Volatility Estimate After the Regime Changed | `problem2.csv` | `problem2_returns.png` (Figure 1) |
| `problem3.py` | 3. When Diversification Raises VaR | `problem3.csv` | `problem3_pnl.png` (Figure 2) |
| `problem4.py` | 4. Gaussian or t Copula | `problem4.csv` | `problem4_ranks.png` (Figure 3) |
| `problem5.py` | 5. Model Based Simulation and Residual Correlation | `problem5.csv` | — |

The CSV files must sit in the same folder as the scripts. Figures are written there too.

### What each script prints

- **problem1.py**: joint observation counts and the number of days all five trade; the
  complete-case and pairwise correlation matrices with their eigenvalues; Cholesky
  attempts on each; the covariance matrix from pairwise correlations and full-history
  standard deviations; the tracking-portfolio variance; the Rebonato–Jäckel and Higham
  repairs (smallest eigenvalue, Frobenius distance, tracking variance); the Higham change
  for every entry; and the estimator-vs-repair Frobenius comparison.
- **problem2.py**: the returns plot; effective sample size, half life and weight on the
  last 40 days for each λ, with a variance break-point check on the 40-day choice; the
  first four moments with Jarque–Bera; the five VaR estimates and the fitted t
  parameters; the standard deviation inside each regime and the normal-mixture
  moments; and the EWMA VaR gap against its estimation noise.
- **problem3.py**: the P&L plot; moments of each bond; counts of scenarios losing more
  than 20%; historical and normal VaR/ES at 5% and 1% for all four positions; the
  subadditivity checks; and which choice each measure prefers.
- **problem4.py**: moments and the effect of dropping the most extreme day; rank plots
  and joint-tail counts; margin fits by AICc; R from Kendall's τ and both copula fits
  (log likelihood, ν̂, AICc, BIC); simulated VaR/ES under each copula and historically;
  implied joint-tail days; the change from Gaussian to t; the split of ℓ_t − ℓ_Gauss
  between middle and outer days; and the tail dependence coefficient.
- **problem5.py**: the part (a) prediction written out from the Week 2 variance formula;
  both regressions (α, β, residual standard deviations, residual correlation);
  simulated VaR of both portfolios with full and with independent residuals; and the
  delta normal VaR from the sample covariance.

## Conventions

These choices are also stated in comments at the top of each script.

- **VaR and ES** are positive numbers for a loss, at α = 5% unless a part says otherwise.
- **Historical / simulated VaR** is minus the ⌊αn⌋-th smallest P&L, and **ES** is minus
  the mean of the ⌊αn⌋ smallest, matching the class code. Problem 2 also prints the
  interpolated-quantile version for comparison.
- **Returns** are arithmetic, P_t / P_{t−1} − 1. Problem 2 removes the sample mean;
  problem 5 assumes zero expected returns, so its simulations drop the fitted alphas.
- **Variances and covariances** use n − 1. Normal and t fits are by maximum likelihood.
- **AICc** = −2ℓ + 2k + 2k(k+1)/(n−k−1); **BIC** = −2ℓ + k ln n.
- **Problem 4 copulas**: R = sin(πτ/2) from Kendall's τ is held fixed for both copulas,
  so the Gaussian has 3 parameters and the t has 4 (adding ν). Ranks are scaled as
  rank / (n + 1).
- **Problem 5 residual standard deviations** in the regression table are the regression
  standard error (n − 2); the simulations use the n − 1 residual covariance so that the
  full model reproduces the sample covariance.
- **Random numbers**: problems 4 and 5 simulate 100,000 days with a fixed seed (545), so
  every run prints the same numbers. Exact simulated digits can differ slightly across
  numpy versions or platforms; parts that use no simulation are unaffected.

## The `qrmlib` library

| Module | Contents | Used in |
|---|---|---|
| `missing.py` | joint-observation counts, complete rows | 1 |
| `correlation.py` | complete-case and pairwise covariance/correlation, eigenvalues, PSD check | 1, 4 |
| `cholesky.py` | Cholesky for PD and PSD matrices, with a non-raising attempt | 1 |
| `psd_repair.py` | Rebonato–Jäckel and Higham repairs, Frobenius distance | 1, 4 |
| `portfolio.py` | covariance from correlations, portfolio variance, scenario P&L | 1, 3, 4, 5 |
| `returns.py` | arithmetic and log returns from prices, demeaning | 2, 5 |
| `ewma.py` | exponential weights, EW variance, effective sample size, half life | 2 |
| `moments.py` | four moments, normality standard errors, Jarque–Bera, extreme-day influence | 2, 3, 4 |
| `fitting.py` | normal and Student t maximum likelihood fits | 2, 4 |
| `var.py` | normal, t and historical VaR; historical and normal ES | 2, 3, 4, 5 |
| `tails.py` | tail-event counts, subadditivity check | 3 |
| `regime.py` | variance break point, regime standard deviations, normal-mixture moments | 2 |
| `copula.py` | margins by AICc, Kendall R, Gaussian and t copulas, joint-tail counts, tail dependence | 4 |
| `factor.py` | single-index OLS model, residual covariance, factor simulation | 5 |

Every function has a docstring; `import qrmlib; help(qrmlib.copula)` lists a module's
contents.

## Tests

```bash
python3 -m pytest tests
```

The tests check the library against known answers and simulated data, for example that a
Higham repair is PSD, that ES ≥ VaR, that the t copula fit recovers a known ν, and that the
full-residual factor model reproduces the sample covariance exactly.

## Folder layout

```
Assignment 2/
├── README.md
├── problem1.py … problem5.py     one script per problem
├── problem1.csv … problem5.csv   data
├── problem2_returns.png, problem3_pnl.png, problem4_ranks.png
├── pyproject.toml                package metadata for qrmlib
├── qrmlib/                       the library
└── tests/                        pytest checks
```
