# Assignment 1 — Reproduction Instructions

## Setup

Requires Python 3 with the following packages:

```
pip install numpy pandas scipy statsmodels seaborn matplotlib
```

(`statsmodels` is needed for Problem 2, as a sanity check against the
by-hand OLS formulas, and again for Problem 5, to fit the AR/MA models and
read off AICc. `seaborn` and `matplotlib` are needed for Problem 3, to plot
the pairwise scatterplots; `matplotlib` alone is used again for the plots
in Problems 4 and 5.)

## Problem 1: First four moments, Normal fit, quantile check

**File:** `problem1.py`

The script currently reads the input CSV from a hardcoded absolute path:

```python
df = pd.read_csv("/Users/ivyzhu/Downloads/FinTech-545-QRM/Assignment1/problem1.csv")
```

To run it, either place `problem1.csv` at that exact path, or edit that
line to point at wherever the file lives on your machine.

Run:

```
python3 problem1.py
```

This reproduces every number reported in the write-up for Problem 1:

- Mean, variance, skewness, and kurtosis of the sample, computed by hand
  via `first4_moments()` (matching the definitions from lecture).
- A Normal distribution fit to the sample by matching its mean and
  variance (method of moments), with `mu` and `sigma` reported.
- The count of observations falling below the fitted Normal's 1%
  quantile, compared against the theoretically expected count
  (`0.01 * n`).

**Conventions used:**

- Variance uses `ddof=1` (the unbiased/"sample" estimator, dividing by
  `n - 1`), computed manually inside `first4_moments`.
- `first4_moments()` reports **raw (Pearson) kurtosis**, where a Normal
  distribution reads as 3.0 — not excess kurtosis (where a Normal reads
  as 0.0). The `excess_kurt_hat = kurt_hat - 3` line inside the function
  is computed but intentionally not returned; excess kurtosis is instead
  derived in the write-up as `raw kurtosis - 3` (e.g. 5.3398 - 3 =
  2.3398), rather than in code.
- The 1% quantile check uses strict `<` (not `<=`) when counting
  observations below the threshold.

## Problem 2: OLS vs. MLE (Normal) vs. MLE (Student's t)

**File:** `problem2.py`

Reads `problem2.csv` (columns `x` and `y`). If the file isn't in the same
directory as the script, edit the `pd.read_csv(...)` line near the top to
point at it.

Run:

```
python3 problem2.py
```

This reproduces every number reported in the write-up for Problem 2:

- **OLS**, fit via the closed-form normal equations, with `alpha`, `beta`,
  and their standard errors reported using the formula
  `SE(beta_hat) = sqrt(s^2 / sum((x_i - xbar)^2))`. Cross-checked against
  `statsmodels.api.OLS` immediately after, so the by-hand formula's
  correctness is verified against a trusted library implementation.
- **MLE under a Normal error**, fit by numerically maximizing the Normal
  log-likelihood (via `scipy.optimize.minimize`, minimizing the negative
  log-likelihood since `minimize()` only minimizes) over `alpha`, `beta`,
  and `sigma`.
- **MLE under a Student's t error**, fit the same way over `alpha`,
  `beta`, a scale parameter, and `nu` (degrees of freedom) using a
  location-scale t distribution (`scipy.stats.t.logpdf`).
- **AICc** for the Normal and t models, used to choose between them.

**Conventions used:**

- OLS's residual variance `s^2` divides by `(n - p)` where `p = 2` (alpha,
  beta) — the standard unbiased estimator, matching `statsmodels`.
- Both MLE optimizations are started from the OLS estimates (a natural,
  well-motivated starting point) and use `scipy.optimize.minimize` with
  bounds enforcing positive scale parameters (and `sigma > 0` for
  the Normal model).
- AICc parameter counts: `k = 3` for the Normal model (alpha, beta,
  sigma), `k = 4` for the t model (alpha, beta, scale, nu).
- OLS is not given its own separate AICc entry. Its `alpha`/`beta` point
  estimates are mathematically identical to the Normal MLE's (least
  squares *is* the Normal MLE, up to how sigma is normalized), so it
  isn't a distinct model to score here — its role in this assignment is
  to provide the classical SE formula used in parts (b)/(c), not an
  AICc comparison point.

## Problem 3: Pearson vs. Spearman correlation

**File:** `problem3.py`

Reads `problem3.csv` (columns `x1`–`x4`). If the file isn't in the same
directory as the script, edit the `pd.read_csv(...)` line to point at it.

Run:

```
python3 problem3.py
```

This reproduces every number and figure reported in the write-up for
Problem 3:

- A full pairwise scatterplot grid of `x1`–`x4` via `seaborn.pairplot`
  (the **Predict** step), used to visually judge which pairs look linear,
  monotonic-but-nonlinear, or unrelated before any correlation is
  computed.
- Written predictions (part a), committed to before computing anything,
  of which pair would show the largest Pearson/Spearman disagreement and
  which pairs would show close agreement.
- Both correlation matrices (part b), via `df.corr(method="pearson")`
  and `df.corr(method="spearman")`, with the largest-gap pair identified
  by masking `(pearson - spearman).abs()` to its upper triangle (so the
  diagonal and duplicate symmetric entries are excluded) and calling
  `.stack().idxmax()`.
- LaTeX exports of both matrices (`pearson_corr.tex`, `spearman_corr.tex`)
  via a small custom `to_latex_grid()` helper, since pandas'
  `DataFrame.to_latex()` defaults to booktabs-style rules
  (`\toprule`/`\midrule`/`\bottomrule`) rather than a fully gridded table;
  the helper instead emits `\hline` after every row with `|`-separated
  columns.
- Written explanation (part c) of the mechanism behind the largest-gap
  pair, and which of the two coefficients is the honest description of
  that pair's association versus which is answering a narrower,
  linear-fit-specific question.

**Conventions used:**

- Predictions in part (a) are committed to *before* the correlation
  matrices are computed, per the assignment's "Predict" structure.
- The largest-gap search only considers the upper triangle of the
  absolute-difference matrix, so each pair is counted once and the
  (always-zero) diagonal never wins by default.
- Spearman is treated as the "honest" coefficient for a pair whenever the
  true relationship is monotonic-but-nonlinear (rank order fully
  preserved); Pearson is treated as answering the linear-fit /
  variance-explained question instead, which is a legitimate but
  different question from "are these related at all."

## Problem 4: Conditional distributions

**File:** `problem4.py`

Reads `problem4.csv` (columns `x1`, `x2`) from a hardcoded absolute path,
following the same convention as Problem 1 — edit the `pd.read_csv(...)`
line to point at wherever the file lives on your machine.

Run:

```
python3 problem4.py
```

This reproduces every number and figure reported in the write-up for
Problem 4:

- The sample covariance matrix (the **Predict** step), via
  `df[["x1","x2"]].cov()`.
- Written derivation (part a) of `Var(x2 | x1)` using the partitioned
  Normal result from Week 2. Because the lecture's formula is written for
  `X1 | X2 = a` (i.e. block 1 unknown, block 2 known) while this problem
  asks for `x2 | x1` (i.e. `x2` unknown, `x1` known), the block labels are
  swapped when substituting: `x1` plays the lecture's "known/`X2`" role
  and `x2` plays the "unknown/`X1`" role.
- Written argument (part b) that this factor does **not** depend on the
  observed value of `x1`: the conditional-variance formula
  `Sigma_11 - Sigma_12 Sigma_22^-1 Sigma_21` contains no `a` term, unlike
  the conditional-mean formula `mu_1 + Sigma_12 Sigma_22^-1 (a - mu_2)`,
  which does.
- The conditional mean formula (part c),
  `E[x2|x1] = alpha + beta*x1` with `beta = Cov(x1,x2)/Var(x1)` identified
  as the OLS regression slope of `x2` on `x1`, plotted as a red line with
  a constant-width 95% band (`+/- 1.96 * sqrt(Var(x2|x1))`) over a scatter
  of the raw data (`problem4_conditional.png`).
- The overall coverage fraction (part d): the fraction of observations
  that fall inside the band, evaluated at each point's own `x1` value.
- Bucketed coverage (part e), splitting on
  `|z1| = |x1 - mean(x1)| / std(x1)` into the three regions named in the
  prompt (inside one standard deviation, between one and two, beyond
  two), with the count and coverage reported for each.
- Written explanation (part f) of what the uneven coverage across
  buckets implies about `Var(x2|x1)`, which assumption behind the
  constant-width band fails as a result, and which parts of the part (c)
  answer survive that failure and which do not.

**Conventions used:**

- Covariance uses `ddof=1` (pandas' default), consistent with Problem 1's
  variance convention.
- The 95% band in part (c) is a *prediction* band for individual
  observations (constant half-width `1.96 * sqrt(Var(x2|x1))`), not a
  confidence band for the mean function — this is what makes the
  part (d)/(e) coverage checks meaningful.
- Part (e)'s buckets are defined on `|z1|` (absolute standardized
  distance from `x1`'s own mean), matching the "inside one standard
  deviation / between one and two / beyond two" wording in the prompt.

## Problem 5: Identifying an AR or MA order

**File:** `problem5.py`

Reads `problem5.csv` (column `x`) from a hardcoded absolute path, same
convention as Problems 1 and 4.

Run:

```
python3 problem5.py
```

This reproduces every number and figure reported in the write-up for
Problem 5:

- A combined series/ACF/PACF plot (the **Predict** step) via a `plot_ts()`
  helper adapted from Week 2's class code, extended to draw the
  `+/- 1.96/sqrt(n)` significance band on the ACF and PACF bars
  (`problem5_acf_pacf.png`).
- Written commitment (part a), made before fitting anything, to which
  function cuts off and which decays, what order that implies, and
  whether it's an AR or MA process.
- Written statement (part b) of the significance rule used: a lag is
  judged significant if it exceeds `+/- 1.96/sqrt(n)`, the standard
  large-sample 95% band for sample autocorrelations under the
  white-noise null — the same dashed-line convention used by default in
  `statsmodels.graphics.tsaplots.plot_acf`/`plot_pacf`.
- AICc for all six candidate models (part c), fit via
  `SARIMAX(x, order=..., trend="c")` and read from each result's built-in
  `.aicc` attribute.
- Model selection (part d): whichever model AICc selects (lowest value),
  compared against the part (a) prediction.
- Comparison of the AR(2) and AR(3) fits (part e), and a written
  explanation of the AICc/parameter-count trade-off: AICc charges a
  fixed penalty (`2k`, inflated by the small-sample correction
  `2k(k+1)/(n-k-1)`) per parameter, whereas `R^2` (an unpenalized
  function of the residual variance) can only weakly favor the larger
  model, since AR(3) nests AR(2) and its optimizer can always fall back
  to `phi_3 = 0`.

**Conventions used:**

- The significance band is the standard `+/- 1.96/sqrt(n)` large-sample
  rule, matching `statsmodels`' default ACF/PACF plotting and rendered as
  dashed red lines by `plot_ts()`.
- AICc values are taken directly from `SARIMAXResults.aicc` rather than
  computed by hand.
- All six fitted models include a constant term (`trend="c"`), consistent
  with Week 2's AR1/MA1 examples.
