# How to use `qrmlib`

A step-by-step guide to calling the library from your own scripts or a Jupyter notebook.
Every snippet here also appears in `examples/cookbook.py`, which runs them all on practice data.

---

## 1. One-time setup: install the library

Open **Terminal** and run:

```bash
cd ~/Downloads/FinTech-545-QRM/"QRM Library"
python3 -m pip install -e .
```

What this does:

- `pip install` makes `qrmlib` importable from **any** folder: a new script, a notebook, or the midterm folder.
- `-e` ("editable") means pip *points at* this folder instead of copying it. If you fix or add a function in `qrmlib/`, the change takes effect the next time you run your script, with no reinstall.
- If `python3 -m pip` complains about an "externally managed environment", add `--user`.
  If you use Anaconda, run the same command in the Anaconda prompt / the environment your notebook uses.

Check it worked (from any folder):

```bash
python3 -c "import qrmlib; print(qrmlib.__version__)"     # -> 0.7.0
```

**Jupyter tip:** if a notebook says `No module named 'qrmlib'`, the notebook kernel is a different Python
from the one you installed into. In a notebook cell run `%pip install -e ~/Downloads/FinTech-545-QRM/"QRM Library"`
and restart the kernel.

**Without installing:** put this at the very top of the script or notebook instead:

```python
import sys
sys.path.insert(0, "/Users/ivyzhu/Downloads/FinTech-545-QRM/QRM Library")
```

---

## 2. Three ways to import

```python
import qrmlib as q                      # 1. everything, under a short name
q.risk_table(x)

from qrmlib import risk_table, fit_t    # 2. just the functions you need
risk_table(x)

from qrmlib import var                  # 3. a whole module
var.es_t(fit, 0.01)
```

Every public function is available directly as `q.<name>` (style 1), so you don't need to remember
which file it lives in. Style 1 is the easiest during an exam.

## 3. Finding things and reading help

```python
help(q.copula_risk)        # what a function does, its arguments and what it returns
help(q.var)                # every function in one module
[n for n in dir(q) if "es_" in n]   # search names: es_historical, es_normal, es_t, var_es_table
```

In Jupyter, `q.copula_risk?` shows the help, and typing `q.` then **Tab** lists every function.

---

## 4. Recipes by topic

The data in the examples is the practice data in `examples/`. Swap in your own CSV.

### Loading data

```python
import pandas as pd
import qrmlib as q

x = pd.read_csv("problem1.csv").iloc[:, 0]          # first column as a Series
prices = pd.read_csv("prices.csv")                   # keeps the Date column
```

### Moments of one series

| Want | Call |
|---|---|
| mean, variance, skew, **excess** kurtosis | `q.four_moments(x)` |
| same, Assignment 1 style (**raw** kurtosis, normal = 3) | `q.first4_moments(x)` → tuple |
| normality test | `q.jarque_bera(x)` → (JB, p-value) |
| standard errors of skew / kurt under normality | `q.skew_kurt_std_errors(n)` |

### Normal vs t: fitting and choosing

```python
q.compare_normal_t(x)      # table: mu, sigma, nu, std, loglik, k, AIC, AICc, BIC
mu, sigma, ll = q.fit_normal(x)      # MLE (sigma uses n)
tf = q.fit_t(x)                      # tf.mu, tf.sigma (SCALE), tf.nu, tf.loglik, tf.std
```

Lower AICc / BIC wins. For a t, `tf.sigma` is the **scale**, and `tf.std` is the implied standard deviation.

### VaR and ES for one series

```python
q.risk_table(x, alphas=(0.05, 0.01))                 # rows: Normal, Student t, Historical
q.risk_table(x, alphas=(0.05, 0.01), value=1e6)     # in $ on a $1M position
q.risk_table(x, demean=True)                         # remove the mean first
q.risk_table(x, from_mean=True)                      # "VaR diff from mean" convention
```

Defaults follow the class code. The normal uses σ with n − 1, which matches course tests 8.1 and 8.4
exactly; pass `normal_sd="mle"` for n. VaR is absolute (loss measured from 0). `from_mean=True`
measures from the mean instead, i.e. VaR + μ.

Pieces individually (positive = loss):

| | VaR | ES |
|---|---|---|
| normal | `q.var_normal(sigma, alpha, mu)` | `q.es_normal(sigma, alpha, mu)` |
| t | `q.var_t(tf, alpha)` | `q.es_t(tf, alpha)` |
| historical / simulated | `q.var_historical(x, alpha)` | `q.es_historical(x, alpha)` |

Historical uses the class convention: VaR = −(⌊αn⌋-th smallest), ES = −mean of the ⌊αn⌋ smallest.
Pass `method="quantile"` for numpy's interpolated quantile instead.

### Exponentially weighted estimates

```python
q.effective_sample_size(0.94); q.half_life(0.94)
q.ew_covariance(df, 0.94)          # matrix (numpy)
q.ew_cov_df(df, 0.94)              # same, with column labels
q.ew_correlation(df, 0.97)         # EW correlation
q.ew_std_vector(df, 0.94) ** 2     # EW variance of each column
cov, corr, sd = q.ew_mixed_covariance(df, lam_corr=0.97, lam_var=0.94)   # D C D
```

Correlations move slowly, so they get the **higher** λ (0.97). Variances move fast, so they get the **lower** λ (0.94).
Deviations are taken from the EW-weighted mean, like the class `ewCovar` (matches course tests 2.1–2.3).
Pass `mean="simple"` for the plain mean.

### Non-PSD matrices

```python
q.eigenvalues(S)                       # ascending
q.is_psd(S); q.min_eigenvalue(S)
H = q.near_psd_cov(S, "higham")        # works on a covariance OR a correlation matrix
R = q.near_psd_cov(S, "rj")            # Rebonato-Jackel
q.eigen_compare(original=S, higham=H, rj=R)     # eigenvalues side by side
q.frobenius_distance(H, S)
```

`near_psd_cov` splits Σ = D C D, repairs C and puts the variances back. The diagonal never changes.
Higham is the nearest **correlation** matrix, so measured on the covariance it can sit slightly further
away than Rebonato–Jäckel. Say which space you measured in.
For a correlation matrix directly: `q.higham_nearest_correlation(C).matrix`, `q.rebonato_jackel(C)`.

### Returns from prices

```python
rets = q.return_calculate(prices, "DISCRETE", date_column="Date")   # arithmetic; "LOG" for log
r = rets.drop(columns="Date")
r = r - r.mean()                                                    # demean
rets.head(); rets.tail()                                            # first / last 5 rows
```

### Portfolio: shares and $ VaR / ES

```python
last = prices.drop(columns="Date").iloc[-1]          # current prices
shares = q.shares_for_weights(last, 100_000)         # equal weights; weights={...} otherwise
values = q.position_values(shares, last)

hist = q.historical_risk(r, values, alpha=0.01)                  # historical simulation
dn   = q.delta_normal_risk(values, r.cov(), alpha=0.01)          # closed-form normal
mc, sims = q.normal_mc_risk(values, r.cov(), alpha=0.01, rng=545)
gc, info = q.copula_risk(r, values, alpha=0.01, family="t", copula="gaussian", rng=545)
q.compare_methods({"Gaussian copula": gc, "Historical": hist, "Delta normal": dn})
```

Each table has one row per stock plus `Portfolio`, with columns `VaR` and `ES` in $.

`copula_risk` options:

- `family`: `"t"`, `"normal"`, `"aicc"` (best of the two per stock), or a dict like `{"AAA": "t", "BBB": "normal"}`
- `copula`: `"gaussian"` or `"t"`
- `corr`: `"spearman"` (default for Gaussian) or `"kendall"` (default for t)
- `n_sim`: number of draws (default 100,000)
- `rng`: seed (default none). Set it so reruns give the same numbers.
- `normal_sd`: `"sample"` (n − 1, default, class convention) or `"mle"`

`info["params"]` is the fitted-margin table, `info["R"]` the copula correlation, `info["sims"]` the simulated returns.

### Fitting margins and copulas by hand

```python
margins = q.fit_margins(r, "t")              # {column: MarginFit}
q.margin_table(margins)                      # mu, sigma, nu, loglik, AICc per stock
U = q.to_uniforms(r, margins)                # U = F(x)
R = q.spearman_corr(U)                       # or q.kendall_corr(U)
gfit = q.fit_gaussian_copula(U.to_numpy(), R); tfit = q.fit_t_copula(U.to_numpy(), R)
gfit.aicc, tfit.aicc, tfit.nu
u_sim = q.simulate_gaussian_copula(R, 100_000, rng=545)
sim_returns = q.from_uniforms(u_sim, margins, list(r.columns))
q.t_tail_dependence(rho=0.6, nu=5)
```

### Simulation and PCA

```python
q.variance_explained(S)                                      # eigenvalues, %, cumulative %
X = q.simulate_normal(S, 10_000, method="cholesky", rng=1)
X = q.simulate_normal(S, 10_000, method="pca", pct_explained=0.9, rng=1)
B = q.pca_root(S, 0.9); L = q.chol_psd(S)
T = q.simulate_mvt(R, nu=5, n_sim=10_000, rng=1)             # multivariate t
```

### Assignment 1 tools

```python
o = q.ols_se(x, y)                       # o.alpha, o.beta, o.se_alpha, o.se_beta, o.s
n = q.mle_regression_normal(x, y); t = q.mle_regression_t(x, y)     # .aicc, .error_dist
pearson, spearman = q.corr_matrices(df); q.largest_gap(pearson, spearman)
line = q.conditional_line(df, "x1", "x2")       # E[x2|x1], Var(x2|x1)
q.conditional_normal(mu, Sigma, known=[0], a=[1.5])
results = q.fit_ar_ma(x); q.aicc_table(results); q.best_model(results)   # needs statsmodels
```

---

## 5. Common errors

| Error | Fix |
|---|---|
| `No module named 'qrmlib'` | Do the install in section 1, or use the `sys.path` lines. In Jupyter, install from inside the notebook. |
| `CholeskyError: non-positive pivot` | The matrix is not PSD. Repair it with `q.near_psd_cov` first. |
| `KeyError` in a portfolio function | The `values` index (stock names) must match the return columns. Drop the `Date` column first. |
| Numbers change every run | Pass `rng=<seed>` to the simulation functions. |
| `could not convert string to float` | A date column is still in the data you passed in. |
