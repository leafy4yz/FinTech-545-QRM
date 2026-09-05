#Assignment 1
#Problem 2

#Load relevant packages and CSV file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sma

df = pd.read_csv("/Users/ivyzhu/Downloads/FinTech-545-QRM/Assignment1/problem2.csv")

#Extract data
x = df["x"].to_numpy()
y = df["y"].to_numpy()
n = x.shape[0]

#Plot y against x
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("y vs x")
plt.show()

#Estimating the models

#1. OLS
X = np.column_stack([np.ones(n), x])
 
ols_check = sma.OLS(y, X).fit()
alpha_ols, beta_ols = ols_check.params
se_alpha_ols, se_beta_ols = ols_check.bse
 
#residual std, for use as an MLE starting value below -- mse_resid is
#statsmodels' (n - p)-normalized residual variance, same convention as
#SE(beta_hat) = sqrt(s^2 / sum((x_i - xbar)^2))
s_ols = np.sqrt(ols_check.mse_resid)
 
print("--- 1) OLS ---")
print(f"alpha = {alpha_ols}, SE(alpha) = {se_alpha_ols}")
print(f"beta  = {beta_ols}, SE(beta)  = {se_beta_ols}")
print(f"Residual std dev = {s_ols}")

#2. Maximum likelihood under a Normal error
def normal_negll(params, x=x, y=y):
    a, b, s = params
    e = y - a - b * x
    n = e.shape[0]
    ll = -n / 2 * np.log(2 * np.pi * s ** 2) - (e @ e) / (2 * s ** 2)
    return -ll  # minimize() minimizes, so hand it the negative log-likelihood
 
 
#start from the OLS estimates -- a good starting point since OLS and the
#Normal MLE share the same solution for parameter estimates
x0_norm = [alpha_ols, beta_ols, s_ols]
res_norm = minimize(
    normal_negll, x0=x0_norm, bounds=[(None, None), (None, None), (1e-8, None)]
)
alpha_norm, beta_norm, sigma_norm = res_norm.x
ll_norm = -res_norm.fun
 
print("\n--- 2) MLE under Normal error ---")
print(f"alpha = {alpha_norm}")
print(f"beta  = {beta_norm}")
print(f"sigma = {sigma_norm}")
print(f"log-likelihood = {ll_norm}")
 
 
#3. Maximum likelihood under a Student's t error
def t_negll(params, x=x, y=y):
    a, b, s, nu = params
    e = y - a - b * x
    #location-scale t distribution: pdf(e) = (1/s) * t.pdf(e/s, df=nu)
    ll = np.sum(stats.t.logpdf(e / s, df=nu) - np.log(s)) 
    #using scipy.stats.t to avoid hand-coding Gamma function
    #then modified to change from standard t distribution to t that fits parameters
    #then sum up log-densities to get total log-likelihood
    return -ll
 
 #Using OLS as starting guess
x0_t = [alpha_ols, beta_ols, s_ols, 4.0]  #nu=4 as a reasonable starting guess 
res_t = minimize(
    t_negll,
    x0=x0_t,
    bounds=[(None, None), (None, None), (1e-8, None), (1e-3, None)], #constraining s and nu to be positive
)
alpha_t, beta_t, s_t, nu_t = res_t.x
ll_t = -res_t.fun
 
print("\n--- 3) MLE under Student's t error ---")
print(f"alpha = {alpha_t}")
print(f"beta  = {beta_t}")
print(f"scale (s) = {s_t}")
print(f"nu (degrees of freedom) = {nu_t}")
print(f"log-likelihood = {ll_t}")
 
 
#Choose between models using AICc
def aicc(loglik, k, n):
    aic = 2 * k - 2 * loglik
    return aic + (2 * k ** 2 + 2 * k) / (n - k - 1)
 
 
k_norm = 3  #k = p + d = q + 1 + d = 1 + 1 + 1 = 3
k_t = 4     #k = p + d = q + 1 + 2 = 1 + 1 + 2 = 4
 
aicc_norm = aicc(ll_norm, k_norm, n)
aicc_t = aicc(ll_t, k_t, n)
 
print("\n--- AICc comparison ---")
print(f"Normal AICc: {aicc_norm}")
print(f"t AICc:      {aicc_t}")
preferred = "Normal" if aicc_norm < aicc_t else "Student's t" #Lower AICc is better
print(f"Preferred model: {preferred}")
 
#Export results as LaTeX tables (report a, b, and the fitted error params)
#Table 2: alpha, beta, and fitted error parameters for each across all three models
results_table = pd.DataFrame({
    "Model": ["OLS", "MLE (Normal)", "MLE (Student's t)"],
    "alpha": [alpha_ols, alpha_norm, alpha_t],
    "SE(alpha)": [se_alpha_ols, np.nan, np.nan],
    "beta": [beta_ols, beta_norm, beta_t],
    "SE(beta)": [se_beta_ols, np.nan, np.nan],
    "Root MSE": [s_ols, sigma_norm, s_t],
    "nu": [np.nan, np.nan, nu_t],
})
 
with open("results_table.tex", "w") as f:
    f.write(results_table.to_latex(index=False, float_format="%.6f", na_rep="--"))

#e) Computing quantiles
fitted_normal_err = stats.norm(loc=0, scale=sigma_norm)
fitted_t_err = stats.t(df=nu_t, loc=0, scale=s_t)
 
q95_norm = fitted_normal_err.ppf(0.95)
q995_norm = fitted_normal_err.ppf(0.995)
q95_t = fitted_t_err.ppf(0.95)
q995_t = fitted_t_err.ppf(0.995)
 
print("\n--- e) Quantile comparison ---")
print(f"95%   quantile -> Normal: {q95_norm}, t: {q95_t}")
print(f"99.5% quantile -> Normal: {q995_norm}, t: {q995_t}")
wider_95 = "Normal" if q95_norm > q95_t else "t"
wider_995 = "Normal" if q995_norm > q995_t else "t"
print(f"Wider at 95%:   {wider_95}")
print(f"Wider at 99.5%: {wider_995}")