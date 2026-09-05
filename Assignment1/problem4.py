#Assignment 1
#Problem 4

#Load relevant packages and CSV file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sma
from statsmodels.tsa.stattools import acf, pacf as pacf_fn
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("/Users/ivyzhu/Downloads/FinTech-545-QRM/Assignment1/problem4.csv")

#Extract data
x1 = df["x1"].to_numpy()
x2 = df["x2"].to_numpy()

#pandas' .cov() uses ddof=1 by default (the unbiased/"sample" estimator, dividing by n-1)
Sigma = df[["x1", "x2"]].cov()

print(Sigma)

#c)
#Calculating the conditional mean of x2 given x1, plotted with a 95% band over the data
mu1, mu2 = x1.mean(), x2.mean()
s11, s22, s12 = Sigma.loc["x1", "x1"], Sigma.loc["x2", "x2"], Sigma.loc["x1", "x2"]

beta = s12 / s11          #regression coefficient: Cov(x1,x2)/Var(x1), the OLS slope of x2 on x1
alpha = mu2 - beta * mu1  #intercept
cond_var = s22 - s12**2 / s11
cond_std = np.sqrt(cond_var)

print(f"E[x2|x1] = {alpha:.4f} + {beta:.4f} * x1")
print(f"Var(x2|x1) = {cond_var:.4f}, band half-width = {1.96 * cond_std:.4f}")

x1_grid = np.linspace(x1.min(), x1.max(), 200)
cond_mean = alpha + beta * x1_grid
upper = cond_mean + 1.96 * cond_std
lower = cond_mean - 1.96 * cond_std

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1, x2, alpha=0.5, s=15, label="data")
ax.plot(x1_grid, cond_mean, color="red", label="E[x2 | x1]")
ax.fill_between(x1_grid, lower, upper, color="red", alpha=0.15, label="95% band")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
plt.show()

#d) 
#Calculating the fraction of observations inside the 95% band
fitted = alpha + beta * x1
lower_pts = fitted - 1.96 * cond_std
upper_pts = fitted + 1.96 * cond_std

inside = (x2 >= lower_pts) & (x2 <= upper_pts)
coverage = inside.mean()

print(f"Fraction of observations inside the 95% band: {coverage:.4f}")

#e)
sd1 = np.sqrt(s11)
z1 = np.abs(x1 - mu1) / sd1

bucket1 = z1 < 1
bucket2 = (z1 >= 1) & (z1 < 2)
bucket3 = z1 >= 2

for name, mask in [("|z1| < 1", bucket1), ("1 <= |z1| < 2", bucket2), ("|z1| >= 2", bucket3)]:
    n_b = mask.sum()
    cov_b = inside[mask].mean() if n_b > 0 else np.nan
    print(f"{name:<15} n={n_b:<4} coverage={cov_b:.4f}")