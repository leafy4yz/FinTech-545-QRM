#Assignment 1
#Problem 5

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

df = pd.read_csv("/Users/ivyzhu/Downloads/FinTech-545-QRM/Assignment1/problem5.csv")

#Extract data
x = df["x"].to_numpy()
n = x.shape[0]

#Plot the series, its ACF, and its PACF
#adding ±1.96/sqrt(n) significance band needed for part b)
def plot_ts(y, img_name="series.png", lags=10, title=None):
    n = len(y)
    l = np.arange(1, lags + 1)
    acf_vals = acf(y, nlags=lags, fft=False)[1:]
    pacf_vals = pacf_fn(y, nlags=lags)[1:]
    band = 1.96 / np.sqrt(n)  # large-sample 95% significance band

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(np.arange(1, n + 1), y)
    if title is not None:
        ax0.set_title(title)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.bar(l, acf_vals)
    ax1.axhline(band, color="red", linestyle="--", linewidth=1)
    ax1.axhline(-band, color="red", linestyle="--", linewidth=1)
    ax1.set_title("ACF")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.bar(l, pacf_vals)
    ax2.axhline(band, color="red", linestyle="--", linewidth=1)
    ax2.axhline(-band, color="red", linestyle="--", linewidth=1)
    ax2.set_title("PACF")

    fig.savefig(img_name)
    return fig

plot_ts(x, img_name="problem5_acf_pacf.png", lags=10, title="Problem 5 Series")
plt.show()

#Fitting AR() and MA() models
results = {}
for p in range(1, 4):
    results[f"AR({p})"] = SARIMAX(x, order=(p, 0, 0), trend="c").fit(disp=False)
for q in range(1, 4):
    results[f"MA({q})"] = SARIMAX(x, order=(0, 0, q), trend="c").fit(disp=False)

#Finding AICc
print("Model     AICc")
for name, fit in results.items():
    print(f"{name:<9} {fit.aicc:.4f}")

best_name = min(results, key=lambda k: results[k].aicc)
print(f"\nAICc-selected model: {best_name}")

#Getting the parameters to compare AR(2) vs AR(3)
print("\nAR(2) params:\n", results["AR(2)"].params)
print("\nAR(3) params:\n", results["AR(3)"].params)