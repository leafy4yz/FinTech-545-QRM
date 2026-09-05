#Assignment 1
#Problem 3

#Load relevant packages and CSV file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sma

df = pd.read_csv("/Users/ivyzhu/Downloads/FinTech-545-QRM/Assignment1/problem3.csv")

#Extract data
x1 = df["x1"].to_numpy()
x2 = df["x2"].to_numpy()
x3 = df["x3"].to_numpy()
x4 = df["x4"].to_numpy()

#Plot every pair
sns.pairplot(df)
plt.show()

#Compute both correlation matrices
pearson = df.corr(method="pearson")
spearman = df.corr(method="spearman")

print("Pearson:\n", pearson.round(4))
print("\nSpearman:\n", spearman.round(4))

#Absolute gap between the two matrices
gap = (pearson - spearman).abs()

#Only looking at the upper triangle (excluding diagonal) so each pair is counted once
mask = np.triu(np.ones(gap.shape, dtype=bool), k=1)
gap_upper = gap.where(mask)

#Finding the pair with the largest disagreement
max_pair = gap_upper.stack().idxmax()
max_gap = gap_upper.stack().max()

print(f"\nLargest gap: {max_pair}")
print(f"  Pearson  = {pearson.loc[max_pair]:.4f}")
print(f"  Spearman = {spearman.loc[max_pair]:.4f}")
print(f"  |gap|    = {max_gap:.4f}")

#Export coefficient matrices to LaTeX
def to_latex_grid(df, float_format="%.4f", caption=None, label=None):
    cols = df.columns.tolist()
    col_format = "|l|" + "c|" * len(cols)

    lines = [r"\begin{table}[h]", r"\centering"] if caption or label else []
    lines.append(r"\begin{tabular}{" + col_format + "}")
    lines.append(r"\hline")
    lines.append(" & ".join([""] + cols) + r" \\")
    lines.append(r"\hline")
    for idx, row in df.iterrows():
        vals = " & ".join([str(idx)] + [float_format % v for v in row])
        lines.append(vals + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(r"\caption{" + caption + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    if caption or label:
        lines.append(r"\end{table}")
    return "\n".join(lines)

with open("pearson_corr.tex", "w") as f:
    f.write(to_latex_grid(pearson, caption="Pearson correlation matrix", label="tab:pearson"))

with open("spearman_corr.tex", "w") as f:
    f.write(to_latex_grid(spearman, caption="Spearman correlation matrix", label="tab:spearman"))

print("Wrote pearson_corr.tex and spearman_corr.tex")