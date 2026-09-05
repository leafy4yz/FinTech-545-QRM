#Assignment 1
#Problem 1

#Load relevant packages and CSV file
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

df = pd.read_csv("/Users/ivyzhu/Downloads/FinTech-545-QRM/Assignment1/problem1.csv")

x = df["x"].to_numpy()
 
 #a) Manually calculating the first four moments
def first4_moments(sample):
    n = sample.shape[0]
 
    #mean
    mu_hat = np.sum(sample) / n
 
    #remove the mean from the sample
    sample_corrected = sample - mu_hat
    cm2 = sample_corrected @ sample_corrected / n
 
    #variance
    sigma2_hat = sample_corrected @ sample_corrected / (n - 1)
 
    #skew
    skew_hat = np.sum(sample_corrected ** 3) / n / np.sqrt(cm2 * cm2 * cm2)
 
    #kurtosis (excess)
    kurt_hat = np.sum(sample_corrected ** 4) / n / cm2 ** 2
    excess_kurt_hat = kurt_hat - 3
 
    return mu_hat, sigma2_hat, skew_hat, kurt_hat
 
 
m, s2, sk, k = first4_moments(x)
 
print(f"Mean {m})")
print(f"Variance {s2}")
print(f"Skew {sk}")
print(f"Kurtosis {k}")

#Fit a Normal distribution by matching the mean and variance
mu_fit = m
sigma_fit = np.sqrt(s2)
fitted_normal = stats.norm(loc=mu_fit, scale=sigma_fit)
 
print(f"\nFitted Normal: mu = {mu_fit}, sigma = {sigma_fit}")

#b) & c)
n = x.shape[0]
q01 = fitted_normal.ppf(0.01)  #the value such that 1% of the fitted Normal's mass lies below it

n_below = np.sum(x < q01)      #counting number of observations below 1% quantile
n_expected = 0.01 * n          #expected count if the data really follows Normal

print(f"1% quantile of fitted Normal: {q01}")
print(f"Observations below it: {n_below}")
print(f"Expected under the fitted Normal: {n_expected}")