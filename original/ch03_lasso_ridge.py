# ch03_lasso_ridge.py
# Regularized regression : LASSO, Ridge, Elastic net

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet

#######################################
# Examples 3.1 - 3.2
#######################################

# Load data
dat1 = pd.read_csv("data/ch3_dat1.csv")
dat1

# Standardize inputs
# Let us convert data frame into input data matrix `x` and output array `y`. 
x = dat1[['x1', 'x2']].to_numpy()
y = dat1['y'].to_numpy()
N = len(y)

std_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)

# ex 3.1: Lasso : l1_ratio=1 for a function `ElasticNet()`
# Lasso : l1_ratio=1 Let us fit models by varying penalization size.

lambdas = np.arange(4) / (2 * N)

lasso_model = [ElasticNet(alpha=l, l1_ratio=1) for l in lambdas]
lasso_fit = [model.fit(std_x, y) for model in lasso_model]
[fit.coef_ for fit in lasso_fit]

# ex 3.2: Ridge : l1_ratio=0 for a function `ElasticNet()`
# Lasso : l1_ratio=1 Let us fit models by varying penalization size.
lambdas = np.arange(4) / N

ridge_model = [ElasticNet(alpha=l, l1_ratio=0) for l in lambdas]
ridge_fit = [model.fit(std_x, y) for model in ridge_model]
[fit.coef_ for fit in ridge_fit]

# example : ElasticNet : scikit-learn
# `l1_ratio=1` for Lasso, l1_ratio=0 for Ridge, 0<l1_ratio=<1 for ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(std_x, y) 
# Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# R-squared
r2 = model.score(std_x, y)
print("R²:", r2)

