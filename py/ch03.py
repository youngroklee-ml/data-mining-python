# Regularized regression

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
# Additional, let `N` be the number of training data.
x = dat1[['x1', 'x2']].to_numpy()
y = dat1['y'].to_numpy()
N = len(y)

std_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)

# Ex 3.1: Lasso

# First, let us estimate a model by using `{statsmodels}`. After defining an OLS model, use `fit_regularized()` method with `L1_wt=1` to give penalty on the L1-norm of cofficients.
def lasso(X, y, alpha):
    model_fit = (sm
                 .OLS(y, X)
                 .fit_regularized(alpha=alpha, L1_wt=1))

    return model_fit

# Let us fit models by varying penalization size.
lambdas = np.arange(4) / (2 * N)
lasso_fit = [lasso(std_x, y, l) for l in lambdas]
[fit.params for fit in lasso_fit]

# `{scikit-learn}` provides `ElasticNet()` to fit regularized regression model. 
# Pass `l1_ratio=1` to fit a Lasso model.
lambdas = np.arange(4) / (2 * N)
lasso_model = [ElasticNet(alpha=l, l1_ratio=1) for l in lambdas]
lasso_fit = [model.fit(std_x, y) for model in lasso_model]
[fit.coef_ for fit in lasso_fit]


# Ex 3.2: Ridge

# Do similarly as Lasso, but use different argument `L1_wt=0` and `l1_ratio=0`.

# statsmodels
def ridge(X, y, alpha):
    model_fit = (sm
                 .OLS(y, X)
                 .fit_regularized(alpha=alpha, L1_wt=0))

    return model_fit

lambdas = np.arange(4) / N
ridge_fit = [ridge(std_x, y, l) for l in lambdas]
[fit.params for fit in ridge_fit]


# scikit-learn
lambdas = np.arange(4) / N
lasso_model = [ElasticNet(alpha=l, l1_ratio=0) for l in lambdas]
lasso_fit = [model.fit(std_x, y) for model in lasso_model]
[fit.coef_ for fit in lasso_fit]
