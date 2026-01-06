# Dimension reduction

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression


#######################################
# Example 4.10
#######################################

# Load data
dat2 = pd.read_csv("data/ch4_dat2.csv", encoding="euc-kr")
dat2

# Principal component analysis with `statsmodels`

# Train a PCA model.
pca_model = sm.PCA(dat2.drop("ID", axis=1))

# Print a loading matrix.
pca_model.loadings

# Draw a scree plot.
fig = pca_model.plot_scree(log_scale=False)

# The x-axis represents number of principal component - 1. 
# 0 in the x-axis represents the 1st principal component.

# The y-axis values are different from the book example, 
# because `statsmodels` returns eigenvalues multiplied by the number of observations.

print("Eigenvalue from statsmodels:")
print(pca_model.eigenvals)

print("Eigenvalue in the book example:")
print(pca_model.eigenvals / len(dat2))


# Contribution
pca_model.rsquare


#######################################
# Example 4.12
#######################################

# Load data
dat3 = pd.read_csv("data/ch4_dat3.csv")
dat3

X = dat3.drop('y', axis=1)
y = dat3['y']

# Principal component analysis with `scikit-learn`
# Use `PCA()` from `skearn.decomposition` to define a principal component analysis model. 
# Pass `n_components=2` argument to find up to two components.

# Extract principal component scores for training data by calling `fit_transform()` method. 
# The results will be used as input to a linear regression model.
pca_model = PCA(n_components=2)
pc = pca_model.fit_transform(X)
pc

# Use `LinearRegression()` from `sklearn.linear_model` to define a linear regression model. 
# Use principal component scores as input matrix.
pcr_model = LinearRegression()
pcr_model.fit(pc, y)

# See estimated intercept.
pcr_model.intercept_

# Also, see estimated coefficient on each of two principal component scores.
pcr_model.coef_


#######################################
# Examples 4.14 - 4.15
#######################################

# Load data
dat3 = pd.read_csv("data/ch4_dat3.csv")
dat3

X = dat3.drop('y', axis=1)
y = dat3['y']

# Partial least squares regression:
# Use `PLSRegression()` from `sklearn.cross_decomposition` module 
# to define a partial least squares (PLS) regression model. 

# In this example, pass `n_components=2` argument to use two latent variables. 
# In addition, set `scale=False` to override default argument `scale=True`. 
# Typically use the default argument that standardize inputs and outputs, 
# but we will not do the standardization in this example.
pls_model = PLSRegression(n_components=2, scale=False)
pls_model.fit(X, y)

# See latent variable matrix.
pls_model.x_scores_

# X-loading matrix.
pls_model.x_loadings_

# Weight matrix for X.
pls_model.x_weights_

# y-loading
pls_model.y_loadings_

# Intecept and coefficients on original input variables.
print("Intercept")
print(pls_model.intercept_)

print("Coefficients")
print(pls_model.coef_)

# Prediction:
# Make a prediction by calling `predict()` method.
dat3.assign(
    pred_y = pls_model.predict(X)
)

