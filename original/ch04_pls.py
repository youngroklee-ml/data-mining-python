# ch04_pls.py
# ch4.9 Partial least squares

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression

# ex 4.14 PLS
# Load data
dat3 = pd.read_csv("data/ch4_dat3.csv")
dat3

X = dat3.drop('y', axis=1)
y = dat3['y']

# X: predictor matrix
# y: response vector
# dat3 pandas DataFrame format
X = dat3.drop(columns=['y']).values
y = dat3['y'].values

# estimate PLS model (ncomp = 2) only centering
# PLSRegression(scale=True)   # scaling is ON by default
pls = PLSRegression(n_components=2, scale=False)
pls.fit(X, y)

#pls.x_scores_ → T matrix (scores)
#pls.x_weights_ → W matrix (weights)
#pls.x_loadings_ → P matrix (loadings)

T_scores = pls.x_scores_
print("T scores:\n", T_scores)

P_loadings = pls.x_loadings_
print("P loadings:\n", P_loadings)

W_weights = pls.x_weights_
print("W weights:\n", W_weights)

#Python and R may choose different signs internally
#Opposite signs in coefficients do NOT affect predictions.
coef_T = np.linalg.pinv(pls.x_scores_) @ y
print("Coefficients:\n", coef_T)

# ex 4.16 : the predicted y for new data
# new data
X_new = np.array([
  [-1.5, -2, 4],
  [1, 1, 3]
])

y_pred = pls.predict(X_new)
print(y_pred)
