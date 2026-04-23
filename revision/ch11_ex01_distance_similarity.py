# ch11_ex01_distance_similarity.py
# ch11.2 similarity/distance metrics

# import modules
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise

# load data
dat1 = pd.read_csv("data/ch11_dat1.csv")
X = dat1[['X1', 'X2', 'X3']]

# ex 11.1

# compute Euclidean distance
D1 = pairwise.euclidean_distances(X)
print(np.round(D1, 2))

# Distance between 2nd object and 4th object. 
D1[1, 3]

# Standardized Euclidean distance
# Using `StandardScaler` from `sklearn.preprocessing` module provides different results
# because it uses population variance instead of sample variance
x_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
std_D1 = pairwise.euclidean_distances(x_std)
print(np.round(std_D1, 2))

# Standardized distance between 2nd obs and 4th obs
std_D1[1, 3]

# ex 11.3
# Correlation coefficients
row_cor = np.corrcoef(X)
print(row_cor.round(4))

# Correlation coefficients between 1st object and 8th object.
row_cor[0, 7]

# After standardization
std_row_cor = np.corrcoef(x_std)
print(std_row_cor.round(4))

# Correlation coefficients between 1st obs and 8th obs
std_row_cor[0, 7]
