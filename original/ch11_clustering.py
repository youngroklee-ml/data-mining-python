# ch11_clustering.py
# ch11.2 distance metrics
# ch11.3 similarity metrics on discrete variables

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform


#######################################
# Examples 11.1, 11.3
#######################################

# Load data
dat1 = pd.read_csv("data/ch11_dat1.csv")
X = dat1[['X1', 'X2', 'X3']]

# Ex 11.1

# Euclidean distance
D1 = pairwise.euclidean_distances(X)
np.round(D1, 2)

# Distance between 2nd object and 4th object. 
D1[1, 3]

# Standardized Euclidean distance
# may use `StandardScaler` for standazation

x_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
std_D1 = pairwise.euclidean_distances(x_std)
np.round(std_D1, 2)

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

#######################################
# Example 11.4
#######################################

# Load data
dat2 = pd.read_csv("data/ch11_dat2.csv")

# Jaccard distance (same definition as R's dist(method="binary"))
jaccard_dist = pdist(dat2, metric="jaccard")

# Convert to similarity
jaccard_sim = 1 - jaccard_dist

# Convert to square matrix
jaccard_sim_mat = squareform(jaccard_sim)

# similarity between the first and the second rows
jaccard_sim_mat[0, 1]
