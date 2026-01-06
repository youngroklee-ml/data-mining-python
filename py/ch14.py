# Cluster validation

import pandas as pd
import numpy as np
from sklearn.metrics import (
  rand_score, adjusted_rand_score, 
  silhouette_samples, silhouette_score, 
  calinski_harabasz_score
)

#######################################
# Examples 14.1, 14.3
#######################################

# Data
sol1 = np.array([1, 1, 2, 2, 2, 3, 3])
sol2 = np.array([1, 1, 2, 3, 2, 1, 3])

# Ex 14.1 Rand index

rand_score(sol1, sol2)

# Ex 14.3 Adjusted Rand index

adjusted_rand_score(sol1, sol2)


#######################################
# Example 14.4
#######################################

# Data
sol1 = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
sol2 = np.array([1, 1, 2, 3, 1, 2, 3, 1, 2, 3])

# Rand index
rand_score(sol1, sol2)

# Adjusted Rand index
adjusted_rand_score(sol1, sol2)


#######################################
# Example 14.5
#######################################

# Load data
df = pd.read_csv("data/ch14_dat1.csv")

# Cluster solutions
sol1 = np.array([1, 2, 1, 3, 2, 1, 2, 3])
sol2 = np.array([1, 2, 1, 2, 2, 1, 2, 2])

# Calinski-Harabasz index

calinski_harabasz_score(df, sol1)

calinski_harabasz_score(df, sol2)

# Average silhouette width

silhouette_score(df, sol1)

silhouette_score(df, sol2)

# Silhouette for each sample

df.assign(
  silhouetee = silhouette_samples(df, sol1)
)

df.assign(
  silhouetee = silhouette_samples(df, sol2)
)
