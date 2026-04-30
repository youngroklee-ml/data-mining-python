# ch14_ex05_internal.py
# ch14.1 cluster solution evaluation (14.1.2 internal index)

# load modules
import pandas as pd
import numpy as np
from sklearn.metrics import (
  silhouette_samples, silhouette_score, calinski_harabasz_score
)

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

# Silhouette for each sample with solution 1
silhouette_samples(df, sol1)

