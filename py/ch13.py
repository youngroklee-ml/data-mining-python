# Non-hierarchical clustering

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

#######################################
# Example 13.1
#######################################

# Load data
dat1 = pd.read_csv("data/ch12_dat1.csv")
X = dat1.drop('ID', axis=1)

# K-means
km = KMeans(n_clusters=3)
km.fit(X)

# Cluster centers:
km.cluster_centers_

# Cluster labels:
dat1.assign(
  cluster = km.labels_
)

#######################################
# Example 13.8
#######################################

# Load data
dat1 = pd.read_csv("data/ch12_dat1.csv")
X = dat1.drop('ID', axis=1)

# DBSCAN
db = DBSCAN(eps=2.5, min_samples=3)
db.fit(X)

# Core samples:
db.core_sample_indices_

# Cluster labels:
dat1.assign(
  cluster = db.labels_
)

