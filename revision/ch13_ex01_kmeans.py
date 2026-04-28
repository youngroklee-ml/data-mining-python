# ch13_ex01_kmeans.py
# ch13.1 k-means: Non-hierarchical clustering

# ex13.1
import pandas as pd
from sklearn.cluster import KMeans

# Load data
dat1 = pd.read_csv("data/ch12_dat1.csv")
X = dat1.drop('ID', axis=1)

# K-means
km = KMeans(n_clusters=3)
km.fit(X)

# Cluster centers:
print(km.cluster_centers_)

# Cluster labels:
dat1_cluster = dat1.assign(cluster = km.labels_)
print(dat1_cluster)

