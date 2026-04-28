# ch13_ex08_dbscan.py
# ch13.5 DBSCAN

# ex13.8
import pandas as pd
from sklearn.cluster import DBSCAN

# Load data
dat1 = pd.read_csv("data/ch12_dat1.csv")
X = dat1.drop('ID', axis=1)

# DBSCAN
db = DBSCAN(eps=2.5, min_samples=3)
db.fit(X)

# Core samples:
print(db.core_sample_indices_)

# Cluster labels:
dat1_cluster = dat1.assign(cluster = db.labels_)
print(dat1_cluster)

