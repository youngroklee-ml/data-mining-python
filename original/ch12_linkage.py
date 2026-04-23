# ch12_linkage.py
# ch12.2-12.3 linkage methods : average, ward
# ch12.5 optimal k

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

#######################################
# Example 12.1
######################################

# Load data
dat1 = pd.read_csv("data/ch12_dat1.csv")
X = dat1[['X1', 'X2']]

# to keep ID i to n
X.index = range(1, len(X) + 1)
n = X.shape[0]

# Euclidean distance 
D1 = pdist(X, metric="euclidean")
print(squareform(np.round(D1, 2)))

# linkage method is 'average'
Z = linkage(D1, method="average")

plt.figure(figsize=(8, 4))
dendrogram(
    Z,
    labels=[str(i) for i in range(1, n + 1)], 
    leaf_rotation=90,
    leaf_font_size=9
)

plt.title("Average linkage with Euclidean distance")
plt.xlabel("Observation")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

#######################################
# Example 12.2
#######################################

# Load data
dat2 = pd.read_csv("data/ch12_dat2.csv")
X = dat2.set_index('ID')
n = X.shape[0]

# Euclidean distance 
D2 = pdist(X, metric="euclidean")

# Print squared Euclidean distance
print(squareform(np.round(D2**2, 2)))

# linkage method is 'ward' with Euclidean distance
Z = linkage(D2, method="ward")

plt.figure(figsize=(8, 4))
dendrogram(
    Z,
    labels=[str(i) for i in range(1, n + 1)], 
    leaf_rotation=90,
    leaf_font_size=9
)
plt.title("Ward's linkage with Euclidean distance")
plt.xlabel("Observation")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

#######################################
# Example 12.5
#######################################

# Load data
dat2 = pd.read_csv("data/ch12_dat2.csv")
X = dat2.drop('ID', axis=1)

# Average silhouette width 
# Use `silhouette_score()` from `sklearn.metrics` module.
n_clusters = np.arange(start=2, stop=7)
avg_silhouette = list()
for k in n_clusters:
  hc_c = AgglomerativeClustering(n_clusters=k, linkage='ward')
  hc_c.fit(X)
  avg_silhouette.append(silhouette_score(X, hc_c.labels_))

# Visualize average silhouette for the optimal k
plt.figure()
plt.scatter(
  x=n_clusters,
  y=avg_silhouette
)
plt.plot(
    n_clusters,
    avg_silhouette)
plt.xticks(n_clusters)
plt.show()
