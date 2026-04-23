# ch12_ex05_choose_k.py
# ch12.5 How to choose k

# import modules
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

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

