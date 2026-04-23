# ch12_ex01_average_linkage.py
# ch12.1 cluster distance metrics and linkage methods
# ch12.2 algorithm of linkage methods

# ex12.1
# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# read csv file
dat1 = pd.read_csv("data/ch12_dat1.csv")
X = dat1[['X1', 'X2']]

# to keep ID i to n
X.index = range(1, len(X) + 1)
n = X.shape[0]

# Euclidean distance 
D1 = pdist(X, metric='euclidean')
print(squareform(np.round(D1, 2)))

# Average linkage method
# linkage method is 'average'
Z = linkage(D1, method='average')

# plot dendrogram
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

