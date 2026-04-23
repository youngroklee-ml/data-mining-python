# ch12_ex02_ward.py
# ch12.3 Ward's method

# ex12.2
# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# load data
dat2 = pd.read_csv("data/ch12_dat2.csv")
X = dat2.set_index('ID')
n = X.shape[0]

# Euclidean distance 
D2 = pdist(X, metric='euclidean')

# print squared Euclidean distance
print(squareform(np.round(D2**2, 2)))

# linkage method is 'ward' with Euclidean distance
Z = linkage(D2, method='ward')

# plot dendrogram
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


