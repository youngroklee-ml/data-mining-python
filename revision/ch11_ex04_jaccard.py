# ch11_ex04_jaccard.py
# ch11.3 similarity metrics on discrete variables

import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Load data
dat2 = pd.read_csv("data/ch11_dat2.csv")

# Jaccard distance (same definition as R's dist(method="binary"))
jaccard_dist = pdist(dat2, metric='jaccard')

# Convert to similarity
jaccard_sim = 1 - jaccard_dist

# Convert to square matrix
jaccard_sim_mat = squareform(jaccard_sim)

# similarity between the first and the second rows
jaccard_sim_mat[0, 1]
