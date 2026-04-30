# ch14_ex04_rand.py
# ch14.1 cluster solution evaluation (14.1.1 external index)

# load modules
import numpy as np
from sklearn.metrics import rand_score, adjusted_rand_score

# Data
sol1 = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
sol2 = np.array([1, 1, 2, 3, 1, 2, 3, 1, 2, 3])

# Rand index
rand_score(sol1, sol2)

# Adjusted Rand index
adjusted_rand_score(sol1, sol2)

