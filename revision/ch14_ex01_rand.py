# ch14_ex01_rand.py
# ch14.1 cluster solution evaluation (14.1.1 external index)

# load modules
import numpy as np
from sklearn.metrics import rand_score, adjusted_rand_score

# Data
sol1 = np.array([1, 1, 2, 2, 2, 3, 3])
sol2 = np.array([1, 1, 2, 3, 2, 1, 3])

# Ex 14.1 Rand index
rand_score(sol1, sol2)

# Ex 14.3 Adjusted Rand index
adjusted_rand_score(sol1, sol2)

