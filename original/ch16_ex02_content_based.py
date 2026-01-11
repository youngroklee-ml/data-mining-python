# ch16_ex02_content_based.py
# ch16.2 content-based recommender system

# ex16.2

# load required modules
import pandas as pd
import numpy as np
from scipy.linalg import norm

# load data
df = pd.read_csv("data/ch16_content.csv")
user_weights = np.array(df[['weight']])
doc_weights = np.array(df.iloc[:, 2:])

# compute utility
numerator = np.matmul(user_weights.T, doc_weights)
denominator = norm(user_weights, ord=2) * norm(doc_weights, ord=2, axis=0)
utility = numerator / denominator
np.round(utility, 4)
