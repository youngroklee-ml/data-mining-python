# ch04_svd_pca.py
# Matrix decomposition
# Principal component score

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

# ex 4.5
# load data
dat1 = pd.read_csv("data/ch4_dat1.csv")
dat1

# define as matrix
x = dat1.to_numpy()

# X is your matrix
U, d, Vt = np.linalg.svd(x, full_matrices=False)

# Same as diag(s$d) in R
D = np.diag(d)
print(D)

# Same as s$u in R
U_matrix = U
print(U)

# Same as s$v in R  (note: R gives V, Python gives Vᵀ)
V_matrix = Vt.T
print(V_matrix)

# ex 4-6
# covariance matrix (same as R cov(), normalized by n−1)
cov_x = np.cov(x, rowvar=False)

# SVD of covariance matrix
U_cov, singular_cov, Vt_cov = np.linalg.svd(cov_x)

print("=== Covariance Matrix ===")
print(cov_x)

print("\n=== U (Left Singular Vectors) ===")
print(U_cov)

print("\n=== Singular Values ===")
print(singular_cov)

print("\n=== Vt (Right Singular Vectors) ===")
print(Vt_cov)

# ex 4.10 PCA
# Load data
dat2 = pd.read_csv("data/ch4_dat2.csv", encoding="euc-kr")
dat2

# correlation matrix
x = dat2.drop("ID", axis=1)   # this is a pandas dataframe
cor_x = x.corr()
print(cor_x)

# PCA by default, it centers for data
# Need StandardScaler for scaling
# principal component analysis with centering and scaling
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

# correlation matrix
cor_x = x.corr()
print(np.round(cor_x, 3))

# eigenvalue of cor(x)
pca_model = PCA()
pca_model.fit(x_std)

# pca_model.explained_variance_ : eigenvalues 
# ev = eigenvalues 
ev = pca_model.explained_variance_
#print(ev)

# scree plot : to choose the number of components
plt.figure()
plt.plot(range(1, len(ev)+1), ev, marker='o')   # connected line + dots
plt.title("Scree Plot")
plt.xlabel("Component")
plt.ylabel("Variance")
plt.xticks(range(1, len(ev)+1))
plt.show()

# bar plot of variance proportion explained by each PC
rate_var = ev / np.sum(ev)
categories = ["PC1", "PC2", "PC3", "PC4", "PC5"]
plt.bar(categories, rate_var)
plt.xlabel("Principal Components")
plt.ylabel("Proportion of Variance Explained")
plt.title("Variance explained by PCs")
plt.ylim(0, 1)
plt.show()

# ex 4.12 PCR
# Load data
dat3 = pd.read_csv("data/ch4_dat3.csv")
dat3

X = dat3.drop('y', axis=1)
y = dat3['y']

# Principal component analysis with `scikit-learn` 
# PCA by default, it centers the data. 
# PCA by default, it does not scale to unit variance; use StandardScaler

pca_model = PCA(n_components=2)
pc = pca_model.fit_transform(X)
print(pc)

# Use `LinearRegression()` from `sklearn.linear_model` 
# Use principal component scores as input matrix.
pcr_model = LinearRegression()
pcr_model.fit(pc, y)

# intercept.
print(pcr_model.intercept_)

# coefficient of principal component regression
print(pcr_model.coef_)
