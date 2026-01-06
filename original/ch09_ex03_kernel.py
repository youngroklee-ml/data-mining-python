# ch09_ex03_kernel.py
# ch9.3 nonlinear SVM

# ex9.3 kernel
# load modules
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel, linear_kernel

# load data
dat = np.array([[1, 2], 
                [2, 2], 
                [2, -1]])

# Gaussian RBF kernel
KG = rbf_kernel(dat, gamma=1/2)
print(KG)

# Polynomial kernel: 2nd order
KQ = polynomial_kernel(dat, degree=2, gamma=1, coef0=1)
print(KQ)

# Sigmoid kernel
KS = sigmoid_kernel(dat, gamma=1, coef0=0)
print(KS)

# Linear kernel
KL = linear_kernel(dat)
print(KL)
