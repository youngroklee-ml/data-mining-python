# ch09_ex01_linear_separable.py
# ch9.1 linear SVM - separable

# ex9.1

# load modules
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer

# load data
dat = pd.read_csv("data/ch9_dat1.csv")
X = dat[['x1', 'x2']]
y = LabelBinarizer(neg_label=-1, pos_label=1).fit_transform(dat['class']).ravel()

# train SVM
svm_model = SVC(kernel='linear', tol=1e-5)
svm_model.fit(X, y)


# support vectors
# NOTE: i-th observation has index i-1
print(svm_model.support_)

# alpha values for support vectors
alphas = np.abs(svm_model.dual_coef_)
print(alphas)

# ensure separable
assert (np.sign(svm_model.dual_coef_) == y[svm_model.support_]).all()

# objective value
obj_value = (
  np.sum(alphas) - 
    np.sum(
      np.matmul(svm_model.dual_coef_.T, svm_model.dual_coef_) * 
      np.matmul(svm_model.support_vectors_, svm_model.support_vectors_.T)
    ) / 2.
)
print(obj_value)

# hyperplane coefficient vector w
w = svm_model.coef_
print(w)

# hyperplane intercept b
b = svm_model.intercept_
print(b)
