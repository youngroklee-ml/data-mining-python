# ch09_ex02_linear_inseparable.py
# ch9.2 linear SVM - inseparable

# ex9.2
# load modules
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer

# load data
dat = pd.read_csv("data/ch9_dat2.csv")
X = dat[['x1', 'x2']]
y = LabelBinarizer(neg_label=-1, pos_label=1).fit_transform(dat['class']).ravel()

# train SVM
svm_model = SVC(
  C=1, # please also try 5 and 100
  kernel='linear',
  tol=1e-5
)
svm_model.fit(X, y)

# support vectors
# NOTE: i-th observation has index i-1
print(svm_model.support_)

# alpha values for support vectors
alphas = np.abs(svm_model.dual_coef_)
print(alphas)

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

# misclassified objects
# NOTE: i-th observation has index i-1
np.where(svm_model.predict(X) != y)

