# ch09_ex07_nonlinear.py
# ch9.3 nonlinear SVM

# ex9.7
# load modules
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer

# load data
dat = pd.read_csv("data/ch9_dat3.csv")
X = dat[['x1', 'x2']]
y = LabelBinarizer(neg_label=-1, pos_label=1).fit_transform(dat['class']).ravel()

# SVM with 2nd order polynomial kernel
svm_model = SVC(
  C=1, # please also try 5 and 100
  kernel='poly',
  degree=2, 
  gamma=1, 
  coef0=1,
  tol=1e-5
)
svm_model.fit(X, y)

# support vectors
# NOTE: i-th observation has index i-1
print(svm_model.support_)

# alpha values for support vectors
alphas = np.abs(svm_model.dual_coef_)
print(alphas)

# misclassified objects
# NOTE: i-th observation has index i-1
np.where(svm_model.predict(X) != y)
