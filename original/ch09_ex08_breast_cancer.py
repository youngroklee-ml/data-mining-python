# ch09_ex08_breast_cancer.py
# ch9.3 nonlinear SVM

# ex9.8

# load modules
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

# load data 
# with removing rows that include missing values
# and removing ID column
dat = pd.read_csv("data/breast-cancer-wisconsin.csv").dropna()
X = dat.iloc[:, 1:10]
y = LabelBinarizer(neg_label=2, pos_label=4).fit_transform(dat['class']).ravel()


# split train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=849716)

# train SVM with Gaussian kernel with sigma = 0.5
# See [kernel function parameterization](https://scikit-learn.org/stable/modules/svm.html#svm-kernels)
svm_model = SVC(
  C=10, 
  kernel='rbf', 
  gamma=0.5, # Please also try 2
  tol=1e-5
  )
svm_model.fit(X_train, y_train)

# confusion matrix on train data
cm_train = confusion_matrix(y_train, svm_model.predict(X_train))
print(cm_train)

# confusion matrix on test data
cm_test = confusion_matrix(y_test, svm_model.predict(X_test))
print(cm_test)

