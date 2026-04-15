# ch07_LDA.py
# Discriminant Analysis

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix

############################################
# Examples 7.4
############################################
# load data
dat1 = pd.read_csv("data/ch7_dat1.csv")
X = dat1[['X1', 'X2']]
y = LabelBinarizer().fit_transform(dat1['class']).ravel()

# Linear Discriminant Analysis (LDA)
lda_model = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
lda_fit = lda_model.fit(X, y)

# intercept
lda_fit.intercept_

# coefficients
lda_fit.coef_

# predicted posterior probabilities
lda_fit.predict_proba(X)

# predicted class
lda_fit.predict(X)


############################################
# Examples 7.6
############################################
# load data
dat1 = pd.read_csv("data/ch7_dat1.csv")
X = dat1[['X1', 'X2']]
y = LabelBinarizer().fit_transform(dat1['class']).ravel()

# Quadratic Discriminant Analysis (QDA)
qda_model = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], store_covariance=True)
qda_fit = qda_model.fit(X, y)

# variance-covariance matrix within each class
qda_fit.covariance_

# predicted posterior probabilities
qda_fit.predict_proba(X)

# predicted class
qda_fit.predict(X)

############################################
# Examples 7.7
############################################
# load data
iris = pd.read_csv("data/iris.csv")
X = iris[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
y = LabelEncoder().fit_transform(iris['Species']).ravel()

# train/test split
random.seed(478245)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

# Linear Discriminant Analysis (LDA)
lda_model = LinearDiscriminantAnalysis()
lda_fit = lda_model.fit(X_train, y_train)

# intercept
lda_fit.intercept_

# coefficients
lda_fit.coef_

# predicted class on test set
y_pred = lda_fit.predict(X_test)

X_test.assign(
    y = y_test,
    y_pred = y_pred
)

# confusion matrix
confusion_matrix(y_test, y_pred)