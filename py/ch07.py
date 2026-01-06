# Discriminant analysis

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix


#######################################
# Example 7.4
#######################################

# Load data
dat1 = pd.read_csv("data/ch7_dat1.csv")
X = dat1[['X1', 'X2']]
y = LabelBinarizer().fit_transform(dat1['class']).ravel()

# Use `scikit-learn` for LDA

# Training with `priors=[0.5, 0.5]`:
lda_model = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
lda_fit = lda_model.fit(X, y)

# Intercept:
lda_fit.intercept_

# Coefficients:
lda_fit.coef_

# Posterior probability:
lda_fit.predict_proba(X)

# Classification:
lda_fit.predict(X)


#######################################
## Example 7.6
#######################################

# Load data
dat1 = pd.read_csv("data/ch7_dat1.csv")
X = dat1[['X1', 'X2']]
y = LabelBinarizer().fit_transform(dat1['class']).ravel()

# Use `scikit-learn` for QDA

# Training with `priors=[0.5, 0.5]`:
qda_model = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], store_covariance=True)
qda_fit = qda_model.fit(X, y)

# Variance-covariance matrix within each class:
qda_fit.covariance_

# Posterior probabilty:
qda_fit.predict_proba(X)

# Classification:
qda_fit.predict(X)


#######################################
# Example 7.7
#######################################

# Load data
iris = pd.read_csv("data/iris.csv")
X = iris[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
y = LabelEncoder().fit_transform(iris['Species']).ravel()

# Train/test split
random.seed(478245)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

# LDA

# Training:
lda_model = LinearDiscriminantAnalysis()
lda_fit = lda_model.fit(X_train, y_train)

# Intercept:
lda_fit.intercept_

# Coefficients:
lda_fit.coef_

# Prediction:
y_pred = lda_fit.predict(X_test)

X_test.assign(
    y = y_test,
    y_pred = y_pred
)

# Confusion matrix:
confusion_matrix(y_test, y_pred)
