# Classification

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score, roc_curve, auc, 
    RocCurveDisplay, confusion_matrix
)

#######################################
# Example 5.2
#######################################

# Load data
dat1 = pd.read_csv("data/ch7_dat1.csv")
dat1

X = dat1[['X1', 'X2']]
y = LabelBinarizer().fit_transform(dat1['class'])


# Training and testing data:
# Let us use the first 7 observations as training data, 
# while remaining 2 observations as testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=2, train_size=7, 
    random_state=None, shuffle=False
)

# k-nearest neighbors classifier:
# Use `KNeighborsClassifier()` from `sklearn.neighbors` module 
# to define a kNN model for a classification problem.
# Set `n_neighbors=3` to train 3-NN in this example.
knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the 3-NN classifier on training data.
knn_model.fit(X_train, y_train)

# Use `predict()` method to make a prediction on training data.
pd.DataFrame(X_train).assign(
    observed_class = y_train,
    pred_class = knn_model.predict(X_train)
)

# Also, make a prediction on testing data.
pd.DataFrame(X_test).assign(
    observed_class = y_test,
    pred_class = knn_model.predict(X_test)
)


#######################################
# Examples 5.3 - 5.4
#######################################

# Load data
dat3 = pd.read_csv("data/ch5_dat3.csv")
dat3

# Preprocess data:
# Encode categorical values to integer variable by using `factorize()` method from `pandas` module.
X = dat3[['gender', 'age_gr']].copy()
X.gender, _ = pd.factorize(X.gender)
X.age_gr, _ = pd.factorize(X.age_gr)


# Ex 5.3: Naive Bayes classifier

# Use `CategoricalNB()` to define a naive Bayes classifier. 
# Pass `alpha=0` argument to do not apply smoothing.
nb_model = CategoricalNB(alpha=0)
nb_model.fit(X, dat3['class'])

# Call `predict_proba()` method to predict posterior probability of belonging to each class.
nb_model.predict_proba(X)

# Call `predict()` method to make classification.
nb_model.predict(X)


# Ex 5.4: Evaluation metrics

y = dat3['class']
y_pred = nb_model.predict(X)

# Confusion matrix
confusion_matrix(y, y_pred)

# Accuracy
accuracy_score(y, y_pred)

# Sensitivity (Recall)
# Sensitivity is the same to precision. 
# Use `precision_score()` in `sklearn.metrics` module. 
# Set class `1` to be a positive event in this example by passing `pos_label=1` argument.
precision_score(y, y_pred, pos_label=1)

# Specificity
# Specificity is the same to sensitivity with different event level, 
# by passing `pos_label=2` argument.
precision_score(y, y_pred, pos_label=2)

# F1-score
# F1 score is a harmonic mean of precision and recall.
# You can either indirectly compute by using precision and recall
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
2 * precision * recall / (precision + recall)

# or directly compute by using `f1_score` function.
f1_score(y, y_pred)

# ROC curve and AUC

# Computing a ROC curve requires posterior probability.
prob = nb_model.predict_proba(X)[:, 0]

# Visualize ROC curve.
fpr, tpr, thresholds = roc_curve(y, y_score = prob, pos_label=1)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, pos_label=1)
display.plot()
