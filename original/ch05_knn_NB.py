# ch05_knn_NB.py
# 5.2.1 KNN
# 5.2.2 Naive Bayes

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, 
    precision_score, recall_score, roc_auc_score, roc_curve, auc, 
    RocCurveDisplay, confusion_matrix
)

# ex5.2 (5.2.1.knn)

# Load data
dat1 = pd.read_csv("data/ch7_dat1.csv")
dat1 

# predictors
X = dat1[['X1', 'X2']]

# target (converted to 0/1)
y = LabelEncoder().fit_transform(dat1['class'])

# train and test : the first 7 obs as train & the remaining 2 as test, no shuffle
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=2, train_size=7, 
    random_state=None, shuffle=False
)

dist_matrix = pairwise_distances(X_train)
print(np.round(dist_matrix,3))

# knn (k=3)
# from sklearn.neighbors import KNeighborsClassifier 
knn_model = KNeighborsClassifier(n_neighbors=3)

# 3-NN classifier on training data.
knn_model.fit(X_train, y_train)

# `predict()` for train data
pd.DataFrame(X_train).assign(
    observed_class = y_train,
    pred_class = knn_model.predict(X_train)
)

# predict for test data
pd.DataFrame(X_test).assign(
    observed_class = y_test,
   pred_class = knn_model.predict(X_test)
)

#######################################
# Examples 5.3 - 5.4
#######################################

# ex5.3 (5.2.2.Naive Bayes)

# Load data
dat3 = pd.read_csv("data/ch5_dat3.csv")
dat3

# Preprocess data:
# Encode categorical values to integer variable by using `factorize()` method from `pandas` module.
X = dat3[['gender', 'age_gr']].copy()
X.gender, _ = pd.factorize(X.gender)
X.age_gr, _ = pd.factorize(X.age_gr)

# `CategoricalNB()` for Naive Bayes classifier. 
# Pass `alpha=0` argument to do not apply smoothing.
nb_model = CategoricalNB(alpha=0)
nb_model.fit(X, dat3['class'])

# `predict_proba()` for posterior probability of each class.
post=nb_model.predict_proba(X)

# `predict()` to make classification.
pred=nb_model.predict(X)

combined = np.column_stack((np.round(post,3), pred))
print(combined)


# ex 5.4: Evaluation metrics

y = dat3['class']
y_pred = nb_model.predict(X)

# Confusion matrix
confusion_matrix(y, y_pred)

# Accuracy
accuracy_score(y, y_pred)

# Sensitivity (Recall)
# Use `precision_score()` in `sklearn.metrics` module. 
# Set class `1` to be a positive event (`pos_label=1`)
precision_score(y, y_pred, pos_label=1)

# Specificity
# Specificity is the same to sensitivity with different event level, 
# by passing `pos_label=2` argument.
precision_score(y, y_pred, pos_label=2)

# F1-score
# F1 score is a harmonic mean of precision and recall.
# compute by using precision and recall
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
