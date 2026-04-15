# ch10_ex04_adaboost.py
# ch10.4 AdaBoost

# load packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# load data
dat = pd.read_csv("data/ch10_dat1.csv")
dat["X1"] = dat["X1"].astype("category")
dat["X2"] = dat["X2"].astype("category")
dat["X3"] = dat["X3"].astype("category")
dat["Y"] = dat["Y"].astype("category")

# create training data
dat["class"] = np.where(dat["Y"].astype(str) == "yes", 1, -1)
X = pd.get_dummies(dat.drop(columns=["Y", "class"]), drop_first=False)
y = dat["class"].to_numpy()

# estimate AdaBoost model
# with 3 trees
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=3,
)
ada.fit(X, y)

# plot each score tree
for m in ada.estimators_:
  plot_tree(m)
  plt.show()

# values of alpha
ada.estimator_weights_

# predicted probability of y = 1 on training data
y_posterior = ada.predict_proba(X)[:, 1]
y_posterior

# predicted class on training data
pred_class = ada.predict(X)
pred_class

confusion_matrix(y, pred_class)