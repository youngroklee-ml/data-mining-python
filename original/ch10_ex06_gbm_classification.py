# ch10_ex06_gbm_classification.py
# Ch10.5 Gradient boosting

# ex10.6

# load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# load data
dat = pd.read_csv("data/ch8_dat1.csv")
dat["class"] = dat["class"] - 1  # set class to be 0 or 1


# evaluate base model performance (i.e. p = 0.5)
# based on ROC AUC
result0 = pd.DataFrame({
    "class": pd.Categorical(dat["class"], categories=[1, 0], ordered=True),
    "posterior1": np.repeat(0.5, len(dat))
})
roc_auc_score(result0["class"].astype(int), result0["posterior1"])
fpr0, tpr0, _ = roc_curve(result0["class"].astype(int), result0["posterior1"])
plt.plot(fpr0, tpr0)
plt.xlabel("1-specificity")
plt.ylabel("sensitivity")
plt.show()


# estimate gradient boosting model
# with 1 tree
fit1 = GradientBoostingClassifier(
    loss="log_loss", # distribution = "bernoulli"
    n_estimators=1, # n.trees = 1
    learning_rate=1.0, # shrinkage = 1 (step size = 1)
    max_depth=1, # interaction.depth = 1
    min_samples_leaf=1, # n.minobsinnode = 1
    subsample=1.0, # bag.fraction = 1 (no subsampling of training data)
    random_state=0
)
fit1.fit(dat[["x1", "x2"]].to_numpy(), dat["class"].to_numpy().astype(int))

# evaluate classification performance
# based on ROC AUC
result1 = pd.DataFrame({
    "class": pd.Categorical(dat["class"], categories=[1, 0], ordered=True),
    "posterior1": fit1.predict_proba(dat[["x1", "x2"]].to_numpy())[:, 1]
})
roc_auc_score(result1["class"].astype(int), result1["posterior1"])
fpr1, tpr1, _ = roc_curve(result1["class"].astype(int), result1["posterior1"])
plt.plot(fpr1, tpr1)
plt.xlabel("1-specificity")
plt.ylabel("sensitivity")
plt.show()


# estimate gradient boosting model
# with 2 trees
fit2 = GradientBoostingClassifier(
    loss="log_loss", # distribution = "bernoulli"
    n_estimators=2, # n.trees = 2
    learning_rate=1.0, # shrinkage = 1 (step size = 1)
    max_depth=1, # interaction.depth = 1
    min_samples_leaf=1, # n.minobsinnode = 1
    subsample=1.0, # bag.fraction = 1 (no subsampling of training data)
    random_state=0
)
fit2.fit(dat[["x1", "x2"]].to_numpy(), dat["class"].to_numpy().astype(int))

# evaluate classification performance
# based on ROC AUC
result2 = pd.DataFrame({
    "class": pd.Categorical(dat["class"], categories=[1, 0], ordered=True),
    "posterior1": fit2.predict_proba(dat[["x1", "x2"]].to_numpy())[:, 1]
})
roc_auc_score(result2["class"].astype(int), result2["posterior1"])
fpr2, tpr2, _ = roc_curve(result2["class"].astype(int), result2["posterior1"])
plt.plot(fpr2, tpr2)
plt.xlabel("1-specificity")
plt.ylabel("sensitivity")
plt.show()
