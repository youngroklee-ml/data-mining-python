# ch10_ex02_rf.py
# ch10.2 Bootstrapping

# load packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt

# load data
dat = pd.read_csv("data/ch10_dat1.csv")
dat["X1"] = dat["X1"].astype("category")
dat["X2"] = dat["X2"].astype("category")
dat["X3"] = dat["X3"].astype("category")
dat["Y"]  = dat["Y"].astype("category")

# ex10.2

# try random forest with 4 trees, 2 candidate variables at each split
# results depend on random seed number
np.random.seed(5678)
rf = RandomForestClassifier(
    n_estimators=4, # ntree = 4 (4 trees)
    max_features=2, # mtry = 2 (2 candidate variables at each split)
    bootstrap=True, # samptype = "swr" (sampling with replacement)
    oob_score=True,
    random_state=5678
)

X = pd.get_dummies(dat[["X1", "X2", "X3", "X4"]], drop_first=False)
y = dat["Y"].astype(str).to_numpy()
rf.fit(X, y)

# observations used in training data
# results would be different from book example
# due to randomness in resampling
from sklearn.ensemble._forest import _generate_sample_indices
n = X.shape[0]
samples_per_tree = [_generate_sample_indices(est.random_state, n, n) for est in rf.estimators_]
inbag_counts = np.column_stack([np.bincount(idx, minlength=n) for idx in samples_per_tree])
inbag_counts

# plot each tree
plt.figure()
plot_tree(rf.estimators_[0], feature_names=list(X.columns))
plt.show()
plt.figure()
plot_tree(rf.estimators_[1], feature_names=list(X.columns))
plt.show()
plt.figure()
plot_tree(rf.estimators_[2], feature_names=list(X.columns))
plt.show()
plt.figure()
plot_tree(rf.estimators_[3], feature_names=list(X.columns))
plt.show()

# ex10.3

# out-of-bag prediction: probability
oob_prob = rf.oob_decision_function_.copy()
if np.isnan(oob_prob).any():
    nan_rows = np.isnan(oob_prob).any(axis=1)
    oob_prob[nan_rows] = 1.0 / oob_prob.shape[1]
rf_predicted_oob = pd.DataFrame(oob_prob, columns=[str(c) for c in rf.classes_])
rf_predicted_oob

# out-of-bag prediction: classification
rf_class_oob = rf.classes_[np.argmax(oob_prob, axis=1)]
rf_class_oob

# out-of-bag prediction performance
results = dat.copy()
results["pred_class"] = rf_class_oob
cm = confusion_matrix(results["Y"].astype(str), results["pred_class"].astype(str), labels=["yes", "no"])
cm

# summary(cm) - confusion matrix summary
truth = results["Y"].astype(str)
estimate = results["pred_class"].astype(str)
levels = sorted(pd.unique(truth))
if len(levels) != 2:
    raise ValueError(f"Binary classification expected, got levels={levels}")
pos_label, neg_label = levels[0], levels[1]
summary_tbl = pd.DataFrame({
    ".metric": [
        "accuracy", "kap", "sens", "spec", "ppv", "npv",
        "mcc", "j_index", "bal_accuracy", "detection_prevalence",
        "precision", "recall", "f_meas"
    ],
    ".estimator": ["binary"] * 13,
    ".estimate": [
        accuracy_score(truth, estimate),
        cohen_kappa_score(truth, estimate, labels=[pos_label, neg_label]),
        recall_score(truth, estimate, pos_label=pos_label), # sens
        recall_score(truth, estimate, pos_label=neg_label), # spec
        precision_score(truth, estimate, pos_label=pos_label), # ppv
        precision_score(truth, estimate, pos_label=neg_label), # npv
        matthews_corrcoef(truth, estimate),
        (recall_score(truth, estimate, pos_label=pos_label) +
         recall_score(truth, estimate, pos_label=neg_label) - 1),
        balanced_accuracy_score(truth, estimate),
        np.mean(estimate.to_numpy() == pos_label),
        precision_score(truth, estimate, pos_label=pos_label), # precision
        recall_score(truth, estimate, pos_label=pos_label), # recall
        f1_score(truth, estimate, pos_label=pos_label), # f_meas
    ]
})
summary_tbl[".estimate"] = summary_tbl[".estimate"].astype(float).round(3)
print(f"# A tibble: {summary_tbl.shape[0]} × {summary_tbl.shape[1]}")
print(summary_tbl.to_string(index=False))
