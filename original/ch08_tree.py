# ch08_tree.py
# Decision Tree

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

############################################
# Examples 8.2 - 8.7
############################################
# load data
df_train = pd.read_csv("data/ch8_dat1.csv")
df_test = pd.read_csv("data/ch8_dat2.csv")

X_train = df_train[['x1', 'x2']]
X_test = df_test[['x1', 'x2']]

label = LabelBinarizer()
y_train = label.fit_transform(df_train['class']).ravel()
y_test = label.transform(df_test['class']).ravel()

tree_depth1 = DecisionTreeClassifier(max_depth=1)
tree_depth1.fit(X_train, y_train)

plot_tree(tree_depth1)

# Ex 8.3: Maximal tree
tree_maximal = DecisionTreeClassifier()
tree_maximal.fit(X_train, y_train)
plot_tree(tree_maximal)

# Ex 8.6: pruning
path = tree_maximal.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

print(ccp_alphas)

tree_prune = DecisionTreeClassifier(ccp_alpha = 0.1)
tree_prune.fit(X_train, y_train)
plot_tree(tree_prune)

# Ex 8.7: prediction
df_test.assign(
  pred_class = label.inverse_transform(tree_prune.predict(X_test))
)

# prediction accuracy
accuracy_score(
  y_test,
  tree_prune.predict(X_test)
)