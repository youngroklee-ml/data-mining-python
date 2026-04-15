# ch10_ex05_gbm_regression.py
# Ch10.5 Gradient boosting

# ex10.5

# load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree

# load data
dat = pd.read_csv("data/ch10_dat3.csv")

# use {sklearn} package
gbm_fit = GradientBoostingRegressor(
    loss="squared_error", # distribution = "gaussian"
    n_estimators=5, # n.trees = 5
    learning_rate=1.0, # shrinkage = 1 (step size = 1)
    max_depth=1, # interaction.depth = 1
    min_samples_leaf=1, # n.minobsinnode = 1
    subsample=1.0 # bag.fraction = 1 (no subsampling of training data)
)
gbm_fit.fit(dat[["X"]].to_numpy(), dat["Y"].to_numpy())

# prediction for new data
dat_p = pd.DataFrame({
    "X": np.linspace(dat["X"].min(), dat["X"].max(), 1000)
})

plt.scatter(dat["X"], dat["Y"],
            s=20, marker="o")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Observed vs Prediction: {sklearn} package")
plt.plot(dat_p["X"], gbm_fit.predict(dat_p[["X"]].to_numpy()), color="red")
plt.show()

# implement gradient boosting from scratch

# initial model
dat_m = dat.copy()
dat_m["pred"] = dat["Y"].mean()
dat_p["pred"] = dat["Y"].mean()

iter = 5

for i in range(1, iter + 1):
    # negative gradients
    dat_m["ngrad"] = dat["Y"] - dat_m["pred"]

    # plot layout
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # plot observed point and prediction
    ax1.scatter(dat["X"], dat["Y"], s=20, marker="o")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title(f"Observed vs Prediction: Iteration {i}")
    ax1.plot(dat_p["X"], dat_p["pred"], color="red")

    # plot residuals
    ax2.scatter(dat_m["X"], dat_m["ngrad"], s=20, marker="o")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Residual")
    ax2.set_title(f"Residual: Iteration {i}")


    # fit C to negative gradients
    fit = DecisionTreeRegressor(max_depth=1)
    fit.fit(dat_m[["X"]].to_numpy(), dat_m["ngrad"].to_numpy())

    # plot tree
    plot_tree(fit, feature_names=["X"], filled=False, rounded=True, ax=ax3)
    ax3.set_title(f"Residual estimation: Iteration {i}")
    plt.show()

    # update prediction
    dat_m["pred"] = dat_m["pred"] + fit.predict(dat_m[["X"]].to_numpy())
    dat_p["pred"] = dat_p["pred"] + fit.predict(dat_p[["X"]].to_numpy())

# plot final prediction
plt.scatter(dat["X"], dat["Y"], s=20, marker="o")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Observed vs Prediction: Final")
plt.plot(dat_p["X"], dat_p["pred"], color="red")
plt.show()