# Regression

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

#######################################
# Examples 2.3 - 2.5, 2.7, 2.10 - 2.11
#######################################


# load data
dat1 = pd.read_csv("data/ch2_reg1.csv")

# print data
dat1

# Ex 2.3: Regression coefficient

# Define a linear regression model with `smf.ols()` 
# by passing formula string and training data as arguments.
model = smf.ols("weight ~ age + height", data = dat1)

# Estimate the model by calling `fit()` method.
model_fit = model.fit()

# Let's see a summary of model estimation results.
model_fit.summary()

# Coefficient estimates can be accessed with `params` attribute.
model_fit.params


# Ex 2.4: Variance of error terms

# `scale` attribute represents the estimate of error variance.
model_fit.scale

# Ex 2.5: Test a model

# Use `anova_lm()` from `statsmodels.api.stats` module 
# and pass the fitted linear regression model to conduct ANOVA test.
sm.stats.anova_lm(model_fit)

# Ex 2.7 Test regression coefficients

# One of tables from `summary()` output represents coefficient-level statistics 
# including t-test statstics.
model_fit.summary().tables[1]


# Ex 2.10 - 2.11 Mean prediction, confidence interval and prediction interval

# Variance-covariance matrix of coefficients:
model_fit.cov_params()

# Create new data to predict:
newdata = pd.DataFrame({'age': [40, ], 'height': [170, ]})

# Predict mean response by calling `predict()` method.
model_fit.predict(newdata)

# Let's also get 95% confidence interval of mean response and 95% prediction interval. 
# To do so, first let's produce a prediction object by calling `get_prediction()` method.
prediction_results = model_fit.get_prediction(newdata)

# `summary_frame()` method produces data frame that include 
# mean prediction, confidence interval, and prediction interval. 
# To compute 95% confidence/prediction intervals, pass `alpha=0.95` as an argument.
prediction_results.summary_frame(alpha=0.95)


#######################################
# Example 2.14, 2.16
#######################################

# Load data
dat1 = pd.read_csv("data/ch2_coil.csv")
dat1

# Ex 2.14: Estimate a model with an indicator variable

# Using `C()` inside a formula means that the variable will be considered 
# as a categorical variable. In such case, dummy variables are automatically 
# created during the process of model estimation. 
# Inside `C()`, you can pass the second argument `Treatment(reference=6)` 
# to specify `6` as a base category, so the dummy variable value is `0` 
# when original variable value is `6`.
model_fit = smf.ols("y ~ temp + C(thick, Treatment(reference=6))", data=dat1).fit()

# Let's check the estimated regression coefficients.
model_fit.summary().tables[1]

# Ex 2.16: Interaction term
# Use `*` to estimate not only main effects but only interactions.
model_fit = smf.ols("y ~ temp*C(thick, Treatment(reference=6))", data=dat1).fit()
model_fit.summary().tables[1]

