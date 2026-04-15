# ch02_reg.py
# Regression

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor # for VIF

#############################
# Examples 2.7, 2.10 - 2.11
#############################

# load data
dat1 = pd.read_csv("data/ch2_reg1.csv")

# ex 2.7: Regression coefficient

# Define a linear regression model with `smf.ols()` 
model = smf.ols("weight ~ age + height", data=dat1).fit()

# summary2() produces a tidy table of coefficients
# R-squared, and standard errors without the extra notes.
#print(model.summary())           # Regression table including R²
print(model.summary2())

print("===== ANOVA =====")
print(sm.stats.anova_lm(model))  # ANOVA table

# ex 2.8 & ex 2.9 Stepwise regression : 
# Python doesn’t have a built-in stepwise regression function

# ex 2.10 : Variance-covariance matrix of coefficients
model.cov_params()
print("===== Variance-covariance matrix =====")
print(model.cov_params())

# ex2.10 : the predicted mean response for new data 
new_data = pd.DataFrame({'age': [40], 'height': [170]})

# 95% confidence interval for mean response
# predicted response using `get_prediction()`
pred = model.get_prediction(new_data)
pred_summary = pred.summary_frame(alpha=0.05)  # alpha=0.05 → 95% CI

#  95% confidence interval for mean response
print("===== 95% confidence interval =====")
print(round(pred_summary,2))

# ex 2.11 : 95% prediction interval for individual future observations
print("===== 95% prediction interval for individual future obs =====")
print(pred_summary[['obs_ci_lower', 'obs_ci_upper']])


# ex 2.13 Salary data of Baseball player : Hitters data

# load data
hitter = pd.read_csv("data/ch2_hitter.csv")

# Define a linear regression model with `smf.ols()` 
model = smf.ols("Salary ~ Hits+ Walks+ CRuns+HmRun+CWalks", data=hitter).fit()

# summary2() produces a tidy table : R-squared, and standard errors without extra notes.
#print(model.summary())           # Regression table including R²
print(model.summary2())

print("===== ANOVA =====")
print(sm.stats.anova_lm(model))  # ANOVA table

# Select independent variables
X = hitter[['Hits','Walks','CRuns','HmRun','CWalks']]

# Add constant for intercept
X = sm.add_constant(X)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("===== VIF =====")
print(vif_data)

# The reason of slight differences for VIF in R and Python
# usually appear in the first or second decimal place in R and Python
# Python’s statsmodels uses OLS with least squares via QR decomposition or LAPACK routines.
# R’s lm() uses its own optimized linear algebra routines.


#######################################
# Example 2.14, 2.16
#######################################

# load data
dat1 = pd.read_csv("data/ch2_coil.csv")
dat1

print(dat1.head())

# ex 2.14: Estimate a model with an indicator variable

# By default: lowest value is the base for categorical variable. 
# If you want to make the base =6, `Treatment(reference=6)` 

model = smf.ols("y ~ temp + C(thick, Treatment(reference=6))", data=dat1).fit()

# summary2() produces a tidy table
print(model.summary2())

# ex 2.16: Interaction term
# Use `*` to estimate not only main effects but only interactions.
model_fit = smf.ols("y ~ temp*C(thick, Treatment(reference=6))", data=dat1).fit()
model_fit.summary().tables[1]
