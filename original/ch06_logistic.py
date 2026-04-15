# ch06_logistic.py
# logistic regression

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OrdinalEncoder

############################################
# Examples 6.1, 6.3
############################################
# load data
dat1 = pd.read_csv("data/ch6_dat1.csv")
dat1

X = dat1[['Break', 'Sleep', 'Circle']]
y = LabelBinarizer().fit_transform(dat1['Class']).ravel()
dat1['Class'] = y

# Ex 6.1: Logit model
# estimate binary logistic regression
logit_model = LogisticRegression(penalty = None)
logit_fit = logit_model.fit(X, y)

# intercept
logit_fit.intercept_

# regression coefficients
logit_fit.coef_

# logit(P), predicted probability, predicted class
dat1.assign(
    pred_logit = np.diff(logit_fit.predict_log_proba(X), axis=1),
    pred_prob = logit_fit.predict_proba(X)[:, 1],
    pred_class = logit_fit.predict(X)
)

# using statsmodels
logit_fit = smf.logit("Class ~ Break + Sleep + Circle", data = dat1).fit()

# regression coefficients with significance tests
print(logit_fit.summary().tables[1])

# predicted probability (P)
dat1.assign(
    pred_logit = logit_fit.predict(dat1, which = 'linear'),
    pred_prob = logit_fit.predict(dat1, which = 'mean'),
)

# Ex 6.3: Probit model
# estimate with probit(normit) link function
probit_fit = smf.probit("Class ~ Break + Sleep + Circle", data = dat1).fit()

# regression coefficients with significance tests
print(probit_fit.summary().tables[1])

# logit-like index and predicted probability
dat1.assign(
    pred_probit = probit_fit.predict(dat1, which = 'linear'),
    pred_prob = probit_fit.predict(dat1, which = 'mean'),
)


############################################
# Examples 6.4
############################################
# load data
dat2 = pd.read_csv("data/ch6_dat2.csv")
dat2

X = dat2[['X1', 'X2']]
y = LabelEncoder().fit_transform(dat2['Y']).ravel()
dat2['Y'] = y

# still use LogisticRegression() to fit multinomial logistic regression.
mnlogit_model = LogisticRegression(penalty=None)
mnlogit_fit = mnlogit_model.fit(X, y)

# intercepts
mnlogit_fit.intercept_

# coefficients on variables
mnlogit_fit.coef_

pd.concat(
    [dat2, 
     pd.DataFrame(
        mnlogit_fit.predict_proba(X), 
        columns=['pred_prob_0', 'pred_prob_1', 'pred_prob_2'])],
    axis=1
).assign(
    pred_class = mnlogit_fit.predict(X)
)

# use mnlogit() to train a model
mnlogit_fit = smf.mnlogit("Y ~ X1 + X2", data=dat2).fit()

# coefficient table
print(mnlogit_fit.summary().tables[1])

pd.concat([dat2, mnlogit_fit.predict(dat2)], axis=1)

# confusion matrix
mnlogit_fit.pred_table()


############################################
# Example 6.5
############################################
# load data
dat3 = pd.read_csv("data/ch6_dat3.csv")
X = dat3[['N', 'L']]
y = OrdinalEncoder().fit_transform(dat3[['Y']])
dat3['Y'] = y

# use statsmodels.miscmodels.ordinal_model.OrderedModel(). Set distr='logit' to use logit link function
ordinal_fit = OrderedModel(y, X, distr='logit').fit()

# coefficient table
print(ordinal_fit.summary().tables[1])

ordinal_fit.predict(X)

# confusion matrix
ordinal_fit.pred_table()