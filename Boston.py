import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import statsmodels.api as sm
from statsmodels.stats.api import anova_lm
from statsmodels.formula.api import ols
from scipy import stats
import numpy as np
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.metrics import confusion_matrix, mean_squared_error

boston = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Boston.csv")
boston.dropna() 

# ----------------------------------------------------------------------------
print("Linear Regression with Stat")

x_lstat = boston.lstat.values.reshape(-1, 1)
y = boston.medv

X2 = sm.add_constant(x_lstat)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

x_pred = np.array([5,10,15]).reshape(-1, 1)
X3 = sm.add_constant(x_pred)

y_pred = est2.predict(X3)
print(y_pred)

predictions = est2.get_prediction(X3)
print(predictions.summary_frame(alpha=0.05)[['mean_ci_lower', 'mean_ci_upper']])
print(predictions.summary_frame(alpha=0.05)[['obs_ci_lower','obs_ci_upper']])
print(predictions.summary_frame(alpha=0.05)[['mean', 'mean_se']])

plt.scatter(x_lstat, y)
plt.plot(boston.lstat, est2.fittedvalues, 'r')
plt.grid(True)
plt.show()

# influence = est2.get_influence()
# standardized_residuals = influence.resid_studentized_internal
# leverage = influence.hat_matrix_diag
# max_leverage = np.argmax(leverage)+1

# plt.scatter(est2.fittedvalues,standardized_residuals)

print("----------------------------------------------------------------------------")
print("Multiple Linear Regression with Stat and Age")

x_lstat_age = boston[['lstat', 'age']]

X4 = sm.add_constant(x_lstat_age)
est3 = sm.OLS(y, X4)
est4 = est3.fit()
print(est4.summary())

y_pred = est4.predict(X4)
print(y_pred)
print("\nTest MSE:\n")
print(mean_squared_error(y, y_pred))

print("----------------------------------------------------------------------------")
print("Multiple Linear Regression with all predictors")

x_all = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis',
       'rad', 'tax', 'ptratio', 'black', 'lstat']]

X5 = sm.add_constant(x_all)
est5 = sm.OLS(y, X5)
est6 = est5.fit()
print(est6.summary())

y_pred = est6.predict(X5)
print(y_pred)
print("\nTest MSE:\n")
print(mean_squared_error(y, y_pred))

print("----------------------------------------------------------------------------")
print("Multiple Linear Regression with all predictors but age")

x_all_age = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'dis',
       'rad', 'tax', 'ptratio', 'black', 'lstat']]

X6 = sm.add_constant(x_all_age)
est7 = sm.OLS(y, X6)
est8 = est7.fit()
print(est8.summary())

print("----------------------------------------------------------------------------")
print("Multiple Linear Regression with all predictors but age, indus")

x_all_age_indus = boston[['crim', 'zn', 'chas', 'nox', 'rm', 'dis',
       'rad', 'tax', 'ptratio', 'black', 'lstat']]

X7 = sm.add_constant(x_all_age_indus)
est9 = sm.OLS(y, X7)
est10 = est9.fit()
print(est10.summary())

influence = est10.get_influence()
standardized_residuals = influence.resid_studentized_internal
leverage = influence.hat_matrix_diag
max_leverage = np.argmax(leverage)+1

plt.scatter(est10.fittedvalues,standardized_residuals)


print("----------------------------------------------------------------------------")
print("Including interation term")

model = ols('medv ~ lstat + age + lstat:age', data=boston).fit()
print(sm.stats.anova_lm(model, typ=2))
print(model.summary())

print("----------------------------------------------------------------------------")
print("Non linear transformation of lstat predictor")

model1 = ols('medv ~ lstat +I(lstat**2)', data=boston).fit()
print(model1.summary())
table = anova_lm(est2,model1)
print(table)

influence = model1.get_influence()
standardized_residuals = influence.resid_studentized_internal
plt.figure()
plt.scatter(model1.fittedvalues,standardized_residuals)

plt.figure()
plt.scatter(x_lstat, y)
plt.plot(boston.lstat, model1.fittedvalues, '.r')
plt.grid(True)
plt.show()

print("---------------------------------------------------------------------")
print("\nCarseats data + Qualitative predictors\n")

carseats = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Carseats.csv")
carseats.dropna()

print(sm.OLS.from_formula('Sales ~ Income:Advertising+Price:Age + ' + "+".join(carseats.columns.difference(['Sales'])), carseats).fit().summary())