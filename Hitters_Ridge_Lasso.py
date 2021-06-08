import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
import random

random.seed(0)
base = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Hitters.csv")
base = base.dropna()

dummies = pd.get_dummies(base[['League', 'Division', 'NewLeague']])

y = base.Salary
X_ = base.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Define the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

print("\n--------Ridge Regression--------\n")

# Covers the full range from only intercept model to least square fit
# 100 values
# matrix of coefs of 19 (predictors) x 100 (lambda)
# It is needed to normalize the variables

alphas = 10**np.linspace(10,-2,100)*0.5

ridge = Ridge(normalize = True, fit_intercept = True)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

print("\nAlpha = 4\n")
ridge2 = Ridge(alpha = 4, normalize = True)
# Fit a ridge regression on the training data
ridge2.fit(X_train, y_train)      
# Use this model to predict the test data       
pred2 = ridge2.predict(X_test)
# Print coefficients           
print(pd.Series(ridge2.coef_, index = X.columns)) 
print("Intercept: " + str(ridge2.intercept_))
# Calculate the test MSE
print("\ntest MSE: ")
print(mean_squared_error(y_test, pred2))  

print("\nAlpha = 10^10\n")       

ridge3 = Ridge(alpha = 10**10, normalize = True)
# Fit a ridge regression on the training data
ridge3.fit(X_train, y_train) 
# Use this model to predict the test data            
pred3 = ridge3.predict(X_test)     
# Print coefficients      
print(pd.Series(ridge3.coef_, index = X.columns)) 
# Calculate the test MSE 
print("\ntest MSE: ")
print(mean_squared_error(y_test, pred3))     

# This big penalty shrinks the coefficients to a very large degree, 
# essentially reducing to a model containing just the intercept. 
# This over-shrinking makes the model more biased, resulting in a higher MSE. 

print("\nAlpha = 0 (Least Square)\n")    

ridge2 = Ridge(alpha = 0, normalize = True)
# Fit a ridge regression on the training data
ridge2.fit(X_train, y_train)   
# Use this model to predict the test data          
pred = ridge2.predict(X_test)   
# Print coefficients         
print(pd.Series(ridge2.coef_, index = X.columns)) 
# Calculate the test MSE
print("\ntest MSE: ")
print(mean_squared_error(y_test, pred))  

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(X_train, y_train)
print("\nValue of alpha that results in the smallest cross-validation error: "  + 
      str(ridgecv.alpha_))         

ridge4 = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge4.fit(X_train, y_train)
print("\nMSE associated with this value of alpha is: " + 
      str(mean_squared_error(y_test, ridge4.predict(X_test))))

print("\nCoefficients using the full data and alpha by CV: ")

ridge4.fit(X, y)
print(pd.Series(ridge4.coef_, index = X.columns))
print("Intercept: " + str(ridge4.intercept_))
#%%
print("\n--------Lasso Regression--------\n")

# We saw that ridge regression with a wise choice of alpha can outperform 
# least squares as well as the null model on the Hitters data set. 
# We now ask whether the lasso can yield either a more accurate or 
# a more interpretable model than ridge regression. In order to fit a lasso model, 
# we'll use the Lasso() function; however, this time we'll need to include 
# the argument max_iter = 10000. Other than that change, we proceed just as we did 
# in fitting a ridge model:
    
lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)


ax = plt.figure()    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

# Não precisa de colocar a coluna de alpha e nem scoring, apenas o numero maximo de iteração
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X, y)

print("\nValue of alpha that results in the smallest cross-validation error: "  + 
      str(lassocv.alpha_)) 

print("\nMSE associated with this value of alpha is: " + 
      str(mean_squared_error(y, lasso.predict(X))))

# However, the lasso has a substantial advantage over ridge regression in 
# that the resulting coefficient estimates are sparse. Here we see that 13 
# of the 19 coefficient estimates are exactly zero:

print("\nCoefficients using the full data and alpha by CV: ")
# Some of the coefficients are now reduced to exactly zero.
print(pd.Series(lasso.coef_, index=X.columns))
print("Intercept: " + str(lasso.intercept_))



