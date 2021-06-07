import pandas as pd
import numpy as np
import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

auto = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Auto.csv")
auto.dropna()
auto.info()

for i in range(2):
    print("\nSeed: " + str(i+1))
    train_auto = auto.sample(196, random_state = i+1)
    test_auto = auto[~auto.isin(train_auto)].dropna(how = 'all')
    
    X_train = train_auto['horsepower'].values.reshape(-1,1)
    y_train = train_auto['mpg']
    X_test = test_auto['horsepower'].values.reshape(-1,1)
    y_test = test_auto['mpg']
    
    lm = skl_lm.LinearRegression()
    model = lm.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    MSE = mean_squared_error(y_test, pred)
    print("\nMean squared error for linear regression:")  
    print(MSE)
    
    # Order 2
    poly = PolynomialFeatures(degree=2)
    X_train2 = poly.fit_transform(X_train)
    X_test2 = poly.fit_transform(X_test)
    
    model = lm.fit(X_train2, y_train)
    print("\nMean squared error for linear regression order 2:") 
    print(mean_squared_error(y_test, model.predict(X_test2)))
    
    # Order 3
    poly = PolynomialFeatures(degree=3)
    X_train3 = poly.fit_transform(X_train)
    X_test3 = poly.fit_transform(X_test)
    
    model = lm.fit(X_train3, y_train)
    print("\nMean squared error for linear regression order 3:")
    print(mean_squared_error(y_test, model.predict(X_test3)))
