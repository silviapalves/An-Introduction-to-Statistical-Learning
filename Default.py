import pandas as pd
import numpy as np
import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

default = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Default.csv")
default.dropna()
default.info()
default.describe()

for i in range(1,11):
    train_default = default.sample(8000, random_state = i)
    test_default = default[~default.isin(train_default)].dropna(how = 'all')
    
    # Fit a logistic regression to predict default using balance
    model = smf.glm('default~balance', data=train_default, family=sm.families.Binomial())
    result = model.fit()
    predictions_nominal = [ "Yes" if x < 0.5 else "No" for x in result.predict(test_default)]
    print("----------------")
    print("Random Seed = " + str(i) + "")
    print("----------------")
    print(confusion_matrix(test_default["default"], 
                       predictions_nominal))
    print(classification_report(test_default["default"], 
                            predictions_nominal, 
                            digits = 3))

    print()
    
print("\n-----------k-Fold Cross-Validation---------------\n")
    

train_default = default.sample(8000, random_state = i)
test_default = default[~default.isin(train_default)].dropna(how = 'all')
model = LogisticRegression()
# result = model.fit()


crossvalidation = KFold(n_splits=5, random_state=1, shuffle=True)

scores = cross_val_score(model, train_default.balance.values.reshape(-1,1), train_default.default.values.reshape(-1,1), cv=crossvalidation)

# print("Degree-"+str(i)+" polynomial MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))
