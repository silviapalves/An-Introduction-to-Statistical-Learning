import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


smarket = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Smarket.csv", index_col=0, parse_dates=True)
smarket.dropna() 



print(smarket.columns)
print(smarket.shape)
print(smarket.describe())
print(smarket.corr())

smarket.Volume.plot()

formula = 'Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume'
model = smf.glm(formula = formula, data=smarket, family=sm.families.Binomial())
result = model.fit()
print(result.summary())
print("Coefficeients")
print(result.params)
print()
print("p-Values")
print(result.pvalues)
print()
print("Dependent variables")
print(result.model.endog_names)

predictions = result.predict()
print(predictions[0:10])

print(np.column_stack(smarket[["Direction"]]).flatten(), result.model.endog)
predictions_nominal = [ "Up" if x < 0.5 else "Down" for x in predictions]
print(confusion_matrix(smarket["Direction"], predictions_nominal))
print(classification_report(smarket["Direction"],predictions_nominal,digits = 3)) 

x_train = smarket[:'2004'][:]
y_train = smarket[:'2004']['Direction']

x_test = smarket['2005':][:]
y_test = smarket['2005':]['Direction']

model_train = smf.glm(formula = formula,data = x_train,family = sm.families.Binomial())
result_train = model_train.fit()

predictions_train = result_train.predict(x_test)
predictions_nominal = [ "Up" if x < 0.5 else "Down" for x in predictions_train]
print(classification_report(y_test, predictions_nominal,digits = 3))

formula = 'Direction ~ Lag1+Lag2'
model_train = smf.glm(formula = formula,data = x_train,family = sm.families.Binomial())
result_train = model_train.fit()
                    
predictions_train = result_train.predict(x_test)
predictions_nominal = [ "Up" if x < 0.5 else "Down" for x in predictions_train]
print(classification_report(y_test, predictions_nominal,digits = 3)) 
print(confusion_matrix(y_test, predictions_nominal)) 

print(result_train.predict(pd.DataFrame([[1.2, 1.1], 
                                   [1.5, -0.8]], 
                                  columns = ["Lag1","Lag2"])))




                             
                            

