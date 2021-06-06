import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt

smarket = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Smarket.csv", index_col=0, parse_dates=True)
smarket.dropna() 

print("\n----------------------------------------------------------------")
print("\nLogistic Regression\n")

# print(smarket.columns)
# print(smarket.shape)
# print(smarket.describe())
# print(smarket.corr())
# smarket.Volume.plot()


print("\n------------------All lag and data-------------------------\n")
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
# print(predictions[0:10])

print(np.column_stack(smarket[["Direction"]]).flatten(), result.model.endog)
predictions_nominal = [ "Up" if x < 0.5 else "Down" for x in predictions]
print(confusion_matrix(smarket["Direction"], predictions_nominal))
print(classification_report(smarket["Direction"],predictions_nominal,digits = 3)) 


print("\n------------------All lag and training data-------------------------\n")

x_train = smarket[:'2004'][:]
y_train = smarket[:'2004']['Direction']

x_test = smarket['2005':][:]
y_test = smarket['2005':]['Direction']

model_train = smf.glm(formula = formula,data = x_train,family = sm.families.Binomial())
result_train = model_train.fit()

predictions_train = result_train.predict(x_test)
predictions_nominal = [ "Up" if x < 0.5 else "Down" for x in predictions_train]
print(classification_report(y_test, predictions_nominal,digits = 3))

print("\n------------------Lag1 and lag2 and training data-------------------------\n")

formula = 'Direction ~ Lag1+Lag2'
model_train = smf.glm(formula = formula,data = x_train,family = sm.families.Binomial())
result_train = model_train.fit()
                    
predictions_train = result_train.predict(x_test)
predictions_nominal = [ "Up" if x < 0.5 else "Down" for x in predictions_train]
print(classification_report(y_test, predictions_nominal,digits = 3)) 
print(confusion_matrix(y_test, predictions_nominal)) 

print("\n------------------Future data-------------------------\n")

print(result_train.predict(pd.DataFrame([[1.2, 1.1], 
                                   [1.5, -0.8]], 
                                  columns = ["Lag1","Lag2"])))

print("\n----------------------------------------------------------------")
print("\nLinear Discriminant Analysis LDA\n")

X_train = smarket[:'2004'][['Lag1','Lag2']]
y_train = smarket[:'2004']['Direction']

X_test = smarket['2005':][['Lag1','Lag2']]
y_test = smarket['2005':]['Direction']

lda = LinearDiscriminantAnalysis()
model = lda.fit(X_train, y_train)
print("Prior probability to going down and up:")
print(model.priors_)
print("Average of each predictor used as estimate of mik (Lines: Down, UP; Columns: Lag1, Lag2):")
print(model.means_)
print("Coefficients of linear discriminants:")
print(model.coef_)

pred=model.predict(X_test)
print(np.unique(pred, return_counts=True))

print(confusion_matrix(pred, y_test))
print(classification_report(y_test, pred, digits=3))
probs_positive_class = model.predict_proba(X_test)[:, 1]
# say default is the positive class and we want to make few false positives
prediction = probs_positive_class > 0.9
print("Number of downs with 90% of trheshould (posterior probability):")
print(np.sum(prediction))


                             
                            

