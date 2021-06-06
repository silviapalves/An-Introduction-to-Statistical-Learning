import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import preprocessing

caravan = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Caravan.csv", index_col=0, parse_dates=True)
caravan.dropna()

# Diferença nas escalas dos predictors impacta os resultados. 
# Ex: 1000 reais impactariam mais que 50 anos, mas o inverso é que deveria ser verdade
# O mesmo acontece se eu mudo as escalas para yins e segundos, respectivamente
# Logo é necessario normalizar os dados para ter mean = 0 e SD = 1

caravan["Purchase"].value_counts()
print("% to purchase a caravan:")
print(caravan["Purchase"].value_counts()/(caravan.shape[0])*100)

y = caravan.Purchase
X = caravan.drop('Purchase', axis=1).astype('float64')
X_scaled = preprocessing.scale(X)

X_train = X_scaled[1000:,:]
y_train = y[1000:]

X_test = X_scaled[:1000,:]
y_test = y[:1000]

print("\nK = 1\n")

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
pred = knn.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, pred, digits=3))

# 12% de erro parece ser bom mais apenas 6% comprariam a caravan
# mas o que se precisa é apenas reduzir o grupo o qual se vai oferecer o seguro, logo é melhor que randomico

# KNN precisa fazer a matriz transposta?

print(confusion_matrix(y_test, pred).T)