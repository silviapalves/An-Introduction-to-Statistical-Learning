import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import graphviz
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error

boston = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Boston.csv")
boston.dropna() 

X = boston.drop('medv', axis = 1)
y = boston.medv
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, random_state = 0)

print("\n--------Regression Tree--------\n")

# Pruning not supported. Choosing max depth 2)
regr_tree_boston = DecisionTreeRegressor(max_depth = 3)
regr_tree_boston.fit(X_train, y_train)
print("Training accurancy: \n"+ str(regr_tree_boston.score(X_train, y_train)*100)+"%")


export_graphviz(regr_tree_boston, 
                out_file = "boston_tree.dot", 
                feature_names = X_train.columns)

with open("boston_tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

pred = regr_tree_boston.predict(X_test)

plt.scatter(pred, 
            y_test, 
            label = 'medv')

plt.plot([0, 1], 
         [0, 1], 
         '--k', 
         transform = plt.gca().transAxes)

plt.xlabel('pred')
plt.ylabel('y_test')


print("\nTest MSE:\n")
print(mean_squared_error(y_test, pred))

print("\n--------Bagging--------\n")

# Bagging: using all features
bagged_boston = RandomForestRegressor(max_features = 13, random_state = 1)
bagged_boston.fit(X_train, y_train)
print("\nTraining MSE:\n")
print(mean_squared_error(y_train, bagged_boston.predict(X_train)))
print("\nTraining accurancy: \n"+ str(bagged_boston.score(X_train, y_train)*100)+"%")

pred = bagged_boston.predict(X_test)

plt.figure() 
plt.scatter(pred, 
            y_test, 
            label = 'medv')
plt.title("Bagging")
plt.plot([0, 1], 
         [0, 1], 
         '--k', 
         transform = plt.gca().transAxes)

plt.xlabel('pred')
plt.ylabel('y_test')

print("\nTest MSE:\n")
print(mean_squared_error(y_test, pred))

print("\n--------Random Forest--------\n")

# Random forests: using 6 features
random_forest_boston = RandomForestRegressor(max_features = 6, random_state = 1)

random_forest_boston.fit(X_train, y_train)

pred = random_forest_boston.predict(X_test)

print("\nTest MSE:\n")
print(mean_squared_error(y_test, pred))

plt.figure()
plt.title("Random Forest") 
plt.scatter(pred, 
            y_test, 
            label = 'medv')

plt.plot([0, 1], 
         [0, 1], 
         '--k', 
         transform = plt.gca().transAxes)

plt.xlabel('pred')
plt.ylabel('y_test')

Importance = pd.DataFrame({'Importance':random_forest_boston.feature_importances_*100}, 
                          index = X.columns)

Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None

print("\n--------Boosting--------\n")

boosted_boston = GradientBoostingRegressor(n_estimators = 200, 
                                           learning_rate = 0.01, 
                                           max_depth = 4, 
                                           random_state = 1)

boosted_boston.fit(X_train, y_train)

feature_importance = boosted_boston.feature_importances_*100

rel_imp = pd.Series(feature_importance, 
                    index = X.columns).sort_values(inplace = False)

rel_imp.T.plot(kind = 'barh', 
               color = 'r', )

plt.xlabel('Variable Importance')

plt.gca().legend_ = None


print("\nTest MSE:\n")
print(mean_squared_error(y_test, boosted_boston.predict(X_test)))

plt.figure() 
plt.title("Boosting") 
plt.scatter(boosted_boston.predict(X_test), 
            y_test, 
            label = 'medv')

plt.plot([0, 1], 
         [0, 1], 
         '--k', 
         transform = plt.gca().transAxes)

plt.xlabel('pred')
plt.ylabel('y_test')

# n_estimators = number of trees
boosted_boston2 = GradientBoostingRegressor(n_estimators = 5000, 
                                            learning_rate = 0.2, 
                                            max_depth = 4, 
                                            random_state = 1)
boosted_boston2.fit(X_train, y_train)

mean_squared_error(y_test, boosted_boston2.predict(X_test))

print("\nTest MSE:\n")
print(mean_squared_error(y_test, boosted_boston2.predict(X_test)))