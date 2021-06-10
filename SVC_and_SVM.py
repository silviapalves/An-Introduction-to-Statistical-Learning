import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# We'll define a function to draw a nice plot of an SVM
def plot_svc(svc, X, y, c, h=0.02, pad=0.25):
    plt.figure() 
    plt.title('C = ' + str(c))
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='x', s=100, linewidths=1)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
    print('Number of support vectors: ', svc.support_.size)
    
# Generating random data: 20 observations of 2 features and divide into two classes.
np.random.seed(5)
X = np.random.randn(20,2)
y = np.repeat([1,-1], 10)

X[y == -1] = X[y == -1]+1

plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')


# c  small, margins will be wide and many support vectors 
# will be on the margin or will violate the margin

# c large, then the margins will be narrow and there will be few 
# support vectors on the margin or violating the margin.

print("\n-----------------------------------------------\n")
print("-------- Example non-linear separable ---------")
print("\n-----------------------------------------------\n")

print("\n-------- C = 1 --------\n")

C = 1
svc = SVC(C=C, kernel='linear')
svc.fit(X, y)

plot_svc(svc, X, y, C)


print("Indexes of the support vectors")
print(svc.support_)

print("\n-------- C = 0.1 --------\n")

C = 0.1
svc2 = SVC(C=C, kernel='linear')
svc2.fit(X, y)
plot_svc(svc2, X, y, C)
print("Indexes of the support vectors")
print(svc2.support_)

print("\n-------- Cross Validation for C --------\n")

# Select the optimal C parameter by cross-validation
tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}]
clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X, y)

clf_results = clf.cv_results_

print("The best C resulted from the Cross Validation is: ")
print(clf.best_params_)

print("\n-------- Predicting with the best C --------\n")

# only changing the random seed
np.random.seed(1)
X_test = np.random.randn(20,2)
y_test = np.random.choice([-1,1], 20)
X_test[y_test == 1] = X_test[y_test == 1]-1

svc2 = SVC(C=0.001, kernel='linear')
svc2.fit(X, y)
y_pred = svc2.predict(X_test)
ConfMat = pd.DataFrame(confusion_matrix(y_test, y_pred), index=svc2.classes_, columns=svc2.classes_)
print(ConfMat)

print("\n-----------------------------------------------\n")
print("-------- Example linear separable ---------")
print("\n-----------------------------------------------\n")

X_test[y_test == 1] = X_test[y_test == 1]-1
plt.figure()
plt.scatter(X_test[:,0], X_test[:,1], s=70, c=y_test, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')

C =1e5
svc3 = SVC(C=C, kernel='linear')
svc3.fit(X_test, y_test)
plot_svc(svc3, X_test, y_test,C)

C =1
svc4 = SVC(C=C, kernel='linear')
svc4.fit(X_test, y_test)
plot_svc(svc4, X_test, y_test,C)


# C menor aumenta a margem o faz o test performar melhor