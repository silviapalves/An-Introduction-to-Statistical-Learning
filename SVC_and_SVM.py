import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


print("----------------------------------------------------------------------")
print("                          Linear Kernel                               ")
print("----------------------------------------------------------------------")

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

print("----------------------------------------------------------------------")
print("                          Polinomial Kernel                           ")
print("----------------------------------------------------------------------")

# for polinomial:specify a degree for the polynomial kernel (d)
# for radial: specify a value of  γ  for the radial basis kernel.
   
np.random.seed(8)
X = np.random.randn(200,2)
X[:100] = X[:100] +2
X[101:150] = X[101:150] -2
y = np.concatenate([np.repeat(-1, 150), np.repeat(1,50)])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2)


plt.figure()
plt.title("Data plot")
plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')

print("\n-------- Gama = 1, C = 1 --------\n")

C = 1
svm = SVC(C=C, kernel='rbf', gamma=1)
svm.fit(X_train, y_train)
plot_svc(svm, X_test, y_test,C)

print("\n-------- Gama = 1, C = 100 --------\n")

# Increasing C parameter, allowing more flexibility
C=100
svm2 = SVC(C=C, kernel='rbf', gamma=1.0)
svm2.fit(X_train, y_train)
plot_svc(svm2, X_test, y_test,C)

print("\n-------- Cross Validation for X and gama --------\n")

tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100],
                     'gamma': [0.5, 1,2,3,4]}]
clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X_train, y_train)

print("The best C resulted from the Cross Validation is: ")
print(clf.best_params_)

C= clf.best_params_["C"]
plot_svc(clf.best_estimator_, X_test, y_test, C)
print(confusion_matrix(y_test, clf.best_estimator_.predict(X_test)))
print(clf.best_estimator_.score(X_test, y_test))

print("\n-------- ROC Curves --------\n")

# ROC Curve: false positive rate versus true positive rate

# More constrained model

C=1
svm3 = SVC(C=C, kernel='rbf', gamma=1)
svm3.fit(X_train, y_train)

# More flexible model
svm4 = SVC(C=C, kernel='rbf', gamma=50)
svm4.fit(X_train, y_train)

# tem como se obter o valor fittado de cada input (aquele que determina a distancia do plano)
# e que se for positivo mostra que esta de um lado e negativo de outro

# if the fitted value exceeds zero then the observation is assigned to one class, 
# and if it is less than zero than it is assigned to the other.

# For training data

y_train_score3 = svm3.decision_function(X_train)
y_train_score4 = svm4.decision_function(X_train)

false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,5))
ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
ax1.set_title('Training Data')

# For test data

y_test_score3 = svm3.decision_function(X_test)
y_test_score4 = svm4.decision_function(X_test)

false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
ax2.set_title('Test Data')

for ax in fig.axes:
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    
# quanto mais acima e a esquerda melhor

print("----------------------------------------------------------------------")
print("                   Radial Kernel with 3 classes                       ")
print("----------------------------------------------------------------------")

np.random.seed(8)
XX = np.vstack([X, np.random.randn(50,2)])
yy = np.hstack([y, np.repeat(0,50)])
XX[yy ==0] = XX[yy == 0] +4

plt.figure()
plt.scatter(XX[:,0], XX[:,1], s=70, c=yy, cmap=plt.cm.prism)
plt.xlabel('XX1')
plt.ylabel('XX2')
C=1
svm5 = SVC(C=C, kernel='rbf')
svm5.fit(XX, yy)
plot_svc(svm5, XX, yy, C)

