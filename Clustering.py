import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


print("----------------------------------------------------------------------")
print("                          k-Means Clustering                          ")
print("----------------------------------------------------------------------")

print("\n-------- K = 2 --------\n")

np.random.seed(123)
X = np.random.randn(50,2)
X[0:25, 0] = X[0:25, 0] + 3
X[0:25, 1] = X[0:25, 1] - 4

f, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:,0], X[:,1], s=50) 
ax.set_xlabel('X0')
ax.set_ylabel('X1')

kmeans = KMeans(n_clusters = 2, random_state = 123).fit(X)

print(kmeans.labels_)

plt.figure(figsize=(6,5))
plt.title("K = 2")
plt.scatter(X[:,0], X[:,1], s = 50, c = kmeans.labels_, cmap = plt.cm.bwr) 
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            marker = '*', 
            s = 150,
            color = 'cyan', 
            label = 'Centers')
plt.legend(loc = 'best')
plt.xlabel('X0')
plt.ylabel('X1')

print("\n-------- K = 3 --------\n")

kmeans_3_clusters = KMeans(n_clusters = 3, random_state = 123)
kmeans_3_clusters.fit(X)

plt.figure(figsize=(6,5))
plt.title("K = 3")
plt.scatter(X[:,0], X[:,1], s=50, c=kmeans_3_clusters.labels_, cmap=plt.cm.prism) 
plt.scatter(kmeans_3_clusters.cluster_centers_[:, 0], kmeans_3_clusters.cluster_centers_[:, 1], marker='*', s=150,
            color='blue', label='Centers')
plt.legend(loc='best')
plt.xlabel('X0')
plt.ylabel('X1')

print("\n-------- Varying the initial random --------\n")

km_out_single_run = KMeans(n_clusters = 3, n_init = 1, random_state = 123).fit(X)

print("n_init = 1: ")
print(km_out_single_run.inertia_)

km_out_single_run = KMeans(n_clusters = 3, n_init = 20, random_state = 123).fit(X)

print("n_init = 20: ")
print(km_out_single_run.inertia_)

km_out_single_run = KMeans(n_clusters = 3, n_init = 50, random_state = 123).fit(X)

print("n_init = 50: ")
print(km_out_single_run.inertia_)