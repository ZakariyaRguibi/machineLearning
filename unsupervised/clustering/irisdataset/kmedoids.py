from sklearn_extra.cluster import KMedoids
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()

X = iris.data
labels_true = iris.target
cobj = KMedoids(n_clusters=3).fit(X)
labels = cobj.labels_

unique_labels = set(labels)
colors = [
    plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
]
plt.figure(figsize=(15, 10))

for k, col in zip(unique_labels, colors):
    condition1 = labels == k
    condition2 = labels_true == 0
    class_member_mask = (condition1 & condition2)
    xy = X[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 3],
        "s",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )
    condition2 = labels_true == 1
    class_member_mask = (condition1 & condition2)
    xy = X[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 3],
        "v",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )
    condition2 = labels_true == 2
    class_member_mask = (condition1 & condition2)
    xy = X[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 3],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )


plt.title("KMedoids clustering. Medoids are represented in cyan.")
plt.show()
colors = ["red", "green", "blue"]

# plot a scatter plot of the data points
for j in range(0, 3):
    plt.figure(figsize=(10, 5))
    plt.title("class number"+str(j))
    for i in range(0, len(X)):
        if labels_true[i] == j:
            plt.plot(X[i][0], X[i][3], "o", color=colors[labels[i]])
    plt.show()
