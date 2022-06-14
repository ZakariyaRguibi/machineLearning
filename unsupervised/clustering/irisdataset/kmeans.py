from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('Iris.csv')
unique_data = data['Id'].unique()
# print(data.shape)
# print(data.head)


X = data


y = data['Species']


le = LabelEncoder()
X['Species'] = le.fit_transform(X['Species'])
y = le.transform(y)

X.head()


kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(X)

kmeans.cluster_centers_

dataset2 = pd.read_csv("Iris.csv")

X = dataset2.loc[:, ['SepalLengthCm', 'PetalWidthCm']]


km = KMeans(n_clusters=3)
km.fit(X)


plt.figure(figsize=(10, 5))
plt.scatter(x=X.iloc[:, 0], y=X.iloc[:, 1], c=km.labels_)


plt.scatter(x=X.iloc[:, 0], y=X.iloc[:, 1], c=km.labels_)
plt.xlabel('SepalLengthCm')
plt.ylabel('PetalWidthCm')


plt.show()
