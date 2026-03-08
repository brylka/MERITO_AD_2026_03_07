from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, 2:]
y_true = iris.target

print(X)

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Prawdziwe etykiety
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.xlabel('Długość działki kielicha')
plt.ylabel('Szerokość działki kielicha')
plt.title('Prawdziwe gatunki irysów')
plt.show()

# Klasteryzacja K-means
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Centroidy')
plt.xlabel('Długość działki kielicha')
plt.ylabel('Szerokość działki kielicha')
plt.title('K-means na irysach')
plt.legend()
plt.show()