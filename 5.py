from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X_blobs, y_true = make_blobs(n_samples=600, centers=4,
                cluster_std=0.6, random_state=42)

plt.figure(figsize=(8,5))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true)
plt.show()

kmeans = KMeans(
    n_clusters=4,
    max_iter=300,
    random_state=42
)

y_kmeans = kmeans.fit_predict(X_blobs)

print(y_true, '\n', y_kmeans)