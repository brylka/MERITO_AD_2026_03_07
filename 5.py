from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

X_blobs, y_true = make_blobs(n_samples=400, centers=4,
                cluster_std=1, random_state=40)

plt.figure(figsize=(8,5))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true)
plt.show()

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, max_iter=300, random_state=42)
    kmeans.fit(X_blobs)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_blobs, kmeans.labels_))

# print(inertias)
# print(silhouette_scores)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertias, 'ro-')
plt.xlabel('Liczba klastrów')
plt.ylabel('Inercja (suma kwadratów odległości)')
plt.title('Metoda łokcia')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Liczka klastrów')
plt.ylabel('Współczynnik sylwetkowy')
plt.title('Współczynnik sylwetkowy dla różnych k')
plt.grid(True)
plt.show()

kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)

y_kmeans = kmeans.fit_predict(X_blobs)

plt.figure(figsize=(8,5))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='X', c='red', s=100)
plt.show()


# print(y_true, '\n', y_kmeans)