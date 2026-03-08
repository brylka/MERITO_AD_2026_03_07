from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_blobs, y_true = make_blobs(n_samples=600, centers=2,
                cluster_std=3.6, random_state=42)

plt.figure(figsize=(8,5))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true)
plt.show()