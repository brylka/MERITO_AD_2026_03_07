from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    # print(f"{k}: {accuracy_score(y_test, y_pred)}")
    scores.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10,5))
plt.plot(k_range, scores, marker='o')
plt.title('Dokładność KNN dla różnych wartości k')
plt.xlabel('Wartość k')
plt.ylabel('Dokładność')
plt.grid(True)
plt.show()

knn_classifier = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance',     # 'uniform'
    metric='euclidean'      # 'manhattan', 'minkowski'
)

knn_classifier.fit(X_train_scaled, y_train)

y_pred = knn_classifier.predict(X_test_scaled)

print(f"Dokładność: {accuracy_score(y_test, y_pred)}")
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))



