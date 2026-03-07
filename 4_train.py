import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


data = load_iris()
X, y = data.data, data.target
print("Wczytano dane...")

model = DecisionTreeClassifier()
model.fit(X, y)
print("Wytrenowano model...")

joblib.dump(model, 'model.joblib')
print("Zapisano model...")