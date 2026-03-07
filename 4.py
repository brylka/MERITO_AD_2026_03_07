from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


data = load_iris()
X, y = data.data, data.target

model = DecisionTreeClassifier()

model.fit(X, y)

y_pred = model.predict(...)





