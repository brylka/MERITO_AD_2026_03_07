from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


data = load_iris()
X, y = data.data, data.target

model = DecisionTreeClassifier()
model.fit(X, y)

# print(data)

sepal_length = 6
sepal_width = 3
petal_length = 4
petal_width = 1.5

cechy = [[sepal_length, sepal_width, petal_length, petal_width]]
y_pred = model.predict(cechy)[0]

print(f"Wynik: {data.target_names[y_pred]}")





