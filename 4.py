from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


data = load_iris()
X, y = data.data, data.target

model = DecisionTreeClassifier()
model.fit(X, y)

#print(data)
# [4.9, 3. , 1.4, 0.2]
# [6.1, 3. , 4.6, 1.4]
# [6.9, 3.1, 5.4, 2.1]

while True:
    sepal_length = float(input("Sepal lenght: "))
    sepal_width =  float(input("Sepal width:  "))
    petal_length = float(input("Petal lenght: "))
    petal_width =  float(input("Petal width:  "))

    cechy = [[sepal_length, sepal_width, petal_length, petal_width]]
    y_pred = model.predict(cechy)[0]

    print(f"Wynik: {data.target_names[y_pred]}")
