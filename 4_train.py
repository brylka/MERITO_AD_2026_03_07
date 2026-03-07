import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# pamiętaj przy KNN należy także przeskalować dane - więc zapis/odczyt scalar


data = load_iris()
X, y = data.data, data.target
print("Wczytano dane...")

model = DecisionTreeClassifier()
model.fit(X, y)
print("Wytrenowano model...")

joblib.dump(model, 'model.joblib')
print("Zapisano model...")