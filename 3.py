from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                test_size=0.3, random_state=42)


gnb_classifier = GaussianNB()

gnb_classifier.fit(X_train, y_train)

y_pred = gnb_classifier.predict(X_test)

print(f"Dokładność: {accuracy_score(y_test, y_pred)}")
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
