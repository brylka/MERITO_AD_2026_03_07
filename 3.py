from sklearn.datasets import load_iris
from sklearn.metrics import auc, accuracy_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                test_size=0.3, random_state=42)


gnb_classifier = GaussianNB()

gnb_classifier.fit(X_train, y_train)

y_pred = gnb_classifier.predict(X_test)
y_proba = gnb_classifier.predict_proba(X_test)

# print(y_test)
# print(y_pred)
# print(y_proba)

print(f"Dokładność: {accuracy_score(y_test, y_pred)}")
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

y_test_bin = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test_bin.shape[1]



plt.figure(figsize=(10,5))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{iris.target_names[i]}")


plt.plot([0,1], [0,1], 'k--', label="Klasyfikator losowy")
plt.title('Krzywa ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
