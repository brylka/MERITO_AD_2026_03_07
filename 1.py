from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

iris = load_iris()
X = iris.data
y = iris.target

dt_classifier = DecisionTreeClassifier(random_state=42)

dt_classifier.fit(X, y)

plt.figure(figsize=(15,10))
tree.plot_tree(dt_classifier, feature_names=iris.feature_names,
               class_names=iris.target_names, filled=True)
plt.show()
