"""
Iris flower data set

https://en.wikipedia.org/wiki/Iris_flower_data_set
"""

from sklearn import tree
from sklearn.datasets import load_iris

iris_dataset = load_iris()
_classifier = tree.DecisionTreeClassifier()

# print the list of labels available for this dataset
print(list(iris_dataset.target_names))

classifier = _classifier.fit(
  iris_dataset.data,
  iris_dataset.target
)

examples = [
  [5.1, 3.5, 1.4, 0.2],
  [5.1, 3.5, 1.4, 0.2],
  [5.1, 3.5, 1.4, 0.2],
  [5.1, 3.5, 1.4, 0.2],
  [5.1, 3.5, 1.4, 1.5],
  [6.5, 3.0, 5.8, 2.2]
]
prediction = classifier.predict(examples)
print(prediction)
# the indices match the labels in `iris_dataset.target_names`
