# Practical 8: Decision Tree Classifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a flower
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)[0]

print("Predicted Flower:", iris.target_names[prediction])