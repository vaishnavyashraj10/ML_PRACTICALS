# Practical 12: Demonstration of NumPy, Pandas,
# Scikit-learn and TensorFlow

import numpy as np
import pandas as pd
import sklearn

print("1. NumPy Demonstration")

numbers = np.array([10, 20, 30, 40, 50])

print("Array:", numbers)
print("Mean:", np.mean(numbers))
print("Sum:", np.sum(numbers))


print("\n2. Pandas Demonstration")

data = {
    "Name": ["Amit", "Riya", "Raj"],
    "Marks": [80, 90, 75]
}

df = pd.DataFrame(data)

print(df)
print("Average Marks:", df["Marks"].mean())


print("\n3. Scikit-learn Demonstration")

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

print("Scikit-learn Version:", sklearn.__version__)
print("Predicted Flower:", iris.target_names[prediction[0]])


print("\n4. TensorFlow Demonstration")

try:
    import tensorflow as tf

    print("TensorFlow Version:", tf.__version__)

    tensor = tf.constant([10, 20, 30])

    print("Tensor:", tensor.numpy())
    print("Tensor Sum:", tf.reduce_sum(tensor).numpy())

except ImportError:
    print("TensorFlow is not installed.")
    print("NumPy, Pandas and Scikit-learn demonstrations completed.")