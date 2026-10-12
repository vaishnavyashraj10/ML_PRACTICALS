# Practical 10: Within-Class, Between-Class and Total Scatter

import numpy as np
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Overall mean
overall_mean = np.mean(X, axis=0)

# Initialize scatter matrices
within_class_scatter = np.zeros((X.shape[1], X.shape[1]))
between_class_scatter = np.zeros((X.shape[1], X.shape[1]))

# Calculate scatter matrices
for class_label in np.unique(y):
    class_data = X[y == class_label]

    class_mean = np.mean(class_data, axis=0)
    class_size = len(class_data)

    # Within-class scatter
    centered_data = class_data - class_mean
    within_class_scatter += centered_data.T @ centered_data

    # Between-class scatter
    mean_difference = (class_mean - overall_mean).reshape(-1, 1)
    between_class_scatter += (
        class_size * mean_difference @ mean_difference.T
    )

# Total scatter
centered_all = X - overall_mean
total_scatter = centered_all.T @ centered_all

print("Within-Class Scatter Matrix:")
print(within_class_scatter)

print("\nBetween-Class Scatter Matrix:")
print(between_class_scatter)

print("\nTotal Scatter Matrix:")
print(total_scatter)

# Check the scatter relationship
print(
    "\nScatter relationship holds:",
    np.allclose(
        total_scatter,
        within_class_scatter + between_class_scatter
    )
)