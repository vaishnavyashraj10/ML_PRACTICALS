# Practical 3: Dimensionality Reduction using PCA

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
iris = load_iris()
X = iris.data

print("Original number of features:", X.shape[1])

# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# Reduce 4 features to 2
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print("Reduced number of features:", X_reduced.shape[1])
print("Reduced dataset shape:", X_reduced.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)