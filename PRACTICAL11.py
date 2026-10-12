# Practical 11: K-Means Clustering

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
X = iris.data

# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# Create K-Means model with 3 clusters
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Perform clustering
clusters = model.fit_predict(X_scaled)

print("First 10 Cluster Labels:")
print(clusters[:10])

print("\nNumber of Clusters:", len(set(clusters)))
print("Cluster Centers:")
print(model.cluster_centers_)

print("\nInertia:", round(model.inertia_, 4))