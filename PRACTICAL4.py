# Practical 4: Similarity and Dissimilarity Measures

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Two data vectors
A = np.array([[1, 2, 3]])
B = np.array([[2, 3, 4]])

# Euclidean distance
distance = euclidean_distances(A, B)[0][0]

# Cosine similarity
similarity = cosine_similarity(A, B)[0][0]

print("Vector A:", A[0])
print("Vector B:", B[0])
print("Euclidean Distance:", round(distance, 4))
print("Cosine Similarity:", round(similarity, 4))