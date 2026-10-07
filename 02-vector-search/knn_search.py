import numpy as np
vectors = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [8, 8],
    [9, 10]
])


#Create a query vector
query = np.array([2,2])

#Use Euclidean Distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

#Calculate every distance
distances = []

for vector in vectors:
    distance = euclidean_distance(query, vector)
    distances.append((distance, vector))

    distances.sort(key=lambda x: x[0])


k = 3

nearest_neighbors = distances[:k]


print("Query:", query)
print(f"\n{k} nearest neighbors:")

for distance, vector in nearest_neighbors:
    print(f"Vector: {vector}, Distance: {distance:.3f}")

