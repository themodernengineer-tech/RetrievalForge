from sentence_transformers import SentenceTransformer
import numpy as np


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Our tiny document database
documents = [
    "Python is commonly used for machine learning",
    "Football teams have eleven players",
    "Neural networks learn patterns from data",
    "Paris is the capital of France",
    "Transformers are widely used in modern AI",
]


# Convert documents into vectors
document_vectors = model.encode(documents)


# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# User query
query = "How are neural networks used in AI?"


# Convert query into vector
query_vector = model.encode(query)


# Compare query against every document
scores = []

for document, vector in zip(documents, document_vectors):
    score = cosine_similarity(query_vector, vector)
    scores.append((score, document))


# Rank highest similarity first
scores.sort(reverse=True)


# Return Top-K results
top_k = 3

print(f"\nQuery: {query}")
print(f"\nTop {top_k} results:\n")

for score, document in scores[:top_k]:
    print(f"{score:.3f} | {document}")