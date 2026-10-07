# Day 01 — Embeddings & Semantic Search

First lab in my **Vector Database & AI Retrieval Infrastructure** journey.

## Objective

Understand how text becomes vectors and how semantic search retrieves similar documents **without using a vector database**.

## What I Learned

- Text embeddings and vector dimensions
- Cosine similarity
- Query vs. document embeddings
- Semantic search
- Top-K retrieval
- Why brute-force vector search does not scale efficiently

## Search Pipeline

```text
Documents → Embeddings → Document Vectors
                           ↑
Query → Embedding → Query Vector
                           ↓
                   Cosine Similarity
                           ↓
                     Rank → Top-K
```

## Tech Stack

- Python
- NumPy
- Sentence Transformers
- `all-MiniLM-L6-v2`

## Run

```bash
pip install sentence-transformers numpy
python semantic_search.py
```

## Key Takeaway

The current implementation compares the query against **every document vector**.

This works for small datasets but becomes expensive at scale.

That leads to the next problem:

> How do we search millions of vectors efficiently?

**Next → kNN, ANN and vector indexing**
