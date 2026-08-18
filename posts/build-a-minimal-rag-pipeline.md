---
title: "Build a minimal RAG pipeline in 200 lines of Python"
date: 2026-08-18
tags: [rag, embeddings, ai, llm, retrieval-augmented-generation, openai-compatible]
canonical: https://github.com/Rawbeew/flippy
---

# Build a minimal RAG pipeline in 200 lines of Python

Retrieval-augmented generation is the single most useful pattern in
production LLM work. The good news: the bones of it are tiny. You do not
need Pinecone, Weaviate, or any other vector database to ship something
real. You need:

1. A way to **embed** your text.
2. A way to **store** the vectors.
3. A way to **retrieve** the top-k by cosine similarity.
4. A way to **inject** the retrieved text into your LLM prompt.

That is the whole pipeline. Everything else is engineering.

## Step 1 — Embed

Pick any model with an OpenAI-compatible `/v1/embeddings` endpoint. The
`bge-m3` model on freeinference returns 1024-dim vectors and has good
multilingual coverage, which is usually what you want. One POST request,
one vector per string.

```python
import urllib.request, json
def embed(texts):
    req = urllib.request.Request("https://freeinference.org/v1/embeddings",
        data=json.dumps({"model": "bge-m3", "input": texts}).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 Chrome/126.0"}, method="POST")
    return [d["embedding"] for d in json.load(urllib.request.urlopen(req))["data"]]
```

## Step 2 — Store

For anything under a million vectors, a JSON file is fine. SQLite + numpy is
better. Don't reach for a vector DB until you've profiled and seen actual
problems.

## Step 3 — Retrieve

Cosine similarity over numpy arrays. Sub-millisecond at 10k vectors.

```python
import numpy as np
def cosine_search(query_vec, store, top_k=3):
    A = np.array([r["vector"] for r in store])
    q = np.array(query_vec) / np.linalg.norm(query_vec)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    scores = A @ q
    idx = np.argsort(-scores)[:top_k]
    return [(store[i], float(scores[i])) for i in idx]
```

## Step 4 — Inject

A system message at the top of your messages list. That's it.

```python
ctx = "\n\n".join(f"- {r['text']}" for r, _ in hits)
messages = [
    {"role": "system",
     "content": f"Use the following retrieved context to answer. "
                f"If it's irrelevant, say so and answer generally.\n\n{ctx}"},
    {"role": "user", "content": user_query},
]
```

## When you outgrow it

The first thing that breaks is **re-ranking**. Pure cosine similarity over
embeddings is fine for "find documents about X," mediocre for "find the
exact clause that contradicts the user's claim." Adding a re-ranker (often
a smaller LLM call) in front of the final answer is the most reliable
single upgrade you can make.

The second thing that breaks is **chunking strategy**. 500-char chunks with
no overlap is the lazy default; it loses information across chunk
boundaries. Semantic chunking (split on paragraph/sentence boundaries,
respect heading structure) is a one-day improvement that gives you
double-digit accuracy gains for free.

The third thing that breaks is **freshness**. Embeddings don't update
themselves; you have to re-embed when the source changes. A scheduled job
that watches your source folder is the answer.

The full working pipeline, with persistence, retrieval, and re-embed on
demand, is in [`src/aihub.py`](https://github.com/Rawbeew/flippy) (~400
lines, file-backed vector store, ready to swap for a real DB later).
