# Lokal-Intern-Assignment
# Support Bot

This Python script implements a Retrieval-Augmented Generation (RAG) based support assistant that leverages FAISS for similarity search and a transformer-based language model to answer user queries using previously resolved support tickets.

---

## Features

-  Loads and processes resolved support tickets from a file.
-  Encodes ticket "problem" fields into vector embeddings using SentenceTransformer.
-  Uses FAISS for fast similarity-based retrieval of relevant tickets.
-  Generates support responses using a local language model (DistilGPT2).
-  Filters and formats final answers for clarity and accuracy.

---

## Requirements
- python
- faiss-cpu
- numpy
- transformers
- sentence-trasformers
---
## How It Works

The support bot uses a Retrieval-Augmented Generation (RAG) architecture with the following steps:

### 1. Load Resolved Tickets
- The bot reads past resolved support tickets from a file called `resolved_tickets.txt`.
- Each line in the file should be a JSON object with `problem` and `resolution` keys.

### 2. Embedding
- The `problem` field from each ticket is converted into a dense vector embedding using the all-MiniLM-L6-v2 model from sentence-transformers.
- These embeddings capture semantic meaning of the issues.

### 3. Build FAISS Index
- A FAISS (Facebook AI Similarity Search) index is created using the embeddings.
- This allows fast retrieval of tickets with problems similar to a new user query.

### 4. Retrieve Top Matches
- When the user enters a new query, it is also embedded into a vector.
- The FAISS index is searched for the top k most similar past problems.
- The corresponding `resolution`s are extracted and used as context.

### 5. Generate Answer
- A prompt is constructed using the retrieved resolutions and the user's query.
- This prompt is passed to a local language model (distilgpt2) using the Hugging Face pipeline for text generation.
- The model generates a concise resolution, strictly based on the past examples.

### 6. Post-Processing
- The response is cleaned and truncated to keep it short and relevant.
- If a retrieved ticket has very high similarity (cosine similarity > 0.7) to the user query, its resolution is returned directly for maximum confidence.

### 7. Fallback Handling
- If the model fails to generate a meaningful answer or the context lacks relevant information, the bot responds with a clear fallback message:
  
  “I am sorry, I cannot provide a specific resolution based on the available past tickets for this issue.”
----
# How to run
`python RAG.py`

