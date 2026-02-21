# LocalDoc-QA

**A Minimal Retrieval-Augmented Generation (RAG) System Built From Scratch**

---

## Overview

LocalDoc-QA is a locally hosted **Retrieval-Augmented Generation (RAG)** system built using:

- FastAPI  
- PostgreSQL  
- Sentence Transformers  
- SQLAlchemy  
- Manual cosine similarity search  

This project demonstrates practical knowledge in:

- Retrieval-Augmented Generation (RAG)  
- Large Language Model (LLM) workflows  
- Machine Learning embeddings  
- Vector similarity algorithms  
- PostgreSQL database design  
- Backend API development  

The system ingests documents, stores vector embeddings, and retrieves semantically relevant chunks via similarity search.

---

## Architecture

```
User Query
    ↓
FastAPI Endpoint
    ↓
Embed Query (SentenceTransformer)
    ↓
Fetch Stored Embeddings from PostgreSQL
    ↓
Cosine Similarity (Manual Implementation)
    ↓
Top-K Ranking
    ↓
Return Relevant Chunks
```

---

## Features

- Document ingestion pipeline  
- Text chunking with sliding window overlap  
- Transformer-based embedding generation  
- Manual cosine similarity computation  
- PostgreSQL storage of embeddings (`FLOAT8[]`)  
- Ranked semantic retrieval  
- FastAPI search endpoint  
- Fully local deployment (no Docker required)  

---

## Technical Demonstrations

### Retrieval-Augmented Generation (RAG)

This project implements the core RAG pipeline:

1. Convert documents into embeddings  
2. Store embeddings in a database  
3. Convert query into embedding  
4. Compute similarity against stored vectors  
5. Return top-k semantically similar results  

Demonstrates understanding of:

- Dense retrieval  
- Embedding-based search  
- Ranking pipelines  
- Separation of retrieval from generation  

---

## Large Language Model (LLM Concepts)

Although this project focuses on retrieval, it demonstrates:

- Transformer-based embedding usage  
- Sentence-level semantic encoding  
- Integration-ready pipeline for LLM completion  
- Query-context augmentation design  

Structurally ready to plug into an LLM generation layer:

```python
create_answer(query, retrieved_chunks)
```

---

## Machine Learning Concepts

### Embedding Model

Uses:

```
sentence-transformers/all-MiniLM-L6-v2
```

Demonstrates understanding of:

- Transformer encoders  
- Vector representations of text  
- 384-dimensional semantic embeddings  
- Model loading and inference  

### Vector Storage

Embeddings are stored as:

```
FLOAT8[]
```

Demonstrates:

- Numeric array storage  
- Structured ML data persistence  
- Manual similarity evaluation  

---

## Algorithm Implementation

### Sliding Window Chunking

```python
def chunk_text(text, chunk_size=500, overlap=100):
```

Demonstrates:

- Overlapping segmentation  
- Context preservation  
- Controlled memory sizing  

**Time complexity:** `O(n)`

---

### Cosine Similarity

Manually implemented:

```python
np.dot(a, b) / (||a|| * ||b||)
```

Demonstrates:

- Linear algebra fundamentals  
- Vector normalization  
- Similarity scoring  
- Ranking by descending similarity  

**Time complexity per query:**

```
O(N × D)
```

Where:

- `N` = number of stored chunks  
- `D` = embedding dimension (384)  

---

## Database Design (PostgreSQL)

### Schema

#### documents

| Field      | Type       |
|------------|------------|
| id         | SERIAL     |
| title      | TEXT       |
| created_at | TIMESTAMP  |

#### chunks

| Field       | Type      |
|-------------|-----------|
| id          | SERIAL    |
| document_id | INTEGER   |
| chunk_text  | TEXT      |
| embedding   | FLOAT8[]  |

#### queries

| Field       | Type       |
|-------------|------------|
| id          | SERIAL     |
| query_text  | TEXT       |
| created_at  | TIMESTAMP  |

Demonstrates:

- Relational modeling  
- Foreign key constraints  
- Cascading deletes  
- ML feature storage in SQL  

---

## Backend Engineering (FastAPI)

### API Endpoints

#### Health Check

```
GET /
```

#### Semantic Search

```
GET /search?query=...
```

Demonstrates:

- REST API design  
- Dependency injection  
- JSON serialization  
- Local server deployment  

### Run Server

```bash
uvicorn app.api:app --reload
```

### Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## Setup Instructions

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/localdoc-qa.git
cd localdoc-qa
```

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install PostgreSQL (Homebrew)

```bash
brew install postgresql@14
brew services start postgresql@14
```

### Create Database

```bash
psql postgres
CREATE DATABASE localdocqa;
\q
```

### Create Tables

```bash
psql -h localhost -U raguser -d localdocqa
```

Create schema (see SQL above).

### Ingest a Document

```bash
python3 -m scripts.ingest "Test Doc" sample.txt
```

### Run API

```bash
uvicorn app.api:app --reload
```

---

## Performance Characteristics

Current implementation uses:

- Full table scan for similarity  
- In-memory ranking  
- No vector index  

Scales well for:

- Small datasets  
- Prototyping  
- Learning  

Future optimizations:

- pgvector  
- Approximate nearest neighbor (ANN)  
- HNSW indexing  
- Batch retrieval  
- GPU acceleration  

---

## Why This Project Matters

This project demonstrates:

- Understanding of RAG architecture  
- Transformer-based embedding pipelines  
- Linear algebra for similarity search  
- Database-backed ML systems  
- Backend API development  
- Algorithmic reasoning  
- End-to-end ML system design  

It shows capability beyond using frameworks — implementation from first principles.

---

## Future Extensions

- Integrate OpenAI or local LLM generation  
- Add vector indexing (pgvector)  
- Implement hybrid search (BM25 + dense)  
- Add authentication  
- Add document upload endpoint  
- Add caching layer  
- Implement pagination  

---

## Tech Stack

- Python  
- FastAPI  
- PostgreSQL  
- SQLAlchemy  
- Sentence Transformers  
- NumPy  
