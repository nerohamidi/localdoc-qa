# LocalDoc-QA

**A Production-Ready Retrieval-Augmented Generation (RAG) System**

---

## Overview

LocalDoc-QA is a fully local **RAG system** built with:

- FastAPI  
- PostgreSQL 17  
- pgvector (HNSW index)  
- Sentence Transformers  
- SQLAlchemy  
- Ollama (local LLM)

Documents are embedded, stored as `vector(384)`, and retrieved using indexed cosine similarity directly inside PostgreSQL.

---

## Architecture

```
User Query
    ↓
FastAPI
    ↓
Embed (MiniLM, 384-dim)
    ↓
PostgreSQL (pgvector)
    ↓
HNSW ANN Search
    ↓
Top-K Chunks
    ↓
LLM (Ollama)
```

---

## Key Features

- Sliding window document chunking  
- 384-dimensional transformer embeddings  
- Native `vector(384)` storage  
- Cosine similarity via `<=>` operator  
- HNSW approximate nearest neighbor index  
- Database-level ranking (no Python full scan)  
- Local LLM generation via Ollama  
- Fully local deployment  

---

## Embedding Model

```
sentence-transformers/all-MiniLM-L6-v2
```

- 384-dimensional semantic vectors  
- Stored using pgvector  
- Indexed with HNSW  

---

## Database Schema

### documents

| Field | Type |
|-------|------|
| id | SERIAL |
| title | TEXT |

### chunks

| Field | Type |
|-------|------|
| id | SERIAL |
| document_id | INTEGER |
| chunk_text | TEXT |
| embedding | vector(384) |

### queries

| Field | Type |
|-------|------|
| id | SERIAL |
| query_text | TEXT |

---

## Retrieval Implementation

Similarity search runs entirely in PostgreSQL:

```python
db.query(Chunk) \
  .order_by(Chunk.embedding.cosine_distance(query_embedding)) \
  .limit(top_k)
```

Indexed using:

```sql
CREATE INDEX chunks_embedding_hnsw
ON chunks
USING hnsw (embedding vector_cosine_ops);
```

This enables:

- Approximate nearest neighbor search  
- Sublinear scaling  
- Millisecond retrieval  

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install PostgreSQL 17

```bash
brew install postgresql@17
brew services start postgresql@17
```

### 3. Create Database

```bash
psql postgres
CREATE DATABASE localdocqa;
```

### 4. Enable pgvector

```sql
\c localdocqa
CREATE EXTENSION vector;
```

### 5. Create Tables

```bash
python
```

```python
from app.database import engine
from app.models import Base
Base.metadata.create_all(bind=engine)
```

### 6. Ingest a Document

```bash
python3 -m scripts.ingest "Test Doc" sample.txt
```

### 7. Run API

```bash
uvicorn app.api:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## Performance

**Before (manual cosine):**

- Full table scan  
- O(N × D) complexity  
- Python-side ranking  

**Now (pgvector + HNSW):**

- Indexed ANN search  
- Database-level cosine distance  
- Sublinear scaling  
- Millisecond retrieval  

---

## Tech Stack

- Python  
- FastAPI  
- PostgreSQL 17  
- pgvector  
- SQLAlchemy  
- Sentence Transformers  
- Ollama  

---

## Future Extensions

- Hybrid search (BM25 + vector)  
- Re-ranking layer  
- Async database queries  
- Pagination  
- Authentication  
- Document upload endpoint  
