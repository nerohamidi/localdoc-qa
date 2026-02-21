⸻

LocalDoc-QA

A Minimal Retrieval-Augmented Generation (RAG) System Built From Scratch

Overview

LocalDoc-QA is a locally hosted Retrieval-Augmented Generation (RAG) system built using:
	•	FastAPI
	•	PostgreSQL
	•	Sentence Transformers
	•	SQLAlchemy
	•	Manual cosine similarity search

This project demonstrates practical knowledge in:
	•	Retrieval-Augmented Generation (RAG)
	•	Large Language Model (LLM) workflows
	•	Machine Learning embeddings
	•	Vector similarity algorithms
	•	PostgreSQL database design
	•	Backend API development

The system ingests documents, stores vector embeddings, and retrieves semantically relevant chunks via similarity search.

⸻

Architecture

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


⸻

Features
	•	Document ingestion pipeline
	•	Text chunking with sliding window overlap
	•	Transformer-based embedding generation
	•	Manual cosine similarity computation
	•	PostgreSQL storage of embeddings (FLOAT8[])
	•	Ranked semantic retrieval
	•	FastAPI search endpoint
	•	Fully local deployment (no Docker required)

⸻

Technical Demonstrations

1. Retrieval-Augmented Generation (RAG)

This project implements the core RAG pipeline:
	1.	Convert documents into embeddings
	2.	Store embeddings in a database
	3.	Convert query into embedding
	4.	Compute similarity against stored vectors
	5.	Return top-k semantically similar results

This demonstrates understanding of:
	•	Dense retrieval
	•	Embedding-based search
	•	Ranking pipelines
	•	Separation of retrieval from generation

⸻

2. Large Language Models (LLM Concepts)

Although this project focuses on retrieval, it demonstrates:
	•	Transformer-based embedding usage
	•	Sentence-level semantic encoding
	•	Integration-ready pipeline for LLM completion
	•	Query-context augmentation design

This is structurally ready to plug into an LLM generation layer such as:

create_answer(query, retrieved_chunks)


⸻

3. Machine Learning Concepts

Embedding Model

Uses:

sentence-transformers/all-MiniLM-L6-v2

Demonstrates understanding of:
	•	Transformer encoders
	•	Vector representations of text
	•	384-dimensional semantic embeddings
	•	Model loading and inference

Vector Storage

Embeddings are stored as:

FLOAT8[]

This demonstrates:
	•	Numeric array storage
	•	Structured ML data persistence
	•	Manual similarity evaluation

⸻

4. Algorithm Implementation

Sliding Window Chunking

def chunk_text(text, chunk_size=500, overlap=100):

Demonstrates:
	•	Overlapping segmentation
	•	Context preservation
	•	Controlled memory sizing

Time complexity: O(n)

⸻

Cosine Similarity

Manually implemented:

np.dot(a, b) / (||a|| * ||b||)

Demonstrates:
	•	Linear algebra fundamentals
	•	Vector normalization
	•	Similarity scoring
	•	Ranking by descending similarity

Time complexity per query:

O(N × D)

Where:
	•	N = number of stored chunks
	•	D = embedding dimension (384)

⸻

5. Database Design (PostgreSQL)

Schema

documents

Field	Type
id	SERIAL
title	TEXT
created_at	TIMESTAMP

chunks

Field	Type
id	SERIAL
document_id	INTEGER
chunk_text	TEXT
embedding	FLOAT8[]

queries

Field	Type
id	SERIAL
query_text	TEXT
created_at	TIMESTAMP

Demonstrates:
	•	Relational modeling
	•	Foreign key constraints
	•	Cascading deletes
	•	ML feature storage in SQL

⸻

6. Backend Engineering (FastAPI)

API Endpoints:

Health Check

GET /

Semantic Search

GET /search?query=...

Demonstrates:
	•	REST API design
	•	Dependency injection
	•	JSON serialization
	•	Local server deployment

Run server:

uvicorn app.api:app --reload

Swagger UI:

http://127.0.0.1:8000/docs


⸻

Setup Instructions

1. Clone Repository

git clone https://github.com/YOUR_USERNAME/localdoc-qa.git
cd localdoc-qa


⸻

2. Create Virtual Environment

python3 -m venv venv
source venv/bin/activate


⸻

3. Install Dependencies

pip install -r requirements.txt


⸻

4. Install PostgreSQL (Homebrew)

brew install postgresql@14
brew services start postgresql@14


⸻

5. Create Database

psql postgres
CREATE DATABASE localdocqa;
\q


⸻

6. Create Tables

psql -h localhost -U raguser -d localdocqa

Create schema (see SQL above).

⸻

7. Ingest a Document

python3 -m scripts.ingest "Test Doc" sample.txt


⸻

8. Run API

uvicorn app.api:app --reload


⸻

Performance Characteristics

Current implementation uses:
	•	Full table scan for similarity
	•	In-memory ranking
	•	No vector index

Scales well for:
	•	Small datasets
	•	Prototyping
	•	Learning

Future optimizations:
	•	pgvector
	•	Approximate nearest neighbor (ANN)
	•	HNSW indexing
	•	Batch retrieval
	•	GPU acceleration

⸻

Why This Project Matters

This project demonstrates:
	•	Understanding of RAG architecture
	•	Transformer-based embedding pipelines
	•	Linear algebra for similarity search
	•	Database-backed ML systems
	•	Backend API development
	•	Algorithmic reasoning
	•	End-to-end ML system design

It shows capability beyond using frameworks — it demonstrates implementation from first principles.

⸻

Future Extensions
	•	Integrate OpenAI or local LLM generation
	•	Add vector indexing (pgvector)
	•	Implement hybrid search (BM25 + dense)
	•	Add authentication
	•	Add document upload endpoint
	•	Add caching layer
	•	Implement pagination

⸻

Tech Stack
	•	Python
	•	FastAPI
	•	PostgreSQL
	•	SQLAlchemy
	•	Sentence Transformers
	•	NumPy

⸻

