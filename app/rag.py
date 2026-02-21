import requests
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model once (global)
model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Split text into overlapping chunks.
    Demonstrates algorithmic thinking (sliding window).
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def embed_text(text: str):
    """
    Convert text to embedding vector.
    ML usage: transformer-based embeddings.
    """
    embedding = model.encode(text)
    return embedding.tolist()


def cosine_similarity(a, b):
    """
    Manual similarity calculation.
    Demonstrates vector math understanding.
    """
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


from app.database import SessionLocal
from app.models import Chunk

def retrieve(query: str, top_k: int = 3):
    """
    Retrieve top-k most similar chunks for a query.
    Demonstrates similarity search + ranking.
    """

    db = SessionLocal()

    # 1. Embed query
    query_embedding = embed_text(query)

    # 2. Load all chunks
    chunks = db.query(Chunk).all()

    scored_chunks = []

    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk.embedding)
        scored_chunks.append((score, chunk))

    db.close()

    # 3. Sort descending by similarity
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # 4. Return top-k chunks
    top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]

    return top_chunks

def generate_answer(query: str, chunks: list):
    """
    Generate answer using local Ollama LLM.
    """

    context = "\n\n".join(
        [f"Source {i+1}:\n{chunk.chunk_text}" for i, chunk in enumerate(chunks)]
    )

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided sources to answer the question.
If the answer is not in the sources, say you do not know.

Sources:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]
