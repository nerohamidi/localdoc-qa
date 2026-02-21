from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import retrieve, generate_answer
import asyncio

app = FastAPI(title="LocalDocQA")


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    query = request.question

    # Run retrieval in threadpool (CPU-bound embedding work)
    chunks = await asyncio.to_thread(retrieve, query, 3)

    # Run generation in threadpool (blocking HTTP call to Ollama)
    answer = await asyncio.to_thread(generate_answer, query, chunks)

    return {
        "question": query,
        "answer": answer,
        "sources": [
            {
                "document_id": chunk.document_id,
                "chunk_text": chunk.chunk_text
            }
            for chunk in chunks
        ]
    }
