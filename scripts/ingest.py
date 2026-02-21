import sys
from app.database import SessionLocal
from app.models import Document, Chunk
from app.rag import chunk_text, embed_text


def ingest_document(title: str, content: str):
    db = SessionLocal()

    # Create document record
    document = Document(title=title)
    db.add(document)
    db.commit()
    db.refresh(document)

    # Chunk the text
    chunks = chunk_text(content)

    for chunk in chunks:
        embedding = embed_text(chunk)

        chunk_record = Chunk(
            document_id=document.id,
            chunk_text=chunk,
            embedding=embedding
        )

        db.add(chunk_record)

    db.commit()
    db.close()

    print(f"Ingested {len(chunks)} chunks.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m scripts.ingest 'Title' path_to_text_file")
        sys.exit(1)

    title = sys.argv[1]
    file_path = sys.argv[2]

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    ingest_document(title, content)
