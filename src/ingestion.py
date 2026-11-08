import os
import json
import faiss

from src.pdf_loader import extract_text
from src.text_splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store


def ingest_document(pdf_path):

    # Extract text
    pages = extract_text(pdf_path)

    # Split text into chunks
    chunks = split_text(pages)

    # Create embeddings
    embeddings = create_embeddings(chunks)

    # Create FAISS index
    index = create_vector_store(embeddings)

    # Create storage folder
    os.makedirs("storage", exist_ok=True)

    # Save FAISS index
    faiss.write_index(
        index,
        "storage/index.faiss"
    )

    # Save chunks
    with open("storage/chunks.json", "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print("Document ingestion complete.")
    print("Total pages:", len(pages))
    print("Total chunks:", len(chunks))
    print("FAISS vectors:", index.ntotal)