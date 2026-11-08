import os

from src.pdf_loader import extract_text
from src.text_splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import (
    create_vector_store,
    save_vector_store
)


def ingest_document(pdf_path):

    # Extract text
    pages = extract_text(pdf_path)

    # Split into chunks
    chunks = split_text(pages)

    # Create embeddings
    embeddings = create_embeddings(chunks)

    # Create FAISS index
    index = create_vector_store(
        embeddings
    )

    # Create a unique folder for this document
    document_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    storage_path = os.path.join(
        "storage",
        document_name
    )

    # Save index and chunks
    save_vector_store(
        index,
        chunks,
        storage_path
    )

    return storage_path