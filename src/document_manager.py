import os


DATA_DIR = "data"
STORAGE_DIR = "storage"


def get_documents():
    """Return all uploaded PDF filenames."""

    if not os.path.exists(DATA_DIR):
        return []

    documents = []

    for filename in os.listdir(DATA_DIR):

        if filename.lower().endswith(".pdf"):
            documents.append(filename)

    return sorted(documents)


def get_document_storage_path(filename):
    """Return the storage folder for a document."""

    document_name = os.path.splitext(
        filename
    )[0]

    return os.path.join(
        STORAGE_DIR,
        document_name
    )


def document_is_ingested(filename):
    """Check whether a document already has a FAISS index."""

    storage_path = get_document_storage_path(
        filename
    )

    index_path = os.path.join(
        storage_path,
        "index.faiss"
    )

    chunks_path = os.path.join(
        storage_path,
        "chunks.json"
    )

    return (
        os.path.exists(index_path)
        and
        os.path.exists(chunks_path)
    )