import faiss
import json
import os
import numpy as np


def create_vector_store(embeddings):
    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


def search_vector_store(
    index,
    query_embedding,
    k=3
):
    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    return distances, indices


def save_vector_store(
    index,
    chunks,
    storage_path
):

    os.makedirs(
        storage_path,
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        index,
        os.path.join(
            storage_path,
            "index.faiss"
        )
    )

    # Save chunk metadata
    with open(
        os.path.join(
            storage_path,
            "chunks.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


def load_vector_store(
    storage_path
):

    index = faiss.read_index(
        os.path.join(
            storage_path,
            "index.faiss"
        )
    )

    with open(
        os.path.join(
            storage_path,
            "chunks.json"
        ),
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return index, chunks