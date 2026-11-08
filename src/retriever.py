from src.embeddings import create_embeddings
from src.vector_store import search_vector_store
import json
import faiss


class Retriever:

    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def retrieve(self, query, k=3):

        query_embedding = create_embeddings([
            {"text": query}
        ])

        distances, indices = search_vector_store(
            self.index,
            query_embedding,
            k=k
        )

        results = []

        for i, chunk_index in enumerate(indices[0]):

            results.append({
                "text": self.chunks[chunk_index]["text"],
                "page": self.chunks[chunk_index]["page"],
                "distance": float(distances[0][i])
            })

        return results

def load_vector_store():

    # Load FAISS index
    index = faiss.read_index(
        "storage/index.faiss"
    )

    # Load chunks
    with open(
        "storage/chunks.json",
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return index, chunks