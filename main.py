from src.retriever import Retriever, load_vector_store
from src.rag_pipeline import RAGPipeline


# Load saved vector store
index, chunks = load_vector_store()

print("FAISS vectors:", index.ntotal)
print("Loaded chunks:", len(chunks))


# Create retriever
retriever = Retriever(
    index,
    chunks
)


# Create RAG pipeline
pipeline = RAGPipeline(
    retriever
)


print("\n==============================")
print("RAG RESEARCH ASSISTANT")
print("==============================")

print("Ask questions about the document.")
print("Type 'exit' to quit.")


while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        print("\nGoodbye!")
        break

    if not query.strip():
        print("Please enter a question.")
        continue

    answer, sources = pipeline.ask(
        query,
        k=3
    )

    print("\n==============================")
    print("ANSWER")
    print("==============================")

    print(answer)

    print("\n==============================")
    print("SOURCES")
    print("==============================")

    for i, source in enumerate(sources):

        print(
            f"Result {i + 1}: "
            f"Page {source['page']}"
        )