from src.pdf_loader import extract_text
from src.text_splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, search_vector_store
from src.gemini import generate_answer


PDF_PATH = "data/sample.pdf"

# 1. Extract PDF
pages = extract_text(PDF_PATH)
print("Total pages:", len(pages))

# 2. Split into chunks
chunks = split_text(pages)
print("Total chunks:", len(chunks))

# 3. Create embeddings
embeddings = create_embeddings(chunks)

# 4. Create FAISS index
index = create_vector_store(embeddings)
print("FAISS vectors:", index.ntotal)

# 5. User question
query = "What is the BB84 protocol?"

# 6. Embed the question
query_embedding = create_embeddings([
    {"text": query}
])

# 7. Search FAISS
distances, indices = search_vector_store(
    index,
    query_embedding,
    k=3
)

# 8. Retrieve relevant chunks
retrieved_chunks = []

for chunk_index in indices[0]:
    retrieved_chunks.append(
        chunks[chunk_index]["text"]
    )

# 9. Combine retrieved chunks
context = "\n\n".join(retrieved_chunks)

# 10. Send context + question to Gemini
answer = generate_answer(
    query,
    context
)

# 11. Display answer
print("\n==============================")
print("QUESTION")
print("==============================")
print(query)

print("\n==============================")
print("ANSWER")
print("==============================")
print(answer)

print("\n==============================")
print("SOURCES")
print("==============================")

for i, chunk_index in enumerate(indices[0]):
    print(
        f"Result {i + 1}: "
        f"Page {chunks[chunk_index]['page']}"
    )