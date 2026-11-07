from src.pdf_loader import extract_text
from src.text_splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.vector_store import create_vector_store, search_vector_store

PDF_PATH = "data/sample.pdf"

# Step 1: Extract text
pages = extract_text(PDF_PATH)
print("Total pages:", len(pages))

# Step 2: Split into chunks
chunks = split_text(pages)
print("Total chunks:", len(chunks))

# Step 3: Create embeddings
embeddings = create_embeddings(chunks)

# step 4: Create FAISS vector store
index = create_vector_store(embeddings)
print("FAISS vectors:", index.ntotal)

# Step 5: Search for relevant chunks
query = "What is the BB84 protocol?"

query_embedding = create_embeddings([
    {"text": query}
])

distances, indices = search_vector_store(
    index,
    query_embedding,
    k=3
)

print("\nTop 3 relevant chunks:")

for i, chunk_index in enumerate(indices[0]):
    print(f"\nRESULT {i + 1}")
    print("Distance:", distances[0][i])
    print("Page:", chunks[chunk_index]["page"])
    print("Content:", chunks[chunk_index]["text"])



print("Embedding shape:", embeddings.shape)
# print("First embedding:")
# print(embeddings[0])