
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(extracted_pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    for page in extracted_pages:

        chunks = splitter.split_text(page["text"])

        for chunk in chunks:

            all_chunks.append({
                "page": page["page"],
                "text": chunk
            })

    return all_chunks
