
from src.pdf_loader import extract_text
from src.text_splitter import split_text


PDF_PATH = "data/sample.pdf"


# Step 1: Extract PDF text
pages = extract_text(PDF_PATH)

print("Total pages:", len(pages))


# Step 2: Split extracted text
chunks = split_text(pages)

print("Total chunks:", len(chunks))


# Step 3: Display sample chunks
for i, chunk in enumerate(chunks[:5]):

    print(f"\nCHUNK {i + 1}")
    print("Page:", chunk["page"])
    print("Content:", chunk["text"])
    print("-" * 50)
