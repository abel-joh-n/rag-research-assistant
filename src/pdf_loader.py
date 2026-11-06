from pypdf import PdfReader


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)

    extracted_pages = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()

        extracted_pages.append({
            "page": page_number + 1,
            "text": text or ""
        })

    return extracted_pages