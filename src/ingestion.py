"""
ingestion.py
------------
Responsible for:
1. Loading the PDF using LangChain's PyPDFLoader
2. Cleaning the raw extracted text (artifacts, special chars, etc.)
"""

import re
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    """
    Load PDF using LangChain PyPDFLoader.
    Returns list of Document objects (one per page).
    Each Document has:
        - page_content : text of that page
        - metadata     : { source, page }
    """
    print(f"[Ingestion] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[Ingestion] Loaded {len(documents)} pages")
    return documents


def clean_text(text: str) -> str:
    """Clean raw PDF text — fix encoding issues, line breaks, whitespace."""

    # Smart quotes → straight quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')

    # Non-breaking space, null bytes
    text = text.replace('\u00a0', ' ').replace('\x00', '')

    # Em/en dashes → hyphen
    text = text.replace('\u2014', '-').replace('\u2013', '-')

    # Bullet variants → standard bullet
    text = text.replace('\uf0b7', '•')

    # Fix hyphenated line breaks: "non-\ninfringement" → "non-infringement"
    text = re.sub(r'-\n', '', text)

    # Fix mid-sentence line breaks (single \n → space, preserve \n\n)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Remove page numbers
    text = re.sub(r'\bPage\s+\d+(\s+of\s+\d+)?\b', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove repeated document title header
    text = re.sub(r'\bUser Agreement\b', '', text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def clean_documents(documents):
    """Apply clean_text() to all loaded Document objects."""
    print(f"[Ingestion] Cleaning {len(documents)} pages...")
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    print("[Ingestion] Cleaning complete.")
    return documents


def load_and_clean(pdf_path: str):
    """Full ingestion: load PDF + clean all pages."""
    documents = load_pdf(pdf_path)
    documents = clean_documents(documents)
    return documents