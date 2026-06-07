"""
chunking.py
-----------
Responsible for:
Splitting cleaned LangChain Documents into smaller chunks
using RecursiveCharacterTextSplitter.

One job only — takes documents, returns chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    """
    Split documents into overlapping sentence-aware chunks.
    """
    print("[Chunking] Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n• ", "\n", ". ", " "],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"[Chunking] Created {len(chunks)} chunks.")
    return chunks