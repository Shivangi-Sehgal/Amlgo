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

    RecursiveCharacterTextSplitter tries separators in order:
      \n\n  → paragraph breaks (highest priority)
      \n•   → bullet points
      \n    → line breaks
      .     → sentence endings
      ' '   → word boundaries (last resort)

    chunk_size=1200 chars ≈ 150-200 words
    chunk_overlap=200 chars ≈ 2-3 sentences of overlap
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