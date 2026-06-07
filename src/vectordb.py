"""
vectordb.py
-----------
Responsible for:
1. Creating a FAISS vector store from document chunks + embeddings
2. Saving the FAISS index to disk (/vectordb)
3. Loading an existing FAISS index from disk

Why FAISS?
- Runs fully locally (no server needed)
- Fast similarity search even on CPU
- Simple to save/load
- Officially supported by LangChain
"""

import os
from langchain_community.vectorstores import FAISS


def create_vectordb(chunks, embeddings, save_path: str):
    """
    Create a FAISS vector store from chunks and save to disk.

    Args:
        chunks     : list of LangChain Document objects (from chunking.py)
        embeddings : embedding model (from embedding.py)
        save_path  : directory path to save the FAISS index

    Returns:
        FAISS vector store object
    """
    print(f"[VectorDB] Creating FAISS index from {len(chunks)} chunks...")
    vectordb = FAISS.from_documents(chunks, embeddings)

    os.makedirs(save_path, exist_ok=True)
    vectordb.save_local(save_path)
    print(f"[VectorDB] FAISS index saved → {save_path}")
    return vectordb


def load_vectordb(save_path: str, embeddings):
    """
    Load an existing FAISS index from disk.

    Args:
        save_path  : directory where FAISS index was saved
        embeddings : same embedding model used during creation

    Returns:
        FAISS vector store object ready for querying
    """
    print(f"[VectorDB] Loading FAISS index from: {save_path}")
    vectordb = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True  # required by LangChain for local FAISS
    )
    print(f"[VectorDB] FAISS index loaded successfully.")
    return vectordb


def get_chunk_count(save_path: str) -> int:
    """Return number of vectors stored in the FAISS index."""
    try:
        import faiss
        index = faiss.read_index(os.path.join(save_path, "index.faiss"))
        return index.ntotal
    except Exception:
        return -1