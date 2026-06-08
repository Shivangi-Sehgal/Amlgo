"""
vectordb.py
-----------
Responsible for:
1. Creating a ChromaDB vector store from document chunks + embeddings
2. Persisting the Chroma collection to disk (/vectordb)
3. Loading an existing Chroma collection from disk
"""

import os
from langchain_chroma import Chroma

# Fixed collection name for our eBay document
COLLECTION_NAME = "ebay_user_agreement"


def create_vectordb(chunks, embeddings, save_path: str):
    """
    Create a ChromaDB vector store from chunks and persist to disk.

    Args:
        chunks     : list of LangChain Document objects (from chunking.py)
        embeddings : embedding model (from embedding.py)
        save_path  : directory path to persist the Chroma collection

    Returns:
        Chroma vector store object
    """
    print(f"[VectorDB] Creating ChromaDB from {len(chunks)} chunks...")
    print(f"[VectorDB] Persisting to: {save_path}")

    os.makedirs(save_path, exist_ok=True)

    # Chroma automatically persists when persist_directory is set
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=save_path,
    )

    count = vectordb._collection.count()
    print(f"[VectorDB] ChromaDB created with {count} vectors.")
    return vectordb


def load_vectordb(save_path: str, embeddings):
    """
    Load an existing ChromaDB collection from disk.

    Args:
        save_path  : directory where Chroma collection was persisted
        embeddings : same embedding model used during creation

    Returns:
        Chroma vector store object ready for querying
    """
    print(f"[VectorDB] Loading ChromaDB from: {save_path}")

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=save_path,
    )

    count = vectordb._collection.count()
    print(f"[VectorDB] ChromaDB loaded — {count} vectors ready.")
    return vectordb


def get_chunk_count(save_path: str) -> int:
    """
    Return number of vectors stored in the ChromaDB collection.
    Used by Streamlit sidebar to display indexed chunk count.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=save_path)
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return -1