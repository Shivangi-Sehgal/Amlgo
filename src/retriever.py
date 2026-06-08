"""
retriever.py
------------
Responsible for:
Performing semantic search on the ChromaDB vector store
to find the most relevant chunks for a user query.

How it works:
1. User query - embedding vector (same model used during indexing)
2. ChromaDB finds top-k most similar chunk vectors
3. Returns those chunks as source context for the LLM
"""

from langchain_chroma import Chroma


def get_retriever(vectordb: Chroma, k: int = 4):
    """
    Create a LangChain retriever from the ChromaDB vector store.

    Args:
        vectordb : loaded Chroma vector store
        k        : number of top relevant chunks to retrieve (default 4)

    Returns:
        LangChain retriever object
    """
    retriever = vectordb.as_retriever(
        search_type="similarity",   # cosine similarity search
        search_kwargs={"k": k}      # return top-k chunks
    )
    print(f"[Retriever] Retriever ready (top-{k} chunks per query).")
    return retriever


def retrieve_chunks(retriever, query: str):
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        retriever : LangChain retriever object
        query     : user's natural language question

    Returns:
        List of relevant Document objects with page_content + metadata
    """
    print(f"[Retriever] Searching for: '{query}'")
    docs = retriever.invoke(query)
    print(f"[Retriever] Found {len(docs)} relevant chunks.")
    return docs


def format_context(docs) -> str:
    """
    Combine retrieved chunks into a single context string
    to inject into the LLM prompt.
    """
    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(f"[Chunk {i+1}]\n{doc.page_content.strip()}")
    return "\n\n".join(context_parts)