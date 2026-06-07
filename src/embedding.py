"""
embedding.py
------------
Responsible for:
Generating vector embeddings for text chunks using
HuggingFace's all-MiniLM-L6-v2 model (free, no API key needed).

Why all-MiniLM-L6-v2?
- Fast and lightweight (22M params)
- Great semantic similarity performance
- Runs locally — no cost
- Ideal for RAG over legal/policy documents
"""

from langchain_community.embeddings import HuggingFaceEmbeddings


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load HuggingFace sentence-transformer embedding model.
    Downloads automatically on first use (~90MB).

    Returns a LangChain-compatible embedding object
    that can be passed directly to FAISS.
    """
    print(f"[Embedding] Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},   # use CPU (no GPU needed)
        encode_kwargs={"normalize_embeddings": True}  # normalize for cosine similarity
    )
    print("[Embedding] Embedding model loaded.")
    return embeddings