"""
pipeline.py
-----------
Master RAG Pipeline — connects ALL modules together.

Flow:
  PDF → ingestion → chunking → embedding → vectordb (build once)
  Query → retriever → prompt → generator → streamed response (every query)

Two modes:
  1. build()  → run once to process PDF and build FAISS index
  2. query()  → run every time a user asks a question (streaming)
"""

import os
import sys

# Add src/ to path so imports work
sys.path.append(os.path.dirname(__file__))

from ingestion import load_and_clean
from chunking import chunk_documents
from embedding import load_embedding_model
from vectordb import create_vectordb, load_vectordb, get_chunk_count
from retriever import get_retriever, retrieve_chunks, format_context
from prompt import build_prompt
from generator import load_llm, stream_response, generate_response
from utils import save_chunks_to_json, print_chunk_stats


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH    = os.path.join(BASE_DIR, "data", "AI Training Document.pdf")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks", "chunks.json")
VECTORDB_PATH = os.path.join(BASE_DIR, "vectordb")


# ── Global state (loaded once, reused across queries) ─────────────────────────

_embeddings = None
_vectordb   = None
_retriever  = None
_llm        = None


# ── Step 1: Build pipeline (run once) ─────────────────────────────────────────

def build_pipeline():
    """
    Full pipeline build:
    Load PDF → Clean → Chunk → Embed → Save to FAISS

    Run this ONCE to set up the vector database.
    After this, use query_pipeline() for all user queries.
    """
    print("\n" + "="*55)
    print("   BUILDING RAG PIPELINE")
    print("="*55 + "\n")

    # 1. Load & clean PDF
    documents = load_and_clean(PDF_PATH)

    # 2. Chunk documents
    chunks = chunk_documents(documents)

    # 3. Save chunks metadata to JSON
    chunks_data = save_chunks_to_json(chunks, CHUNKS_PATH)
    print_chunk_stats(chunks_data)

    # 4. Load embedding model
    embeddings = load_embedding_model()

    # 5. Create & save FAISS vector store
    vectordb = create_vectordb(chunks, embeddings, VECTORDB_PATH)

    print("\n[Pipeline] Build complete! FAISS index ready.")
    print(f"[Pipeline] Indexed {len(chunks)} chunks from document.")
    print("="*55 + "\n")
    return vectordb


# ── Step 2: Load pipeline (after build) ───────────────────────────────────────

def load_pipeline():
    """
    Load the pre-built FAISS index + embedding model + LLM.
    Call this at app startup (in app.py via Streamlit session state).

    Returns:
        retriever, llm  → used by query_pipeline()
    """
    global _embeddings, _vectordb, _retriever, _llm

    if _retriever is not None and _llm is not None:
        print("[Pipeline] Already loaded — reusing.")
        return _retriever, _llm

    print("[Pipeline] Loading pipeline components...")

    # Load embedding model
    _embeddings = load_embedding_model()

    # Load FAISS index from disk
    _vectordb = load_vectordb(VECTORDB_PATH, _embeddings)

    # Create retriever (top 4 chunks)
    _retriever = get_retriever(_vectordb, k=4)

    # Load LLM (Mistral via Groq)
    _llm = load_llm()

    print("[Pipeline] All components loaded. Ready for queries.\n")
    return _retriever, _llm


# ── Step 3: Query pipeline (every user message) ───────────────────────────────

def query_pipeline(question: str, retriever, llm, stream: bool = True):
    """
    Run a full RAG query:
    1. Retrieve relevant chunks from FAISS
    2. Build prompt with context
    3. Stream response from Mistral via Groq

    Args:
        question  : user's natural language query
        retriever : loaded retriever (from load_pipeline)
        llm       : loaded LLM (from load_pipeline)
        stream    : if True, yields tokens; if False, returns full string

    Yields (stream=True):
        str token chunks

    Returns (stream=False):
        full response string, source_docs list
    """
    # Step 1: Retrieve relevant chunks
    source_docs = retrieve_chunks(retriever, question)

    # Step 2: Format context from retrieved chunks
    context = format_context(source_docs)

    # Step 3: Build the Mistral prompt
    prompt = build_prompt(context=context, question=question)

    # Step 4: Generate response
    if stream:
        return stream_response(llm, prompt), source_docs
    else:
        response = generate_response(llm, prompt)
        return response, source_docs


# ── Utility: get chunk count for Streamlit sidebar ────────────────────────────

def get_indexed_chunk_count() -> int:
    """Return number of chunks in the FAISS index."""
    return get_chunk_count(VECTORDB_PATH)


# ── Run standalone (build pipeline from terminal) ─────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the FAISS index from PDF")
    parser.add_argument("--query", type=str, help="Ask a question (non-streaming test)")
    args = parser.parse_args()

    if args.build:
        build_pipeline()

    elif args.query:
        retriever, llm = load_pipeline()
        response, sources = query_pipeline(args.query, retriever, llm, stream=False)
        print(f"\n[Answer]\n{response}")
        print(f"\n[Sources] {len(sources)} chunks used")
        for i, doc in enumerate(sources):
            print(f"  [{i+1}] Page {doc.metadata.get('page', 0)+1}: {doc.page_content[:100]}...")

    else:
        print("Usage:")
        print("  python pipeline.py --build          # build FAISS index")
        print("  python pipeline.py --query 'text'   # test a query")