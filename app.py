"""
app.py
------
Streamlit Chatbot with Real-Time Streaming Responses.

Features:
- Natural language query input
- Real-time streaming response (token by token)
- Source chunks display (grounding evidence)
- Sidebar: model info, chunk count, clear chat button
- Session state: chat history persists during session

Run with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st

# Add src/ to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pipeline import load_pipeline, query_pipeline, get_indexed_chunk_count, build_pipeline
from utils import format_source_chunks

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="eBay Document RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4a90d9;
        padding: 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .sidebar-info {
        background-color: #eef2ff;
        padding: 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize session state ──────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ RAG Chatbot Info")
    st.markdown("---")

    # Model info
    st.markdown("**🧠 Model**")
    st.markdown("`llama-3.3-70b-versatile` via Groq")

    st.markdown("**📦 Embedding Model**")
    st.markdown("`all-MiniLM-L6-v2`")

    st.markdown("**🗄️ Vector Store**")
    st.markdown("`ChromaDB (local)`")

    st.markdown("---")

    # Chunk count
    chunk_count = get_indexed_chunk_count()
    if chunk_count > 0:
        st.markdown(f"**📄 Indexed Chunks:** `{chunk_count}`")
    else:
        st.markdown("**📄 Indexed Chunks:** `Not built yet`")

    st.markdown("---")

    # Build index button
    st.markdown("**🔧 Setup**")
    if st.button("🏗️ Build FAISS Index", use_container_width=True):
        with st.spinner("Building index from PDF... (takes ~1 min first time)"):
            try:
                build_pipeline()
                st.success("✅ Index built successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Build failed: {e}")

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class='sidebar-info'>
    <b>📌 How to use:</b><br>
    1. Click <b>Build FAISS Index</b> once<br>
    2. Type your question below<br>
    3. Watch the answer stream live!
    </div>
    """, unsafe_allow_html=True)


# ── Main header ───────────────────────────────────────────────────────────────

st.markdown("<div class='main-header'>🤖 eBay Document RAG Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Ask questions about the eBay User Agreement — powered by Mistral via Groq</div>", unsafe_allow_html=True)

# ── Load pipeline on startup ──────────────────────────────────────────────────

vectordb_path = os.path.join(os.path.dirname(__file__), "vectordb")
vectordb_exists = os.path.exists(vectordb_path) and any(
    f.endswith(".sqlite3") for f in os.listdir(vectordb_path)
) if os.path.exists(vectordb_path) else False

if not st.session_state.pipeline_loaded and vectordb_exists:
    with st.spinner("🔄 Loading RAG pipeline..."):
        try:
            retriever, llm = load_pipeline()
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.pipeline_loaded = True
        except Exception as e:
            st.error(f"❌ Failed to load pipeline: {e}")

elif not vectordb_exists:
    st.warning("⚠️ FAISS index not found. Click **'Build FAISS Index'** in the sidebar first.")


# ── Display chat history ──────────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Source Chunks Used"):
                st.markdown(
                    f"<div class='source-box'>{message['sources']}</div>",
                    unsafe_allow_html=True
                )


# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about the eBay User Agreement..."):

    # Block if pipeline not loaded
    if not st.session_state.pipeline_loaded:
        st.error("❌ Please build the FAISS index first using the sidebar button.")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        source_docs = []

        try:
            # Get streaming generator + source docs
            stream_gen, source_docs = query_pipeline(
                question=prompt,
                retriever=st.session_state.retriever,
                llm=st.session_state.llm,
                stream=True
            )

            # Stream tokens into placeholder
            for token in stream_gen:
                full_response += token
                response_placeholder.markdown(full_response + "▌")  # blinking cursor effect

            # Final response (remove cursor)
            response_placeholder.markdown(full_response)

            # Show source chunks in expander
            formatted_sources = format_source_chunks(source_docs)
            with st.expander("📚 View Source Chunks Used"):
                st.markdown(
                    f"<div class='source-box'>{formatted_sources}</div>",
                    unsafe_allow_html=True
                )

        except Exception as e:
            full_response = f"❌ Error generating response: {str(e)}"
            response_placeholder.markdown(full_response)
            formatted_sources = ""

        # Save assistant message + sources to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": formatted_sources if source_docs else ""
        })