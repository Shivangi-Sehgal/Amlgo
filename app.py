import streamlit as st
import sys, os

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Setup paths
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pipeline import load_pipeline, query_pipeline
from utils import format_source_chunks

# Page Config
st.set_page_config(
    page_title="AMLGO Labs —  Assistant",
    page_icon="💠",
    layout="centered"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    /* ── Base & Background ───────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(145deg, #e8e4f8 0%, #ede9f9 25%, #f4f0fc 55%, #faf8ff 100%);
        min-height: 100vh;
    }

    /* ── Header Banner ───────────────────────────────── */
    .amlgo-header {
        background: linear-gradient(120deg, #c8bff5 0%, #d8d0f8 40%, #e8e0fb 70%, #f0ecfe 100%);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 32px rgba(120, 90, 220, 0.12), 0 2px 8px rgba(120, 90, 220, 0.06);
        border: 1px solid rgba(255,255,255,0.7);
        position: relative;
        overflow: hidden;
    }

    .amlgo-header::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(180,120,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }

    .amlgo-header::after {
        content: '';
        position: absolute;
        bottom: -30px; left: 30%;
        width: 120px; height: 120px;
        background: radial-gradient(circle, rgba(0,210,255,0.10) 0%, transparent 70%);
        border-radius: 50%;
    }

    .header-left { display: flex; align-items: center; gap: 18px; z-index: 1; }

    .logo-diamond {
        width: 52px; height: 52px;
        background: linear-gradient(135deg, #00d2ff 0%, #e040fb 100%);
        clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
        flex-shrink: 0;
        box-shadow: 0 4px 16px rgba(100, 60, 200, 0.3);
    }

    .header-text-group { display: flex; flex-direction: column; gap: 2px; }

    .brand-name {
        font-family: 'Sora', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #1a1040;
        letter-spacing: -0.3px;
    }

    .brand-name span { color: #7c5cbf; }

    .brand-tagline {
        font-size: 12px;
        font-weight: 400;
        color: #6b5f8a;
        letter-spacing: 0.2px;
    }

    .header-right {
        font-family: 'Sora', sans-serif;
        font-size: 15px;
        font-weight: 500;
        color: #2d1f5e;
        text-align: right;
        z-index: 1;
        line-height: 1.5;
    }

    .header-right strong { color: #7c3fbf; }

    /* ── Subtitle strip ──────────────────────────────── */
    .page-subtitle {
        text-align: center;
        font-family: 'Sora', sans-serif;
        font-size: 13px;
        font-weight: 500;
        color: #8b79b8;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    /* ── Chat container ──────────────────────────────── */
    .stChatMessage {
        background: rgba(255,255,255,0.72) !important;
        backdrop-filter: blur(10px);
        border-radius: 16px !important;
        border: 1px solid rgba(200, 185, 245, 0.4) !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 12px rgba(120, 90, 210, 0.07) !important;
        padding: 16px 20px !important;
    }

    [data-testid="stChatMessageContent"] p {
        font-size: 15px;
        line-height: 1.7;
        color: #1e1535;
    }

    /* User bubble accent */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: rgba(200, 185, 248, 0.25) !important;
        border-color: rgba(160, 130, 230, 0.3) !important;
    }

    /* ── Chat input ──────────────────────────────────── */
    .stChatInputContainer {
        background: rgba(255,255,255,0.85) !important;
        border-radius: 16px !important;
        border: 1.5px solid rgba(160,130,230,0.35) !important;
        box-shadow: 0 4px 20px rgba(120, 90, 210, 0.10) !important;
        backdrop-filter: blur(10px);
    }

    .stChatInputContainer:focus-within {
        border-color: rgba(120, 60, 210, 0.55) !important;
        box-shadow: 0 4px 24px rgba(120, 90, 210, 0.18) !important;
    }

    textarea[data-testid="stChatInputTextArea"] {
        color: #1e1535 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
    }

    textarea[data-testid="stChatInputTextArea"]::placeholder {
        color: #a898cc !important;
    }

    /* ── Expander (references) ───────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(200, 185, 248, 0.15) !important;
        border-radius: 10px !important;
        color: #6b52a8 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        border: 1px solid rgba(160, 130, 230, 0.2) !important;
    }

    .streamlit-expanderContent {
        background: rgba(255,255,255,0.6) !important;
        border-radius: 0 0 10px 10px !important;
        border: 1px solid rgba(160, 130, 230, 0.15) !important;
        border-top: none !important;
    }

    /* ── Sidebar ─────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8e0f8 0%, #f0ecfe 100%) !important;
        border-right: 1px solid rgba(160, 130, 230, 0.2) !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #9b7fe8 0%, #c070e8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 10px 20px !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(140, 80, 220, 0.3) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(140, 80, 220, 0.4) !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Sora', sans-serif !important;
        color: #2d1f5e !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stMarkdown {
        color: #5a4a7a !important;
        font-size: 13.5px !important;
    }

    /* ── Info / Alert boxes ──────────────────────────── */
    .stAlert {
        background: rgba(200, 185, 248, 0.2) !important;
        border: 1px solid rgba(160, 130, 230, 0.3) !important;
        border-radius: 12px !important;
        color: #3d2d6a !important;
    }

    /* ── Spinner ─────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: #9b7fe8 !important;
    }

    /* ── Scrollbar ───────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(155, 127, 232, 0.35);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(155, 127, 232, 0.6);
    }

    /* ── Hide Streamlit default elements ─────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; }
    </style>
""", unsafe_allow_html=True)

# ── Header Banner ──────────────────────────────────────────────
st.markdown("""
<div class="amlgo-header">
    <div class="header-left">
        <div class="logo-diamond"></div>
        <div class="header-text-group">
            <div class="brand-name"><span>AMLGO</span> LABS</div>
            <div class="brand-tagline">Data with purpose, <strong>Impact with vision.</strong></div>
        </div>
    </div>
    <div class="header-right">
        RAG Policy Assistant<br>
        <strong>User Agreement</strong>
    </div>
</div>
<div class="page-subtitle">✦ How Can I Help You? ✦</div>
""", unsafe_allow_html=True)

# ── Initialize Pipeline ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.spinner("Initializing AI engine..."):
        try:
            st.session_state.retriever, st.session_state.llm = load_pipeline()
        except Exception as e:
            st.error(f"Error loading model: {e}")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💠 About")
    st.info(
        "Ask natural language questions about the **User Agreement**. "
        "The assistant retrieves relevant policy sections and answers using your local LLM."
    )
    st.markdown("---")
    st.markdown("### ⚙️ Controls")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px; color:#9080b8; text-align:center;'>Built by AMLGO Labs</div>",
        unsafe_allow_html=True
    )

# ── Chat History ───────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📎 View References"):
                st.caption(message["sources"])

# ── Chat Input ─────────────────────────────────────────────────
if prompt := st.chat_input("Ask about the user agreement…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            stream, source_docs = query_pipeline(
                prompt, st.session_state.retriever, st.session_state.llm, stream=True
            )

            for chunk in stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            sources = format_source_chunks(source_docs)
            if sources:
                with st.expander("📎 View References"):
                    st.caption(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })

        except Exception as e:
            st.error(f"Error: {e}")