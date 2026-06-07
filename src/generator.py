"""
generator.py
------------
Responsible for:
1. Loading Mistral-7B-Instruct via Groq API
2. Generating streaming responses token by token
3. Used by pipeline.py and app.py (Streamlit)

Why Groq?
- Free API (sign up at console.groq.com)
- Extremely fast inference (LPU hardware)
- Supports Mistral, LLaMA, Gemma models
- No GPU needed on your machine

Setup:
1. Go to https://console.groq.com
2. Create a free account
3. Generate an API key
4. Add to your .env file: GROQ_API_KEY=your_key_here
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env file


def load_llm(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.2):
    """
    Load LLM via Groq API using LangChain's ChatGroq.

    Args:
        model_name  : Groq model ID
                      Current active options (as of 2025):
                        "llama-3.3-70b-versatile"  → LLaMA 3.3 70B (best quality, recommended)
                        "llama3-8b-8192"            → LLaMA 3 8B (fastest, lightweight)
                        "gemma2-9b-it"              → Google Gemma 2 9B (good alternative)
        temperature : 0.0 = deterministic, 1.0 = creative
                      Use 0.1-0.2 for factual RAG tasks

    Returns:
        ChatGroq LLM object (LangChain compatible)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "[Generator] GROQ_API_KEY not found.\n"
            "→ Create a free key at https://console.groq.com\n"
            "→ Add to .env file: GROQ_API_KEY=your_key_here"
        )

    print(f"[Generator] Loading LLM: {model_name} via Groq API...")
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key,
        streaming=True          # enable token-by-token streaming
    )
    print("[Generator] LLM ready.")
    return llm


def generate_response(llm, prompt: str) -> str:
    """
    Generate a full (non-streaming) response from the LLM.
    Used for testing and evaluation.

    Args:
        llm    : loaded ChatGroq LLM
        prompt : full prompt string (from prompt.py build_prompt)

    Returns:
        Response string
    """
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def stream_response(llm, prompt: str):
    """
    Stream response token by token from the LLM.
    Used by Streamlit app for real-time output.

    Args:
        llm    : loaded ChatGroq LLM
        prompt : full prompt string

    Yields:
        str token chunks as they arrive
    """
    from langchain_core.messages import HumanMessage
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content