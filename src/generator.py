"""
generator.py
------------
Responsible for:
1. Loading Mistral-7B-Instruct via Groq API
2. Generating streaming responses token by token
3. Used by pipeline.py and app.py (Streamlit)
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
        temperature : creativity control (0.0 = deterministic, 1.0 = creative)
                      

    Returns:
        ChatGroq LLM object (LangChain compatible)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "[Generator] GROQ_API_KEY not found.\n"
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