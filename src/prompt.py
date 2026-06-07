"""
prompt.py
---------
Responsible for:
Building the prompt template for Mistral-7B-Instruct.

Mistral uses a specific chat format:
[INST] ... [/INST]

Our prompt injects:
- Retrieved context chunks from FAISS
- The user's question
- Instructions to stay grounded in the document
"""

from langchain_core.prompts import PromptTemplate

# Mistral-optimized RAG prompt template
# {context} → retrieved chunks from FAISS
# {question} → user's query
RAG_PROMPT_TEMPLATE = """<s>[INST]
You are a helpful assistant that answers questions strictly based on the provided document context.

Instructions:
- Answer ONLY using the information in the context below.
- If the answer is not found in the context, say: "I could not find that information in the provided document."
- Be concise, factual, and clear.
- Do not make up or assume any information not present in the context.

Context from document:
{context}

Question: {question}
[/INST]
"""


def get_prompt_template() -> PromptTemplate:
    """
    Return a LangChain PromptTemplate with {context} and {question} variables.
    Used by the RAG chain to inject retrieved chunks + user query.
    """
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return prompt


def build_prompt(context: str, question: str) -> str:
    """
    Manually build the prompt string (used for streaming with Groq directly).

    Args:
        context  : formatted retrieved chunks (from retriever.format_context)
        question : user's question

    Returns:
        Full prompt string ready to send to Groq API
    """
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)