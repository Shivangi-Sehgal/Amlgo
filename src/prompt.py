"""
prompt.py
---------
Responsible for:
Building the prompt template for LLaMA 3.3 70B via Groq API.

Our prompt injects:
- Retrieved context chunks from ChromaDB
- The user's question
- Instructions to stay grounded in the document
"""

from langchain_core.prompts import PromptTemplate


RAG_PROMPT_TEMPLATE = """
You are an intelligent document question-answering assistant.

Your task is to answer the user's question ONLY using the information provided in the document context below.


DOCUMENT CONTEXT
{context}

USER QUESTION
{question}

INSTRUCTIONS

1. Answer strictly from the provided document context.
2. Do NOT use external knowledge.
3. Do NOT make up facts, assumptions, or explanations.
4. If the answer is not present in the context, reply exactly:
   "Sorry, I do not know the answer."

5. Keep answers:
   - Clear
   - Concise
   - Factually accurate
   - Well-structured

6. When possible:
   - Mention important clauses, conditions, or exceptions
   - Preserve legal/business terminology from the document
   - Summarize long sections into readable language

7. If the question is ambiguous:
   - Use the most relevant interpretation from the retrieved context
   - Do not invent missing details

8. If multiple chunks contain relevant information:
   - Combine them into a single coherent answer
   - Avoid repetition

9. Never mention:
   - embeddings
   - vector databases
   - retrieval systems
   - chunking
   - internal implementation details

10. Do not say:
   - "Based on the context provided..."
   - "According to the retrieved chunks..."
   - "The document says..."

Return only the final answer.

ANSWER:
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