# PDF Chatbot using RAG (Retrieval-Augmented Generation)

An AI-powered PDF Question-Answering chatbot built using LangChain, ChromaDB, HuggingFace embeddings, and Groq-hosted LLaMA models.

The chatbot allows users to upload/query a PDF document and ask natural language questions. Responses are generated strictly from the document context using a Retrieval-Augmented Generation (RAG) pipeline.

---

# Features

* PDF document ingestion and cleaning
* Sentence-aware chunking with overlap
* Semantic search using vector embeddings
* ChromaDB vector storage and retrieval
* Grounded response generation using LLaMA 3.3
* Streaming responses for better UX
* Modular and scalable architecture

---

# Tech Stack

* Python
* LangChain
* ChromaDB
* HuggingFace Embeddings
* Groq API
* LLaMA 3.3 70B
* Streamlit

---

# Project Architecture

```text
PDF
 ↓
Ingestion & Cleaning
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB Vector Store
 ↓
Retriever
 ↓
Prompt Builder
 ↓
LLaMA 3.3 via Groq API
 ↓
Final Response
```

---

# Folder Structure

```text
project/
│
├── data/
│   └── AI Training Document.pdf
│
├── chunks/
│   └── chunks.json
│
├── vectordb/
│
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── vectordb.py
│   ├── retriever.py
│   ├── prompt.py
│   ├── generator.py
│   ├── pipeline.py
│   └── utils.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# How It Works

## 1. PDF Ingestion

The PDF is loaded using LangChain’s `PyPDFLoader`.
Extracted text is cleaned to remove formatting issues, special characters, repeated headers, and page artifacts.

---

## 2. Chunking

The cleaned document is split into overlapping chunks using `RecursiveCharacterTextSplitter`.

### Configuration

* Chunk Size: `1200`
* Chunk Overlap: `200`

This preserves semantic continuity and improves retrieval accuracy.

---

## 3. Embeddings

Each chunk is converted into vector embeddings using:

```text
all-MiniLM-L6-v2
```

The embeddings are normalized for cosine similarity search.

---

## 4. Vector Database

Embeddings are stored in ChromaDB for efficient semantic retrieval.

The retriever fetches the top-k most relevant chunks for every query.

---

## 5. Response Generation

Retrieved chunks and user queries are combined into a grounded prompt and sent to:

```text
llama-3.3-70b-versatile
```

via the Groq API.

Streaming responses are enabled for real-time interaction.

---

# Prompt Engineering Strategy

The prompt was designed to:

* Restrict responses to retrieved document context
* Reduce hallucinations
* Prevent use of external knowledge
* Generate concise and factual answers
* Return fallback responses when information is unavailable

---

# Installation

## 1. Clone Repository

```bash
git clone <your-github-repo-link>
cd <repo-name>
```

---

## 2. Create Virtual Environment

```bash
python -m venv my_venv
```

Activate environment:

### Windows

```bash
my_venv\Scripts\activate
```

### Mac/Linux

```bash
source my_venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add Environment Variables

Create a `.env` file:

```text
GROQ_API_KEY=your_api_key
```

Get a free API key from:

```text
https://console.groq.com
```

---

# Build Vector Database

Run once to process the PDF and create embeddings:

```bash
python src/pipeline.py --build
```

---

# Run Query from Terminal

```bash
python src/pipeline.py --query "Your question here"
```

---

# Run Streamlit App

```bash
streamlit run app.py
```

---

# Example Queries

### Query

```text
What are the payment terms?
```

### Query

```text
How are disputes handled?
```

### Query

```text
What are the listing fees?
```

---

# Limitations

* Retrieval quality depends on chunk relevance
* Full-document summarization may be incomplete
* CPU embeddings increase indexing time
* Context window limitations affect long responses

---

# Future Improvements

* Hybrid search (BM25 + vector search)
* Metadata filtering
* Source citations in responses
* Multi-PDF support
* Conversational memory
* Re-ranking models for retrieval optimization

---

# Author

Shivangi Sehgal

AI/ML | GenAI | RAG Systems | LLM Applications
