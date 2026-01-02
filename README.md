# RAG Agent with Groq & Local Embeddings

A fast, privacy-focused RAG (Retrieval-Augmented Generation) agent that lets you chat with your documents (PDF, TXT, MD).

## Features
- **LLM**: Groq (Llama 3.3 70B) - Ultra-fast inference.
- **Embeddings**: Local (HuggingFace `all-MiniLM-L6-v2`) - Unlimited, free, private.
- **Interface**: Streamlit Dashboard (`dashboard.py`) & CLI (`main.py`).

## Setup
1.  Clone repo.
2.  `pip install -r requirements.txt`
3.  Set `GROQ_API_KEY` in `.env`.
4.  Run: `streamlit run dashboard.py`
