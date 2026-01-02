import os
# Force CPU immediately before any other imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import sys
from dotenv import load_dotenv, set_key
from agent import RAGAgent
from ingest import ingest_docs
import time

# Page Config
st.set_page_config(page_title="RAG Agent Dashboard (Groq)", page_icon="⚡", layout="wide")
st.title("⚡ RAG Agent Dashboard (Groq)")

# Load Environment
load_dotenv(override=True)
chroma_path = "chroma_db"
env_path = ".env"

# --- Sidebar: Configuration ---
st.sidebar.header("⚙️ Configuration")
groq_key = os.environ.get("GROQ_API_KEY", "")

st.sidebar.subheader("API Keys")
new_groq_key = st.sidebar.text_input("GROQ_API_KEY", value=groq_key, type="password")

if st.sidebar.button("Save Keys"):
    set_key(env_path, "GROQ_API_KEY", new_groq_key)
    st.sidebar.success("Keys saved! Reloading...")
    time.sleep(1)
    st.rerun()

# --- Diagnostics Section ---
st.header("🔍 Diagnostics")
col1, col2, col3 = st.columns(3)

# 1. API Key Check
with col1:
    st.subheader("1. Groq API Key")
    if not groq_key:
        st.error("❌ GROQ_API_KEY Missing")
    elif groq_key.startswith("xai-") or groq_key.startswith("sk-") or groq_key.startswith("AIza"):
        st.error("❌ Invalid Groq Key Format")
        st.warning("Groq keys usually start with 'gsk_'")
    elif not groq_key.startswith("gsk_"):
         st.warning("⚠️ Unusual Token Format (Expected 'gsk_')")
    else:
        st.success(f"✅ Groq Key Loaded ({groq_key[:8]}...)")

# 2. Database Check
with col2:
    st.subheader("2. Knowledge Base")
    if os.path.exists(chroma_path):
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                model_kwargs={'device': 'cpu'}
            )
            db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
            count = db._collection.count()
            st.success(f"✅ Active ({count} chunks)")
        except Exception as e:
            st.error(f"⚠️ Error reading DB: {e}")
    else:
        st.warning("⚠️ No Database Found")

# 3. Model Check
with col3:
    st.subheader("3. Model Status")
    st.info("Embedding: Local (HuggingFace)\nLLM: Groq (Llama 3.3 70B)")

st.divider()

# --- Ingestion Section ---
st.header("📚 Knowledge Base Ingestion")
st.markdown("Use this to process documents in the `data/` folder.")

if st.button("🚀 Run Ingestion", type="primary"):
    with st.status("Ingesting documents...", expanded=True) as status:
        st.write("Initializing...")
        # Redirect stdout to capture logs
        class StreamToLogger(object):
            def write(self, buf):
                for line in buf.rstrip().splitlines():
                    st.write(f"👉 {line.strip()}")
            def flush(self): pass
        
        old_stdout = sys.stdout
        sys.stdout = StreamToLogger()
        
        try:
            ingest_docs()
            status.update(label="Ingestion Complete!", state="complete", expanded=False)
            st.success("Ingestion Finished.")
        except Exception as e:
            st.error(f"Ingestion Failed: {e}")
        finally:
            sys.stdout = old_stdout
            st.rerun()

st.divider()

# --- Chat Interface ---
st.header("💬 Chat with Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Response logic
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            agent = RAGAgent()
            with st.spinner("Thinking (Llama 3)..."):
                full_response = agent.ask(prompt)
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {e}")
            if "401" in str(e) or "authentication" in str(e).lower():
                st.error("🚨 **CRITICAL**: usage of Invalid API Key detected.")
