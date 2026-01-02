import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def load_documents():
    documents = []
    # Load .txt
    for file_path in glob.glob(os.path.join(DATA_PATH, "*.txt")):
        print(f"Loading {file_path}...")
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    
    # Load .pdf
    for file_path in glob.glob(os.path.join(DATA_PATH, "*.pdf")):
        print(f"Loading {file_path}...")
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        
    # Load .md
    for file_path in glob.glob(os.path.join(DATA_PATH, "*.md")):
        print(f"Loading {file_path}...")
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
        
    return documents

def ingest_docs():
    print("Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("No documents found in data/ folder.")
        return

    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Creating vector store with Local Embeddings (HuggingFace)...")
    # Using a small, fast, and free local model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # No need for rate limit logic with local embeddings!
    if os.path.exists(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)
        
    db.add_documents(chunks)
    print(f"Saved to {CHROMA_PATH}.")

if __name__ == "__main__":
    ingest_docs()
