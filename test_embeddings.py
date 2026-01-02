from langchain_community.embeddings import HuggingFaceEmbeddings
import time

print("Testing HuggingFace Embeddings on CPU...")

try:
    start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print(f"Model loaded in {time.time() - start:.2f}s")
    
    print("Embedding test string...")
    res = embeddings.embed_query("Hello world")
    print(f"Success! Vector length: {len(res)}")
except Exception as e:
    print(f"FAILED: {e}")
