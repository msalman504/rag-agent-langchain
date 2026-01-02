import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_PATH = "chroma_db"

class RAGAgent:
    def __init__(self):
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not set in .env")
            
        # Initialize Groq (Llama 3.3)
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        
        # Initialize Local Embeddings
        # Must match the one used in ingest.py
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Chroma client
        if os.path.exists(CHROMA_PATH):
             self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        else:
             print(f"Warning: {CHROMA_PATH} does not exist. Please run 'ingest' first.")
             self.db = Chroma(embedding_function=self.embeddings)

        self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
        
        # Define Prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def ask(self, question):
        response = self.rag_chain.invoke({"input": question})
        return response["answer"]
