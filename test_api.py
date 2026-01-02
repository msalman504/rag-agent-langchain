import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

key = os.environ.get("GOOGLE_API_KEY")
if not key:
    print("Error: GOOGLE_API_KEY not found in .env")
    exit(1)

print(f"API Key found: {key[:5]}...{key[-5:]}")

try:
    print("Initializing Gemini...")
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash") 
    print("Sending test request: 'Say Hello World'...")
    result = llm.invoke("Say Hello World")
    print("-" * 20)
    print(f"Success! Response:\n{result.content}")
    print("-" * 20)
except Exception as e:
    print(f"\nAPI Test Failed: {e}")
