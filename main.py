import argparse
from ingest import ingest_docs
from agent import RAGAgent
from dotenv import load_dotenv

# Force load .env
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="RAG Agent CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Ingest command
    subparsers.add_parser("ingest", help="Ingest documents from data/ directory")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", type=str, help="The question to ask")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_docs()
    elif args.command == "ask":
        if not args.question:
             print("Please provide a question.")
             return
        try:
            agent = RAGAgent()
            print(f"Agent: {agent.ask(args.question)}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
