import os
from dotenv import load_dotenv
from rag import get_rag_chain

def main():
    """
    Main function to run the interactive chatbot.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if the API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return

    # Check if the vector store exists
    if not os.path.exists("faiss_index"):
        print("Error: Vector store 'faiss_index' not found.")
        print("Please run 'python vector.py' first to create it.")
        return
        
    # Get the initialized RAG chain
    print("Loading the Christite Assistant...")
    rag_chain = get_rag_chain()
    print("Assistant is ready! 🤖")
    
    # --- Interactive Chat Loop ---
    print("\n--- Welcome to The Christite Assistant! ---")
    print("I can answer questions based on the University Handbook and Academic Calendar.")
    print("Type 'quit' or 'exit' to end the chat.\n")

    while True:
        user_query = input("Ask your question: ")

        if user_query.lower() in ['quit', 'exit']:
            print("Thank you for using The Christite Assistant. Goodbye!")
            break

        if user_query.strip() == "":
            continue

        print("🤖 Thinking...")
        response = rag_chain.invoke(user_query)
        print("\n🤖 Assistant:", response)
        print("-" * 50)


if __name__ == "__main__":
    main()