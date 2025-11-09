import os
from dotenv import load_dotenv # <-- Add this import
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths
PDF_DIRECTORY = "data/"
VECTOR_STORE_PATH = "faiss_index"

def create_vector_store():
    """
    Loads PDFs, splits them into chunks, creates embeddings,
    and saves them to a local FAISS vector store.
    """
    # --- Start of The Fix ---
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if the API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
        return
    # --- End of The Fix ---

    print("Starting vector store creation...")

    # 1. Load Documents
    all_docs = []
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in the '{PDF_DIRECTORY}' directory.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(PDF_DIRECTORY, pdf_file))
        all_docs.extend(loader.load())
        print(f"Loaded '{pdf_file}'.")

    # 2. Split Documents
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(split_chunks)} document chunks.")

    # 3. Create Embeddings
    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Create and Save Vector Store
    print("Creating FAISS vector store and saving it locally...")
    vector_store = FAISS.from_documents(split_chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector store created and saved at '{VECTOR_STORE_PATH}'.")

if __name__ == "__main__":
    create_vector_store()