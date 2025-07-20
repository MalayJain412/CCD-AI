# build_db.py - One-time script to build the vector store

import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the path for the persistent database
CHROMA_DB_PATH = "./chroma_db"

def build_vector_store():
    """
    Builds the vector store from the knowledge base and saves it to disk.
    """
    print("--- Starting Vector Store Build Process ---")
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("CRITICAL ERROR: GOOGLE_API_KEY not found in environment. Build cannot proceed.")
        exit(1)
    else:
        print("SUCCESS: GOOGLE_API_KEY loaded successfully.")

    # 1. Load documents
    print("Step 1: Loading documents from './knowledge_base/'...")
    loader = DirectoryLoader(
        './knowledge_base/', 
        glob="**/*.txt", 
        show_progress=True,
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    docs = loader.load()
    if not docs:
        print("CRITICAL ERROR: No documents found in 'knowledge_base'. Build failed.")
        exit(1)
    print(f"SUCCESS: Loaded {len(docs)} document(s).")

    # 2. Split documents
    print("Step 2: Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"SUCCESS: Split documents into {len(splits)} chunks.")

    # 3. Create embeddings model
    print("Step 3: Initializing Google Generative AI Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    print("SUCCESS: Embeddings model initialized.")

    # 4. Create Chroma vector store and PERSIST it to disk
    print(f"Step 4: Creating and persisting vector store at: {CHROMA_DB_PATH}")
    try:
        Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=CHROMA_DB_PATH
        )
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create Chroma database. Error: {e}")
        exit(1)
    
    # 5. Verification Step
    print("Step 5: Verifying database creation...")
    if os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0:
        print("--- SUCCESS: Vector store built and verified successfully! ---")
    else:
        print("--- CRITICAL ERROR: Vector store directory is empty or was not created. Build failed. ---")
        exit(1)

if __name__ == '__main__':
    build_vector_store()
