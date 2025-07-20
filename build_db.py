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
    print("Starting to build the vector store...")
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("CRITICAL ERROR: GOOGLE_API_KEY not found. Cannot build embeddings.")
        return

    # 1. Load documents
    loader = DirectoryLoader(
        './knowledge_base/', 
        glob="**/*.txt", 
        show_progress=True,
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    docs = loader.load()
    if not docs:
        print("Error: No documents found in 'knowledge_base'.")
        return

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Create embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    # 4. Create Chroma vector store and PERSIST it to disk
    print(f"Creating and persisting vector store at: {CHROMA_DB_PATH}")
    Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=CHROMA_DB_PATH
    )

    print("Vector store built successfully!")

if __name__ == '__main__':
    build_vector_store()
