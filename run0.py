# run.py - AI Chatbot Backend with API Key Fallback Logic

import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Setup ---
app = Flask(__name__)
CORS(app)
load_dotenv()

# --- Global Variables ---
rag_chain = None
CHROMA_DB_PATH = "./chroma_db"
response_cache = {}
api_keys = []
current_key_index = 0

# --- NEW: Load all API keys from .env file ---
def load_api_keys():
    """Loads all GOOGLE_API_KEY_n variables into a list."""
    global api_keys
    print("Loading API keys...")
    i = 1
    while True:
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key:
            api_keys.append(key)
            i += 1
        else:
            break
    if not api_keys:
        print("CRITICAL ERROR: No API keys found in the format GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.")
    else:
        print(f"SUCCESS: Loaded {len(api_keys)} API key(s).")

# --- LangChain RAG Pipeline Initialization ---
def initialize_rag_pipeline(api_key):
    """
    This function builds the RAG chain using a specific Google Gemini API key.
    """
    global rag_chain
    
    # CORRECTED: Use the provided api_key argument
    if not api_key:
        print("Error: A valid API key was not provided to initialize the pipeline.")
        return False
        
    print(f"Initializing RAG pipeline with key ending in ...{api_key[-4:]}")
    
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"CRITICAL ERROR: Chroma DB not found at {CHROMA_DB_PATH}.")
        return False

    try:
        # CORRECTED: Use the api_key passed into this function
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        retriever = vectorstore.as_retriever()

        system_prompt = (
            "You are a helpful and friendly AI assistant for the Google Cloud Community Day Bhopal event. "
            "Your goal is to answer questions accurately based on the provided context. "
            "You must follow these rules:\n"
            "1.  **CRITICAL RULE: When asked for a list of speakers, you MUST list ALL speakers found in the context. Do not summarize or shorten the list.**\n"
            "2.  **Always use Markdown for formatting.** Use bullet points (`-` or `*`) for lists and bold (`**text**`) for names and important terms.\n"
            "3.  If a user asks for a speaker's social media like LinkedIn, you MUST reply with: 'For the most up-to-date professional profiles, please check the official event website.' Do not provide the direct link.\n"
            "4.  If the user asks in a mix of Hindi and English (Hinglish), understand it and reply in clear, simple English.\n"
            "5.  If asked about the event location, provide the answer from the context and also add: 'You can find a direct link to the location at the bottom of the chat window.'\n"
            "6.  If asked if the event is free, you MUST reply: 'The event requires a ticket for entry. For details on pricing and registration, please visit the official website.'\n"
            "7.  If the user says 'thank you' or a similar phrase of gratitude, you MUST reply with: 'You\\'re welcome! See you at GCCD Bhopal!'\n"
            "8.  If the context does not contain the answer, politely say: 'I don\\'t have that specific information.'\n"
            "9.  Keep your answers concise but complete."
            "\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("Gemini RAG pipeline initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing RAG pipeline with key index {current_key_index}: {e}")
        return False

# --- Flask API Endpoints ---
@app.route('/')
def home():
    """Serves the simple chatbot HTML interface."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_bot():
    """
    This endpoint receives a user's message, processes it, and handles API key fallback.
    """
    global rag_chain, current_key_index
    
    if not rag_chain:
        return jsonify({'reply': 'Sorry, the AI model is not ready yet. Please try again in a moment.'}), 503

    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    if user_message in response_cache:
        print(f"DEBUG: Returning cached response for: '{user_message}'")
        return jsonify({'reply': response_cache[user_message]})

    max_retries = len(api_keys)
    for i in range(max_retries):
        try:
            print(f"DEBUG: Attempting request with API key index {current_key_index}")
            response = rag_chain.invoke({"input": user_message})
            bot_reply = response.get("answer", "I couldn't find an answer to that.")
            response_cache[user_message] = bot_reply
            return jsonify({'reply': bot_reply})
        
        except google_exceptions.ResourceExhausted as e:
            print(f"WARNING: Rate limit exceeded for key index {current_key_index}. Error: {e}")
            # Move to the next key, wrapping around if necessary
            current_key_index = (current_key_index + 1) % len(api_keys)
            print(f"Switching to new API key index: {current_key_index}")
            
            # Re-initialize the pipeline with the new key
            success = initialize_rag_pipeline(api_keys[current_key_index])
            if not success:
                return jsonify({'reply': 'Sorry, there was an issue switching AI providers. Please try again.'}), 500
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return jsonify({'reply': 'An unexpected error occurred while processing your request.'}), 500

    # If all keys have failed
    return jsonify({'reply': 'Sorry, all our AI services are currently busy. Please try again in a few moments.'}), 503

# --- SCRIPT EXECUTION ---
load_api_keys() 
if api_keys:
    initialize_rag_pipeline(api_keys[current_key_index])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
