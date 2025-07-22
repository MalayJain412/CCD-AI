# run.py - AI Chatbot Backend with Conversation Turn History

import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions
from datetime import datetime

# Import the db object and the new Conversation model
from models import db, Conversation
from models import db, Conversation

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Setup ---
app = Flask(__name__)
CORS(app)
load_dotenv()

# Load configuration from the Config object
app.config.from_object(Config)
# Initialize the database with the app
db.init_app(app)

# --- Global Variables ---
rag_chain = None
CHROMA_DB_PATH = "./chroma_db"
api_keys = []
current_key_index = 0

# --- Load API Keys ---
def load_api_keys():
    global api_keys
    i = 1
    while True:
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key:
            api_keys.append(key)
            i += 1
        else:
            break
    if not api_keys:
        print("CRITICAL ERROR: No API keys found.")
    else:
        print(f"SUCCESS: Loaded {len(api_keys)} API key(s).")

# --- LangChain RAG Pipeline Initialization ---
def initialize_rag_pipeline(api_key):
    global rag_chain
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"CRITICAL ERROR: Chroma DB not found at {CHROMA_DB_PATH}.")
        return False
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly assistant for the Google Cloud Community Day Bhopal event.\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        print("Gemini RAG pipeline initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        return False

# --- Flask API Endpoints ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_bot():
    global rag_chain, current_key_index
    if not rag_chain:
        return jsonify({'reply': 'AI model not ready.'}), 503

    data = request.get_json()
    user_message = data.get('message')
    session_id = data.get('session_id')
    client_history = data.get('history', [])

    if not user_message or not session_id:
        return jsonify({'error': 'Missing message or session_id'}), 400

    chat_history_for_chain = []
    for msg in client_history:
        content = msg.get('message')
        if msg.get('sender') == 'user':
            chat_history_for_chain.append(HumanMessage(content=content))
        else:
            chat_history_for_chain.append(AIMessage(content=content))

    bot_reply = "Sorry, all our AI services are currently busy."
    max_retries = len(api_keys)
    for _ in range(max_retries):
        try:
            response = rag_chain.invoke({"input": user_message, "chat_history": chat_history_for_chain})
            bot_reply = response.get("answer", "I couldn't find an answer to that.")
            break # Success
        except google_exceptions.ResourceExhausted:
            current_key_index = (current_key_index + 1) % len(api_keys)
            initialize_rag_pipeline(api_keys[current_key_index])
        except Exception as e:
            print(f"Unexpected error: {e}")
            bot_reply = 'An error occurred. Please try again.'
            break
    
    # NEW: Save the entire conversation turn to the DB in a single operation
    conversation_turn = Conversation(
        session_id=session_id,
        question=user_message,
        answer=bot_reply
    )
    db.session.add(conversation_turn)
    db.session.commit()

    return jsonify({'reply': bot_reply})

@app.route('/history', methods=['POST'])
def get_chat_history():
    data = request.json
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'session_id is required'}), 400
    
    # Query the new Conversation table
    turns = Conversation.query.filter_by(session_id=session_id).order_by(Conversation.timestamp).all()
    
    # "Unroll" the data back into a linear chat format for the frontend
    history = []
    for turn in turns:
        history.append({'sender': 'user', 'message': turn.question})
        history.append({'sender': 'bot', 'message': turn.answer})
        
    return jsonify({'history': history})

# --- SCRIPT EXECUTION ---
load_api_keys()
if api_keys:
    initialize_rag_pipeline(api_keys[current_key_index])

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
