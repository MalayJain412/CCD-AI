# run.py - AI Chatbot Backend using Google Gemini API - ENHANCED

import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain Imports for Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Setup ---
app = Flask(__name__)
CORS(app)
load_dotenv()

# --- Global Variables for LangChain Components ---
rag_chain = None
# Get the absolute path of the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load Event Data ---
event_data = {}
try:
    with open('data.json', 'r', encoding='utf-8') as f:
        event_data = json.load(f)
    print("DEBUG: data.json loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load data.json. Error: {e}")

# --- LangChain RAG Pipeline Initialization ---
def initialize_rag_pipeline():
    """
    This function loads the pre-built RAG chain using the persisted Chroma database.
    """
    global rag_chain
    
    print("Loading pre-built RAG pipeline with Google Gemini...")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

    # Load the pre-built vector store from disk
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    # --- Use the same prompt as before ---
    system_prompt = (
        "You are a helpful and friendly assistant for the Google Cloud Community Day Bhopal event. "
        "Your goal is to answer questions accurately based on the provided context. "
        "You must follow these rules:\n"
        "1.  **CRITICAL RULE: When asked for a list of speakers ('who are the speakers', 'speakers kon kon he', etc.), you MUST list ALL speakers found in the context. Do not summarize or shorten the list under any circumstances.**\n"
        "2.  **Always use Markdown for formatting.** Use bullet points (`-` or `*`) for lists and bold (`**text**`) for names, titles, and important terms. For the speaker list, format each entry as: `- **Speaker Name**: Title at Company, speaking on \"Topic\".`\n"
        "3.  If the user asks in a mix of Hindi and English (Hinglish), understand it converting it to english and reply in clear, simple English.\n"
        "4.  If the context does not contain the answer, politely say 'I don't have that specific information, but I can tell you about speakers, the agenda, or the venue.\n For more details, please check the official website provided on top right corner or the event's social media.'\n"
        "5.  Keep your answers concise but complete."
        "6.  If the user asks for socials such as linkedin, reply with 'Please check the official website provided on top right corner or the event's social media for such information.'"
        "7. When asked for location of the event reply provide full location as addredded in the knowledge_base with text 'Please check at bottom of the page forevent location'."
        "8. Remember the event is not free if asked is the event free? Reply with 'Tickets are available plase visit the official page for registration'."
        "9. When the user says 'thank you' reply with 'You are welcome, see you at CCD bhopal!'"
        "\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("Gemini RAG pipeline loaded successfully.")


# --- Bot Response Logic ---
def get_bot_response(user_message):
    """
    Acts as a router. First checks for structured queries, then falls back to RAG.
    """
    msg = user_message.lower()

    # Priority 1: Check for a request for the full speaker list
    if "speakers" in msg or "who is speaking" in msg or "kon kon he" in msg:
        reply_text = "We have an amazing lineup of speakers at Google Cloud Community Day Bhopal! They include:\n"
        for speaker in event_data.get('speakers', []):
            reply_text += f"- **{speaker['name']}**: {speaker['title']} at {speaker['company']}, speaking on \"{speaker['topic']}\"\n"
        reply_text += "\nWould you like to know more about a specific speaker?"
        return reply_text

    # Priority 2: Check if asking about a specific speaker by name
    for speaker in event_data.get('speakers', []):
        if speaker['name'].lower() in msg:
            reply_text = (
                f"Here are the details for **{speaker['name']}**:\n"
                f"- **Title**: {speaker['title']} at {speaker['company']}\n"
                f"- **Topic**: \"{speaker['topic']}\"\n"
            )
            return reply_text

    # Priority 3: Fallback to the RAG pipeline for all other questions
    if not rag_chain:
        return "My conversational AI is still warming up. Please try again in a moment."
    try:
        response = rag_chain.invoke({"input": user_message})
        return response.get("answer", "I couldn't find an answer to that.")
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return 'An error occurred while processing your request.'

# --- Flask API Endpoints ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_bot():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data.get('message')
    bot_reply = get_bot_response(user_message)
    return jsonify({'reply': bot_reply})

initialize_rag_pipeline()
if __name__ == '__main__':
    app.run(debug=True, port=5000)
