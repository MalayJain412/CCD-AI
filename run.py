# run.py - AI Chatbot Backend using a PRE-BUILT RAG pipeline

import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Setup ---
app = Flask(__name__)
CORS(app)
load_dotenv()

# --- Global Variables ---
rag_chain = None
CHROMA_DB_PATH = "./chroma_db" # Path to the pre-built database

# --- LangChain RAG Pipeline Initialization ---
def initialize_rag_pipeline():
    """
    This function now LOADS the pre-built RAG chain from the persisted Chroma database.
    This is a very fast operation.
    """
    global rag_chain
    
    print("Loading pre-built RAG pipeline with Google Gemini...")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return
    
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"CRITICAL ERROR: Chroma DB not found at {CHROMA_DB_PATH}. Please ensure the build_db.py script ran successfully during deployment.")
        return

    # 1. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

    # 2. Initialize the embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    # 3. Load the Chroma vector store from disk
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

    # 4. Create the retriever
    retriever = vectorstore.as_retriever()

    # 5. Create the prompt template (same as before)
    system_prompt = (
        "You are a helpful and friendly AI assistant for the Google Cloud Community Day Bhopal event. "
        "Your goal is to answer questions accurately based on the provided context. "
        "You must follow these rules:\n"
        "1.  **CRITICAL RULE: When asked for a list of speakers ('who are the speakers', 'speakers kon kon he', etc.), you MUST list ALL speakers found in the context. Do not summarize or shorten the list.**\n"
        "2.  **Always use Markdown for formatting.** Use bullet points (`-`) for lists and bold (`**text**`) for names, titles, and important terms. For the speaker list, format each entry as: `- **Speaker Name**: Title at Company, speaking on \"Topic\".`\n"
        "3.  If a user asks for a speaker's social media like LinkedIn, you MUST reply with: 'For the most up-to-date professional profiles, please check the official event website.' Do not provide the direct link.\n"
        "4.  If the user asks in a mix of Hindi and English (Hinglish), like 'event kaha par he', understand it and reply in clear, simple English.\n"
        "5.  If asked about the event location, provide the answer from the context and also add: 'You can find a direct link to the location at the bottom of the chat window.'\n"
        "6.  If asked if the event is free, you MUST reply: 'The event requires a ticket for entry. For details on pricing and registration, please visit the official website.'\n"
        "7.  If the user says 'thank you' or a similar phrase of gratitude, you MUST reply with: 'You're welcome! See you at GCCD Bhopal!'\n"
        "8.  If the context does not contain the answer to a question, politely say: 'I don't have that specific information, but I can tell you about speakers, the agenda, or the venue.'\n"
        "9.  Keep your answers concise but complete."
        "\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 6. Create the RAG chain (same as before)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("Gemini RAG pipeline loaded successfully.")


# --- Flask API Endpoints ---
@app.route('/')
def home():
    """Serves the simple chatbot HTML interface."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_bot():
    """
    This endpoint receives a user's message, processes it using the RAG chain,
    and returns the bot's reply.
    """
    if not rag_chain:
        return jsonify({'reply': 'Sorry, the AI model is not ready yet. Please try again in a moment.'}), 503

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data.get('message')

    try:
        response = rag_chain.invoke({"input": user_message})
        bot_reply = response.get("answer", "I couldn't find an answer to that.")
        return jsonify({'reply': bot_reply})
        
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return jsonify({'reply': 'An error occurred while processing your request.'}), 500

# --- SCRIPT EXECUTION ---
# This ensures the pipeline is initialized when Gunicorn imports the file on Render.
initialize_rag_pipeline()

# This block is now ONLY for local development.
if __name__ == '__main__':
    app.run(debug=True, port=5000)
