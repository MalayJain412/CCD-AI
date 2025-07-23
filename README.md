# **GCCD Bhopal AI Chatbot**

A web-based AI assistant for the Google Cloud Community Day Bhopal event, built with Flask, Google Gemini, and a Retrieval-Augmented Generation (RAG) pipeline.

## **🚀 Project Overview**

This repository contains the source code for the GCCD Bhopal AI Chatbot, an intelligent assistant designed to provide attendees with comprehensive information about the event. The application leverages a sophisticated AI architecture to understand natural language queries (including Hinglish) and provide accurate, context-aware responses.

The project is deployed on the Google Cloud Platform, utilizing a robust stack to ensure scalability and reliability.

### **Key Features**

* **Conversational AI:** Powered by Google's Gemini 1.5 Flash model to understand and respond to a wide range of questions.  
* **Retrieval-Augmented Generation (RAG):** The AI uses a pre-built vector database created from a text-based knowledge base, ensuring answers are factually grounded in the event's official information.  
* **API Key Fallback:** A resilient API key pool system automatically switches between multiple API keys to handle rate limits and ensure high availability.  
* **Responsive Frontend:** A clean, modern user interface built with HTML and Tailwind CSS that works seamlessly on both desktop and mobile devices.  
* **Production-Ready Deployment:** Deployed on Google Cloud Platform using Gunicorn, Nginx (as a reverse proxy), and secured with a free SSL certificate from Let's Encrypt.

## **🛠️ Technology Stack**

* **Backend:** Python, Flask  
* **AI / LLM:** Google Gemini 1.5 Flash  
* **AI Framework:** LangChain (for the RAG pipeline)  
* **Vector Database:** ChromaDB  
* **Frontend:** HTML, Tailwind CSS, JavaScript  
* **Deployment:** Google Cloud Platform (GCP), Gunicorn, Nginx, Docker (for n8n)  
* **(Optional) WhatsApp Integration:** n8n (self-hosted)

## **⚙️ Local Development Setup**

Follow these steps to run the project on your local machine.

### **Prerequisites**

* Python 3.9+  
* pip (Python package installer)  
* A Google Gemini API Key

### **Instructions**

1. **Clone the Repository:**  
   git clone https://github.com/your-username/CCD-AI.git  
   cd CCD-AI

2. **Create a Virtual Environment:**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows: venv\\Scripts\\activate

3. **Install Dependencies:**  
   pip install \-r requirements.txt

4. **Create the .env File:**  
   * Create a file named .env in the root directory.  
   * Add your Google Gemini API key(s) to this file:  
     GOOGLE\_API\_KEY\_1="your\_first\_api\_key\_here"  
     GOOGLE\_API\_KEY\_2="your\_second\_api\_key\_here"

5. **Build the AI's Knowledge Base:**  
   * Run the build\_db.py script once to create the chroma\_db folder. This script reads knowledge\_base/knowledge\_base.txt, creates the AI embeddings, and saves them to disk.  
     python build\_db.py

6. **Run the Application:**  
   * Start the Flask development server:  
     python run.py

   * The chatbot will be accessible at http://127.0.0.1:5000.

## **☁️ Deployment**

This application is deployed on a **Google Cloud Platform (GCP) Compute Engine** VM instance. The deployment process involves:

1. Provisioning an e2-medium VM.  
2. Cloning the repository and setting up the Python environment.  
3. Creating the .env file with the API keys on the server.  
4. Running the build\_db.py script to create the production vector store.  
5. Running the application with a **Gunicorn** production server inside a **tmux** session for persistence.  
6. Configuring **Nginx** as a reverse proxy to handle traffic on port 80\.  
7. Securing the custom domain with a free SSL certificate from **Let's Encrypt** via Certbot.

## **📄 License**

This project is licensed under the MIT License. See the LICENSE file for details.