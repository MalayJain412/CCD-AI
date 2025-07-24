# **GCCD Bhopal AI Chatbot**

A web-based AI assistant for the Google Cloud Community Day Bhopal event, built with Flask, Google Gemini, and a Retrieval-Augmented Generation (RAG) pipeline.

---

## **🚀 Project Overview**

This repository contains the source code for the GCCD Bhopal AI Chatbot, an intelligent assistant designed to provide attendees with comprehensive information about the event. The application leverages a sophisticated AI architecture to understand natural language queries (including Hinglish) and provide accurate, context-aware responses.

The project is deployed on both **Google Cloud Platform (GCP)** and **Microsoft Azure**, ensuring scalability and reliability.

---

## **✨ Key Features**

- **Conversational AI:** Powered by Google's Gemini 1.5 Flash model to understand and respond to a wide range of questions.  
- **Retrieval-Augmented Generation (RAG):** Uses a vector store built from event-specific knowledge to provide factually grounded answers.  
- **API Key Pool & Fallback:** Automatically switches between multiple Gemini API keys to handle rate limits.  
- **Responsive Frontend:** Modern UI with Tailwind CSS, works across all screen sizes.  
- **Multi-Cloud Ready:** Supports deployment on GCP Compute Engine or Azure App Service via Docker.

---

## **🛠️ Technology Stack**

- **Backend:** Python, Flask  
- **AI / LLM:** Google Gemini 1.5 Flash  
- **AI Framework:** LangChain (RAG pipeline)  
- **Vector DB:** ChromaDB  
- **Frontend:** HTML, Tailwind CSS, JavaScript  
- **Deployment:** GCP Compute Engine, Azure App Service (Docker)  
- **DevOps:** Gunicorn, Nginx, tmux, Docker  
- **(Optional):** WhatsApp Integration via n8n

---

## **⚙️ Local Development Setup**

### **📦 Prerequisites**

- Python 3.9+  
- pip  
- Google Gemini API Key(s)

### **🧪 Steps to Run Locally**

1. **Clone the Repository:**

```bash
git clone https://github.com/your-username/CCD-AI.git
cd CCD-AI
````

2. **Create Virtual Environment:**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables:**

Create a `.env` file in root with your keys:

```env
GOOGLE_API_KEY_1="your_first_key"
GOOGLE_API_KEY_2="your_second_key"
SECRET_KEY="your_secret_here"
```

5. **Build Vector Store:**

```bash
python build_db.py
```

6. **Run the App:**

```bash
python run.py
# App available at http://127.0.0.1:5000
```

---

## **☁️ Deployment Options**

### ✅ **1. Google Cloud Platform (GCP)**

Deployed on a Compute Engine instance using the following stack:

* Gunicorn server inside `tmux`
* Nginx reverse proxy on port 80
* Let's Encrypt SSL for HTTPS
* Custom domain mapping via Cloud DNS

Steps:

1. SSH into VM and clone repo
2. Setup Python environment and `.env`
3. Run `build_db.py`
4. Serve with Gunicorn in `tmux`
5. Configure Nginx and Certbot

📍 Example Domain: `https://chatbot.bhopal.dev`

---

### ✅ **2. Microsoft Azure (Docker + App Service)**

Deployed using Docker container to **Azure App Service** with optional **custom domain** mapping (e.g., `https://bot.ccd.bhopal.dev`)

Steps:

1. Create `Dockerfile` and `.dockerignore`
2. Build Docker image locally
3. Push to Azure Container Registry (ACR)
4. Create App Service Plan and Web App
5. Set environment variables using `.env`
6. (Optional) Map custom domain via TXT + CNAME

📍 Example Domain: `https://bot.ccd.bhopal.dev`

---

## **📊 Logging & Monitoring**

### Azure:

```bash
az webapp log config \
  --name ccd-ai-chatbot \
  --resource-group ccd-chatbot-rg \
  --web-server-logging filesystem \
  --detailed-error-messages true \
  --failed-request-tracing true

az webapp log tail \
  --name ccd-ai-chatbot \
  --resource-group ccd-chatbot-rg
```

### GCP:

* Logs handled via `journalctl` or `gunicorn` logs inside `tmux` session

---

## **📄 License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 🙌 Acknowledgements

Special thanks to Google Developers Group Bhopal and the Cloud Community Day team for organizing this initiative.

---

> Made with ❤️ for GCCD Bhopal 2025

