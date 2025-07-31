# ♻️ Updating Chroma DB and Redeploying to Azure

This guide explains how to **update the Chroma vector store (**``**)** for your chatbot and redeploy your Docker container to **Azure App Service**.

---

## 📆 Prerequisites

Make sure you have:

- Docker Desktop installed and running
- Azure CLI installed
- Logged into Azure via CLI:
  ```bash
  az login
  ```
- Valid `.env` file in your local project root with updated Google API keys
- Access to your Azure Container Registry (ACR): `ccdchatbotacr`
- Your Azure App Service is configured: `ccd-ai-chatbot`

---

## 🔧 Step 1: Rebuild Docker Image with Updated Chroma DB

Inside your project root:

### PowerShell:

```powershell
docker build `
  --build-arg GOOGLE_API_KEY_1=$((Get-Content .env | Select-String "GOOGLE_API_KEY_1").Line.Split('=')[1].Trim('"')) `
  -t ccdchatbotacr.azurecr.io/chatbot:latest .
```

This command:

- Uses `GOOGLE_API_KEY_1` from your local `.env`
- Re-runs `build_db.py` inside the Docker container

---

## 🚀 Step 2: Push to Azure Container Registry (ACR)

```bash
az acr login --name ccdchatbotacr
```

```bash
docker push ccdchatbotacr.azurecr.io/chatbot:latest
```

---

## ⚙️ Step 3: Update Azure Web App with New Image

> **Note**: Use `--container-*` flags (the older `--docker-*` flags are deprecated)

```powershell
az webapp config container set `
  --resource-group ccd-chatbot-rg `
  --name ccd-ai-chatbot `
  --container-image-name ccdchatbotacr.azurecr.io/chatbot:latest `
  --container-registry-url https://ccdchatbotacr.azurecr.io `
  --container-registry-user ccdchatbotacr `
  --container-registry-password <YOUR_ACR_PASSWORD>
```

> Get ACR password:

```bash
az acr credential show --name ccdchatbotacr
```

---

## 🔄 Step 4: Restart Web App

```bash
az webapp restart --resource-group ccd-chatbot-rg --name ccd-ai-chatbot
```

---

## 🔎 Step 5: Confirm with Live Logs

```bash
az webapp log tail --name ccd-ai-chatbot --resource-group ccd-chatbot-rg
```

Watch for:

- `build_db.py` success logs
- Any Chroma loading errors
- Google API key errors (expired/invalid)

Visit: [https://bot.ccd.bhopal.dev](https://bot.ccd.bhopal.dev)

---

## ✅ Done!

You've successfully:

- Rebuilt Chroma vector DB
- Packaged it into Docker
- Pushed and redeployed to Azure

---

## 📅 Optional: Automate via GitHub Actions

Ask me if you'd like a `.github/workflows/deploy.yml` CI/CD file!

---

