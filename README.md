# 📘 KeanGlobal – Multilingual Campus Concierge & Navigation System

KeanGlobal is a chat-first, AI-powered campus concierge system with an optional context-aware campus map. The system assists students with Kean University policies, services, and navigation using a locally hosted open-source language model.

---

## 🧰 Tech Stack

- Frontend: React + Vite  
- Backend: FastAPI (planned)  
- AI Model: Mistral (local via Ollama)  
- Vector Database: Chroma (planned)  
- Mapping: Leaflet / OpenStreetMap (planned)

---

## ✅ Prerequisites

Install the following:

- Node.js (v18 or newer)  
  https://nodejs.org  
- Git  
  https://git-scm.com  

Verify installation:

```bash
node -v
npm -v
git --version
```

---

## 📥 Clone Repository

```bash
git clone https://github.com/Dav1dKean/KeanGlobal_Chatbot.git
cd KeanGlobal_Chatbot
```

---

## 📦 Install Frontend Dependencies

React and Vite are already defined in `package.json`.

Run:

```bash
npm install
```

---

## ▶️ Run Frontend

```bash
npm run dev
```

Open:

```
http://localhost:5173
```

If the chat interface loads → success.

---

## ⚙️ Run Backend (FastAPI)

From project root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
http://localhost:8000/health
```

Frontend sends chat requests to:

```bash
http://localhost:8000/chat
```

Optional frontend env override:

```bash
VITE_API_URL=http://localhost:8000
```

Optional backend Ollama timeout tuning (useful for first model load):

```bash
export OLLAMA_CONNECT_TIMEOUT_SECONDS=5
export OLLAMA_READ_TIMEOUT_SECONDS=120
export OLLAMA_MAX_RETRIES=1
```

---

## 🤖 Install Local AI Model (Ollama)

### macOS

```bash
brew install ollama
```

### Windows

Download installer:

```
https://ollama.com
```

---

### Pull Mistral Model

```bash
ollama run mistral
```

---

### Test Model

Type:

```bash
Hello
```

If model responds → working.

---

## 🧠 How AI Will Be Used

The backend will call the locally running Mistral model through Ollama and combine it with university documents using Retrieval-Augmented Generation (RAG).

No API keys are required.

---

## ⬇️ Pull Latest Changes

Always pull the newest code before starting work:

```bash
git checkout main
git pull origin main
```

---

## 🌿 Creating and Switching Branches

Create a new branch:

```bash
git checkout -b feature-name
```

Example:

```bash
git checkout -b backend-ai
```

Switch to an existing branch:

```bash
git checkout backend-ai
```

List branches:

```bash
git branch
```

---

## 💾 Save Your Work (Commit)

```bash
git add .
git commit -m "Describe your changes"
```

Example:

```bash
git commit -m "Add backend skeleton"
```

---

## ⬆️ Push Your Branch

```bash
git push origin feature-name
```

Example:

```bash
git push origin backend-ai
```

---

## 🔀 Create Pull Request (PR)

1. Go to GitHub repository  
2. Click **Compare & Pull Request**  
3. Add short description  
4. Submit PR  
5. Team reviews → Merge to main  

Do NOT push directly to main.

---

## 📁 Project Structure

```
KeanGlobal_Chatbot/
 ├─ src/              # React frontend
 ├─ backend/          # FastAPI backend (planned)
 ├─ public/
 ├─ README.md
```

---

## 👥 Team Roles

- Member 1 – AI & Backend  
- Member 2 – Mapping & Visualization  
- Member 3 – Frontend  
- Member 4 – Database & Admin  

---

My Contribution
I contributed to the backend intelligence layer and multilingual chatbot behavior of KeanGlobal. My work focused on improving how the assistant understands and responds to users across multiple languages, while keeping responses reliable and context-aware.

Key contributions:

Implemented automatic language detection for English, Turkish, Spanish, Mandarin Chinese, Urdu, and Korean
Enforced same-language response behavior so the chatbot replies in the language used by the user
Improved backend prompt flow for more consistent multilingual responses
Added language-aware fallback behavior when model output or retrieval context was unavailable
Integrated retrieval-first chat flow so the assistant checks available university knowledge before generating a final answer
Helped stabilize backend startup and local development setup for running the system with FastAPI, Ollama, and local RAG components

---

## 📅 Weekly Reporting

Each team member documents:

- What they completed  
- What they are working on next  

in the shared weekly tracking document.

---

## 🚀 Future Work

- Document ingestion & embeddings  
- Vector database integration  
- Real campus map with routing  
- Admin dashboard  
- Multilingual support  

---

## 🏆 Academic Statement

This project demonstrates full-stack development, intelligent systems design, retrieval-augmented generation, and human-centered UI design.
