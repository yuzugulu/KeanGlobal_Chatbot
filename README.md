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

Make sure you have:

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

Vite and React are already defined in `package.json`.

Just run:

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

If you see the chat interface → success.

---

## 🤖 Install Local AI Model (Ollama)

### macOS

```bash
brew install ollama
```

### Windows

Download installer from:

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

## 🔁 Daily Git Workflow

Before starting work:

```bash
git checkout main
git pull origin main
git checkout -b your-feature-name
```

After finishing work:

```bash
git add .
git commit -m "Describe your changes"
git push origin your-feature-name
```

Create a Pull Request on GitHub.

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

## 📅 Weekly Reporting

Each team member documents:

- What they completed  
- What they are working on next  

in the shared weekly tracking document.
