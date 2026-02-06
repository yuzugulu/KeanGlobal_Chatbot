KeanGlobal – Multilingual Campus Concierge

KeanGlobal is a chat-first, AI-powered campus concierge system with an optional context-aware map. The system assists students with Kean University policies, services, and navigation.

🧰 Tech Stack

Frontend: React + Vite

Backend: FastAPI (planned)

AI Model: Mistral (local via Ollama)

Vector Database: Chroma (planned)

Map: Leaflet / OpenStreetMap (planned)

🚀 Quick Start (All Team Members)
1️⃣ Clone Repository
git clone https://github.com/Dav1dKean/KeanGlobal_Chatbot.git
cd KeanGlobal_Chatbot

2️⃣ Install Frontend Dependencies
npm install

3️⃣ Run Frontend
npm run dev


Open:

http://localhost:5173


If chat UI loads → success.

🤖 Install Local AI Model (Ollama)
Mac
brew install ollama

Windows

Download from:

https://ollama.com

Pull Model
ollama run mistral


Type:

Hello


If model replies → working.

🔁 Daily Git Workflow

Before starting work:

git checkout main
git pull origin main
git checkout -b your-branch-name


After finishing work:

git add .
git commit -m "Describe your changes"
git push origin your-branch-name


Create Pull Request on GitHub.

📁 Project Structure
KeanGlobal_Chatbot/
 ├─ src/              # React frontend
 ├─ backend/          # FastAPI backend (planned)
 ├─ public/
 ├─ README.md

📌 Team Roles

Member 1: AI & Backend

Member 2: Mapping & Visualization

Member 3: Frontend

Member 4: Database & Admin

🧪 Troubleshooting

If blank screen:

npm run dev


If npm errors:

npm install

📅 Weekly Reporting

Each member records:

What they completed

What they are working on next

in the shared weekly tracking document.
