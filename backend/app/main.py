from typing import Literal, Optional
import os
import re
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import chromadb
from chromadb.utils import embedding_functions

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# CONFIG

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_CONNECT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SECONDS", "5"))
OLLAMA_READ_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_READ_TIMEOUT_SECONDS", "30"))
OLLAMA_HEALTH_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_HEALTH_TIMEOUT_SECONDS", "5"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "1"))
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000",
)

# APP SETUP

app = FastAPI(title="KeanGlobal Backend", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD CALENDARS

DATA_FOLDER = Path("data")
calendar_data = {}

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def tokenize(text: str) -> set[str]:
    return set(normalize(text).split())

def is_term_header(line: str) -> bool:
    return bool(re.match(r"^(Fall|Spring|Winter|Summer)\s\d{4}\s(Semester|Term)$", line.strip()))

def load_calendars():
    for file in DATA_FOLDER.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_term = None
        for raw_line in lines:
            line = raw_line.strip()
            if is_term_header(line):
                current_term = normalize(line)
                calendar_data[current_term] = {}
            elif line.startswith("-") and ":" in line and current_term:
                event, date = line[1:].split(":", 1)
                calendar_data[current_term][normalize(event)] = date.strip()

load_calendars()

# CHROMA RAG

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "app" / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

policy_collection = client.get_collection(
    name="kean_knowledge",
    embedding_function=embedding_function
)

# MODELS

class ChatRequest(BaseModel):
    message: str

# CALENDAR LOGIC

def detect_event_category(question: str) -> Optional[str]:
    q = normalize(question)
    if any(p in q for p in ["begin", "start", "first day", "opening"]):
        return "start"
    if any(p in q for p in ["end", "finish", "last day"]):
        return "end"
    if "recess" in q or "break" in q:
        return "recess"
    if "immunization" in q:
        return "immunization"
    if "registration" in q:
        return "registration"
    if "withdraw" in q:
        return "withdrawal"
    if "exam" in q:
        return "exam"
    return None

def is_calendar_question(text: str) -> bool:
    return bool(re.search(r"(fall|spring|winter|summer)\s\d{4}", normalize(text)))

# TIME / LOCATION

NYC_TIMEZONE = ZoneInfo("America/New_York")

def get_time_response(prompt: str):
    if "time" not in prompt.lower():
        return None
    now = datetime.now(NYC_TIMEZONE)
    return f"The current time in New York is {now.strftime('%I:%M %p')}."

# OLLAMA

def build_ollama_timeout():
    return httpx.Timeout(
        connect=OLLAMA_CONNECT_TIMEOUT_SECONDS,
        read=OLLAMA_READ_TIMEOUT_SECONDS,
        write=OLLAMA_CONNECT_TIMEOUT_SECONDS,
        pool=OLLAMA_CONNECT_TIMEOUT_SECONDS,
    )

async def query_ollama(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=build_ollama_timeout()) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "No response.")

# ROUTES

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    user_text = req.message.strip()

    if is_calendar_question(user_text):
        term_match = re.search(r"(fall|spring|winter|summer)\s(\d{4})", normalize(user_text))
        if not term_match:
            return {"answer": "Please specify a term and year.", "intent": "calendar"}

        term = f"{term_match.group(1)} {term_match.group(2)}"
        matched = next((t for t in calendar_data if term in t), None)
        if not matched:
            return {"answer": "Term not found.", "intent": "calendar"}

        category = detect_event_category(user_text)
        if not category:
            return {"answer": "Event not recognized.", "intent": "calendar"}

        for event, date in calendar_data[matched].items():
            if category in event:
                return {"answer": f"{event.title()}: {date}", "intent": "calendar"}

    reply = await query_ollama(user_text)
    return {"answer": reply, "intent": "general"}