from typing import Optional
import os
import re
import csv
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import chromadb
from chromadb.utils import embedding_functions

from fastapi import FastAPI
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
LOCATION_CSV_PATH = BASE_DIR.parent / "src" / "data" / "campus_locations_main_east_full.csv"
POLICY_FOLDER = BASE_DIR / "Policies"
RAG_DATA_FOLDER = BASE_DIR / "data"
FAQ_INTENT_PATH = RAG_DATA_FOLDER / "faq_intent_keywords.json"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

policy_collection = client.get_or_create_collection(
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

def event_matches_category(event_name: str, category: str) -> bool:
    event_text = normalize(event_name)
    category_tokens = {
        "start": ("start", "begins", "term begins", "first day", "opening"),
        "end": ("end", "ends", "term ends", "last day", "finish"),
        "recess": ("recess", "break"),
        "immunization": ("immunization"),
        "registration": ("registration"),
        "withdrawal": ("withdraw", "withdrawal"),
        "exam": ("exam", "final"),
    }
    return any(token in event_text for token in category_tokens.get(category, (category,)))

def is_calendar_question(text: str) -> bool:
    return bool(re.search(r"(fall|spring|winter|summer)\s\d{4}", normalize(text)))

# TIME / LOCATION

NYC_TIMEZONE = ZoneInfo("America/New_York")
LOCATION_KEYWORDS = (
    "where is",
    "where's",
    "directions",
    "direction",
    "how do i get",
    "how to get",
    "route",
    "navigate",
    "map",
    "locate",
    "find",
    "way to",
    "get to",
)
DIRECTION_KEYWORDS = (
    "directions",
    "direction",
    "how do i get",
    "how to get",
    "route",
    "navigate",
    "way to",
    "get to",
)
DEFAULT_FAQ_INTENT_KEYWORDS = {
    "admissions": ("admission", "apply", "application", "accepted", "enroll"),
    "tuition_fees": ("tuition", "cost", "fees", "payment", "bill", "bursar"),
    "financial_aid": ("financial aid", "fafsa", "scholarship", "grant", "loan"),
    "registration": ("register", "registration", "add/drop", "drop", "schedule"),
    "calendar_deadline": ("deadline", "due date", "when is", "date", "semester starts", "semester ends"),
    "housing": ("housing", "dorm", "residence hall", "roommate", "move in"),
    "parking_transport": ("parking", "permit", "lot", "shuttle", "bus", "train"),
    "library": ("library", "books", "study room", "citation"),
    "it_support": ("wifi", "email", "password", "it", "tech support", "portal"),
    "programs": ("major", "minor", "program", "degree", "curriculum"),
    "policies": ("policy", "policies", "rule", "conduct", "procedure"),
}
FAQ_INTENT_KEYWORDS = {}

campus_locations = []
campus_location_by_id = {}
fallback_rag_docs = []

def load_campus_locations():
    rows = []
    if not LOCATION_CSV_PATH.exists():
        return rows

    with open(LOCATION_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = (row.get("id") or "").strip()
            name = (row.get("name") or "").strip()
            if not row_id or not name:
                continue
            row_type = (row.get("type") or "").strip().lower() or "location"
            rows.append(
                {
                    "id": row_id,
                    "name": name,
                    "campus": (row.get("campus") or "").strip() or "Main",
                    "type": row_type,
                    "parent": (row.get("parent") or "").strip(),
                }
            )
    return rows

def load_fallback_rag_docs():
    docs = []
    for folder, doc_type in ((RAG_DATA_FOLDER, "calendar"), (POLICY_FOLDER, "policy")):
        if not folder.exists():
            continue
        for file in folder.glob("*.txt"):
            try:
                text = file.read_text(encoding="utf-8")
            except Exception:
                continue

            if not text.strip():
                continue

            chunks = []
            step = 1200
            for i in range(0, len(text), step):
                chunk = text[i : i + step].strip()
                if chunk:
                    chunks.append(chunk)

            for idx, chunk in enumerate(chunks):
                docs.append(
                    {
                        "source": file.name,
                        "type": doc_type,
                        "chunk_id": f"{file.stem}_{idx}",
                        "text": chunk,
                        "tokens": tokenize(chunk),
                    }
                )
    return docs

def load_faq_intent_keywords():
    if FAQ_INTENT_PATH.exists():
        try:
            with open(FAQ_INTENT_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                cleaned = {}
                for topic, keywords in raw.items():
                    if not isinstance(topic, str) or not isinstance(keywords, list):
                        continue
                    cleaned_list = [str(k).strip().lower() for k in keywords if str(k).strip()]
                    if cleaned_list:
                        cleaned[topic.strip()] = tuple(cleaned_list)
                if cleaned:
                    return cleaned
        except Exception:
            pass
    return DEFAULT_FAQ_INTENT_KEYWORDS

def build_location_aliases(place: dict) -> list[str]:
    aliases = {
        place["id"],
        place["id"].replace("_", " "),
        place["name"],
    }

    acronym_match = re.findall(r"\(([A-Za-z0-9&/ ]+)\)", place["name"])
    for value in acronym_match:
        aliases.add(value)

    return [normalize(alias) for alias in aliases if normalize(alias)]

def is_location_question(text: str) -> bool:
    q = normalize(text)
    return any(keyword in q for keyword in LOCATION_KEYWORDS)

def should_use_current_location(text: str) -> bool:
    q = normalize(text)
    return any(keyword in q for keyword in DIRECTION_KEYWORDS)

def find_location_destination_id(text: str) -> Optional[str]:
    q = normalize(text)
    padded_query = f" {q} "
    best_place = None
    best_score = 0

    for place in campus_locations:
        aliases = build_location_aliases(place)
        score = 0

        for alias in aliases:
            if not alias:
                continue
            padded_alias = f" {alias} "
            if q == alias:
                score = max(score, 100 + len(alias))
            elif padded_alias in padded_query:
                score = max(score, 85 + len(alias))
            elif alias in q:
                score = max(score, 70 + len(alias))

        if place["type"] == "building":
            score += 10
        elif place["type"] == "entrance":
            score += 2

        if score > best_score:
            best_place = place
            best_score = score

    if best_place and best_score >= 75:
        return best_place["id"]
    return None

def detect_faq_intent(text: str) -> Optional[str]:
    q = normalize(text)
    best_topic = None
    best_score = 0

    for topic, keywords in FAQ_INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            normalized_keyword = normalize(keyword)
            if not normalized_keyword:
                continue
            if normalized_keyword in q:
                score += len(normalized_keyword)
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic if best_score > 0 else None

def get_time_response(prompt: str):
    if "time" not in prompt.lower():
        return None
    now = datetime.now(NYC_TIMEZONE)
    return f"The current time in New York is {now.strftime('%I:%M %p')}."

campus_locations = load_campus_locations()
campus_location_by_id = {place["id"]: place for place in campus_locations}
fallback_rag_docs = load_fallback_rag_docs()
FAQ_INTENT_KEYWORDS = load_faq_intent_keywords()

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

def retrieve_rag_context(question: str, max_results: int = 4) -> list[str]:
    try:
        if policy_collection.count() == 0:
            return []
        results = policy_collection.query(
            query_texts=[question],
            n_results=max_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    context_blocks = []
    for index, doc in enumerate(documents[0]):
        if not doc:
            continue
        metadata = metadatas[0][index] if index < len(metadatas[0]) else {}
        distance = distances[0][index] if index < len(distances[0]) else None
        source = metadata.get("source", "unknown")
        content_type = metadata.get("type", "knowledge")
        score = f"{distance:.3f}" if isinstance(distance, (int, float)) else "na"
        context_blocks.append(
            f"[{index + 1}] source={source} type={content_type} score={score}\n{doc}"
        )
    return context_blocks

def retrieve_fallback_context(question: str, max_results: int = 4) -> list[str]:
    if not fallback_rag_docs:
        return []

    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    scored = []
    for doc in fallback_rag_docs:
        overlap = len(query_tokens & doc["tokens"])
        if overlap == 0:
            continue
        score = overlap / max(1, len(query_tokens))
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:max_results]
    context_blocks = []
    for index, (score, doc) in enumerate(top):
        context_blocks.append(
            f"[{index + 1}] source={doc['source']} type={doc['type']} score={score:.3f}\n{doc['text']}"
        )
    return context_blocks

def build_rag_prompt(user_text: str, context_blocks: list[str], faq_topic: Optional[str] = None) -> str:
    if not context_blocks:
        return user_text

    joined_context = "\n\n".join(context_blocks)
    intent_line = f"Detected FAQ topic: {faq_topic}.\n" if faq_topic else ""
    return (
        "You are KeanGlobal assistant. Use the provided campus context as primary source.\n"
        "If the answer is not in context, say that clearly and then provide a best-effort answer.\n\n"
        f"{intent_line}"
        f"Context:\n{joined_context}\n\n"
        f"User question: {user_text}"
    )

def build_fallback_answer(question: str, context_blocks: list[str]) -> str:
    if not context_blocks:
        return "I couldn't reach Ollama and I don't have matching campus context yet."

    question_tokens = tokenize(question)
    best_line = None
    best_score = 0

    for block in context_blocks:
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("[") or line.startswith("source="):
                continue
            line_tokens = tokenize(line)
            overlap = len(question_tokens & line_tokens)
            if overlap > best_score:
                best_score = overlap
                best_line = line

    if best_line:
        return f"Ollama is unavailable right now. Best match from campus records: {best_line}"
    return f"Ollama is unavailable right now. Top context match: {context_blocks[0][:280]}"

# ROUTES

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    user_text = req.message.strip()
    faq_topic = detect_faq_intent(user_text)

    if is_location_question(user_text):
        destination_id = find_location_destination_id(user_text)
        use_current_location = should_use_current_location(user_text)
        location_mode = "directions" if use_current_location else "highlight"
        if destination_id:
            destination = campus_location_by_id.get(destination_id, {})
            return {
                "answer": f"{destination.get('name', 'That location')} is on {destination.get('campus', 'campus')}. Opening map.",
                "intent": "location",
                "destination_id": destination_id,
                "use_current_location": use_current_location,
                "location_mode": location_mode,
            }
        return {
            "answer": "Opening campus map. Please pick a destination or search the directory.",
            "intent": "location",
            "destination_id": None,
            "use_current_location": use_current_location,
            "location_mode": location_mode,
        }

    if is_calendar_question(user_text):
        term_match = re.search(r"(fall|spring|winter|summer)\s(\d{4})", normalize(user_text))
        if term_match:
            term = f"{term_match.group(1)} {term_match.group(2)}"
            matched = next((t for t in calendar_data if term in t), None)
            category = detect_event_category(user_text)
            if matched and category:
                for event, date in calendar_data[matched].items():
                    if event_matches_category(event, category):
                        return {"answer": f"{event.title()}: {date}", "intent": "calendar"}

    context_blocks = retrieve_rag_context(user_text)
    if not context_blocks:
        fallback_query = f"{user_text} {faq_topic.replace('_', ' ') if faq_topic else ''}".strip()
        context_blocks = retrieve_fallback_context(fallback_query)
    prompt = build_rag_prompt(user_text, context_blocks, faq_topic)

    try:
        reply = await query_ollama(prompt)
    except httpx.HTTPError:
        fallback_answer = build_fallback_answer(user_text, context_blocks)
        return {"answer": fallback_answer, "intent": "faq" if faq_topic else "general", "faq_topic": faq_topic, "sources_used": len(context_blocks)}
    except Exception:
        fallback_answer = build_fallback_answer(user_text, context_blocks)
        return {"answer": fallback_answer, "intent": "faq" if faq_topic else "general", "faq_topic": faq_topic, "sources_used": len(context_blocks)}

    return {"answer": reply, "intent": "faq" if faq_topic else "general", "faq_topic": faq_topic, "sources_used": len(context_blocks)}
