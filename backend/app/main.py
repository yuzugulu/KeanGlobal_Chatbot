from typing import Optional
import os
import re
import json
import asyncio
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# CONFIG

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_HEALTH_URL = os.getenv("OLLAMA_HEALTH_URL", "http://127.0.0.1:11434/api/tags")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_CONNECT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SECONDS", "5"))
OLLAMA_READ_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_READ_TIMEOUT_SECONDS", "30"))
OLLAMA_TOTAL_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TOTAL_TIMEOUT_SECONDS", "35"))
OLLAMA_HEALTH_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_HEALTH_TIMEOUT_SECONDS", "5"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "1"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "160"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
RAG_MAX_RESULTS = int(os.getenv("RAG_MAX_RESULTS", "2"))
RAG_FALLBACK_MAX_RESULTS = int(os.getenv("RAG_FALLBACK_MAX_RESULTS", "3"))
RAG_MAX_CHARS_PER_BLOCK = int(os.getenv("RAG_MAX_CHARS_PER_BLOCK", "650"))
RAG_MAX_PROMPT_CONTEXT_CHARS = int(os.getenv("RAG_MAX_PROMPT_CONTEXT_CHARS", "2200"))
FAQ_FAST_PATH_ENABLED = os.getenv("FAQ_FAST_PATH_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
FAQ_FAST_PATH_MAX_LINES = int(os.getenv("FAQ_FAST_PATH_MAX_LINES", "3"))
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
# API (dont delete this sis)
@app.get("/api/programs")
def get_programs():
    if not PROGRAMS_FILE.exists():
        raise HTTPException(status_code=404, detail="Program data file not found on the server.")
    try:
        with open(PROGRAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# LOAD CALENDARS

DATA_FOLDER = Path("data")
calendar_data = {}

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def tokenize(text: str) -> set[str]:
    return set(normalize(text).split())

GENERIC_QUERY_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "from", "with",
    "is", "are", "was", "were", "be", "can", "do", "does", "did", "how", "what", "when",
    "where", "which", "who", "whom", "why", "as", "i", "me", "my", "we", "our", "you",
    "your", "it", "its", "they", "them", "their", "kean", "university", "campus",
}

def meaningful_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) >= 3 and t not in GENERIC_QUERY_WORDS}

def program_subject_tokens_from_query(text: str) -> set[str]:
    q_norm = normalize(text)
    drop_tokens = {
        "degree", "degrees", "program", "programs", "master", "masters", "graduate",
        "undergraduate", "undergrad", "available", "kean", "major", "bachelor",
        "there", "have", "has", "is", "are", "does", "do",
    }
    base = {
        t
        for t in tokenize(text)
        if t not in drop_tokens and (len(t) >= 3 or t in {"it", "cs"})
    }

    # Keep short but meaningful program acronyms.
    if re.search(r"\bit\b|information technology", q_norm):
        base.add("it")
        base.add("information")
        base.add("technology")
    if re.search(r"\bcs\b|computer science", q_norm):
        base.add("cs")
        base.add("computer")
        base.add("science")
    return base

def extract_degree_subject_phrase(text: str) -> Optional[str]:
    q_norm = normalize(text)
    patterns = [
        r"(?:is there|there is|does .* have|do you have)\s+(?:an?\s+)?(.+?)\s+(?:major|bachelor|undergrad|undergraduate|master|masters|graduate|degree|program)\b",
        r"(?:is there|there is|does .* have|do you have)\s+(?:an?\s+)?(.+?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, q_norm)
        if match:
            phrase = match.group(1).strip()
            if phrase:
                phrase = re.sub(r"^(a|an)\s+", "", phrase).strip()
                phrase = re.sub(r"^(degree|program|major)\s+(in\s+)?", "", phrase).strip()
                phrase = re.sub(r"\s+", " ", phrase).strip()
                return phrase
    return None

def extract_follow_up_subject_phrase(text: str) -> Optional[str]:
    q_norm = normalize(text)
    match = re.search(
        r"(?:about|sobre|hakkinda|hakkında|有关|關於|کے بارے میں)\s+(.+)$",
        q_norm,
    )
    phrase = match.group(1).strip() if match else q_norm
    if not phrase:
        return None

    phrase = re.sub(
        r"\b(master|masters|graduate|undergraduate|undergrad|bachelor|degree|degrees|program|programs|major|ms|ma|mba)\b",
        " ",
        phrase,
    )
    phrase = re.sub(r"\b(tell|me|more|details|detail|please|por favor)\b", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    if len(phrase) < 2:
        return None
    return phrase

def keyword_in_text(normalized_text: str, text_tokens: set[str], keyword: str) -> bool:
    normalized_keyword = normalize(keyword)
    if not normalized_keyword:
        return False

    # Keep CJK/Arabic-script keywords as substring checks because tokenization is limited.
    if re.search(r"[\u4e00-\u9fff\u0600-\u06ff\uac00-\ud7af]", normalized_keyword):
        return normalized_keyword in normalized_text

    # Multi-word keyword: phrase match.
    if " " in normalized_keyword:
        return normalized_keyword in normalized_text

    # Single-word keyword: token match avoids false positives like "eat" in "repeat".
    return normalized_keyword in text_tokens

def is_term_header(line: str) -> bool:
    return bool(re.match(r"^(Fall|Spring|Winter|Summer)\s\d{4}\s(Semester|Term)$", line.strip()))

def read_text_with_fallback(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="cp1252")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
    return (
        text.replace("\xa0", " ")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2022", "*")
    )

def load_calendars():
    for file in DATA_FOLDER.glob("*.txt"):
        try:
            lines = read_text_with_fallback(file).splitlines()
        except Exception:
            continue

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
LOCATION_JSON_PATH = BASE_DIR.parent / "src" / "data" / "kean_locations.json"
POLICY_FOLDER = BASE_DIR / "Policies"
DATA_FOLDER = BASE_DIR / "data"
FAQ_INTENT_PATH = DATA_FOLDER / "faq_intent_keywords.json"
EXCLUDED_RAG_DIR_NAMES = {"venv", ".venv", "chroma_db", "__pycache__"}
EXCLUDED_RAG_FILE_NAMES = {"requirements.txt", "_RAG_TEMPLATE.txt", "faq_intent_keywords.json", "program_info_rag.txt"}
PROGRAMS_FILE = DATA_FOLDER / "program_info.json"

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

SEASON_ALIASES = {
    "fall": ("fall", "autumn", "guz", "güz", "sonbahar", "otono", "otoño", "秋", "秋季", "خزاں", "가을"),
    "spring": ("spring", "bahar", "ilkbahar", "primavera", "春", "春季", "بہار", "봄"),
    "summer": ("summer", "yaz", "verano", "夏", "夏季", "گرما", "여름"),
    "winter": ("winter", "kis", "kış", "invierno", "冬", "冬季", "سردی", "겨울"),
}

def detect_event_category(question: str) -> Optional[str]:
    q = normalize(question)
    if any(p in q for p in ["registration", "register", "kayıt", "inscripcion", "inscripción", "注册", "اندراج", "등록"]):
        return "registration"
    if any(p in q for p in ["immunization", "aşı", "vacuna", "疫苗", "ویکسین", "예방접종"]):
        return "immunization"
    if any(p in q for p in ["withdraw", "withdrawal", "çekil", "retiro", "退课", "واپسی", "수강철회"]):
        return "withdrawal"
    if any(p in q for p in ["exam", "final", "vize", "finaller", "examen", "考试", "امتحان", "시험"]):
        return "exam"
    if any(p in q for p in ["recess", "break", "tatil", "receso", "假期", "تعطیل", "방학"]):
        return "recess"
    if any(p in q for p in ["end", "finish", "bit", "termina", "结束", "ختم", "끝"]):
        return "end"
    if any(
        p in q
        for p in [
            "begin",
            "start",
            "first day",
            "opening",
            "başla",
            "basla",
            "başlıyor",
            "basliyor",
            "başlar",
            "baslar",
            "başlangıç",
            "baslangic",
            "empieza",
            "comienza",
            "comenza",
            "inicia",
            "开始",
            "شروع",
            "시작",
        ]
    ):
        return "start"
    return None

def event_matches_category(event_name: str, category: str) -> bool:
    event_text = normalize(event_name)
    if category == "start":
        return any(token in event_text for token in ("term begins", "semester begins", "classes begin", "class begins", "instruction begins", "begins"))
    if category == "end":
        return any(token in event_text for token in ("term ends", "semester ends", "classes end", "class ends", "ends"))
    if category == "recess":
        return any(token in event_text for token in ("recess", "break"))
    if category == "immunization":
        return "immunization" in event_text
    if category == "registration":
        return "registration" in event_text
    if category == "withdrawal":
        return "withdraw" in event_text
    if category == "exam":
        return any(token in event_text for token in ("exam", "final"))
    return False

def extract_term_from_text(text: str) -> Optional[str]:
    q = normalize(text)
    year_match = re.search(r"(20\d{2})", q)
    if not year_match:
        return None
    year = year_match.group(1)

    for season_en, aliases in SEASON_ALIASES.items():
        if any(alias in q for alias in aliases):
            return f"{season_en} {year}"
    return None

def extract_session_from_text(text: str) -> Optional[str]:
    q = normalize(text)
    if re.search(r"\b(i|1)\b", q) and any(alias in q for alias in SEASON_ALIASES.get("summer", ())):
        return "1"
    if re.search(r"\b(ii|2)\b", q) and any(alias in q for alias in SEASON_ALIASES.get("summer", ())):
        return "2"
    return None

def find_best_calendar_event(events: dict[str, str], category: str, session: Optional[str] = None) -> Optional[tuple[str, str]]:
    best_event = None
    best_date = None
    best_score = -10**9

    for event, date in events.items():
        if not event_matches_category(event, category):
            continue
        e = normalize(event)
        score = 0

        if category == "start":
            if "term begins" in e or "semester begins" in e:
                score += 120
            if "classes begin" in e or "class begins" in e or "instruction begins" in e:
                score += 90
            if session == "1" and "session i begins" in e:
                score += 220
            if session == "2" and "session ii begins" in e:
                score += 220
            if session == "1" and "session ii begins" in e:
                score -= 220
            if session == "2" and "session i begins" in e:
                score -= 220
            if "registration" in e:
                score -= 120
            if "immunization" in e or "deadline" in e or "withdraw" in e:
                score -= 90
            if "begins" in e:
                score += 20
        elif category == "end":
            if "term ends" in e or "semester ends" in e:
                score += 120
            if "classes end" in e or "class ends" in e:
                score += 90
            if "final deadline" in e or "registration" in e:
                score -= 80
            if "ends" in e:
                score += 20
        elif category == "registration":
            if "registration begins" in e:
                score += 120
            elif "registration" in e:
                score += 80
        elif category == "exam":
            if "exam week" in e or "final exam" in e:
                score += 120
            elif "final" in e or "exam" in e:
                score += 70
        else:
            score += 50

        if score > best_score:
            best_score = score
            best_event = event
            best_date = date

    if best_event is None:
        return None
    return best_event, best_date

def is_calendar_question(text: str) -> bool:
    q = normalize(text)
    has_year = bool(re.search(r"(20\d{2})", q))
    has_season = any(alias in q for aliases in SEASON_ALIASES.values() for alias in aliases)
    return has_year and has_season

# TIME / LOCATION

NYC_TIMEZONE = ZoneInfo("America/New_York")
LOCATION_KEYWORDS = (
    "where is",
    "where's",
    "wheres",
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
    "nerede",
    "nerde",
    "nasıl giderim",
    "nasil giderim",
    "cómo llegar",
    "como llegar",
    "dónde está",
    "donde esta",
    "donde esta el",
    "donde queda",
    "donde puedo",
    "donde consigo",
    "where can i",
    "where do i",
    "在哪里",
    "怎么去",
    "کہاں",
    "راستہ",
    "어디",
    "어떻게 가",
    "cafe",
    "coffee",
    "starbucks",
    "cafeteria",
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
    "nasıl giderim",
    "nasil giderim",
    "cómo llegar",
    "como llegar",
    "怎么去",
    "راستہ",
    "어떻게 가",
)

LOCATION_ONLY_KEYWORDS = (
    "where is",
    "where's",
    "wheres",
    "nerede",
    "nerde",
    "dónde está",
    "donde esta",
    "donde esta el",
    "donde queda",
    "在哪里",
    "在哪",
    "어디",
    "کہاں",
)
ENTRANCE_REQUEST_KEYWORDS = (
    "entrance",
    "front entrance",
    "rear entrance",
    "side entrance",
    "main entrance",
    "gate",
    "entrada",
    "giris",
    "giriş",
    "入口",
    "입구",
    "دروازہ",
)

CLOSEST_PARKING_KEYWORDS = (
    "closest parking",
    "nearest parking",
    "closest parking lot",
    "nearest parking lot",
    "parking near",
    "parking lot near",
    "parking lot to",
    "closest lot",
    "nearest lot",
    "estacionamiento mas cercano",
    "aparcamiento mas cercano",
    "parking mas cercano",
    "parqueo mas cercano",
    "parqueo cercano",
    "parqueo mas proximo",
    "parqueadero mas proximo",
    "parqueadero mas cercano",
    "parqueo cerca de",
    "en yakin otopark",
    "最 近 停车",
    "最近停车",
    "가장 가까운 주차",
    "قریب ترین پارکنگ",
)

FOOD_INTENT_KEYWORDS = (
    "food",
    "eat",
    "meal",
    "hungry",
    "snack",
    "cafeteria",
    "restaurant",
    "comida",
    "comer",
    "almuerzo",
    "desayuno",
    "cena",
    "cafeteria",
    "restaurante",
    "yemek",
    "yiyecek",
    "kafeterya",
    "kantin",
    "吃",
    "吃饭",
    "食物",
    "餐厅",
    "食堂",
    "음식",
    "식당",
    "밥",
    "کھانا",
    "خوراک",
)

FOOD_DESTINATION_HINTS = (
    ("starbucks_at_library", ("coffee", "cafe", "café", "starbucks", "cafeteria", "café", "cafe", "café", "café", "comprar cafe", "comprar café", "cafe", "kahve", "咖啡", "커피")),
    ("smash_burgar_at_miron", ("burger", "smash", "hamburger", "sandwich", "hamburguesa", "hamburger", "burger", "برگر")),
    ("Cougar Pantry", ("pantry", "food pantry", "despensa", "banco de comida", "erzak", "gida bankasi", "食物银行", "푸드팬트리", "فوڈ پینٹری")),
    ("cougars_den", ("cougars den", "cougar den", "dining", "eat", "food", "comida", "comer", "yemek", "吃饭", "食堂", "식당", "کھانا")),
)
LOCATION_ALIAS_OVERRIDES = {
    "miron_center": (
        "student center",
        "miron",
        "msc",
        "miron student center",
        "campus center"
    ),
    "library": (
        "library",
        "nancy thompson library",
        "thompson library"
    ),
    "starbucks_at_library": (
        "starbucks",
        "coffee",
        "cafe",
        "cafeteria",
        "coffee shop",
        "buy coffee",
        "comprar cafe",
        "comprar café",
        "donde comprar cafe",
        "donde comprar café",
    ),
    "barnes_nobles_glab": (
        "bookstore",
        "barnes and noble",
        "barnes noble",
        "campus bookstore"
    ),
    "campus_police": (
        "campus police",
        "public safety",
        "police",
        "safety office",
        "security office"
    ),
    "administration": (
        "administration",
        "administration building",
        "financial aid",
        "registrar",
        "student accounts",
        "admissions office"
    ),
    "cas": (
        "cas",
        "center for academic success",
        "academic success center",
        "tutoring center"
    ),
    "glab": (
        "glab",
        "green lane academic building",
        "green lane building"
    ),
    "hri": (
        "hri",
        "human rights institute"
    ),
    "technology_center": (
        "technology center",
        "tech center",
        "tec"
    ),
    "naab": (
        "naab",
        "north avenue academic building",
        "north avenue building"
    ),
    "stem": (
        "stem",
        "stem building",
        "science technology engineering mathematics building",
        "nj stem center"
    ),
    "harwood_arena": (
        "harwood",
        "harwood arena",
        "arena"
    ),
    "wilkins_theatre": (
        "wilkins",
        "wilkins theatre",
        "theater",
        "theatre"
    ),
    "lhac": (
        "lhac",
        "liberty hall academic center"
    ),
    "nathan_weiss_building": (
        "nathan weiss",
        "east campus building",
        "ecb"
    ),
    "union_train_station": (
        "train station",
        "rail station",
        "union train station",
        "nj transit station",
        "transit station",
        "station"
    ),
    "hutchinson_hall Main": (
        "hutchinson",
        "hutchinson hall",
    ),
}
GENERIC_LOCATION_WORDS = {
    "hall",
    "building",
    "center",
    "main",
    "entrance",
    "campus",
    "lot",
    "parking",
    "station",
}
SECONDARY_PLACE_WORDS = {
    "field",
    "turf",
    "room",
    "lawn",
    "statue",
    "court",
    "courts",
    "pantry",
    "starbucks",
    "game",
}
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
conversation_state = {
    "last_degree_subject": None,
    "last_degree_level": None,
}
KEAN_MAIN_URL = "https://www.kean.edu"
KEAN_PROGRAMS_URL = "https://www.kean.edu/academics"
KEAN_CATALOG_URL = "https://kean.smartcatalogiq.com/"

SUPPORTED_LANGS = {"en", "tr", "es", "zh", "ur", "ko"}
LANGUAGE_NAMES = {
    "en": "English",
    "tr": "Turkish",
    "es": "Spanish",
    "zh": "Mandarin Chinese",
    "ur": "Urdu",
    "ko": "Korean",
}
SHORT_LANGUAGE_HINTS = {
    "en": {
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    },
    "tr": {
        "merhaba", "selam", "gunaydin", "günaydın", "iyi aksamlar", "iyi akşamlar",
    },
    "es": {
        "hola", "buenas", "buenos dias", "buenos días", "buenas tardes", "buenas noches",
    },
    "zh": {
        "你好", "您好", "早上好", "下午好", "晚上好",
    },
    "ur": {
        "سلام", "السلام علیکم", "السلام عليكم",
    },
    "ko": {
        "안녕", "안녕하세요", "좋은 아침", "좋은 저녁",
    },
}
LANGUAGE_HINT_KEYWORDS = {
    "tr": {
        "nerede", "nasil", "nasıl", "ne", "zaman", "guz", "güz", "bahar", "donem", "dönem",
        "basliyor", "başlıyor", "kampus", "kampüs", "kayit", "kayıt", "otopark", "yemek",
        "saat", "kutuphane", "kütüphane", "mali", "yardim", "yardım", "mezuniyet",
        "merhaba", "selam", "gunaydin", "günaydın", "iyi aksamlar", "iyi akşamlar",
    },
    "es": {
        "donde", "dónde", "cuando", "cuándo", "que", "qué", "semestre", "otono", "otoño",
        "primavera", "parqueo", "estacionamiento", "comida", "horario", "biblioteca",
        "admisiones", "solicitud", "matricula", "matrícula", "ayuda", "financiera",
        "graduacion", "graduación", "quien", "quién", "eres", "hola", "necesito",
        "buenas", "buenos dias", "buenos días", "buenas tardes", "buenas noches",
        "como", "cómo", "llamas", "nombre", "puedes", "gracias",
    },
    "zh": {"哪里", "在哪", "怎么", "如何", "时间", "开放", "停车", "餐厅", "图书馆", "专业", "招生", "毕业", "你好", "您好"},
    "ur": {"سلام", "کیا", "کہاں", "کب", "داخلہ", "پارکنگ", "لائبریری", "کھانا", "وقت", "مدد", "یونیورسٹی"},
    "ko": {"어디", "어떻게", "시간", "주차", "식당", "도서관", "전공", "입학", "졸업", "등록금", "안녕", "안녕하세요"},
}
FAQ_TOPIC_RETRIEVAL_HINTS = {
    "admissions": "admissions apply application portal undergraduate graduate international",
    "financial_aid": "financial aid fafsa grants scholarships loans",
    "student_accounts": "tuition fees bursar payment refund hold late fee",
    "registration": "registration registrar transcript one stop add drop enrollment",
    "calendar_deadline": "academic calendar deadline semester term due date",
    "housing": "housing dorm residence life move in roommate residential",
    "health_services": "student health wellness counseling immunization vaccine",
    "accessibility": "accessibility accommodations disability office accessibility services",
    "parking_transport": "parking permit shuttle transportation train trolley bus",
    "library": "library books study room citation hours",
    "bookstore": "bookstore barnes noble textbooks apparel",
    "theaters_events": "kean stage theatre theater tickets box office event",
    "it_support": "wifi email password tech support portal",
    "programs": "program degree major minor curriculum catalog academics",
    "policies": "policy policies rule procedure conduct repeat retake",
    "smoking_policy": "smoking policy tobacco vape vaping no smoking",
    "dining": "dining food court marketplace starbucks cafe hours",
    "hours": "hours open opening closing schedule",
}
TRANSLATIONS = {
    "location_opening_specific": {
        "en": "{name} is on {campus}. Opening map.",
        "tr": "{name}, {campus} kampüsünde. Harita açılıyor.",
        "es": "{name} está en {campus}. Abriendo mapa.",
        "zh": "{name} 位于 {campus}。正在打开地图。",
        "ur": "{name} {campus} میں ہے۔ نقشہ کھولا جا رہا ہے۔",
        "ko": "{name}은(는) {campus}에 있습니다. 지도를 여는 중입니다.",
    },
    "location_opening_generic": {
        "en": "Opening campus map. Please pick a destination or search the directory.",
        "tr": "Kampüs haritası açılıyor. Lütfen bir hedef seç veya dizinde ara.",
        "es": "Abriendo el mapa del campus. Elige un destino o busca en el directorio.",
        "zh": "正在打开校园地图。请选择目的地或在目录中搜索。",
        "ur": "کیمپس کا نقشہ کھولا جا رہا ہے۔ براہِ کرم منزل منتخب کریں یا ڈائریکٹری میں تلاش کریں۔",
        "ko": "캠퍼스 지도를 여는 중입니다. 목적지를 선택하거나 디렉터리에서 검색하세요.",
    },
    "closest_parking_found": {
        "en": "The closest parking lot to {target} is {lot}. Opening map.",
        "tr": "{target} için en yakın otopark {lot}. Harita açılıyor.",
        "es": "El estacionamiento más cercano a {target} es {lot}. Abriendo mapa.",
        "zh": "离 {target} 最近的停车场是 {lot}。正在打开地图。",
        "ur": "{target} کے قریب ترین پارکنگ لاٹ {lot} ہے۔ نقشہ کھولا جا رہا ہے۔",
        "ko": "{target}에 가장 가까운 주차장은 {lot}입니다. 지도를 여는 중입니다.",
    },
    "closest_parking_unknown_target": {
        "en": "Please include a campus building name so I can find the closest parking lot.",
        "tr": "En yakın otoparkı bulmam için lütfen kampüsteki bina adını yaz.",
        "es": "Incluye el nombre de un edificio del campus para encontrar el estacionamiento más cercano.",
        "zh": "请提供校园建筑名称，我才能找到最近的停车场。",
        "ur": "قریب ترین پارکنگ تلاش کرنے کے لیے براہِ کرم کیمپس عمارت کا نام دیں۔",
        "ko": "가장 가까운 주차장을 찾으려면 캠퍼스 건물 이름을 입력해 주세요.",
    },
    "closest_parking_not_found": {
        "en": "I found {target}, but I could not find a nearby parking lot in the current map data.",
        "tr": "{target} bulundu, ancak mevcut harita verisinde yakın bir otopark bulunamadı.",
        "es": "Encontré {target}, pero no pude encontrar un estacionamiento cercano en los datos del mapa.",
        "zh": "我找到了 {target}，但在当前地图数据中未找到附近停车场。",
        "ur": "میں نے {target} تلاش کر لیا، مگر موجودہ نقشہ ڈیٹا میں قریب پارکنگ نہیں ملی۔",
        "ko": "{target}은(는) 찾았지만 현재 지도 데이터에서 가까운 주차장을 찾지 못했습니다.",
    },
    "food_suggestions_intro": {
        "en": "Here are food options on campus:",
        "tr": "Kampüsteki yemek seçenekleri:",
        "es": "Estas son opciones de comida en el campus:",
        "zh": "以下是校园内的餐饮选项：",
        "ur": "کیمپس میں کھانے کے یہ آپشنز ہیں:",
        "ko": "캠퍼스 내 식사 옵션입니다:",
    },
    "parking_guidance_student": {
        "en": "Student parking is shown in blue on the map. Opening map.",
        "tr": "Öğrenci otoparkı haritada mavi renkte gösterilir. Harita açılıyor.",
        "es": "El estacionamiento para estudiantes aparece en azul en el mapa. Abriendo mapa.",
        "zh": "学生停车区在地图上以蓝色显示。正在打开地图。",
        "ur": "طلبہ کی پارکنگ نقشے پر نیلے رنگ میں دکھائی جاتی ہے۔ نقشہ کھولا جا رہا ہے۔",
        "ko": "학생 주차 구역은 지도에서 파란색으로 표시됩니다. 지도를 여는 중입니다.",
    },
    "parking_guidance_faculty": {
        "en": "Faculty/Staff parking is shown in orange on the map. Opening map.",
        "tr": "Akademik/Personel otoparkı haritada turuncu renkte gösterilir. Harita açılıyor.",
        "es": "El estacionamiento para personal/docentes aparece en naranja en el mapa. Abriendo mapa.",
        "zh": "教职工停车区在地图上以橙色显示。正在打开地图。",
        "ur": "فیکلٹی/اسٹاف پارکنگ نقشے پر نارنجی رنگ میں دکھائی جاتی ہے۔ نقشہ کھولا جا رہا ہے۔",
        "ko": "교직원 주차 구역은 지도에서 주황색으로 표시됩니다. 지도를 여는 중입니다.",
    },
    "parking_guidance_overnight": {
        "en": "Overnight parking is shown in green on the map. Opening map.",
        "tr": "Gece parkı haritada yeşil renkte gösterilir. Harita açılıyor.",
        "es": "El estacionamiento nocturno aparece en verde en el mapa. Abriendo mapa.",
        "zh": "夜间停车区在地图上以绿色显示。正在打开地图。",
        "ur": "رات بھر کی پارکنگ نقشے پر سبز رنگ میں دکھائی جاتی ہے۔ نقشہ کھولا جا رہا ہے۔",
        "ko": "야간 주차 구역은 지도에서 초록색으로 표시됩니다. 지도를 여는 중입니다.",
    },
    "parking_guidance_general": {
        "en": "Opening map. Parking lots are color-coded: Student (blue), Faculty/Staff (orange), Overnight (green).",
        "tr": "Harita açılıyor. Otopark renkleri: Öğrenci (mavi), Akademik/Personel (turuncu), Gece Parkı (yeşil).",
        "es": "Abriendo mapa. Los estacionamientos están codificados por color: Estudiantes (azul), Personal/Docentes (naranja), Nocturno (verde).",
        "zh": "正在打开地图。停车场颜色：学生（蓝色）、教职工（橙色）、夜间（绿色）。",
        "ur": "نقشہ کھولا جا رہا ہے۔ پارکنگ رنگوں کے مطابق ہے: طلبہ (نیلا)، فیکلٹی/اسٹاف (نارنجی)، رات بھر (سبز)۔",
        "ko": "지도를 여는 중입니다. 주차장은 색상으로 구분됩니다: 학생(파랑), 교직원(주황), 야간(초록).",
    },
    "fallback_no_context": {
        "en": "I couldn't reach Ollama and I don't have matching campus context yet.",
        "tr": "Ollama'ya ulaşamadım ve eşleşen kampüs bağlamı henüz yok.",
        "es": "No pude conectar con Ollama y todavía no tengo contexto del campus que coincida.",
        "zh": "我无法连接到 Ollama，且目前没有匹配的校园上下文。",
        "ur": "میں Ollama تک نہیں پہنچ سکا اور ابھی متعلقہ کیمپس معلومات دستیاب نہیں ہیں۔",
        "ko": "Ollama에 연결할 수 없고, 일치하는 캠퍼스 컨텍스트도 아직 없습니다.",
    },
    "fallback_best_match": {
        "en": "Ollama is unavailable right now. Best match from campus records: {line}",
        "tr": "Ollama şu anda kullanılamıyor. Kampüs kayıtlarından en iyi eşleşme: {line}",
        "es": "Ollama no está disponible ahora. Mejor coincidencia en registros del campus: {line}",
        "zh": "Ollama 当前不可用。校园记录中的最佳匹配：{line}",
        "ur": "فی الحال Ollama دستیاب نہیں۔ کیمپس ریکارڈ سے بہترین مطابقت: {line}",
        "ko": "현재 Ollama를 사용할 수 없습니다. 캠퍼스 기록에서 가장 일치하는 내용: {line}",
    },
    "fallback_top_context": {
        "en": "Ollama is unavailable right now. Top context match: {line}",
        "tr": "Ollama şu anda kullanılamıyor. En iyi bağlam eşleşmesi: {line}",
        "es": "Ollama no está disponible ahora. Mejor contexto encontrado: {line}",
        "zh": "Ollama 当前不可用。最佳上下文匹配：{line}",
        "ur": "فی الحال Ollama دستیاب نہیں۔ بہترین سیاقی مطابقت: {line}",
        "ko": "현재 Ollama를 사용할 수 없습니다. 가장 높은 컨텍스트 일치: {line}",
    },
    "fast_path_prefix": {
        "en": "Here’s what I found in Kean’s records:",
        "tr": "Kampüs kayıtlarına göre en ilgili bilgiler:",
        "es": "Según los registros del campus, estos son los detalles más relevantes:",
        "zh": "根据校园记录，以下是最相关的信息：",
        "ur": "کیمپس ریکارڈ کے مطابق، یہ سب سے متعلقہ معلومات ہیں:",
        "ko": "캠퍼스 기록 기준으로 가장 관련 있는 정보입니다:",
    },
    "faq_no_exact_match": {
        "en": "I’m sorry, I could not find that information in my current Kean records. Please visit the official Kean University website for more information: https://www.kean.edu",
        "tr": "Mevcut kampüs kayıtlarında tam bir eşleşme bulamadım. Lütfen belirli bir program veya politika adıyla tekrar sor.",
        "es": "No encontré una coincidencia exacta en los registros actuales del campus. Reformula con el nombre específico del programa o política.",
        "zh": "我在当前校园记录中未找到精确匹配。请用具体的项目或政策名称重新提问。",
        "ur": "موجودہ کیمپس ریکارڈ میں عین مطابق جواب نہیں ملا۔ براہِ کرم مخصوص پروگرام یا پالیسی کے نام کے ساتھ دوبارہ سوال کریں۔",
        "ko": "현재 캠퍼스 기록에서 정확한 일치를 찾지 못했습니다. 특정 프로그램 또는 정책 이름으로 다시 질문해 주세요.",
    },
    "degree_exists_yes": {
        "en": "Yes, {subject} is listed at the {level} level.",
        "tr": "Evet, {subject} için {level} bir derece programı listeleniyor.",
        "es": "Sí, hay un programa de {level} en {subject}.",
        "zh": "是的，学校列有 {subject} 的{level}学位项目。",
        "ur": "جی ہاں، {subject} میں {level} ڈگری پروگرام موجود ہے۔",
        "ko": "네, {subject} {level} 학위 과정이 있습니다.",
    },
    "degree_exists_no": {
        "en": "Sorry, {subject} is not listed at the {level} level in current records.",
        "tr": "Üzgünüm, mevcut kayıtlarda {subject} için {level} bir derece programı listelenmiyor.",
        "es": "Lo siento, no aparece un programa de {level} en {subject} en los registros actuales.",
        "zh": "抱歉，当前记录中未列出 {subject} 的{level}学位项目。",
        "ur": "معذرت، موجودہ ریکارڈ میں {subject} کے لیے {level} ڈگری درج نہیں ہے۔",
        "ko": "죄송하지만 현재 기록에는 {subject} {level} 학위 과정이 없습니다.",
    },
    "program_follow_up_prompt": {
        "en": "I can share more details if you ask about a specific subject and degree level.",
        "tr": "Belirli bir bölüm ve derece seviyesi sorarsan daha fazla ayrıntı paylaşabilirim.",
        "es": "Puedo compartir más detalles si preguntas por una carrera y nivel específico.",
        "zh": "如果你提供具体专业和学位层次，我可以给出更多细节。",
        "ur": "اگر آپ مخصوص مضمون اور ڈگری لیول بتائیں تو میں مزید تفصیل دے سکتا ہوں۔",
        "ko": "특정 전공과 학위 수준을 말씀해 주시면 더 자세히 안내할 수 있습니다.",
    },
    "library_hours_intro": {
        "en": "Nancy Thompson Library hours:",
        "tr": "Nancy Thompson Kütüphanesi saatleri:",
        "es": "Horario de la Biblioteca Nancy Thompson:",
        "zh": "Nancy Thompson 图书馆开放时间：",
        "ur": "Nancy Thompson لائبریری کے اوقات:",
        "ko": "Nancy Thompson 도서관 운영 시간:",
    },
    "library_hours_unavailable": {
        "en": "I couldn't find the library hours in current records.",
        "tr": "Mevcut kayıtlarda kütüphane saatlerini bulamadım.",
        "es": "No pude encontrar el horario de la biblioteca en los registros actuales.",
        "zh": "我在当前记录中未找到图书馆开放时间。",
        "ur": "موجودہ ریکارڈ میں لائبریری کے اوقات نہیں ملے۔",
        "ko": "현재 기록에서 도서관 운영 시간을 찾지 못했습니다.",
    },
    "gym_hours_intro": {
        "en": "Miron Student Center building hours:",
        "tr": "Miron Student Center bina saatleri:",
        "es": "Horario del edificio Miron Student Center:",
        "zh": "Miron 学生活动中心开放时间：",
        "ur": "Miron Student Center عمارت کے اوقات:",
        "ko": "Miron Student Center 건물 운영 시간:",
    },
    "pool_hours_intro": {
        "en": "D'Angola Pool hours:",
        "tr": "D'Angola Havuzu saatleri:",
        "es": "Horario de la piscina D'Angola:",
        "zh": "D'Angola 泳池开放时间：",
        "ur": "D'Angola پول کے اوقات:",
        "ko": "D'Angola 수영장 운영 시간:",
    },
    "pool_hours_unavailable": {
        "en": "I couldn't find official pool hours in current records. If you want, I can show the nearest athletics facilities on the map.",
        "tr": "Mevcut kayıtlarda resmi havuz saatlerini bulamadım. İstersen haritada en yakın atletizm tesislerini gösterebilirim.",
        "es": "No encontré horario oficial de la piscina en los registros actuales. Si quieres, puedo mostrar las instalaciones deportivas cercanas en el mapa.",
        "zh": "我在当前记录中未找到官方泳池开放时间。如需，我可以在地图上显示附近体育设施。",
        "ur": "موجودہ ریکارڈ میں سوئمنگ پول کے باضابطہ اوقات نہیں ملے۔ چاہیں تو میں نقشے پر قریب ترین کھیلوں کی سہولیات دکھا سکتا ہوں۔",
        "ko": "현재 기록에서 수영장 공식 운영 시간을 찾지 못했습니다. 원하면 지도에서 가까운 체육 시설을 보여드릴 수 있습니다.",
    },
    "hours_target_unavailable": {
        "en": "I couldn't find operating hours for {target} in current records.",
        "tr": "Mevcut kayıtlarda {target} için çalışma saatlerini bulamadım.",
        "es": "No encontré horarios de atención para {target} en los registros actuales.",
        "zh": "我在当前记录中未找到 {target} 的开放时间。",
        "ur": "موجودہ ریکارڈ میں {target} کے اوقات نہیں ملے۔",
        "ko": "현재 기록에서 {target} 운영 시간을 찾지 못했습니다.",
    },
    "hours_target_prompt": {
        "en": "Please specify which place you mean, for example: library, gym, or pool.",
        "tr": "Lütfen hangi yeri kastettiğini belirt: örneğin kütüphane, spor salonu veya havuz.",
        "es": "Especifica qué lugar quieres consultar, por ejemplo: biblioteca, gimnasio o piscina.",
        "zh": "请说明你要查询的地点，例如：图书馆、健身房或游泳池。",
        "ur": "براہِ کرم جگہ واضح کریں، مثال کے طور پر: لائبریری، جم یا پول۔",
        "ko": "어느 장소인지 지정해 주세요. 예: 도서관, 체육관, 수영장.",
    },
    "bot_identity_intro": {
        "en": "I’m the Kean Global concierge assistant for the Kean University website. I answer questions in simple, easy language and can also help with campus locations, maps, and directions.",
        "tr": "Ben Kean University web sitesi için Kean Global danışma asistanıyım. Soruları basit ve anlaşılır bir dille yanıtlarım; ayrıca kampüs konumları, haritalar ve yön tarifleri konusunda da yardımcı olabilirim.",
        "es": "Soy el asistente de conserjería Kean Global para el sitio web de Kean University. Respondo preguntas con un lenguaje simple y claro, y también puedo ayudar con ubicaciones del campus, mapas y direcciones.",
        "zh": "我是 Kean University 网站的 Kean Global 校园礼宾助理。我会用简单易懂的语言回答问题，也可以帮助你查找校园地点、地图和路线。",
        "ur": "میں Kean University ویب سائٹ کے لیے Kean Global کنسیئر اسسٹنٹ ہوں۔ میں سوالوں کے جواب آسان اور سادہ زبان میں دیتا ہوں، اور کیمپس مقامات، نقشے اور راستوں میں بھی مدد کر سکتا ہوں۔",
        "ko": "저는 Kean University 웹사이트용 Kean Global 안내 도우미입니다. 쉽고 간단한 언어로 질문에 답하고, 캠퍼스 위치, 지도, 길찾기도 도와드릴 수 있습니다.",
    },
    "greeting_intro": {
        "en": "Hello! I can help with Kean University questions, campus locations, maps, and directions.",
        "tr": "Merhaba! Kean University soruları, kampüs konumları, haritalar ve yön tarifleri konusunda yardımcı olabilirim.",
        "es": "Hola. Puedo ayudar con preguntas sobre Kean University, ubicaciones del campus, mapas y direcciones.",
        "zh": "你好！我可以帮助解答 Kean University 的问题，也可以协助查询校园地点、地图和路线。",
        "ur": "ہیلو! میں Kean University سے متعلق سوالات، کیمپس مقامات، نقشوں اور راستوں میں مدد کر سکتا ہوں۔",
        "ko": "안녕하세요! Kean University 관련 질문, 캠퍼스 위치, 지도, 길찾기를 도와드릴 수 있습니다.",
    },
    "thanks_reply": {
        "en": "You're welcome. If you want, ask me about admissions, programs, campus locations, parking, dining, hours, or directions.",
        "tr": "Rica ederim. İstersen kabul, programlar, kampüs konumları, otopark, yemek, saatler veya yol tarifi hakkında sorabilirsin.",
        "es": "De nada. Si quieres, puedes preguntarme sobre admisiones, programas, ubicaciones del campus, estacionamiento, comida, horarios o direcciones.",
        "zh": "不客气。如果需要，你可以继续问我招生、专业、校园地点、停车、餐饮、开放时间或路线问题。",
        "ur": "خوش آمدید۔ اگر چاہیں تو آپ مجھ سے داخلہ، پروگرامز، کیمپس مقامات، پارکنگ، کھانے، اوقات یا راستوں کے بارے میں پوچھ سکتے ہیں۔",
        "ko": "천만에요. 원하시면 입학, 전공, 캠퍼스 위치, 주차, 식사, 운영 시간, 길찾기에 대해 계속 물어보세요.",
    },
    "farewell_reply": {
        "en": "Goodbye. If you need anything else about Kean University, I’ll be here.",
        "tr": "Hoşça kal. Kean University hakkında başka bir şeye ihtiyacın olursa buradayım.",
        "es": "Adiós. Si necesitas algo más sobre Kean University, aquí estaré.",
        "zh": "再见。如果你还需要了解 Kean University 的其他信息，我会在这里。",
        "ur": "خدا حافظ۔ اگر Kean University کے بارے میں مزید کسی چیز کی ضرورت ہو تو میں حاضر ہوں۔",
        "ko": "안녕히 가세요. Kean University에 대해 더 필요한 것이 있으면 언제든지 물어보세요.",
    },
    "capabilities_reply": {
        "en": "I can help with Kean University admissions, programs, tuition, registration, housing, health services, dining, bookstore, campus locations, parking, maps, directions, hours, and academic dates.",
        "tr": "Kean University için kabul, programlar, ücretler, kayıt, konaklama, sağlık hizmetleri, yemek, kitapçı, kampüs konumları, otopark, harita, yol tarifi, saatler ve akademik tarihler konusunda yardımcı olabilirim.",
        "es": "Puedo ayudar con admisiones, programas, matrícula y costos, inscripción, vivienda, servicios de salud, comida, librería, ubicaciones del campus, estacionamiento, mapas, direcciones, horarios y fechas académicas de Kean University.",
        "zh": "我可以帮助解答 Kean University 的招生、专业、学费、注册、住宿、健康服务、餐饮、书店、校园地点、停车、地图、路线、开放时间和学术日期等问题。",
        "ur": "میں Kean University کے داخلہ، پروگرامز، فیس، رجسٹریشن، رہائش، ہیلتھ سروسز، کھانے، بک اسٹور، کیمپس مقامات، پارکنگ، نقشوں، راستوں، اوقات اور تعلیمی تاریخوں کے بارے میں مدد کر سکتا ہوں۔",
        "ko": "저는 Kean University의 입학, 전공, 등록금, 수강신청, 주거, 건강 서비스, 식사, 서점, 캠퍼스 위치, 주차, 지도, 길찾기, 운영 시간, 학사 일정 안내를 도와드릴 수 있습니다.",
    },
    "clarify_reply": {
        "en": "Please ask again with a little more detail. You can include a topic like admissions, a building name, a program name, hours, parking, or directions.",
        "tr": "Lütfen biraz daha ayrıntıyla tekrar sor. Kabul, bina adı, program adı, saatler, otopark veya yol tarifi gibi bir konu ekleyebilirsin.",
        "es": "Vuelve a preguntar con un poco más de detalle. Puedes incluir un tema como admisiones, el nombre de un edificio, un programa, horarios, estacionamiento o direcciones.",
        "zh": "请再具体一点提问。你可以说明招生、建筑名称、专业名称、开放时间、停车或路线等主题。",
        "ur": "براہِ کرم تھوڑی مزید تفصیل کے ساتھ دوبارہ پوچھیں۔ آپ داخلہ، عمارت کا نام، پروگرام کا نام، اوقات، پارکنگ یا راستوں میں سے کوئی موضوع شامل کر سکتے ہیں۔",
        "ko": "조금 더 구체적으로 다시 질문해 주세요. 입학, 건물 이름, 전공명, 운영 시간, 주차, 길찾기 같은 주제를 포함하면 더 정확히 도와드릴 수 있습니다.",
    },
    "acknowledgment_reply": {
        "en": "Understood.",
        "tr": "Anlaşıldı.",
        "es": "Entendido.",
        "zh": "明白了。",
        "ur": "سمجھ گیا۔",
        "ko": "알겠습니다.",
    },
    "frustration_reply": {
        "en": "I understand. Try asking with a specific topic, office, building, or program name, and I’ll answer more directly.",
        "tr": "Anlıyorum. Belirli bir konu, ofis, bina veya program adıyla sorarsan daha doğrudan yanıt verebilirim.",
        "es": "Entiendo. Si preguntas con un tema, oficina, edificio o programa específico, podré responder de forma más directa.",
        "zh": "我明白了。如果你用更具体的主题、办公室、建筑或专业名称提问，我可以更直接地回答。",
        "ur": "میں سمجھ گیا۔ اگر آپ کسی مخصوص موضوع، دفتر، عمارت یا پروگرام کے نام کے ساتھ پوچھیں تو میں زیادہ براہِ راست جواب دے سکتا ہوں۔",
        "ko": "이해했습니다. 더 구체적인 주제, 부서, 건물, 전공 이름으로 질문하면 더 정확하게 답할 수 있습니다.",
    },
    "faq_clarify_reply": {
        "en": "I need a little more detail to answer accurately. Please include a specific office, building, service, program, or topic.",
        "tr": "Doğru yanıt verebilmem için biraz daha ayrıntıya ihtiyacım var. Lütfen belirli bir ofis, bina, hizmet, program veya konu yaz.",
        "es": "Necesito un poco más de detalle para responder con precisión. Incluye una oficina, edificio, servicio, programa o tema específico.",
        "zh": "为了更准确地回答，我需要更多一点细节。请说明具体的办公室、建筑、服务、专业或主题。",
        "ur": "درست جواب دینے کے لیے مجھے تھوڑی مزید تفصیل چاہیے۔ براہِ کرم کسی مخصوص دفتر، عمارت، سروس، پروگرام یا موضوع کا ذکر کریں۔",
        "ko": "정확히 답하려면 조금 더 구체적인 정보가 필요합니다. 특정 부서, 건물, 서비스, 전공 또는 주제를 포함해 주세요.",
    },
    "parking_ticket_fees_intro": {
        "en": "Here are the parking ticket fees listed by Kean:",
        "tr": "Kean tarafından listelenen park ihlali ücretleri:",
        "es": "Estas son las tarifas de multas de estacionamiento que publica Kean:",
        "zh": "以下是 Kean 公布的停车罚单费用：",
        "ur": "Kean کے مطابق پارکنگ ٹکٹ فیس یہ ہیں:",
        "ko": "Kean에 게시된 주차 위반 요금은 다음과 같습니다:",
    },
    "parking_ticket_fees_note": {
        "en": "Note: unpaid/late violations can add a $50 late fee.",
        "tr": "Not: geç/ödenmeyen ihlaller için ek $50 gecikme ücreti uygulanabilir.",
        "es": "Nota: las infracciones sin pagar o tardías pueden generar un recargo de $50.",
        "zh": "注意：逾期或未缴可能会增加 50 美元滞纳金。",
        "ur": "نوٹ: غیر ادا شدہ/تاخیر سے ادا جرمانوں پر اضافی $50 لیٹ فیس لگ سکتی ہے۔",
        "ko": "참고: 미납/지연 시 $50의 추가 연체료가 부과될 수 있습니다.",
    },
    "parking_ticket_fees_unavailable": {
        "en": "I can confirm Kean publishes a parking violation schedule, but I couldn't parse exact line-item fees from current records.",
        "tr": "Kean'ın park ihlal ücret çizelgesi yayımladığını doğrulayabiliyorum, ancak mevcut kayıtlardan kalem bazlı ücretleri ayrıştıramadım.",
        "es": "Puedo confirmar que Kean publica una tabla de infracciones de estacionamiento, pero no pude extraer las tarifas exactas por concepto de los registros actuales.",
        "zh": "我可以确认 Kean 公布了停车违规费用表，但当前记录中未能解析出逐项金额。",
        "ur": "میں تصدیق کر سکتا ہوں کہ Kean پارکنگ خلاف ورزی فیس شیڈول شائع کرتا ہے، لیکن موجودہ ریکارڈ سے درست مد وار فیس نہیں نکال سکا۔",
        "ko": "Kean이 주차 위반 요금표를 게시하는 것은 확인했지만, 현재 기록에서 항목별 정확한 요금을 추출하지 못했습니다.",
    },
    "course_repeat_summary": {
        "en": "Here’s a quick summary of Kean’s course repeat policy:\n1. Undergraduate students may repeat a course once if they earned F, D, C, C+, AF, or WD.\n2. Graduate-level coursework cannot be repeated or recalculated.\n3. If you earned B- or higher, you need approval to repeat the course.\n4. For courses completed before Fall 2024, grade recalculation requires a Registrar form.\n5. For more details, you can review Kean’s policy here: https://www.kean.edu",
    },
    "program_not_found": {
        "en": "I’m sorry, I could not find that major or program in my current Kean records. Please visit the official Kean University website for more information: https://www.kean.edu/academics",
    },
    "program_details_intro": {
        "en": "Here’s a quick overview of {name}:",
    },
    "program_more_info": {
        "en": "For more information, you can check: {url}",
    },
    "program_contact": {
        "en": "Program contact: {contact}",
    },
    "admissions_summary": {
        "en": "Here’s a quick admissions overview:\n1. Freshman, transfer, and readmit applicants use Kean’s application portal, while graduate and international applicants use apply.kean.edu.\n2. Most application fees listed in current records are $75.\n3. For general admissions help, contact Kean Admissions at (908) 737-7100 or admitme@kean.edu.\n4. You can start here: https://www.kean.edu/apply-now",
    },
    "graduation_summary": {
        "en": "Here’s a quick graduation overview:\n1. Degree candidates must apply in KeanWISE to be considered for graduation.\n2. The graduation application fee listed in current records is $100.\n3. Current records say degrees are conferred in January, May, and August.\n4. For graduation questions, contact graduation@kean.edu or (908) 737-0400.\n5. More information is available on Kean’s website: https://www.kean.edu",
    },
    "financial_aid_summary": {
        "en": "Here’s what I found about Financial Aid:\n1. The Financial Aid office helps with FAFSA, grants, loans, and work-study.\n2. Location: Administration Building, 1st floor.\n3. Hours: Monday to Thursday 8 a.m. to 6 p.m.; Friday 8 a.m. to 5 p.m.\n4. Contact: (908) 737-3190 or finaid@kean.edu.\n5. More information: https://www.kean.edu/offices/financial-aid",
    },
    "registrar_summary": {
        "en": "Here’s what I found about the Registrar’s Office:\n1. The office handles registration, transcripts, grade recalculation, graduation evaluation, and enrollment verification.\n2. Location: Administration Building, 1st Floor.\n3. Hours: Monday to Thursday 8 a.m. to 6 p.m.; Friday 8 a.m. to 5 p.m.\n4. Contact: regme@kean.edu or (908) 737-0400.",
    },
    "one_stop_summary": {
        "en": "Here’s what I found about the One Stop Service Center:\n1. One Stop helps with forms such as change of major, graduation applications, registration petitions, grade recalculations, and residency petitions.\n2. Location: CAS Lobby.\n3. Hours: Monday to Thursday 9 a.m. to 8 p.m.; Friday 9 a.m. to 5 p.m.; Saturday 9:30 a.m. to 2 p.m.\n4. Contact: regme@kean.edu or (908) 737-0400.",
    },
    "dining_summary": {
        "en": "Here are the main dining locations and hours I found:\n1. Marketplace: Mon-Thu 9 a.m. to 11 p.m.; Fri 9 a.m. to 9 p.m.; Sat 12 p.m. to 6 p.m.; Sun 1 p.m. to 6 p.m.\n2. MSC Food Court: Mon-Thu 7:30 a.m. to 7 p.m.; Fri 7:30 a.m. to 3 p.m.; Sat 8 a.m. to 3 p.m.; Sun closed.\n3. Cougar’s Den: Mon-Thu 11 a.m. to 11 p.m.; Fri 11 a.m. to 5 p.m.; Sat-Sun closed.\n4. CAS Starbucks: Mon-Thu 7:30 a.m. to 5 p.m.; Fri-Sun closed.\n5. Library Starbucks: Mon-Thu 8 a.m. to 10 p.m.; Fri-Sat 8 a.m. to 4 p.m.; Sun 1 p.m. to 8 p.m.",
    },
}

DATE_TRANSLATIONS = {
    "tr": {
        "Monday": "Pazartesi", "Tuesday": "Salı", "Wednesday": "Çarşamba", "Thursday": "Perşembe",
        "Friday": "Cuma", "Saturday": "Cumartesi", "Sunday": "Pazar",
        "January": "Ocak", "February": "Şubat", "March": "Mart", "April": "Nisan",
        "May": "Mayıs", "June": "Haziran", "July": "Temmuz", "August": "Ağustos",
        "September": "Eylül", "October": "Ekim", "November": "Kasım", "December": "Aralık",
    },
    "es": {
        "Monday": "lunes", "Tuesday": "martes", "Wednesday": "miércoles", "Thursday": "jueves",
        "Friday": "viernes", "Saturday": "sábado", "Sunday": "domingo",
        "January": "enero", "February": "febrero", "March": "marzo", "April": "abril",
        "May": "mayo", "June": "junio", "July": "julio", "August": "agosto",
        "September": "septiembre", "October": "octubre", "November": "noviembre", "December": "diciembre",
    },
    "zh": {
        "Monday": "星期一", "Tuesday": "星期二", "Wednesday": "星期三", "Thursday": "星期四",
        "Friday": "星期五", "Saturday": "星期六", "Sunday": "星期日",
        "January": "1月", "February": "2月", "March": "3月", "April": "4月",
        "May": "5月", "June": "6月", "July": "7月", "August": "8月",
        "September": "9月", "October": "10月", "November": "11月", "December": "12月",
    },
    "ur": {
        "Monday": "پیر", "Tuesday": "منگل", "Wednesday": "بدھ", "Thursday": "جمعرات",
        "Friday": "جمعہ", "Saturday": "ہفتہ", "Sunday": "اتوار",
        "January": "جنوری", "February": "فروری", "March": "مارچ", "April": "اپریل",
        "May": "مئی", "June": "جون", "July": "جولائی", "August": "اگست",
        "September": "ستمبر", "October": "اکتوبر", "November": "نومبر", "December": "دسمبر",
    },
    "ko": {
        "Monday": "월요일", "Tuesday": "화요일", "Wednesday": "수요일", "Thursday": "목요일",
        "Friday": "금요일", "Saturday": "토요일", "Sunday": "일요일",
        "January": "1월", "February": "2월", "March": "3월", "April": "4월",
        "May": "5월", "June": "6월", "July": "7월", "August": "8월",
        "September": "9월", "October": "10월", "November": "11월", "December": "12월",
    },
}

CALENDAR_EVENT_TRANSLATIONS = {
    "term begins": {
        "tr": "Dönem Başlangıcı",
        "es": "Inicio del semestre",
        "zh": "学期开始",
        "ur": "سمسٹر کا آغاز",
        "ko": "학기 시작",
    },
    "term ends": {
        "tr": "Dönem Bitişi",
        "es": "Fin del semestre",
        "zh": "学期结束",
        "ur": "سمسٹر کا اختتام",
        "ko": "학기 종료",
    },
    "registration begins": {
        "tr": "Kayıt Başlangıcı",
        "es": "Inicio de inscripción",
        "zh": "注册开始",
        "ur": "رجسٹریشن کا آغاز",
        "ko": "수강신청 시작",
    },
    "exam week": {
        "tr": "Sınav Haftası",
        "es": "Semana de exámenes",
        "zh": "考试周",
        "ur": "امتحانی ہفتہ",
        "ko": "시험 주간",
    },
    "summer session i begins": {
        "tr": "Yaz Dönemi I Başlangıcı",
        "es": "Inicio de Verano I",
        "zh": "夏季第一学段开始",
        "ur": "سمر سیشن اوّل کا آغاز",
        "ko": "여름학기 I 시작",
    },
    "summer session ii begins": {
        "tr": "Yaz Dönemi II Başlangıcı",
        "es": "Inicio de Verano II",
        "zh": "夏季第二学段开始",
        "ur": "سمر سیشن دوم کا آغاز",
        "ko": "여름학기 II 시작",
    },
}
CALENDAR_EVENT_WORD_TRANSLATIONS = {
    "tr": {
        "Spring": "Bahar",
        "Fall": "Güz",
        "Summer": "Yaz",
        "Winter": "Kış",
        "Semester": "Dönemi",
        "Term": "Dönemi",
        "Begins": "Başlangıcı",
        "Ends": "Bitişi",
        "Registration": "Kayıt",
        "Exam": "Sınav",
        "Week": "Haftası",
        "Classes": "Dersler",
    },
    "es": {
        "Spring": "Primavera",
        "Fall": "Otoño",
        "Summer": "Verano",
        "Winter": "Invierno",
        "Semester": "Semestre",
        "Term": "Periodo",
        "Begins": "Inicio",
        "Ends": "Fin",
        "Registration": "Inscripción",
        "Exam": "Exámenes",
        "Week": "Semana",
        "Classes": "Clases",
    },
    "zh": {
        "Spring": "春季",
        "Fall": "秋季",
        "Summer": "夏季",
        "Winter": "冬季",
        "Semester": "学期",
        "Term": "学期",
        "Begins": "开始",
        "Ends": "结束",
        "Registration": "注册",
        "Exam": "考试",
        "Week": "周",
        "Classes": "课程",
    },
    "ur": {
        "Spring": "بہار",
        "Fall": "خزاں",
        "Summer": "گرما",
        "Winter": "سردی",
        "Semester": "سمسٹر",
        "Term": "مدت",
        "Begins": "آغاز",
        "Ends": "اختتام",
        "Registration": "رجسٹریشن",
        "Exam": "امتحان",
        "Week": "ہفتہ",
        "Classes": "کلاسیں",
    },
    "ko": {
        "Spring": "봄",
        "Fall": "가을",
        "Summer": "여름",
        "Winter": "겨울",
        "Semester": "학기",
        "Term": "학기",
        "Begins": "시작",
        "Ends": "종료",
        "Registration": "등록",
        "Exam": "시험",
        "Week": "주간",
        "Classes": "수업",
    },
}

def detect_language(text: str) -> str:
    t = text.strip().lower()
    t_norm = normalize(text)
    t_tokens = set(t_norm.split())
    short_input = re.sub(r"\s+", " ", t.strip())

    for lang_code, phrases in SHORT_LANGUAGE_HINTS.items():
        if short_input in phrases or t_norm in phrases:
            return lang_code

    if re.search(r"[\u4e00-\u9fff]", t):
        return "zh"
    if re.search(r"[\uac00-\ud7af]", t):
        return "ko"
    if re.search(r"[\u0600-\u06ff]", t):
        return "ur"

    tr_chars = set("çğıöşü")
    if any(ch in tr_chars for ch in t):
        return "tr"
    if any(ch in t for ch in ("¿", "¡", "ñ")):
        return "es"

    lang_scores = {"tr": 0, "es": 0, "zh": 0, "ur": 0, "ko": 0}
    for lang_code, keywords in LANGUAGE_HINT_KEYWORDS.items():
        for keyword in keywords:
            if keyword_in_text(t_norm, t_tokens, keyword):
                lang_scores[lang_code] += max(1, len(keyword) // 2)

    best_lang = max(lang_scores, key=lang_scores.get)
    if lang_scores[best_lang] >= 2:
        return best_lang

    return "en"

async def call_ollama(messages: list[dict], num_predict: Optional[int] = None) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": num_predict or OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE,
            "num_ctx": OLLAMA_NUM_CTX,
        },
        "keep_alive": "30m",
    }

    async with httpx.AsyncClient(timeout=build_ollama_timeout()) as client:
        health = await client.get(OLLAMA_HEALTH_URL, timeout=OLLAMA_HEALTH_TIMEOUT_SECONDS)
        health.raise_for_status()

        last_error = None
        for _ in range(max(1, OLLAMA_MAX_RETRIES + 1)):
            try:
                response = await asyncio.wait_for(
                    client.post(OLLAMA_URL, json=payload),
                    timeout=OLLAMA_TOTAL_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "No response.")
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                last_error = exc

        if last_error:
            raise last_error
        raise RuntimeError("Unknown Ollama error")

async def localize_answer_text(answer: str, lang: str) -> str:
    if lang == "en" or not answer.strip():
        return answer

    lang_name = LANGUAGE_NAMES.get(lang, "English")
    try:
        translated = await call_ollama(
            [
                {
                    "role": "system",
                    "content": (
                        "You are translating a Kean University chatbot reply. "
                        f"Translate the reply into {lang_name}. "
                        "Keep the meaning exact. Preserve URLs, phone numbers, emails, bullets, numbering, and line breaks. "
                        "Do not add explanations, do not cite sources, and do not invent facts. "
                        "Keep official building, office, and program names unchanged unless there is a standard translation in the original text."
                    ),
                },
                {"role": "user", "content": answer},
            ],
            num_predict=max(180, min(500, len(answer) // 2 + 80)),
        )
        return translated.strip() or answer
    except Exception:
        return answer

def trn(key: str, lang: str, **kwargs) -> str:
    lang = lang if lang in SUPPORTED_LANGS else "en"
    template = TRANSLATIONS.get(key, {}).get(lang) or TRANSLATIONS.get(key, {}).get("en", "")
    return template.format(**kwargs)

def localize_date_text(text: str, lang: str) -> str:
    mapping = DATE_TRANSLATIONS.get(lang)
    if not mapping:
        return text
    for en_word, local_word in mapping.items():
        text = re.sub(rf"\b{re.escape(en_word)}\b", local_word, text)
    return text

def localize_calendar_event_text(event_text: str, lang: str) -> str:
    if lang == "en":
        return event_text.title()

    normalized_event = normalize(event_text)
    for key, mapping in CALENDAR_EVENT_TRANSLATIONS.items():
        if key in normalized_event:
            return mapping.get(lang, event_text.title())

    text = event_text.title()
    word_mapping = CALENDAR_EVENT_WORD_TRANSLATIONS.get(lang, {})
    for en_word, local_word in word_mapping.items():
        text = re.sub(rf"\b{re.escape(en_word)}\b", local_word, text)
    return text

def parse_position(value: str) -> Optional[tuple[float, float]]:
    raw = str(value or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
    except ValueError:
        return None
    return lat, lon

def haversine_meters(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    lat1, lon1 = point_a
    lat2, lon2 = point_b
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 2 * 6371000 * asin(sqrt(a))

def load_campus_locations():
    rows = []
    if not LOCATION_JSON_PATH.exists():
        return rows

    with open(LOCATION_JSON_PATH, "r", encoding="utf-8") as f:
        try:
            entries = json.load(f)
        except json.JSONDecodeError:
            return rows

    for entry in entries:
        row_id = str(entry.get("id") or "").strip()
        name = str(entry.get("name") or "").strip()
        if not row_id or not name:
            continue

        position = None
        latitude = entry.get("latitude")
        longitude = entry.get("longitude")
        if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
            position = (float(latitude), float(longitude))
        elif entry.get("latlng"):
            position = parse_position(str(entry.get("latlng")))
        else:
            coordinates = entry.get("coordinates")
            if (
                isinstance(coordinates, list)
                and len(coordinates) >= 2
                and isinstance(coordinates[0], (int, float))
                and isinstance(coordinates[1], (int, float))
            ):
                position = (float(coordinates[1]), float(coordinates[0]))

        row_type = str(entry.get("type") or "").strip().lower() or "location"
        rows.append(
            {
                "id": row_id,
                "name": name,
                "campus": str(entry.get("campus") or "").strip() or "Main",
                "type": row_type,
                "parent": str(entry.get("parent") or "").strip(),
                "position": position,
            }
        )
    return rows

def load_program_catalog():
    if not PROGRAMS_FILE.exists():
        return []
    try:
        with open(PROGRAMS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return []

    programs = raw.get("programs", {}) if isinstance(raw, dict) else {}
    catalog = []
    for program_id, payload in programs.items():
        if not isinstance(payload, dict):
            continue
        metadata = payload.get("metadata") or {}
        details = payload.get("details") or {}
        curriculum = payload.get("curriculum") or {}
        full_name = str(metadata.get("full_name") or program_id).strip()
        description = str(details.get("description") or "").strip()
        contact = metadata.get("contact") or {}
        contact_parts = [str(contact.get("email") or "").strip(), str(contact.get("phone") or "").strip()]
        contact_text = " | ".join(part for part in contact_parts if part)

        catalog_url = ""
        for course_group in ("core_courses", "elective_courses"):
            courses = curriculum.get(course_group) or {}
            if isinstance(courses, dict):
                for course in courses.values():
                    if isinstance(course, dict) and course.get("url"):
                        catalog_url = str(course.get("url")).strip()
                        break
            if catalog_url:
                break

        program_url = (
            str(metadata.get("url") or "").strip()
            or str(details.get("url") or "").strip()
            or ""
        )

        name_tokens = meaningful_tokens(full_name)
        catalog.append(
            {
                "id": program_id,
                "name": full_name,
                "description": description,
                "contact": contact_text,
                "url": program_url or KEAN_PROGRAMS_URL,
                "catalog_url": catalog_url,
                "tokens": name_tokens | meaningful_tokens(description) | meaningful_tokens(program_id.replace("_", " ")),
            }
        )
    return catalog

def is_course_repeat_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "retake",
            "repeat a course",
            "repeat course",
            "course repeat",
            "grade recalculation",
            "retakes",
            "repeats",
            "repetir clase",
            "repetir curso",
            "retomar clase",
            "retomar la misma clase",
            "volver a tomar una clase",
            "volver a tomar un curso",
            "tomar la misma clase",
            "tomar una clase",
            "tomar un curso",
            "cuantas veces puedo tomar",
            "cuántas veces puedo tomar",
            "cuantas veces puedo retomar",
            "cuántas veces puedo retomar",
            "tekrar ders",
            "ders tekrari",
            "ders tekrarı",
            "재수강",
            "重修",
        )
    )

def is_admissions_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "admission",
            "admissions",
            "apply",
            "application",
            "accepted",
            "readmit",
            "transfer admission",
            "graduate admission",
            "cougar app",
            "admisiones",
            "solicitud",
            "aplicar",
            "admission office",
            "basvuru",
            "başvuru",
            "kabul",
            "입학",
            "申请",
        )
    )

def is_graduation_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "graduation",
            "graduate",
            "commencement",
            "diploma",
            "degree conferral",
            "honors",
            "graduation fee",
            "graduacion",
            "graduación",
            "diploma",
            "mezuniyet",
            "mezun olma",
            "졸업",
            "毕业",
        )
    ) and not is_degree_availability_question(text)

def is_financial_aid_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "financial aid",
            "fafsa",
            "grant",
            "loan",
            "scholarship",
            "work study",
            "ayuda financiera",
            "beca",
            "prestamo",
            "préstamo",
            "mali yardim",
            "mali yardım",
            "burs",
            "kredi",
            "장학금",
            "학자금",
            "助学金",
            "贷款",
        )
    )

def is_registrar_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "registrar",
            "transcript",
            "enrollment verification",
            "grade recalculation",
            "registro",
            "expediente",
            "certificacion de estudios",
            "certificación de estudios",
            "transkript",
            "ogrenci kayit",
            "öğrenci kayıt",
            "성적표",
            "재학증명",
            "成绩单",
            "在学证明",
        )
    )

def is_one_stop_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "one stop",
            "one-stop",
            "change of major form",
            "registration petition",
            "residency petition",
            "cambio de especialidad",
            "peticion de registro",
            "petición de registro",
            "major degistirme",
            "major değiştirme",
            "kayit dilekcesi",
            "kayıt dilekçesi",
            "원스톱",
            "전공 변경",
            "一站式",
            "转专业",
        )
    )

def is_dining_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "dining",
            "food court",
            "marketplace",
            "starbucks",
            "cougar's den",
            "cougars den",
            "cafe yumba",
            "restaurant",
            "eat on campus",
            "comida",
            "cafeteria",
            "cafetería",
            "restaurante",
            "yemek",
            "kafeterya",
            "kantin",
            "식당",
            "음식",
            "餐厅",
            "食堂",
        )
    )

def is_shuttle_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "shuttle",
            "shuttle service",
            "track my shuttle",
            "trackmyshuttle",
            "resident student shuttle",
            "kean trolley",
            "trolley",
            "bus",
            "transportation",
            "transport request",
            "transporte",
            "autobus",
            "autobús",
            "servicio de transporte",
            "rastrear shuttle",
            "servis",
            "otobus",
            "otobüs",
            "ulasim",
            "ulaşım",
            "셔틀",
            "버스",
            "交通车",
            "班车",
            "接驳",
        )
    )

def is_student_accounts_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "tuition",
            "bursar",
            "student accounting",
            "payment plan",
            "payment plans",
            "monthly payment",
            "installment plan",
            "bill",
            "refund",
            "late payment fee",
            "matricula",
            "matrícula",
            "plan de pago",
            "cuotas",
            "ucret",
            "ücret",
            "odeme plani",
            "ödeme planı",
            "등록금",
            "분할 납부",
            "学费",
            "付款计划",
        )
    )

def is_smoking_policy_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "smoke",
            "smoking",
            "tobacco",
            "vape",
            "vaping",
            "cigarette",
            "cannabis",
            "marijuana",
            "fumar",
            "tabaco",
            "vapear",
            "sigara",
            "tutun",
            "tütün",
            "icmek",
            "içmek",
            "흡연",
            "담배",
            "전자담배",
            "吸烟",
            "抽烟",
        )
    )

def is_housing_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "housing",
            "residence life",
            "dorm",
            "dorms",
            "residence hall",
            "roommate",
            "move in",
            "move-in",
            "meal plan",
            "housing application",
            "vivienda",
            "residencia",
            "dormitorio",
            "companero de cuarto",
            "compañero de cuarto",
            "mudanza",
            "yurt",
            "konut",
            "oda arkadasi",
            "oda arkadaşı",
            "tasinma",
            "taşınma",
            "기숙사",
            "룸메이트",
            "入住",
            "宿舍",
        )
    )

def is_health_services_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "student health",
            "health services",
            "wellness center",
            "counseling center",
            "mental health",
            "immunization",
            "vaccine",
            "health absence",
            "short term leave",
            "semester withdrawal",
            "uwill",
            "salud",
            "servicios de salud",
            "bienestar",
            "consejeria",
            "consejería",
            "vacuna",
            "inmunizacion",
            "inmunización",
            "saglik",
            "sağlık",
            "danismanlik",
            "danışmanlık",
            "asi",
            "aşı",
            "wellness",
            "건강",
            "상담",
            "예방접종",
            "健康",
            "心理咨询",
            "疫苗",
        )
    )

def is_accessibility_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "accessibility",
            "accessibility services",
            "accommodation",
            "accommodations",
            "disability",
            "meal plan exemption",
            "testing room",
            "accommodate portal",
            "oas",
            "accesibilidad",
            "acomodacion",
            "acomodación",
            "discapacidad",
            "exencion de plan de comida",
            "exención de plan de comida",
            "erisilebilirlik",
            "erişilebilirlik",
            "engelli",
            "uyarlama",
            "장애",
            "접근성",
            "편의 제공",
            "无障碍",
            "便利服务",
        )
    )

def is_bookstore_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in (
            "bookstore",
            "barnes",
            "noble",
            "textbook",
            "textbooks",
            "ebook",
            "e-book",
            "yuzu",
            "book shop",
            "libreria",
            "librería",
            "libros",
            "texto",
            "kitabevi",
            "ders kitabi",
            "ders kitabı",
            "서점",
            "교재",
            "电子书",
            "书店",
            "教材",
        )
    )

def build_course_repeat_answer(lang: str) -> str:
    return trn("course_repeat_summary", lang)

def build_admissions_answer(lang: str) -> str:
    return trn("admissions_summary", lang)

def build_graduation_answer(lang: str) -> str:
    return trn("graduation_summary", lang)

def build_financial_aid_answer(lang: str) -> str:
    return trn("financial_aid_summary", lang)

def build_registrar_answer(lang: str) -> str:
    return trn("registrar_summary", lang)

def build_one_stop_answer(lang: str) -> str:
    return trn("one_stop_summary", lang)

def build_dining_answer(lang: str) -> str:
    return trn("dining_summary", lang)

def build_student_accounts_answer(lang: str) -> str:
    return (
        "Yes. Kean offers payment plans through the Office of Student Accounting.\n"
        "1. Fall and spring payment plans are available in 3, 4, or 5 monthly installments.\n"
        "2. Summer payment plans are available in 2 or 3 monthly installments.\n"
        "3. Plans are set up directly with Kean through your Student Account Suite in KeanWISE under View/Pay My Bill > Payment Plan Management.\n"
        "4. A non-refundable $40 plan initiation fee applies.\n"
        "5. More information: https://www.kean.edu/offices/student-accounting"
    )

def build_shuttle_answer(lang: str) -> str:
    return (
        "Here’s what I found about the Kean shuttle service:\n"
        "1. The Union campus shuttle runs Monday through Friday during the fall and spring semesters from 7:30 a.m. to 10:50 p.m.\n"
        "2. After 5 p.m., the shuttle switches to on-demand service, and the last reservation is taken at 10:50 p.m.\n"
        "3. Main shuttle stops listed in the current file include Kean Hall Parking Lot, Hennings Hall, Wilkins Theatre, NAAB, STEM/LHAC, East Campus Parking Lot, East Campus Building, Faculty Housing, and Hynes Hall.\n"
        "4. Track the shuttle at https://www.trackmyshuttle.com using code KEAN. For on-demand service, use code KEANOD.\n"
        "5. Resident students can also use the Residence Life shuttle on Fridays and Sundays: https://www.kean.edu/reslife/amenities/resident-student-shuttle"
    )

def build_smoking_policy_answer(lang: str) -> str:
    return (
        "No. You cannot smoke or vape inside campus buildings, residence halls, offices, state vehicles, or around the Child Care and Development Center.\n"
        "1. If smoking is permitted outdoors, you must stay at least 3 feet from building entrances.\n"
        "2. The policy includes cigarettes, cigars, hookahs, pipes, smokeless tobacco, e-cigarettes, vapes, and similar devices.\n"
        "3. Cannabis or marijuana is not allowed anywhere on Kean property, even with a prescription.\n"
        "4. Students who violate the policy may face fines and conduct sanctions.\n"
        "5. More information: https://www.kean.edu"
    )

def build_housing_answer(lang: str) -> str:
    return (
        "Here’s a quick housing overview:\n"
        "1. Kean offers campus housing for freshmen, upperclassmen, transfer, and some graduate students.\n"
        "2. Students apply for housing through the Residence Life Housing Portal in KeanWISE after admission.\n"
        "3. The housing application includes a non-refundable $125 application fee.\n"
        "4. For housing help, contact Residence Life at (908) 737-1700 or reslife@kean.edu.\n"
        "5. More information: https://www.kean.edu"
    )

def build_health_services_answer(lang: str) -> str:
    return (
        "Here’s a quick health and wellness overview:\n"
        "1. Student Health Services and the Counseling Center are located in Downs Hall through the Kean Wellness Center.\n"
        "2. Student Health Services helps with primary care, vaccines, lab services, sexual health, and health-related absences.\n"
        "3. The Counseling Center offers short-term counseling, crisis support, referrals, and Uwill teletherapy access.\n"
        "4. For counseling and wellness appointments, call (908) 737-4850. For Student Health Services, use the Student Health Portal.\n"
        "5. In an emergency, call 911 or Kean University Police at (908) 737-4800."
    )

def build_accessibility_answer(lang: str) -> str:
    return (
        "Here’s what I found about Accessibility Services:\n"
        "1. The Office of Accessibility Services helps students request academic and housing accommodations.\n"
        "2. Students must submit documentation and complete an application for services.\n"
        "3. The office is in the Kean Wellness Center, Downs Hall Room 122.\n"
        "4. Contact: accessibilityservices@kean.edu or (908) 737-4910.\n"
        "5. More information: https://www.kean.edu"
    )

def build_bookstore_answer(lang: str) -> str:
    return (
        "Here’s what I found about the bookstore:\n"
        "1. The Kean University Official Bookstore is in the Green Lane Building at 1040 Morris Avenue, Union, NJ 07083.\n"
        "2. Store phone: (908) 737-4940.\n"
        "3. General customer support: (877) 420-1734 or customercare@bncservices.com.\n"
        "4. The bookstore supports textbooks, rentals, eBooks, apparel, and spirit items.\n"
        "5. Store hours in the current file are Monday to Thursday 10 a.m. to 5 p.m., Friday 10 a.m. to 3 p.m., Saturday 10 a.m. to 2 p.m., and Sunday closed."
    )

def find_program_match(text: str) -> Optional[dict]:
    if not program_catalog:
        return None

    q = normalize(text)
    q_tokens = meaningful_tokens(text)
    if not q_tokens:
        q_tokens = tokenize(text)

    best_match = None
    best_score = 0
    for program in program_catalog:
        name_norm = normalize(program["name"])
        score = 0
        overlap = len(q_tokens & program["tokens"])
        if overlap:
            score += overlap * 20
        if name_norm and name_norm in q:
            score += 120
        if q in name_norm and len(q) >= 4:
            score += 90
        if score > best_score:
            best_score = score
            best_match = program

    if best_match and best_score >= 40:
        return best_match
    return None

def build_program_answer(program: dict, lang: str) -> str:
    lines = [trn("program_details_intro", lang, name=program["name"])]
    if program.get("description"):
        lines.append(program["description"])
    link_url = program.get("url") or program.get("catalog_url") or KEAN_PROGRAMS_URL
    lines.append(trn("program_more_info", lang, url=link_url))
    if program.get("contact"):
        lines.append(trn("program_contact", lang, contact=program["contact"]))
    return "\n".join(lines)

def build_degree_availability_answer(
    exists: bool,
    subject: str,
    level: str,
    lang: str,
    program: Optional[dict] = None,
) -> str:
    answer = trn("degree_exists_yes" if exists else "degree_exists_no", lang, subject=subject, level=level)
    if exists:
        url = (program or {}).get("url") or (program or {}).get("catalog_url") or KEAN_PROGRAMS_URL
        answer = f"{answer}\n{trn('program_more_info', lang, url=url)}"
    return answer

def is_section_heading(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if text.startswith(("*", "-", "1.", "2.", "3.")):
        return False
    if len(text) > 90:
        return False
    if text.endswith((".", "?", "!")) and len(text.split()) > 8:
        return False
    words = text.split()
    if len(words) <= 8:
        return True
    uppercase_ratio = sum(1 for ch in text if ch.isupper()) / max(1, sum(1 for ch in text if ch.isalpha()))
    return uppercase_ratio > 0.65

def split_text_into_sections(text: str, max_chars: int = 900) -> list[str]:
    lines = [line.strip() for line in str(text or "").splitlines()]
    sections = []
    heading = None
    buffer = []

    def flush():
        nonlocal heading, buffer
        if not heading and not buffer:
            return
        body = " ".join(part for part in buffer if part).strip()
        if heading and body:
            content = f"{heading}\n{body}"
        else:
            content = heading or body
        content = content.strip()
        if not content:
            heading = None
            buffer = []
            return
        if len(content) <= max_chars:
            sections.append(content)
        else:
            paragraphs = [part.strip() for part in re.split(r"\n{2,}", content) if part.strip()]
            current = ""
            for paragraph in paragraphs:
                candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
                if current and len(candidate) > max_chars:
                    sections.append(current.strip())
                    current = paragraph
                else:
                    current = candidate
            if current.strip():
                sections.append(current.strip())
        heading = None
        buffer = []

    for line in lines:
        if not line:
            if buffer:
                flush()
            continue
        if is_section_heading(line):
            flush()
            heading = line
            continue
        buffer.append(line)

    flush()
    return [section for section in sections if section]

def load_fallback_rag_docs():
    docs = []
    for file in BASE_DIR.rglob("*"):
        if not file.is_file():
            continue
        if any(part in EXCLUDED_RAG_DIR_NAMES for part in file.parts):
            continue
        if file.name in EXCLUDED_RAG_FILE_NAMES:
            continue
        if file.suffix.lower() not in {"", ".txt", ".json"}:
            continue
        try:
            text = read_text_with_fallback(file)
        except Exception:
            continue

        if not text.strip():
            continue

        path_lower = file.relative_to(BASE_DIR).as_posix().lower()
        filename_lower = file.name.lower()
        if "policies/" in path_lower or "policy" in filename_lower:
            doc_type = "policy"
        elif "calendar" in filename_lower:
            doc_type = "calendar"
        elif any(keyword in filename_lower for keyword in ("program", "major", "majors", "minor", "minors", "degree", "degrees", "catalog")):
            doc_type = "program"
        else:
            doc_type = "knowledge"

        if file.suffix.lower() == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None

            extracted_lines = []

            def walk(node):
                if isinstance(node, dict):
                    title = node.get("name") or node.get("title") or node.get("program_name")
                    description = node.get("description")
                    if title or description:
                        extracted_lines.append(f"Title: {title or 'Unknown'}")
                        if description:
                            extracted_lines.append(f"Description: {description}")
                        extracted_lines.append("")
                    for value in node.values():
                        walk(value)
                elif isinstance(node, list):
                    for item in node:
                        walk(item)
                elif isinstance(node, (str, int, float, bool)):
                    extracted_lines.append(str(node))

            if data is not None:
                walk(data)
                flat_text = "\n".join(extracted_lines).strip()
                if flat_text:
                    text = flat_text

        chunks = split_text_into_sections(text, max_chars=max(700, RAG_MAX_CHARS_PER_BLOCK + 200))
        if not chunks:
            chunks = [text.strip()]

        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "source": file.relative_to(BASE_DIR).as_posix(),
                    "type": doc_type,
                    "chunk_id": f"{file.stem}_{idx}",
                    "text": chunk,
                    "tokens": tokenize(chunk),
                    "key_tokens": meaningful_tokens(chunk),
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

    for extra_alias in LOCATION_ALIAS_OVERRIDES.get(place["id"], ()):
        aliases.add(extra_alias)

    return [normalize(alias) for alias in aliases if normalize(alias)]

def is_location_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(keyword_in_text(q, q_tokens, keyword) for keyword in LOCATION_KEYWORDS)

def should_use_current_location(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    if any(keyword_in_text(q, q_tokens, keyword) for keyword in LOCATION_ONLY_KEYWORDS):
        return False
    return any(keyword_in_text(q, q_tokens, keyword) for keyword in DIRECTION_KEYWORDS)

def is_closest_parking_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(keyword_in_text(q, q_tokens, keyword) for keyword in CLOSEST_PARKING_KEYWORDS)

def is_food_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(keyword_in_text(q, q_tokens, keyword) for keyword in FOOD_INTENT_KEYWORDS)

def is_parking_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    keywords = (
        "parking",
        "park",
        "ticket",
        "permit",
        "lot",
        "parqueo",
        "estacionamiento",
        "otopark",
        "停车",
        "주차",
        "پارکنگ",
    )
    return any(keyword_in_text(q, q_tokens, keyword) for keyword in keywords)

def is_parking_location_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    has_parking = any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in ("parking", "park", "lot", "parqueo", "estacionamiento", "otopark", "停车", "주차", "پارکنگ")
    )
    has_location_style = any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in ("where", "donde", "nerede", "where can i", "can i park", "map", "near")
    )
    asks_ticket_or_cost = any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in ("ticket", "fine", "citation", "cost", "price", "how much", "multa", "ceza")
    )
    return has_parking and (has_location_style or not asks_ticket_or_cost) and not asks_ticket_or_cost

def is_parking_ticket_fee_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    asks_ticket = any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in ("ticket", "fine", "citation", "multa", "ceza", "罚单", "벌금", "جرمانہ")
    )
    asks_cost = any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in ("cost", "price", "how much", "fee", "cuanto", "costo", "ne kadar", "多少", "کم")
    )
    has_parking = any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in ("parking", "park", "parqueo", "estacionamiento", "otopark", "停车", "주차", "پارکنگ")
    )
    return has_parking and (asks_ticket or asks_cost)

def build_parking_ticket_fee_answer(lang: str) -> str:
    fee_items = []
    late_fee_line = None

    for doc in fallback_rag_docs:
        source = str(doc.get("source", "")).lower()
        if "parking" not in source:
            continue
        for raw in doc.get("text", "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if "$" in line and (" - $" in line or "fee" in line.lower() or "fine" in line.lower() or "late fee" in line.lower()):
                if "late fee" in line.lower():
                    late_fee_line = line
                elif " - $" in line and len(fee_items) < 6:
                    fee_items.append(line)

    if not fee_items and not late_fee_line:
        return trn("parking_ticket_fees_unavailable", lang)

    lines = [trn("parking_ticket_fees_intro", lang)]
    for item in fee_items[:5]:
        lines.append(f"- {item}")
    if late_fee_line:
        lines.append(f"- {late_fee_line}")
    else:
        lines.append(trn("parking_ticket_fees_note", lang))
    return "\n".join(lines)

def parking_audience(text: str) -> str:
    q = normalize(text)
    q_tokens = tokenize(text)
    if any(keyword_in_text(q, q_tokens, k) for k in ("student", "students", "estudiante", "ogrenci", "öğrenci", "学生", "طلبہ")):
        return "student"
    if any(keyword_in_text(q, q_tokens, k) for k in ("faculty", "staff", "docente", "personal", "akademik", "教职工", "fakulte", "فیکلٹی")):
        return "faculty"
    if any(keyword_in_text(q, q_tokens, k) for k in ("overnight", "night", "nocturno", "gece", "夜间", "رات")):
        return "overnight"
    return "general"

def find_food_destination_id(text: str) -> Optional[str]:
    q = normalize(text)
    q_tokens = tokenize(text)
    for location_id, keywords in FOOD_DESTINATION_HINTS:
        if location_id not in campus_location_by_id:
            continue
        if any(keyword_in_text(q, q_tokens, keyword) for keyword in keywords):
            return location_id

    if "cougars_den" in campus_location_by_id:
        return "cougars_den"
    return None

def find_food_suggestions(text: str, max_results: int = 3) -> list[dict]:
    q = normalize(text)
    q_tokens = tokenize(text)
    scored: list[tuple[int, dict]] = []

    for location_id, keywords in FOOD_DESTINATION_HINTS:
        place = campus_location_by_id.get(location_id)
        if not place:
            continue
        score = 0
        for keyword in keywords:
            normalized_keyword = normalize(keyword)
            if normalized_keyword and keyword_in_text(q, q_tokens, keyword):
                score += max(1, len(normalized_keyword))
        if score > 0:
            scored.append((score, place))

    if not scored:
        fallback_ids = ["cougars_den", "starbucks_at_library", "smash_burgar_at_miron", "Cougar Pantry"]
        seen = set()
        results = []
        for location_id in fallback_ids:
            place = campus_location_by_id.get(location_id)
            if place and location_id not in seen:
                results.append(place)
                seen.add(location_id)
            if len(results) >= max_results:
                break
        return results

    scored.sort(key=lambda item: item[0], reverse=True)
    suggestions = []
    seen_ids = set()
    for _, place in scored:
        if place["id"] in seen_ids:
            continue
        suggestions.append(place)
        seen_ids.add(place["id"])
        if len(suggestions) >= max_results:
            break

    for fallback_id in ("cougars_den", "starbucks_at_library", "smash_burgar_at_miron", "Cougar Pantry"):
        if len(suggestions) >= max_results:
            break
        if fallback_id in seen_ids:
            continue
        fallback_place = campus_location_by_id.get(fallback_id)
        if fallback_place:
            suggestions.append(fallback_place)
            seen_ids.add(fallback_id)

    return suggestions

def should_keep_entrance_destination(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(keyword_in_text(q, q_tokens, keyword) for keyword in ENTRANCE_REQUEST_KEYWORDS)

def should_route_destination_without_location_keyword(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    if not q:
        return False

    # Allow direct lookup style messages ("naab", "miron center", etc.)
    if len(q_tokens) <= 4 and not any(
        keyword_in_text(q, q_tokens, keyword)
        for keyword in ("major", "program", "degree", "repeat", "class", "policy", "tuition", "how many")
    ):
        return True

    return False

def find_location_destination_id(text: str, allowed_types: Optional[set[str]] = None) -> Optional[str]:
    q = normalize(text)
    padded_query = f" {q} "
    query_tokens = set(q.split())
    query_requests_primary_building = any(token in query_tokens for token in {"center", "hall", "building"})
    best_place = None
    best_score = 0

    for place in campus_locations:
        if allowed_types and place.get("type") not in allowed_types:
            continue
        aliases = build_location_aliases(place)
        score = 0

        for alias in aliases:
            if not alias:
                continue
            padded_alias = f" {alias} "
            alias_tokens = set(alias.split())
            informative_alias_tokens = {t for t in alias_tokens if len(t) >= 3 and t not in GENERIC_LOCATION_WORDS}
            overlap = len(informative_alias_tokens & query_tokens)
            precision = overlap / max(1, len(informative_alias_tokens))

            if q == alias:
                score = max(score, 220)
            elif padded_alias in padded_query and len(alias) >= 3:
                score = max(score, 160)
            elif alias in q and len(alias) >= 5:
                score = max(score, 125)
            elif overlap > 0:
                score = max(score, 52 + overlap * 18 + int(precision * 24))

            if query_requests_primary_building and any(token in alias_tokens for token in SECONDARY_PLACE_WORDS):
                score -= 28

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

def get_effective_reference_location(location_id: str) -> Optional[dict]:
    place = campus_location_by_id.get(location_id)
    if not place:
        return None
    if place.get("type") == "entrance":
        parent_id = place.get("parent")
        if parent_id and parent_id in campus_location_by_id:
            return campus_location_by_id[parent_id]
    return place

def normalize_location_destination_for_response(destination_id: str, user_text: str) -> str:
    destination = campus_location_by_id.get(destination_id)
    if not destination:
        return destination_id
    if destination.get("type") != "entrance":
        return destination_id
    if should_keep_entrance_destination(user_text):
        return destination_id

    parent_id = destination.get("parent")
    if parent_id and parent_id in campus_location_by_id:
        return parent_id
    return destination_id

def find_closest_parking_lot(reference_location_id: str) -> Optional[dict]:
    reference = get_effective_reference_location(reference_location_id)
    if not reference:
        return None
    reference_position = reference.get("position")
    if not reference_position:
        return None

    best_lot = None
    best_distance = float("inf")
    for place in campus_locations:
        if place.get("type") != "parking":
            continue
        lot_position = place.get("position")
        if not lot_position:
            continue
        distance = haversine_meters(reference_position, lot_position)
        if distance < best_distance:
            best_distance = distance
            best_lot = place

    return best_lot

def detect_faq_intent(text: str) -> Optional[str]:
    q = normalize(text)
    q_tokens = tokenize(text)
    best_topic = None
    best_score = 0

    for topic, keywords in FAQ_INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            normalized_keyword = normalize(keyword)
            if not normalized_keyword:
                continue
            if keyword_in_text(q, q_tokens, normalized_keyword):
                score += len(normalized_keyword)
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic if best_score > 0 else None

def detect_degree_level(text: str) -> str:
    q = normalize(text)
    q_tokens = tokenize(text)
    if any(keyword_in_text(q, q_tokens, k) for k in ("master", "masters", "graduate", "postgraduate", "mba", "ma", "ms", "maestria", "maestría", "posgrado", "yuksek lisans", "yüksek lisans", "lisansustu", "lisansüstü", "석사", "研究生", "硕士")):
        return "graduate/master"
    if any(keyword_in_text(q, q_tokens, k) for k in ("undergraduate", "undergrad", "bachelor", "major", "ba", "bs", "pregrado", "licenciatura", "lisans", "학부", "本科")):
        return "undergraduate"
    return "degree"

def localize_degree_level(level: str, lang: str) -> str:
    mapping = {
        "graduate/master": {
            "en": "graduate/master",
            "tr": "yüksek lisans/lisansüstü",
            "es": "maestría/posgrado",
            "zh": "硕士/研究生",
            "ur": "ماسٹر/گریجویٹ",
            "ko": "석사/대학원",
        },
        "undergraduate": {
            "en": "undergraduate",
            "tr": "lisans",
            "es": "pregrado",
            "zh": "本科",
            "ur": "انڈرگریجویٹ",
            "ko": "학부",
        },
        "degree": {
            "en": "degree",
            "tr": "derece",
            "es": "grado",
            "zh": "学位",
            "ur": "ڈگری",
            "ko": "학위",
        },
    }
    return mapping.get(level, mapping["degree"]).get(lang, level)

def is_degree_availability_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    asks_existence = any(
        keyword_in_text(q, q_tokens, k)
        for k in ("is there", "does kean have", "do you have", "there is", "hay", "var mi", "有没有", "있나요", "کیا")
    )
    asks_degree = any(
        keyword_in_text(q, q_tokens, k)
        for k in ("major", "bachelor", "undergrad", "undergraduate", "master", "graduate", "degree", "program", "carrera", "especialidad", "maestria", "maestría", "lisans", "yuksek lisans", "yüksek lisans", "전공", "학위", "硕士", "专业")
    )
    return asks_existence and asks_degree

def is_program_follow_up_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, k)
        for k in (
            "tell me more",
            "more info",
            "more information",
            "which ones",
            "what are they",
            "give details",
            "details",
            "list them",
            "dime mas",
            "mas info",
            "cuales",
            "hangileri",
            "daha fazla",
            "再多一点",
            "更多信息",
            "مزید",
        )
    )

def is_identity_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, k)
        for k in (
            "who are you",
            "what are you",
            "what is your name",
            "whats your name",
            "what's your name",
            "your name",
            "are you a bot",
            "quien eres",
            "qué eres",
            "que eres",
            "quien sos",
            "como te llamas",
            "cómo te llamas",
            "cual es tu nombre",
            "cuál es tu nombre",
            "sen kimsin",
            "adin ne",
            "adın ne",
            "nesin",
            "你是谁",
            "你是什麼",
            "你叫什么",
            "你叫什麼",
            "کون ہو",
            "آپ کون ہیں",
            "آپ کا نام کیا ہے",
            "너는 누구야",
            "이름이 뭐야",
            "이름이 뭐예요",
            "뭐 할 수 있어",
        )
    )

def is_greeting(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    greeting_keywords = (
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "merhaba",
        "selam",
        "hola",
        "buenas",
        "你好",
        "您好",
        "سلام",
        "안녕",
        "안녕하세요",
    )
    return len(q_tokens) <= 4 and any(keyword_in_text(q, q_tokens, k) for k in greeting_keywords)

def is_thanks(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, k)
        for k in (
            "thanks", "thank you", "thank u",
            "gracias", "muchas gracias",
            "tesekkurler", "teşekkürler", "tesekkur ederim", "teşekkür ederim",
            "谢谢", "多谢", "感谢",
            "감사", "감사합니다", "고마워", "고맙습니다",
            "شکریہ", "بہت شکریہ",
        )
    )

def is_farewell(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, k)
        for k in (
            "bye", "goodbye", "see you", "see ya",
            "adios", "adiós", "hasta luego", "nos vemos",
            "gorusuruz", "görüşürüz", "hosca kal", "hoşça kal",
            "再见", "拜拜",
            "안녕히 가세요", "잘 가", "안녕",
            "خدا حافظ", "الوداع",
        )
    )

def is_help_capabilities_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, k)
        for k in (
            "help", "what can you do", "how can you help", "what do you do",
            "ayuda", "puedes ayudar", "qué puedes hacer", "que puedes hacer", "en que ayudas", "en qué ayudas",
            "yardim", "yardım", "neler yapabilirsin", "ne yapabilirsin", "yardim edebilir misin", "yardım edebilir misin",
            "帮助", "你能做什么", "你可以做什么",
            "도움", "무엇을 도와줄 수 있어", "뭘 할 수 있어", "무엇을 할 수 있어",
            "مدد", "آپ کیا کر سکتے ہیں", "کیا مدد کر سکتے ہیں",
        )
    )

def is_acknowledgment(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return len(q_tokens) <= 4 and any(
        keyword_in_text(q, q_tokens, k)
        for k in (
            "ok", "okay", "got it", "understood",
            "okey", "vale", "entiendo", "entendido", "esta bien", "está bien",
            "tamam", "anladim", "anladım",
            "好", "好的", "明白了",
            "네", "알겠어", "알겠어요", "알겠습니다",
            "ٹھیک ہے", "سمجھ گیا", "سمجھ گئی",
        )
    )

def is_clarification_request(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, k)
        for k in (
            "repeat", "say that again", "explain again", "simpler", "more simply", "i dont understand", "i don't understand",
            "repite", "otra vez", "explica otra vez", "mas simple", "más simple", "no entiendo",
            "tekrar", "yeniden acikla", "yeniden açıkla", "daha basit", "anlamadim", "anlamadım",
            "再说一遍", "再解释", "简单一点", "我不明白",
            "다시 말해", "다시 설명", "더 쉽게", "이해가 안 돼",
            "دوبارہ", "پھر سے بتائیں", "آسان الفاظ میں", "میں نہیں سمجھا",
        )
    )

def is_frustration(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, k)
        for k in (
            "that didnt help", "that didn't help", "wrong", "not helpful", "you are wrong",
            "no ayudo", "no ayudó", "incorrecto", "eso esta mal", "eso está mal",
            "olmadi", "olmadı", "yanlis", "yanlış", "yardimci olmadi", "yardımcı olmadı",
            "没帮助", "不对", "错了",
            "도움이 안 돼", "틀렸어", "잘못됐어",
            "مدد نہیں ملی", "غلط", "یہ ٹھیک نہیں",
        )
    )

def is_hours_question(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, k)
        for k in (
            "hours",
            "open hours",
            "opening hours",
            "what time",
            "schedule",
            "horario",
            "hora",
            "abierto",
            "saat",
            "acik",
            "açık",
            "开放时间",
            "营业时间",
            "시간",
            "영업 시간",
            "운영 시간",
            "اوقات",
        )
    )

def is_library_target(text: str) -> bool:
    q = normalize(text)
    q_tokens = tokenize(text)
    return any(
        keyword_in_text(q, q_tokens, k)
        for k in ("library", "thompson", "nancy thompson library", "biblioteca", "kutuphane", "kütüphane", "图书馆", "도서관", "لائبریری")
    )

def detect_hours_target(text: str) -> Optional[str]:
    q = normalize(text)
    q_tokens = tokenize(text)

    if any(keyword_in_text(q, q_tokens, k) for k in ("library", "thompson", "nancy thompson library", "biblioteca", "kutuphane", "kütüphane", "图书馆", "도서관", "لائبریری")):
        return "library"
    if any(keyword_in_text(q, q_tokens, k) for k in ("gym", "fitness center", "recreation center", "harwood", "arena", "gimnasio", "spor salonu", "체육관", "健身房")):
        return "gym"
    if any(keyword_in_text(q, q_tokens, k) for k in ("pool", "swimming pool", "natatorium", "aquatic", "piscina", "havuz", "수영장", "游泳池")):
        return "pool"
    return None

def build_library_hours_answer(lang: str) -> str:
    hours_file = DATA_FOLDER / "Hours of Operation.txt"
    if not hours_file.exists():
        return trn("library_hours_unavailable", lang)

    try:
        raw = hours_file.read_text(encoding="utf-8")
    except Exception:
        return trn("library_hours_unavailable", lang)

    lines = [line.strip() for line in raw.splitlines()]
    target_idx = -1
    for idx, line in enumerate(lines):
        line_norm = normalize(line)
        if line_norm in {"nancy thompson library", "kean university thompson library"}:
            target_idx = idx
            break
    if target_idx < 0:
        return trn("library_hours_unavailable", lang)

    schedule_lines = []
    for line in lines[target_idx + 1 : target_idx + 18]:
        if not line:
            if schedule_lines:
                break
            continue
        line_norm = normalize(line)
        if "====" in line:
            break
        if any(
            line_norm.startswith(day)
            for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        ):
            schedule_lines.append(line.replace("\t", " "))

    if not schedule_lines:
        return trn("library_hours_unavailable", lang)

    answer_lines = [trn("library_hours_intro", lang)]
    for item in schedule_lines:
        answer_lines.append(f"- {localize_date_text(item, lang)}")
    return "\n".join(answer_lines)

def _extract_hours_section(raw: str, anchor_keywords: tuple[str, ...], stop_markers: tuple[str, ...]) -> list[str]:
    lines = [line.strip() for line in raw.splitlines()]
    start_idx = -1
    for idx, line in enumerate(lines):
        line_norm = normalize(line)
        if any(keyword in line_norm for keyword in anchor_keywords):
            start_idx = idx
            break
    if start_idx < 0:
        return []

    section = []
    for line in lines[start_idx + 1 : start_idx + 24]:
        if not line:
            if section:
                break
            continue
        line_norm = normalize(line)
        if any(marker in line_norm for marker in stop_markers):
            break
        if ":" in line and any(
            day in line_norm
            for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        ):
            section.append(line)
        elif "to" in line_norm and any(
            day in line_norm
            for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        ):
            section.append(line)
    return section

def build_target_hours_answer(target: str, lang: str) -> str:
    if target == "library":
        return build_library_hours_answer(lang)

    hours_file = DATA_FOLDER / "Hours of Operation.txt"
    if not hours_file.exists():
        return trn("hours_target_unavailable", lang, target=target)

    try:
        raw = hours_file.read_text(encoding="utf-8")
    except Exception:
        return trn("hours_target_unavailable", lang, target=target)

    if target == "gym":
        section = _extract_hours_section(
            raw,
            anchor_keywords=("miron student center building hours",),
            stop_markers=("registrar", "office of the registrar", "dining locations"),
        )
        if not section:
            return trn("hours_target_unavailable", lang, target=target)
        answer_lines = [trn("gym_hours_intro", lang)]
        for item in section[:6]:
            answer_lines.append(f"- {localize_date_text(item, lang)}")
        return "\n".join(answer_lines)

    if target == "pool":
        lines = [line.strip() for line in raw.splitlines()]
        target_idx = -1
        for idx, line in enumerate(lines):
            line_norm = normalize(line)
            if "dangola pool" in line_norm or "d angola pool" in line_norm:
                target_idx = idx
                break
        if target_idx < 0:
            return trn("pool_hours_unavailable", lang)

        schedule_lines = []
        for line in lines[target_idx + 1 : target_idx + 14]:
            if not line:
                if schedule_lines:
                    break
                continue
            if "====" in line:
                break
            line_norm = normalize(line)
            if any(
                line_norm.startswith(day)
                for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
            ):
                schedule_lines.append(line.replace("\t", " "))

        if not schedule_lines:
            return trn("pool_hours_unavailable", lang)

        answer_lines = [trn("pool_hours_intro", lang)]
        for item in schedule_lines:
            answer_lines.append(f"- {localize_date_text(item, lang)}")
        return "\n".join(answer_lines)

    return trn("hours_target_prompt", lang)

def degree_exists_in_records(subject_tokens: set[str], level: str) -> bool:
    if not subject_tokens:
        return False

    for doc in fallback_rag_docs:
        source = str(doc.get("source", "")).lower()
        doc_type = str(doc.get("type", "")).lower()
        if doc_type != "program" and "program" not in source and "degree" not in source and "computer science" not in source:
            continue

        text_norm = normalize(doc.get("text", ""))
        tokens = tokenize(doc.get("text", ""))

        if len(subject_tokens & tokens) == 0:
            continue

        if level == "graduate/master":
            if any(term in text_norm for term in ("master of", "masters", "graduate", "mba", "m s", "m a")):
                return True
            continue

        if level == "undergraduate":
            if any(term in text_norm for term in ("bachelor", "undergraduate", "major", "b s", "b a")):
                return True
            continue

        if any(term in text_norm for term in ("degree", "program", "major", "master", "bachelor", "graduate", "undergraduate")):
            return True

    return False

def get_time_response(prompt: str):
    if "time" not in prompt.lower():
        return None
    now = datetime.now(NYC_TIMEZONE)
    return f"The current time in New York is {now.strftime('%I:%M %p')}."

campus_locations = load_campus_locations()
campus_location_by_id = {place["id"]: place for place in campus_locations}
fallback_rag_docs = load_fallback_rag_docs()
program_catalog = load_program_catalog()
FAQ_INTENT_KEYWORDS = load_faq_intent_keywords()

# OLLAMA

def build_ollama_timeout():
    return httpx.Timeout(
        connect=OLLAMA_CONNECT_TIMEOUT_SECONDS,
        read=OLLAMA_READ_TIMEOUT_SECONDS,
        write=OLLAMA_CONNECT_TIMEOUT_SECONDS,
        pool=OLLAMA_CONNECT_TIMEOUT_SECONDS,
    )

async def query_ollama(prompt: str, lang: str):
    lang_name = LANGUAGE_NAMES.get(lang, "English")
    return await call_ollama(
        [
            {
                "role": "system",
                "content": (
                    "You are the Kean Global concierge assistant for the Kean University website. "
                    f"Respond only in {lang_name}. "
                    "Use simple, easy-to-understand language. "
                    "Be concise, factual, polite, and friendly. "
                    "Default to English unless the user asks in another language. "
                    "Do not invent policy or calendar facts. "
                    "Do not cite sources or filenames. "
                    "Keep the response short and easy to read (max 6 sentences)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

def retrieve_rag_context(question: str, max_results: int = RAG_MAX_RESULTS) -> list[str]:
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

def retrieve_fallback_context(
    question: str,
    max_results: int = RAG_FALLBACK_MAX_RESULTS,
    faq_topic: Optional[str] = None,
) -> list[str]:
    if not fallback_rag_docs:
        return []

    query_tokens = meaningful_tokens(question)
    if not query_tokens:
        query_tokens = tokenize(question)
    if not query_tokens:
        return []

    scored = []
    for doc in fallback_rag_docs:
        doc_tokens = doc.get("key_tokens") or doc["tokens"]
        overlap = len(query_tokens & doc_tokens)
        if overlap == 0:
            continue
        score = overlap / max(1, len(query_tokens))

        if faq_topic == "programs":
            source = str(doc.get("source", "")).lower()
            doc_type = str(doc.get("type", "")).lower()
            if doc_type == "program" or any(keyword in source for keyword in ("program", "major", "majors", "minor", "minors", "degree", "degrees", "online")):
                score *= 2.0
            else:
                score *= 0.2

        if faq_topic == "admissions":
            source = str(doc.get("source", "")).lower()
            if "admissions" in source or "applying" in source:
                score *= 2.5

        if faq_topic == "hours":
            source = str(doc.get("source", "")).lower()
            if "hours of operation" in source:
                score *= 2.5

        if faq_topic == "policies":
            source = str(doc.get("source", "")).lower()
            if "course repeats and retakes" in source or "graduation information" in source:
                score *= 2.2

        if faq_topic == "parking_transport":
            source = str(doc.get("source", "")).lower()
            if "parking" in source:
                score *= 2.0
            if "shuttle" in source or "transport" in source or "trolley" in source:
                score *= 2.4

        if faq_topic == "housing":
            source = str(doc.get("source", "")).lower()
            if "housing" in source or "dorm" in source or "residential" in source:
                score *= 2.4

        if faq_topic == "health_services":
            source = str(doc.get("source", "")).lower()
            if "health" in source or "wellness" in source:
                score *= 2.4

        if faq_topic == "accessibility":
            source = str(doc.get("source", "")).lower()
            if "accessibility" in source:
                score *= 2.5

        if faq_topic == "bookstore":
            source = str(doc.get("source", "")).lower()
            if "bookstore" in source or "barnes" in source:
                score *= 2.5

        if faq_topic == "theaters_events":
            source = str(doc.get("source", "")).lower()
            if any(keyword in source for keyword in ("stage", "theater", "theatre", "ticket", "wilkins", "enlow", "bauer")):
                score *= 2.6

        if faq_topic == "student_accounts":
            source = str(doc.get("source", "")).lower()
            if "tuition" in source or "bursar" in source:
                score *= 2.5

        if faq_topic == "smoking_policy":
            source = str(doc.get("source", "")).lower()
            if "smoking" in source:
                score *= 2.5

        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:max_results]
    context_blocks = []
    for index, (score, doc) in enumerate(top):
        context_blocks.append(
            f"[{index + 1}] source={doc['source']} type={doc['type']} score={score:.3f}\n{doc['text']}"
        )
    return context_blocks

def build_retrieval_query(user_text: str, lang: str, faq_topic: Optional[str] = None) -> str:
    parts = [user_text.strip()]
    topic_hint = FAQ_TOPIC_RETRIEVAL_HINTS.get(faq_topic or "")
    if topic_hint:
        parts.append(topic_hint)

    # Most campus source text is English, so augment multilingual queries with English topic hints.
    if lang != "en" and topic_hint:
        parts.append(topic_hint)

    return " ".join(part for part in parts if part).strip()

def best_context_overlap(question: str, context_blocks: list[str]) -> int:
    question_tokens = meaningful_tokens(question)
    if not question_tokens:
        question_tokens = tokenize(question)

    best_score = 0
    for block in context_blocks:
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("[") or line.startswith("source="):
                continue
            line_tokens = tokenize(line)
            best_score = max(best_score, len(question_tokens & line_tokens))
    return best_score

def should_ask_for_clarification(question: str, context_blocks: list[str], faq_topic: Optional[str]) -> bool:
    if not context_blocks:
        return True

    query_tokens = meaningful_tokens(question)
    if not query_tokens:
        query_tokens = tokenize(question)

    if not faq_topic and len(query_tokens) <= 2:
        return True

    overlap = best_context_overlap(question, context_blocks)
    if faq_topic:
        return overlap == 0
    return overlap < 2

def _trim_context_block(block: str, max_chars: int) -> str:
    lines = block.splitlines()
    if not lines:
        return block

    header = lines[0]
    body = " ".join(line.strip() for line in lines[1:] if line.strip())
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0].strip() + " ..."
    return f"{header}\n{body}" if body else header

def build_rag_prompt(user_text: str, context_blocks: list[str], faq_topic: Optional[str] = None) -> str:
    if not context_blocks:
        return user_text

    trimmed_blocks = [_trim_context_block(block, RAG_MAX_CHARS_PER_BLOCK) for block in context_blocks]
    joined_context = ""
    for block in trimmed_blocks:
        candidate = f"{joined_context}\n\n{block}".strip() if joined_context else block
        if len(candidate) > RAG_MAX_PROMPT_CONTEXT_CHARS:
            break
        joined_context = candidate

    intent_line = f"Detected FAQ topic: {faq_topic}.\n" if faq_topic else ""
    return (
        "You are the Kean Global concierge assistant for the Kean University website.\n"
        "Use the provided campus context first.\n"
        "Use simple, easy-to-understand language.\n"
        "Be polite, friendly, clear, and easy to read.\n"
        "Default to English unless the user asks in another language.\n"
        "Do not mention sources or filenames in the answer.\n"
        "If the information is not available in the context, apologize briefly and direct the user to https://www.kean.edu.\n\n"
        f"{intent_line}"
        f"Context:\n{joined_context}\n\n"
        f"User question: {user_text}"
    )

def build_fallback_answer(question: str, context_blocks: list[str], lang: str) -> str:
    if not context_blocks:
        return trn("faq_no_exact_match", lang)

    question_tokens = meaningful_tokens(question)
    if not question_tokens:
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
        return best_line

    first_block_lines = context_blocks[0].splitlines()
    first_content_line = ""
    for raw in first_block_lines[1:]:
        line = raw.strip()
        if line:
            first_content_line = line
            break
    if not first_content_line:
        first_content_line = context_blocks[0][:280]
    return first_content_line[:280]

def _clean_fast_path_line(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"^Title:\s*Unknown\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Description:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def _truncate_snippet(text: str, max_chars: int = 180) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for marker in (". ", "; ", ", "):
        idx = cut.rfind(marker)
        if idx >= int(max_chars * 0.6):
            return cut[: idx + 1].strip()
    return cut.rsplit(" ", 1)[0].strip() + " ..."

def build_fast_path_answer(
    question: str,
    context_blocks: list[str],
    lang: str,
    max_lines: int = FAQ_FAST_PATH_MAX_LINES,
    faq_topic: Optional[str] = None,
) -> Optional[str]:
    if not context_blocks:
        return None

    query_tokens = meaningful_tokens(question)
    if not query_tokens:
        query_tokens = tokenize(question)
    strict_tokens = {t for t in query_tokens if len(t) >= 6 and t not in {"degree", "program", "policy", "campus"}}
    q_norm = normalize(question)
    asks_master_programs = faq_topic == "programs" and any(
        keyword_in_text(q_norm, tokenize(question), k)
        for k in ("master", "graduate", "postgraduate", "ma", "ms", "mba")
    )
    asks_undergrad_programs = faq_topic == "programs" and any(
        keyword_in_text(q_norm, tokenize(question), k)
        for k in ("undergraduate", "bachelor", "bs", "ba")
    )
    program_subject_tokens = program_subject_tokens_from_query(question)
    candidates = []

    for block in context_blocks:
        lines = block.splitlines()
        source = "unknown"
        if lines:
            match = re.search(r"source=(.+?)\s+type=", lines[0])
            if match:
                source = match.group(1)
        for raw_line in lines[1:]:
            line = _clean_fast_path_line(raw_line)
            if not line:
                continue
            if len(line) < 24:
                continue
            if line.lower().startswith("source="):
                continue
            line_tokens = tokenize(line)
            overlap = len(query_tokens & line_tokens)
            strict_overlap = len(strict_tokens & line_tokens) if strict_tokens else 0
            if overlap == 0:
                continue
            if overlap < 2 and strict_overlap == 0 and not (faq_topic == "programs" and asks_master_programs and not program_subject_tokens):
                continue
            if faq_topic == "programs":
                line_norm = normalize(line)
                if asks_master_programs and not any(
                    token in line_norm
                    for token in ("master of", "masters", "mba", "m s", "m a", "graduate")
                ):
                    continue
                if asks_undergrad_programs and not any(
                    token in line_norm
                    for token in ("bachelor", "undergraduate", "b s", "b a")
                ):
                    continue
                if program_subject_tokens and len(program_subject_tokens & line_tokens) == 0:
                    continue
            candidates.append((overlap, len(line), line, source))

    if not candidates:
        # If lexical overlap is weak (e.g., multilingual query vs English docs),
        # still return top informative lines from highest-ranked context blocks.
        for block in context_blocks:
            lines = block.splitlines()
            source = "unknown"
            if lines:
                match = re.search(r"source=(.+?)\s+type=", lines[0])
                if match:
                    source = match.group(1)
            for raw_line in lines[1:]:
                line = _clean_fast_path_line(raw_line)
                if not line or len(line) < 24:
                    continue
                if line.lower().startswith("source="):
                    continue
                line_tokens = tokenize(line)
                strict_overlap = len(strict_tokens & line_tokens) if strict_tokens else 0
                if strict_tokens and strict_overlap == 0 and not (faq_topic == "programs" and asks_master_programs and not program_subject_tokens):
                    continue
                if faq_topic == "programs":
                    line_norm = normalize(line)
                    if asks_master_programs and not any(
                        token in line_norm
                        for token in ("master of", "masters", "mba", "m s", "m a", "graduate")
                    ):
                        continue
                    if asks_undergrad_programs and not any(
                        token in line_norm
                        for token in ("bachelor", "undergraduate", "b s", "b a")
                    ):
                        continue
                    if program_subject_tokens and len(program_subject_tokens & line_tokens) == 0:
                        continue
                candidates.append((0, len(line), line, source))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = []
    seen_lines = set()

    def too_similar(existing: str, candidate: str) -> bool:
        a = tokenize(existing)
        b = tokenize(candidate)
        if not a or not b:
            return False
        overlap = len(a & b)
        ratio = overlap / max(1, min(len(a), len(b)))
        return ratio >= 0.8

    for overlap, _, line, source in candidates:
        key = normalize(line)
        if key in seen_lines:
            continue
        if any(too_similar(prev_line, line) for prev_line, _ in selected):
            continue
        seen_lines.add(key)
        selected.append((line, source))
        if len(selected) >= max_lines:
            break

    if not selected:
        return None

    body_lines = [trn("fast_path_prefix", lang)]
    for idx, (line, source) in enumerate(selected, start=1):
        snippet = _truncate_snippet(line, max_chars=180)
        body_lines.append(f"{idx}. {snippet}")

    return "\n".join(body_lines)

# ROUTES

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    user_text = req.message.strip()
    lang = detect_language(user_text)
    faq_topic = detect_faq_intent(user_text)
    destination_id = find_location_destination_id(user_text)
    has_location_intent = is_location_question(user_text)

    async def localized(answer: str) -> str:
        return await localize_answer_text(answer, lang)

    if is_help_capabilities_question(user_text):
        return {
            "answer": trn("capabilities_reply", lang),
            "intent": "general",
            "response_mode": "capabilities",
        }

    if is_identity_question(user_text):
        return {
            "answer": trn("bot_identity_intro", lang),
            "intent": "general",
            "response_mode": "identity",
        }

    if is_greeting(user_text):
        return {
            "answer": trn("greeting_intro", lang),
            "intent": "general",
            "response_mode": "greeting",
        }

    if is_thanks(user_text):
        return {
            "answer": trn("thanks_reply", lang),
            "intent": "general",
            "response_mode": "thanks",
        }

    if is_farewell(user_text):
        return {
            "answer": trn("farewell_reply", lang),
            "intent": "general",
            "response_mode": "farewell",
        }

    if is_clarification_request(user_text):
        return {
            "answer": trn("clarify_reply", lang),
            "intent": "general",
            "response_mode": "clarify",
        }

    if is_acknowledgment(user_text):
        return {
            "answer": trn("acknowledgment_reply", lang),
            "intent": "general",
            "response_mode": "acknowledgment",
        }

    if is_frustration(user_text):
        return {
            "answer": trn("frustration_reply", lang),
            "intent": "general",
            "response_mode": "frustration",
        }

    if is_shuttle_question(user_text):
        return {
            "answer": await localized(build_shuttle_answer(lang)),
            "intent": "faq",
            "faq_topic": "parking_transport",
            "response_mode": "shuttle_direct",
        }

    if is_hours_question(user_text):
        hours_target = detect_hours_target(user_text)
        if hours_target:
            return {
                "answer": await localized(build_target_hours_answer(hours_target, lang)),
                "intent": "faq",
                "faq_topic": "hours",
                "response_mode": f"hours_{hours_target}",
            }
        return {
            "answer": trn("hours_target_prompt", lang),
            "intent": "faq",
            "faq_topic": "hours",
            "response_mode": "hours_target_prompt",
        }

    if is_course_repeat_question(user_text):
        return {
            "answer": await localized(build_course_repeat_answer(lang)),
            "intent": "faq",
            "faq_topic": "policies",
            "response_mode": "course_repeat_policy",
        }

    if is_admissions_question(user_text):
        return {
            "answer": await localized(build_admissions_answer(lang)),
            "intent": "faq",
            "faq_topic": "admissions",
            "response_mode": "admissions_direct",
        }

    if is_graduation_question(user_text):
        return {
            "answer": await localized(build_graduation_answer(lang)),
            "intent": "faq",
            "faq_topic": "policies",
            "response_mode": "graduation_direct",
        }

    if is_financial_aid_question(user_text):
        return {
            "answer": await localized(build_financial_aid_answer(lang)),
            "intent": "faq",
            "faq_topic": "financial_aid",
            "response_mode": "financial_aid_direct",
        }

    if is_student_accounts_question(user_text):
        return {
            "answer": await localized(build_student_accounts_answer(lang)),
            "intent": "faq",
            "faq_topic": "student_accounts",
            "response_mode": "student_accounts_direct",
        }

    if is_housing_question(user_text):
        return {
            "answer": await localized(build_housing_answer(lang)),
            "intent": "faq",
            "faq_topic": "housing",
            "response_mode": "housing_direct",
        }

    if is_health_services_question(user_text):
        return {
            "answer": await localized(build_health_services_answer(lang)),
            "intent": "faq",
            "faq_topic": "health_services",
            "response_mode": "health_services_direct",
        }

    if is_accessibility_question(user_text):
        return {
            "answer": await localized(build_accessibility_answer(lang)),
            "intent": "faq",
            "faq_topic": "accessibility",
            "response_mode": "accessibility_direct",
        }

    if is_bookstore_question(user_text):
        return {
            "answer": await localized(build_bookstore_answer(lang)),
            "intent": "faq",
            "faq_topic": "bookstore",
            "response_mode": "bookstore_direct",
        }

    if is_registrar_question(user_text):
        return {
            "answer": await localized(build_registrar_answer(lang)),
            "intent": "faq",
            "faq_topic": "registration",
            "response_mode": "registrar_direct",
        }

    if is_one_stop_question(user_text):
        return {
            "answer": await localized(build_one_stop_answer(lang)),
            "intent": "faq",
            "faq_topic": "registration",
            "response_mode": "one_stop_direct",
        }

    if is_dining_question(user_text):
        return {
            "answer": await localized(build_dining_answer(lang)),
            "intent": "faq",
            "faq_topic": "hours",
            "response_mode": "dining_direct",
        }

    if is_smoking_policy_question(user_text):
        return {
            "answer": await localized(build_smoking_policy_answer(lang)),
            "intent": "faq",
            "faq_topic": "smoking_policy",
            "response_mode": "smoking_policy_direct",
        }

    if is_program_follow_up_question(user_text):
        current_subject = extract_degree_subject_phrase(user_text) or extract_follow_up_subject_phrase(user_text)
        current_level = detect_degree_level(user_text)
        follow_up_subject = (current_subject or conversation_state.get("last_degree_subject") or "").strip()
        follow_up_level = current_level if current_subject else (conversation_state.get("last_degree_level") or current_level)

        if follow_up_subject:
            follow_up_query = f"{follow_up_subject} {follow_up_level or ''} degree program details".strip()
            context_blocks = retrieve_fallback_context(follow_up_query, faq_topic="programs")
            follow_up_answer = build_fast_path_answer(
                follow_up_query,
                context_blocks,
                lang,
                max_lines=3,
                faq_topic="programs",
            )
            if follow_up_answer:
                conversation_state["last_degree_subject"] = follow_up_subject
                conversation_state["last_degree_level"] = follow_up_level
                return {
                    "answer": await localized(follow_up_answer),
                    "intent": "faq",
                    "faq_topic": "programs",
                    "sources_used": len(context_blocks),
                    "response_mode": "program_follow_up",
                }
        return {
            "answer": trn("program_follow_up_prompt", lang),
            "intent": "faq",
            "faq_topic": "programs",
            "response_mode": "program_follow_up_prompt",
        }

    if is_food_question(user_text):
        food_suggestions = find_food_suggestions(user_text, max_results=3)
        if food_suggestions:
            primary = food_suggestions[0]
            suggestion_lines = [trn("food_suggestions_intro", lang)]
            for index, place in enumerate(food_suggestions, start=1):
                suggestion_lines.append(f"{index}. {place.get('name', 'Food location')} ({place.get('campus', 'Main')})")
            return {
                "answer": "\n".join(suggestion_lines),
                "intent": "location",
                "destination_id": primary.get("id"),
                "food_suggestions": [
                    {
                        "id": place.get("id"),
                        "name": place.get("name"),
                        "campus": place.get("campus", "Main"),
                    }
                    for place in food_suggestions
                ],
                "use_current_location": False,
                "location_mode": "highlight",
            }

    if is_closest_parking_question(user_text):
        reference_id = find_location_destination_id(
            user_text,
            allowed_types={"building", "entrance"},
        )
        if not reference_id:
            return {"answer": trn("closest_parking_unknown_target", lang), "intent": "general"}

        reference = get_effective_reference_location(reference_id)
        closest_lot = find_closest_parking_lot(reference_id)
        if not reference or not closest_lot:
            return {
                "answer": trn("closest_parking_not_found", lang, target=(reference or {}).get("name", "that location")),
                "intent": "general",
            }

        return {
            "answer": trn("closest_parking_found", lang, target=reference["name"], lot=closest_lot["name"]),
            "intent": "location",
            "destination_id": closest_lot["id"],
            "use_current_location": False,
            "location_mode": "highlight",
        }

    if is_parking_ticket_fee_question(user_text):
        return {
            "answer": await localized(build_parking_ticket_fee_answer(lang)),
            "intent": "faq",
            "faq_topic": "parking_transport",
            "response_mode": "policy_fee_summary",
        }

    if is_parking_location_question(user_text):
        audience = parking_audience(user_text)
        key = {
            "student": "parking_guidance_student",
            "faculty": "parking_guidance_faculty",
            "overnight": "parking_guidance_overnight",
        }.get(audience, "parking_guidance_general")
        return {
            "answer": trn(key, lang),
            "intent": "location",
            "destination_id": None,
            "use_current_location": False,
            "location_mode": "highlight",
        }

    if has_location_intent or (destination_id and should_route_destination_without_location_keyword(user_text)):
        use_current_location = should_use_current_location(user_text)
        location_mode = "directions" if use_current_location else "highlight"
        if destination_id:
            destination_id = normalize_location_destination_for_response(destination_id, user_text)
            destination = campus_location_by_id.get(destination_id, {})
            return {
                "answer": trn(
                    "location_opening_specific",
                    lang,
                    name=destination.get("name", "That location"),
                    campus=destination.get("campus", "campus"),
                ),
                "intent": "location",
                "destination_id": destination_id,
                "use_current_location": use_current_location,
                "location_mode": location_mode,
            }
        return {
            "answer": trn("location_opening_generic", lang),
            "intent": "location",
            "destination_id": None,
            "use_current_location": use_current_location,
            "location_mode": location_mode,
        }

    if is_calendar_question(user_text):
        term = extract_term_from_text(user_text)
        session = extract_session_from_text(user_text)
        if term:
            matched = next((t for t in calendar_data if term in t), None)
            category = detect_event_category(user_text)
            if matched and category:
                best_event = find_best_calendar_event(calendar_data[matched], category, session=session)
                if best_event:
                    event, date = best_event
                    return {
                        "answer": f"{localize_calendar_event_text(event, lang)}: {localize_date_text(date, lang)}",
                        "intent": "calendar",
                    }

    if is_degree_availability_question(user_text):
        subject_phrase = extract_degree_subject_phrase(user_text)
        subject_tokens = program_subject_tokens_from_query(subject_phrase or user_text)
        level = detect_degree_level(user_text)
        localized_level = localize_degree_level(level, lang)
        subject_label = (subject_phrase or "that subject").strip()
        matched_program = find_program_match(subject_phrase or user_text)
        exists = bool(matched_program) or degree_exists_in_records(subject_tokens, level)
        conversation_state["last_degree_subject"] = subject_label
        conversation_state["last_degree_level"] = level
        return {
            "answer": await localized(
                build_degree_availability_answer(
                    exists,
                    matched_program["name"] if matched_program else subject_label,
                    localized_level,
                    lang,
                    matched_program,
                )
            ),
            "intent": "faq",
            "faq_topic": "programs",
            "response_mode": "degree_availability",
        }

    program_match = find_program_match(user_text)
    if program_match and (
        faq_topic == "programs"
        or any(token in normalize(user_text) for token in ("major", "program", "degree", "curriculum"))
        or len(tokenize(user_text)) <= 5
    ):
        conversation_state["last_degree_subject"] = program_match["name"]
        return {
            "answer": await localized(build_program_answer(program_match, lang)),
            "intent": "faq",
            "faq_topic": "programs",
            "response_mode": "program_catalog_direct",
        }

    retrieval_query = build_retrieval_query(user_text, lang, faq_topic)
    context_blocks = retrieve_rag_context(retrieval_query)
    if not context_blocks:
        fallback_query = build_retrieval_query(
            f"{user_text} {faq_topic.replace('_', ' ') if faq_topic else ''}".strip(),
            lang,
            faq_topic=faq_topic,
        )
        context_blocks = retrieve_fallback_context(fallback_query, faq_topic=faq_topic)

    if should_ask_for_clarification(user_text, context_blocks, faq_topic):
        return {
            "answer": trn("faq_clarify_reply", lang),
            "intent": "faq" if faq_topic else "general",
            "faq_topic": faq_topic,
            "sources_used": len(context_blocks),
            "response_mode": "clarify_no_match",
        }

    if FAQ_FAST_PATH_ENABLED and faq_topic and context_blocks:
        fast_answer = build_fast_path_answer(user_text, context_blocks, lang, faq_topic=faq_topic)
        if fast_answer:
            return {
                "answer": await localized(fast_answer),
                "intent": "faq",
                "faq_topic": faq_topic,
                "sources_used": len(context_blocks),
                "response_mode": "fast_path",
            }
        return {
            "answer": await localized(trn("program_not_found", lang) if faq_topic == "programs" else trn("faq_no_exact_match", lang)),
            "intent": "faq",
            "faq_topic": faq_topic,
            "sources_used": len(context_blocks),
            "response_mode": "fast_path_no_match",
        }

    prompt = build_rag_prompt(user_text, context_blocks, faq_topic)

    try:
        reply = await query_ollama(prompt, lang)
    except (httpx.HTTPError, asyncio.TimeoutError):
        fast_answer = build_fast_path_answer(user_text, context_blocks, lang, max_lines=2, faq_topic=faq_topic)
        if fast_answer:
            return {
                "answer": await localized(fast_answer),
                "intent": "faq" if faq_topic else "general",
                "faq_topic": faq_topic,
                "sources_used": len(context_blocks),
                "response_mode": "fast_path_timeout",
            }
        fallback_answer = trn("program_not_found", lang) if faq_topic == "programs" else build_fallback_answer(user_text, context_blocks, lang)
        return {"answer": await localized(fallback_answer), "intent": "faq" if faq_topic else "general", "faq_topic": faq_topic, "sources_used": len(context_blocks)}
    except Exception:
        fast_answer = build_fast_path_answer(user_text, context_blocks, lang, max_lines=2, faq_topic=faq_topic)
        if fast_answer:
            return {
                "answer": await localized(fast_answer),
                "intent": "faq" if faq_topic else "general",
                "faq_topic": faq_topic,
                "sources_used": len(context_blocks),
                "response_mode": "fast_path_timeout",
            }
        fallback_answer = trn("program_not_found", lang) if faq_topic == "programs" else build_fallback_answer(user_text, context_blocks, lang)
        return {"answer": await localized(fallback_answer), "intent": "faq" if faq_topic else "general", "faq_topic": faq_topic, "sources_used": len(context_blocks)}

    return {"answer": await localized(reply), "intent": "faq" if faq_topic else "general", "faq_topic": faq_topic, "sources_used": len(context_blocks)}
