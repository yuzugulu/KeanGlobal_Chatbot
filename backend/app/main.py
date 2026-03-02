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
    if any(p in q for p in ["begin", "start", "first day", "opening", "başla", "empieza", "开始", "شروع", "시작"]):
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

def find_best_calendar_event(events: dict[str, str], category: str) -> Optional[tuple[str, str]]:
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
    "nasıl giderim",
    "cómo llegar",
    "dónde está",
    "在哪里",
    "怎么去",
    "کہاں",
    "راستہ",
    "어디",
    "어떻게 가",
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
    "cómo llegar",
    "怎么去",
    "راستہ",
    "어떻게 가",
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

SUPPORTED_LANGS = {"en", "tr", "es", "zh", "ur", "ko"}
LANGUAGE_NAMES = {
    "en": "English",
    "tr": "Turkish",
    "es": "Spanish",
    "zh": "Mandarin Chinese",
    "ur": "Urdu",
    "ko": "Korean",
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

def detect_language(text: str) -> str:
    t = text.strip().lower()
    if re.search(r"[\u4e00-\u9fff]", t):
        return "zh"
    if re.search(r"[\uac00-\ud7af]", t):
        return "ko"
    if re.search(r"[\u0600-\u06ff]", t):
        return "ur"
    tr_chars = set("çğıöşü")
    if any(ch in tr_chars for ch in t) or any(x in t for x in (" nasıl ", " nerede", " ne zaman", "güz", "bahar")):
        return "tr"
    if any(x in t for x in ("¿", "¡", " dónde", " cuándo", " qué", "becas", "semestre")):
        return "es"
    return "en"

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

async def query_ollama(prompt: str, lang: str):
    lang_name = LANGUAGE_NAMES.get(lang, "English")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are KeanGlobal assistant. "
                    f"Respond only in {lang_name}. "
                    "Be concise, factual, and do not invent policy/calendar facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
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

def build_fallback_answer(question: str, context_blocks: list[str], lang: str) -> str:
    if not context_blocks:
        return trn("fallback_no_context", lang)

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
        return trn("fallback_best_match", lang, line=best_line)
    return trn("fallback_top_context", lang, line=context_blocks[0][:280])

# ROUTES

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    user_text = req.message.strip()
    lang = detect_language(user_text)
    faq_topic = detect_faq_intent(user_text)

    if is_location_question(user_text):
        destination_id = find_location_destination_id(user_text)
        use_current_location = should_use_current_location(user_text)
        location_mode = "directions" if use_current_location else "highlight"
        if destination_id:
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
        if term:
            matched = next((t for t in calendar_data if term in t), None)
            category = detect_event_category(user_text)
            if matched and category:
                best_event = find_best_calendar_event(calendar_data[matched], category)
                if best_event:
                    event, date = best_event
                    return {
                        "answer": f"{localize_date_text(event.title(), lang)}: {localize_date_text(date, lang)}",
                        "intent": "calendar",
                    }

    context_blocks = retrieve_rag_context(user_text)
    if not context_blocks:
        fallback_query = f"{user_text} {faq_topic.replace('_', ' ') if faq_topic else ''}".strip()
        context_blocks = retrieve_fallback_context(fallback_query)
    prompt = build_rag_prompt(user_text, context_blocks, faq_topic)

    try:
        reply = await query_ollama(prompt, lang)
    except httpx.HTTPError:
        fallback_answer = build_fallback_answer(user_text, context_blocks, lang)
        return {"answer": fallback_answer, "intent": "faq" if faq_topic else "general", "faq_topic": faq_topic, "sources_used": len(context_blocks)}
    except Exception:
        fallback_answer = build_fallback_answer(user_text, context_blocks, lang)
        return {"answer": fallback_answer, "intent": "faq" if faq_topic else "general", "faq_topic": faq_topic, "sources_used": len(context_blocks)}

    return {"answer": reply, "intent": "faq" if faq_topic else "general", "faq_topic": faq_topic, "sources_used": len(context_blocks)}
