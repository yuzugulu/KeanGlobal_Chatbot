import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------
# CONFIG
# -----------------------

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

# -----------------------
# APP SETUP
# -----------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# -----------------------
# OLLAMA FUNCTION
# -----------------------

NYC_TIMEZONE = ZoneInfo("America/New_York")
LOCAL_TIME_PATTERNS = [
    re.compile(r"\btime\b.*\b(nyc|new york|new york city|newark|new jersey|nj|est|eastern)\b", re.IGNORECASE),
    re.compile(r"\b(nyc|new york|new york city|newark|new jersey|nj|est|eastern)\b.*\btime\b", re.IGNORECASE),
]
LIBRARY_HOURS_PATTERNS = [
    re.compile(r"\b(library|nancy thompson)\b.*\b(hour|hours|open|opened|close|closing)\b", re.IGNORECASE),
    re.compile(r"\b(hour|hours|open|opened|close|closing)\b.*\b(library|nancy thompson)\b", re.IGNORECASE),
]
LOCATION_INTENT_KEYWORDS = (
    "where",
    "map",
    "location",
    "directions",
    "route",
    "how do i get",
    "how to get",
    "take me to",
    "find",
)
BUILDING_ALIASES = {
    "kean hall": "kean_hall",
    "green lane academic building": "glassman_hall",
    "glassman hall": "glassman_hall",
    "glassman": "glassman_hall",
    "glab": "glassman_hall",
    "nancy thompson library": "library",
    "library": "library",
    "stem building": "stem",
    "stem": "stem",
    "downs hall": "downs_hall",
    "downs": "downs_hall",
    "harwood arena": "harwood",
    "harwood": "harwood",
    "university center": "uc",
    "student center": "uc",
    "miron student center": "uc",
    "msc": "uc",
    "human rights institute": "library",
    "hri": "library",
    "uc": "uc",
}


def get_time_response(prompt: str):
    lowered = prompt.lower()
    if not any(token in lowered for token in ["time", "date", "day", "today"]):
        return None

    is_local_time_question = any(pattern.search(prompt) for pattern in LOCAL_TIME_PATTERNS)
    is_generic_time_question = lowered.strip() in {"what time is it", "what time is it?"}
    if not is_local_time_question and not is_generic_time_question:
        return None

    location_label = "New York City"
    if any(token in lowered for token in ["newark", "new jersey", " nj"]):
        location_label = "Newark, New Jersey"

    now_nyc = datetime.now(NYC_TIMEZONE)
    timezone_name = now_nyc.tzname() or "ET"
    return (
        f"The current time in {location_label} is {now_nyc.strftime('%I:%M %p')} "
        f"on {now_nyc.strftime('%A, %B %d, %Y')} ({timezone_name})."
    )


def get_library_hours_response(prompt: str):
    if not any(pattern.search(prompt) for pattern in LIBRARY_HOURS_PATTERNS):
        return None

    return (
        "I do not have live library opening hours in this build. "
        "Please check the Nancy Thompson Library website or front desk for today's exact times."
    )


def get_location_response(prompt: str):
    lowered = prompt.lower().strip()
    destination_id = None

    for alias, candidate_id in BUILDING_ALIASES.items():
        if alias in lowered:
            destination_id = candidate_id
            break

    is_location_intent = destination_id is not None or any(
        keyword in lowered for keyword in LOCATION_INTENT_KEYWORDS
    )
    if not is_location_intent:
        return None

    if destination_id:
        return {
            "answer": "Map opened. I set directions from your current location to your requested building.",
            "reply": "Map opened. I set directions from your current location to your requested building.",
            "intent": "location",
            "destination_id": destination_id,
            "use_current_location": True,
        }

    return {
        "answer": "Map opened. Choose a destination and I can guide you there from your current location.",
        "reply": "Map opened. Choose a destination and I can guide you there from your current location.",
        "intent": "location",
        "destination_id": None,
        "use_current_location": True,
    }


def build_ollama_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=OLLAMA_CONNECT_TIMEOUT_SECONDS,
        read=OLLAMA_READ_TIMEOUT_SECONDS,
        write=OLLAMA_CONNECT_TIMEOUT_SECONDS,
        pool=OLLAMA_CONNECT_TIMEOUT_SECONDS,
    )


async def query_ollama(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful university policy assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "num_predict": 512,
            "temperature": 0.3
        }
    }

    timeout = build_ollama_timeout()
    for attempt in range(1, OLLAMA_MAX_RETRIES + 2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(OLLAMA_URL, json=payload)

            print("STATUS:", response.status_code)
            print("RAW:", response.text[:300])

            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "No response from model.")
        except httpx.TimeoutException:
            if attempt <= OLLAMA_MAX_RETRIES:
                continue
        except httpx.HTTPStatusError as exc:
            return f"ERROR: Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        except httpx.RequestError as exc:
            return f"ERROR contacting Ollama: {str(exc)}"
        except ValueError:
            return "ERROR: Ollama returned invalid JSON"

    return (
        "ERROR: Ollama timed out. "
        f"Configured read timeout is {OLLAMA_READ_TIMEOUT_SECONDS:.0f}s at {OLLAMA_URL}. "
        "Increase OLLAMA_READ_TIMEOUT_SECONDS or verify Ollama/model is running."
    )


def get_ollama_tags_url() -> str:
    # Convert configured chat endpoint to Ollama tags endpoint for lightweight health checks.
    if OLLAMA_URL.endswith("/api/chat"):
        return OLLAMA_URL[: -len("/api/chat")] + "/api/tags"
    return "http://127.0.0.1:11434/api/tags"

# -----------------------
# ROUTES
# -----------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/ollama")
def health_ollama():
    tags_url = get_ollama_tags_url()
    try:
        response = httpx.get(tags_url, timeout=OLLAMA_HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=503, detail="Ollama health check timed out.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama is unreachable: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama returned HTTP {exc.response.status_code}.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Ollama returned invalid JSON.") from exc

    models = data.get("models", []) if isinstance(data, dict) else []
    model_names = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
    configured_model_found = MODEL_NAME in model_names or any(
        name.startswith(f"{MODEL_NAME}:") for name in model_names
    )
    return {
        "status": "ok",
        "ollama_reachable": True,
        "configured_chat_url": OLLAMA_URL,
        "tags_url": tags_url,
        "configured_model": MODEL_NAME,
        "configured_model_found": configured_model_found,
        "available_models": model_names,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    time_response = get_time_response(req.message)
    if time_response:
        return {"answer": time_response, "reply": time_response, "intent": "general"}

    library_hours_response = get_library_hours_response(req.message)
    if library_hours_response:
        return {"answer": library_hours_response, "reply": library_hours_response, "intent": "general"}

    location_response = get_location_response(req.message)
    if location_response:
        return location_response

    reply = await query_ollama(req.message)
    return {"answer": reply, "reply": reply, "intent": "general"}
