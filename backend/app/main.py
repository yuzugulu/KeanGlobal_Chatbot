from typing import Literal, Optional
import os
import re
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"

app = FastAPI(title="KeanGlobal Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROGRAMS_FILE = DATA_DIR / "program_info.json"
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)


class ChatResponse(BaseModel):
    answer: str
    intent: Literal["location", "policy"]
    destination_id: Optional[str] = None
    use_current_location: bool = False


BUILDING_ALIASES = {
    "kean hall": "kean_hall",
    "glassman hall": "glassman_hall",
    "glab": "glassman_hall",
    "library": "library",
    "nancy thompson library": "library",
    "stem": "stem",
    "stem building": "stem",
    "downs hall": "downs_hall",
    "harwood": "harwood",
    "harwood arena": "harwood",
    "university center": "uc",
    "student center": "uc",
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

@app.get("/api/programs")
def get_programs():
    """
    Returns the parsed JSON data for all Kean programs.
    """
    if not PROGRAMS_FILE.exists():
        raise HTTPException(status_code=404, detail="Program data file not found on the server.")
        
    try:
        with open(PROGRAMS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load program data: {str(e)}")
    
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    user_text = payload.message.strip()

    # Temporary intent routing for map visibility until dedicated intent model is added.
    lowered = user_text.lower()
    intent: Literal["location", "policy"] = (
        "location"
        if any(keyword in lowered for keyword in ["where", "location", "building", "map"])
        else "policy"
    )

    destination_id = None
    for alias, candidate_id in BUILDING_ALIASES.items():
        if alias in lowered:
            destination_id = candidate_id
            intent = "location"
            break

    if intent == "location":
        if destination_id:
            return ChatResponse(
                answer=(
                    "Map updated. I will use your current location as start and draw a campus route "
                    "to the requested building."
                ),
                intent="location",
                destination_id=destination_id,
                use_current_location=True,
            )
        return ChatResponse(
            answer="Map opened. Please choose a destination building to create a route from your location.",
            intent="location",
            use_current_location=True,
        )

    request_body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are KeanGlobal, a concise campus concierge for Kean University. "
                    "Answer clearly and use plain language for students."
                ),
            },
            {"role": "user", "content": user_text},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_URL, json=request_body)
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not reachable on localhost:11434. "
                "Start Ollama and run `ollama run mistral` first."
            ),
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

    answer = data.get("message", {}).get("content", "")
    if not answer:
        raise HTTPException(status_code=502, detail="Ollama returned an empty response.")

    return ChatResponse(answer=answer.strip(), intent=intent)
