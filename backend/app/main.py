from typing import Literal, Optional

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
