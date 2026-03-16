import asyncio
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import ChatRequest, chat  # noqa: E402


CASES = [
    ("how do I apply?", "admissions_direct", "faq"),
    ("library hours", "hours_library", "faq"),
    ("how does the shuttle work?", "shuttle_direct", "faq"),
    ("where is the bookstore?", "bookstore_direct", "faq"),
    ("how do I contact the registrar?", "registrar_direct", "faq"),
    ("tell me about housing", "housing_direct", "faq"),
    ("student health services", "health_services_direct", "faq"),
    ("accessibility services", "accessibility_direct", "faq"),
    ("what dining options are on campus?", "dining_direct", "faq"),
    ("what is the smoking policy?", "smoking_policy_direct", "faq"),
]


async def run_cases() -> int:
    failures = []
    for prompt, expected_mode, expected_intent in CASES:
        response = await chat(ChatRequest(message=prompt))
        actual_mode = response.get("response_mode")
        actual_intent = response.get("intent")
        answer = response.get("answer", "")
        ok = (
            actual_mode == expected_mode
            and actual_intent == expected_intent
            and isinstance(answer, str)
            and bool(answer.strip())
        )
        result = {
            "prompt": prompt,
            "expected_mode": expected_mode,
            "actual_mode": actual_mode,
            "expected_intent": expected_intent,
            "actual_intent": actual_intent,
            "ok": ok,
            "answer": answer,
        }
        print(json.dumps(result, ensure_ascii=False))
        if not ok:
            failures.append(result)

    if failures:
        print(json.dumps({"passed": False, "failures": failures}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({"passed": True, "count": len(CASES)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_cases()))
