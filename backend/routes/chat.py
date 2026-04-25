"""
Chat API routes.

POST /api/chat/message  — process one user turn
POST /api/chat/reset    — wipe session state
POST /api/chat/start    — return the welcome message (no user input yet)
GET  /api/chat/session  — inspect current session (dev helper)
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Optional
import uuid

from ..services.conversation_manager import (
    process_message,
    empty_profile,
    WELCOME_MESSAGE,
    MAX_QUESTIONS,
)
from ..services.recommendation_service import generate_recommendation

router = APIRouter(prefix="/api/chat")

# ── In-memory session store ───────────────────────────────────────────────────
# Replace with Redis / DB for production.
_sessions: dict[str, dict] = {}


def _get_or_create_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "messages": [],
            "profile": empty_profile(),
            "current_step": "time_horizon",
            "is_complete": False,
            "question_count": 0,
        }
    return _sessions[session_id]


# ── Request / Response models ─────────────────────────────────────────────────

class MessageRequest(BaseModel):
    message: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


class MessageResponse(BaseModel):
    reply: str
    profile: dict[str, Any]
    current_step: str
    options: list[str]
    question_count: int
    total_questions: int
    is_complete: bool
    missing_fields: list[str]
    session_id: str


class StartResponse(BaseModel):
    welcome: str
    session_id: str
    profile: dict[str, Any]
    current_step: str
    options: list[str]
    question_count: int
    total_questions: int


class RecommendationRequest(BaseModel):
    profile: dict[str, Any]


class RecommendationResponse(BaseModel):
    recommended_bucket: str
    reasoning: str
    top_stocks: list[str]
    meme_stocks: list[str]
    standard_stocks: list[str]
    investor_tip: str
    risk_score: float
    expected_return: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/start", response_model=StartResponse)
async def start_chat(session_id: Optional[str] = None):
    """Create (or reset) a session and return the opening message."""
    sid = session_id or str(uuid.uuid4())
    _sessions[sid] = {
        "messages": [{"role": "assistant", "content": WELCOME_MESSAGE}],
        "profile": empty_profile(),
        "current_step": "time_horizon",
        "is_complete": False,
        "question_count": 0,
    }
    return StartResponse(
        welcome=WELCOME_MESSAGE,
        session_id=sid,
        profile=_sessions[sid]["profile"],
        current_step="time_horizon",
        options=["1 week", "1 month", "3 months", "6+ months"],
        question_count=0,
        total_questions=MAX_QUESTIONS,
    )


@router.post("/message", response_model=MessageResponse)
async def send_message(body: MessageRequest):
    """Process one user message and return the assistant reply + updated state."""
    session = _get_or_create_session(body.session_id)

    if not body.message.strip():
        return MessageResponse(
            reply="Please type a message so I can help you.",
            profile=session["profile"],
            current_step=session["current_step"],
            options=[],
            question_count=int(session["profile"].get("question_count", 0)),
            total_questions=MAX_QUESTIONS,
            is_complete=session["is_complete"],
            missing_fields=[
                f
                for f in [
                    "time_horizon",
                    "risk_tolerance",
                    "objective",
                    "preference",
                    "loss_comfort",
                    "diversification",
                ]
                if not session["profile"].get(f)
            ],
            session_id=body.session_id,
        )

    # Add user message to history
    session["messages"].append({"role": "user", "content": body.message})

    result = process_message(
        user_text=body.message,
        profile=session["profile"],
        history=session["messages"],
        current_step=session["current_step"],
    )

    # Persist updated state
    session["profile"] = result["profile"]
    session["current_step"] = result["current_step"]
    session["is_complete"] = result["is_complete"]
    session["messages"].append({"role": "assistant", "content": result["reply"]})

    return MessageResponse(
        reply=result["reply"],
        profile=result["profile"],
        current_step=result["current_step"],
        options=result.get("options", []),
        question_count=result.get("question_count", 0),
        total_questions=result.get("total_questions", MAX_QUESTIONS),
        is_complete=result["is_complete"],
        missing_fields=result["missing_fields"],
        session_id=body.session_id,
    )


@router.post("/reset")
async def reset_chat(body: ResetRequest):
    """Wipe session state and return a fresh welcome."""
    _sessions.pop(body.session_id, None)
    new_session = _get_or_create_session(body.session_id)
    new_session["messages"].append({"role": "assistant", "content": WELCOME_MESSAGE})
    return {"message": "Session reset", "session_id": body.session_id, "welcome": WELCOME_MESSAGE}


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Dev endpoint — inspect current session state."""
    if session_id not in _sessions:
        return {"error": "Session not found"}
    return _sessions[session_id]


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(body: RecommendationRequest):
    """Generate portfolio recommendation from profile + pipeline outputs."""
    return generate_recommendation(body.profile or {})


# ── Profile extraction endpoint (optional) ────────────────────────────────────

from fastapi import APIRouter as _Router

profile_router = _Router(prefix="/api/profile")


class ExtractRequest(BaseModel):
    text: str


@profile_router.post("/extract")
async def extract_profile(body: ExtractRequest):
    """Convert arbitrary text into a structured investor profile."""
    from ..services.profile_extractor import extract_all_fields
    return extract_all_fields(body.text)
