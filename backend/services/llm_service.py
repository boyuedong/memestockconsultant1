"""
LLM service — thin wrapper that generates natural assistant replies.

Priority:
  1. OpenAI (if OPENAI_API_KEY is set)
  2. Rule-based fallback (always available)

To plug in a different LLM later, replace the `_call_openai` function or
add a new branch in `generate_reply`.
"""

import os
import json
from urllib import request as urlrequest
from urllib import error as urlerror
from typing import Optional

# ── optional OpenAI import ────────────────────────────────────────────────────
try:
    from openai import OpenAI as _OpenAI  # openai>=1.0
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


def _call_openai(system_prompt: str, messages: list[dict]) -> str:
    client = _OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}, *messages],
        temperature=0.5,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def _openai_available() -> bool:
    return _OPENAI_AVAILABLE and bool(os.environ.get("OPENAI_API_KEY"))


def _ollama_available() -> bool:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        req = urlrequest.Request(f"{base_url}/api/tags", method="GET")
        with urlrequest.urlopen(req, timeout=1.0) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _call_ollama(system_prompt: str, messages: list[dict]) -> str:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_CHAT_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct"))
    timeout_sec = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "30"))

    sanitized_messages = []
    for msg in messages[-12:]:
        role = msg.get("role")
        if role not in {"user", "assistant", "system"}:
            continue
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        sanitized_messages.append({"role": role, "content": content})

    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "system", "content": system_prompt}, *sanitized_messages],
        "options": {"temperature": 0.4},
    }
    req = urlrequest.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urlerror.URLError as e:
        raise RuntimeError(f"Ollama chat request failed: {e}") from e

    content = ((body.get("message") or {}).get("content") or "").strip()
    if not content:
        raise ValueError("Empty Ollama chat response")
    return content


# ── fallback rule-based replies ───────────────────────────────────────────────

_FALLBACK_STEP_REPLIES: dict[str, str] = {
    "time_horizon": (
        "Got it! Now, how would you describe your risk tolerance? "
        "Are you comfortable with high risk for bigger returns, prefer low risk and stability, "
        "or somewhere in the middle?"
    ),
    "risk_tolerance": (
        "Understood. What's your main investment objective — "
        "maximum returns, lower volatility and safety, or a balanced mix?"
    ),
    "objective": (
        "Almost there! Do you have any sector preferences — like tech, healthcare, energy, "
        "or finance? Or are you happy with a broad diversified mix? (You can say 'no preference')"
    ),
    "sector_preferences": None,  # handled by summary
    "complete": None,
}

_CLARIFY_REPLIES: dict[str, str] = {
    "time_horizon": (
        "I didn't quite catch that. Could you tell me your investment time horizon? "
        "For example: 1 month, 3 months, 6 months, or 1 year?"
    ),
    "risk_tolerance": (
        "Could you clarify your risk preference? For example: high risk, "
        "moderate/medium, or low/conservative?"
    ),
    "objective": (
        "What's your primary goal — growth and max returns, stability and safety, "
        "or a balanced mix of both?"
    ),
}


def generate_reply(
    step_just_completed: Optional[str],
    next_step: Optional[str],
    profile: dict,
    clarifying: bool,
    conversation_history: list[dict],
    current_step: Optional[str] = None,
) -> str:
    """
    Generate the next assistant message.
    `step_just_completed` — the field that was just successfully extracted (or None).
    `next_step`           — the next field we need (or None if complete).
    `clarifying`          — True if we need to re-ask the same step.
    """
    if clarifying:
        clarify_step = current_step or next_step or step_just_completed
        fallback = _CLARIFY_REPLIES.get(clarify_step or "", "Could you say that differently?")
    elif next_step is None:
        # Build summary
        fallback = _build_summary(profile)
    else:
        fallback = _FALLBACK_STEP_REPLIES.get(next_step, "Could you tell me more?")

    provider = os.environ.get("LLM_CHAT_PROVIDER", "auto").strip().lower()
    if provider == "fallback":
        return fallback

    system = (
        "You are a concise, friendly financial assistant helping a user build an investor profile. "
        "Ask one focused question at a time. Keep replies under 40 words. "
        "Do not use bullet points or headers. Sound warm but professional."
    )

    candidates = []
    if provider == "openai":
        if _openai_available():
            candidates = [_call_openai]
    elif provider == "ollama":
        if _ollama_available():
            candidates = [_call_ollama]
    else:
        if _openai_available():
            candidates.append(_call_openai)
        if _ollama_available():
            candidates.append(_call_ollama)

    for caller in candidates:
        try:
            return caller(system, conversation_history)
        except Exception:
            continue
    return fallback


def _build_summary(profile: dict) -> str:
    parts = []
    th = profile.get("time_horizon") or "unspecified"
    rt = profile.get("risk_tolerance") or "unspecified"
    obj = profile.get("objective") or profile.get("priority") or "unspecified"
    sectors = profile.get("sector_preferences") or []

    parts.append(f"a {th} investment")
    parts.append(f"{rt} risk")
    parts.append(f"{obj} objective")
    if sectors:
        parts.append(f"focus on {', '.join(sectors)}")

    summary = "Got it — you're looking for " + ", ".join(parts) + ". "
    summary += "I've generated your portfolio recommendation!"
    return summary
