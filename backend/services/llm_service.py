"""
LLM service — thin wrapper that generates natural assistant replies.

Priority:
  1. OpenAI (if OPENAI_API_KEY is set)
  2. Rule-based fallback (always available)

To plug in a different LLM later, replace the `_call_openai` function or
add a new branch in `generate_reply`.
"""

import os
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


def llm_available() -> bool:
    return _OPENAI_AVAILABLE and bool(os.environ.get("OPENAI_API_KEY"))


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
) -> str:
    """
    Generate the next assistant message.
    `step_just_completed` — the field that was just successfully extracted (or None).
    `next_step`           — the next field we need (or None if complete).
    `clarifying`          — True if we need to re-ask the same step.
    """
    if clarifying and step_just_completed:
        fallback = _CLARIFY_REPLIES.get(step_just_completed, "Could you say that differently?")
    elif next_step is None:
        # Build summary
        fallback = _build_summary(profile)
    else:
        fallback = _FALLBACK_STEP_REPLIES.get(next_step, "Could you tell me more?")

    if not llm_available():
        return fallback

    # ── OpenAI path ────────────────────────────────────────────────────────────
    system = (
        "You are a concise, friendly financial assistant helping a user build an investor profile. "
        "Ask one focused question at a time. Keep replies under 40 words. "
        "Do not use bullet points or headers. Sound warm but professional."
    )
    try:
        return _call_openai(system, conversation_history)
    except Exception:
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
