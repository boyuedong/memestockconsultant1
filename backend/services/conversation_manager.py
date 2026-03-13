"""
Stateless conversation step manager.

All session state is kept in the caller (the route handler) and passed in.
This makes the logic easy to test and easy to swap backends.

Conversation steps (in order):
  1. time_horizon
  2. risk_tolerance
  3. objective
  4. sector_preferences   (optional — skipped if user says no preference)
  → complete
"""

from typing import Any, Optional
from .profile_extractor import (
    extract_time_horizon,
    extract_risk_tolerance,
    extract_objective,
    extract_priority,
    extract_sectors,
)
from .llm_service import generate_reply

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ["time_horizon", "risk_tolerance", "objective"]
OPTIONAL_FIELDS = ["sector_preferences", "priority"]
WELCOME_MESSAGE = (
    "Hi! I can help recommend whether social buzz stocks or the Magnificent 7 "
    "better fit your goals.\n\n"
    "First, what's your investment time horizon? "
    "For example: 1 month, 3 months, 6 months, or 1 year?"
)

_STEP_ORDER = ["time_horizon", "risk_tolerance", "objective", "sector_preferences"]

_EXTRACTORS = {
    "time_horizon": extract_time_horizon,
    "risk_tolerance": extract_risk_tolerance,
    "objective": extract_objective,
    "sector_preferences": extract_sectors,
}


def empty_profile() -> dict:
    return {
        "time_horizon": None,
        "risk_tolerance": None,
        "objective": None,
        "sector_preferences": [],
        "priority": None,
        "extra_notes": "",
    }


def _missing_required(profile: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if not profile.get(f)]


def _is_complete(profile: dict) -> bool:
    return len(_missing_required(profile)) == 0


def _next_unfilled_step(profile: dict) -> Optional[str]:
    """Return the next step that still needs answering, or None if done."""
    for step in _STEP_ORDER:
        if step == "sector_preferences":
            # Optional — only ask if not already answered
            if profile.get("sector_preferences") is None:
                return step
            continue
        if not profile.get(step):
            return step
    return None


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_message(
    user_text: str,
    profile: dict,
    history: list[dict],
    current_step: str,
) -> dict[str, Any]:
    """
    Process a single user message and return the updated state.

    Returns:
      {
        "reply": str,
        "profile": dict,
        "current_step": str,
        "is_complete": bool,
        "missing_fields": list[str],
      }
    """
    updated_profile = dict(profile)

    # ── 1. Try to extract a value for the current step ────────────────────────
    extractor = _EXTRACTORS.get(current_step)
    extracted_value = extractor(user_text) if extractor else None

    # Also opportunistically extract other fields from the message
    _opportunistic_extract(user_text, updated_profile)

    step_filled = False
    if extracted_value is not None:
        if current_step == "sector_preferences":
            updated_profile["sector_preferences"] = extracted_value  # may be []
        else:
            updated_profile[current_step] = extracted_value
        step_filled = True
    elif current_step == "sector_preferences":
        # Treat any reply as "handled" — it's optional
        updated_profile["sector_preferences"] = []
        step_filled = True

    # ── 2. Derive priority from objective if not already set ──────────────────
    _derive_priority(updated_profile)

    # ── 3. Determine next step ────────────────────────────────────────────────
    is_complete = _is_complete(updated_profile)
    next_step = _next_unfilled_step(updated_profile) if not is_complete else None
    clarifying = not step_filled and not is_complete

    # ── 4. Build history for LLM ──────────────────────────────────────────────
    updated_history = list(history) + [{"role": "user", "content": user_text}]

    # ── 5. Generate assistant reply ───────────────────────────────────────────
    reply = generate_reply(
        step_just_completed=current_step if step_filled else None,
        next_step=next_step,
        profile=updated_profile,
        clarifying=clarifying,
        conversation_history=updated_history,
    )

    # Advance step only when current is filled
    new_step = next_step if step_filled else current_step
    if is_complete:
        new_step = "complete"

    return {
        "reply": reply,
        "profile": updated_profile,
        "current_step": new_step or "complete",
        "is_complete": is_complete,
        "missing_fields": _missing_required(updated_profile),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opportunistic_extract(text: str, profile: dict) -> None:
    """Try to fill any empty field from a free-form message."""
    if not profile.get("time_horizon"):
        v = extract_time_horizon(text)
        if v:
            profile["time_horizon"] = v
    if not profile.get("risk_tolerance"):
        v = extract_risk_tolerance(text)
        if v:
            profile["risk_tolerance"] = v
    if not profile.get("objective"):
        v = extract_objective(text)
        if v:
            profile["objective"] = v
    if not profile.get("priority"):
        v = extract_priority(text)
        if v:
            profile["priority"] = v


_OBJECTIVE_TO_PRIORITY = {
    "growth": "max_return",
    "stability": "lower_volatility",
    "balanced": "balanced_growth",
    "income": "lower_volatility",
}


def _derive_priority(profile: dict) -> None:
    if not profile.get("priority") and profile.get("objective"):
        profile["priority"] = _OBJECTIVE_TO_PRIORITY.get(profile["objective"])
