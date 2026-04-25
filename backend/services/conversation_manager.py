"""
Stateless conversation step manager.

All session state is kept in the caller (the route handler) and passed in.
This makes the logic easy to test and easy to swap backends.

Structured conversation steps (max 6 questions):
  1. time_horizon
  2. risk_tolerance
  3. objective
  4. preference
  5. loss_comfort
  6. diversification
  → complete
"""

from typing import Any, Optional
from .profile_extractor import (
    extract_time_horizon,
    extract_risk_tolerance,
    extract_objective,
    extract_preference,
    extract_loss_comfort,
    extract_diversification,
)

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

MAX_QUESTIONS = 6
REQUIRED_FIELDS = [
    "time_horizon",
    "risk_tolerance",
    "objective",
    "preference",
    "loss_comfort",
    "diversification",
]
WELCOME_MESSAGE = (
    "Hi! I can give an educational, model-based suggestion between meme/social-buzz stocks, "
    "standard ETF-style options (SPY/QQQ), or a mixed allocation.\n\n"
    "Q1/6: Choose your investment horizon."
)

_STEP_ORDER = [
    "time_horizon",
    "risk_tolerance",
    "objective",
    "preference",
    "loss_comfort",
    "diversification",
]

_EXTRACTORS = {
    "time_horizon": extract_time_horizon,
    "risk_tolerance": extract_risk_tolerance,
    "objective": extract_objective,
    "preference": extract_preference,
    "loss_comfort": extract_loss_comfort,
    "diversification": extract_diversification,
}

_STEP_PROMPTS = {
    "time_horizon": "Q1/6: What is your investment horizon?",
    "risk_tolerance": "Q2/6: What is your risk tolerance?",
    "objective": "Q3/6: What is your main goal?",
    "preference": "Q4/6: Which style do you prefer?",
    "loss_comfort": "Q5/6: How much downside can you tolerate?",
    "diversification": "Q6/6: What diversification style fits you best?",
}

_STEP_OPTIONS = {
    "time_horizon": ["1 week", "1 month", "3 months", "6+ months"],
    "risk_tolerance": ["Low", "Medium", "High"],
    "objective": [
        "Stable growth",
        "High upside",
        "Short-term trend",
        "Learning/experimenting",
    ],
    "preference": ["Meme/social buzz", "Standard ETF", "No preference"],
    "loss_comfort": [
        "Can tolerate small losses",
        "Can tolerate large swings",
        "Prefer safer choice",
    ],
    "diversification": ["Single trend pick", "Basket", "ETF-heavy"],
}


def empty_profile() -> dict:
    return {
        "time_horizon": None,
        "risk_tolerance": None,
        "objective": None,
        "preference": None,
        "loss_comfort": None,
        "diversification": None,
        "extra_notes": "",
        "question_count": 0,
    }


def _missing_required(profile: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if not profile.get(f)]


def _is_complete(profile: dict) -> bool:
    return len(_missing_required(profile)) == 0


def _next_unfilled_step(profile: dict) -> Optional[str]:
    """Return the next step that still needs answering, or None if done."""
    for step in _STEP_ORDER:
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
        "options": list[str],
        "question_count": int,
        "total_questions": int,
        "is_complete": bool,
        "missing_fields": list[str],
      }
    """
    updated_profile = dict(profile)
    question_count = int(updated_profile.get("question_count", 0))

    # ── 1. Try to extract a value for the current step ────────────────────────
    extractor = _EXTRACTORS.get(current_step)
    extracted_value = extractor(user_text) if extractor else None

    step_filled = False
    if extracted_value is not None:
        updated_profile[current_step] = extracted_value
        step_filled = True

    if step_filled:
        question_count += 1

    # ── 2. Determine completion (hard cap at 6 answered questions) ───────────
    is_complete = question_count >= MAX_QUESTIONS or _is_complete(updated_profile)
    next_step = _next_unfilled_step(updated_profile) if not is_complete else None

    # ── 3. Build reply (structured only, no open-ended stock-picking prompts) ─
    if is_complete:
        reply = (
            "Thanks — profile complete. I will now generate a model-based educational recommendation "
            "between meme/social-buzz picks, standard SPY/QQQ-style options, or a mixed allocation."
        )
        options: list[str] = []
        new_step = "complete"
    else:
        if not step_filled:
            # Remind user with explicit choices to prevent infinite clarifying loops.
            prompt = _STEP_PROMPTS.get(current_step, "Please choose one option:")
            opts = _STEP_OPTIONS.get(current_step, [])
            reply = f"{prompt} Please choose one of: {', '.join(opts)}."
            options = opts
            new_step = current_step
        else:
            prompt = _STEP_PROMPTS.get(next_step or "", "Next question:")
            opts = _STEP_OPTIONS.get(next_step or "", [])
            reply = f"{prompt} Options: {', '.join(opts)}."
            options = opts
            new_step = next_step or "complete"

    updated_profile["question_count"] = question_count

    return {
        "reply": reply,
        "profile": updated_profile,
        "current_step": new_step or "complete",
        "options": options,
        "question_count": question_count,
        "total_questions": MAX_QUESTIONS,
        "is_complete": is_complete,
        "missing_fields": _missing_required(updated_profile),
    }
