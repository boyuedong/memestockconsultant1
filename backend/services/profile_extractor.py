"""
Deterministic rule-based extraction of structured investor profile fields.
These rules run BEFORE (and can override) any LLM output, ensuring reliable mapping.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Time horizon
# ---------------------------------------------------------------------------

def extract_time_horizon(text: str) -> Optional[str]:
    t = text.lower().strip()
    if re.search(r"\b1\s*week|one\s*week|7\s*day", t):
        return "1W"
    if re.search(r"\b1\s*month|one\s*month|30\s*day", t):
        return "1M"
    if re.search(r"\b2\s*month|two\s*month", t):
        return "3M"
    if re.search(r"\b3\s*month|three\s*month|quarter", t):
        return "3M"
    if re.search(r"\b6\+?\s*month|six\s*month|half\s*(a\s*)?year|\b6m\+?", t):
        return "6M+"
    if re.search(r"\b1\s*year|one\s*year|12\s*month|annual|yearly", t):
        return "6M+"
    if re.search(r"\b2\s*year|two\s*year|24\s*month", t):
        return "6M+"
    if re.search(r"\blong[- ]?term|long\b", t):
        return "6M+"
    if re.search(r"\bshort[- ]?term|short\b", t):
        return "1M"
    if re.search(r"\bmedium[- ]?term|mid[- ]?term", t):
        return "3M"
    return None


# ---------------------------------------------------------------------------
# Risk tolerance
# ---------------------------------------------------------------------------

def extract_risk_tolerance(text: str) -> Optional[str]:
    t = text.lower().strip()
    t_clean = re.sub(r"[^a-z0-9\s/-]", " ", t)
    t_clean = re.sub(r"\s+", " ", t_clean).strip()

    # Fast-path common short answers
    if t_clean in {"high", "high risk", "aggressive", "risky"}:
        return "high"
    if t_clean in {"low", "low risk", "conservative", "safe"}:
        return "low"
    if t_clean in {"medium", "moderate", "mid", "balanced"}:
        return "medium"

    if re.search(
        r"\bhigh\s*risk|very\s*risk|aggress|i('m|\s+am)\s+(okay|fine|comfortable|good)\s+with\s+risk"
        r"|risk\s*taker|love\s*risk|maximize\s*return|high[-/ ]?medium|medium[-/ ]?high",
        t_clean,
    ):
        return "high"
    if re.search(
        r"\blow\s*risk|very\s*safe|conserv|cautious|avoid\s*risk|minimal\s*risk"
        r"|risk[\s-]?averse|not\s+(comfortable|okay)\s+with\s+risk|safer\s+stock|safe\s+bet"
        r"|not\s+too\s+risky|keep\s+it\s+safe",
        t_clean,
    ):
        return "low"
    if re.search(
        r"\bmedium|moderate|mid\s*risk|some\s*risk|balanced\s*risk|little\s*risk"
        r"|neutral|middle\s*ground|between|not\s+too\s+high|in\s+the\s+middle",
        t_clean,
    ):
        return "medium"
    return None


# ---------------------------------------------------------------------------
# Objective / style
# ---------------------------------------------------------------------------

def extract_objective(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(
        r"\bstable\s*growth|stabilit|safe|lower\s*volatil|minimal\s*loss|preserve\s*capital|steady",
        t,
    ):
        return "stability"
    if re.search(
        r"\bhigh\s*upside|max(imum|imize)?\s*return|growth|capital\s*gain|highest\s*return|short[-\s]?term\s*trend",
        t,
    ):
        return "growth"
    if re.search(r"\blearning|experiment|balanc|both|mix|even|moderate\s*growth", t):
        return "balanced"
    if re.search(r"\bincome|dividend|yield", t):
        return "income"
    return None


# ---------------------------------------------------------------------------
# Priority  (maps to: max_return | lower_volatility | balanced_growth)
# ---------------------------------------------------------------------------

def extract_priority(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\bmax(imum|imize)?\s*(return|profit|gain)|best\s*return|highest", t):
        return "max_return"
    if re.search(r"\blower\s*volatil|low\s*volatil|stabil|safe|protect", t):
        return "lower_volatility"
    if re.search(r"\bbalanced|balanced\s*growth|mix|both", t):
        return "balanced_growth"
    return None


# ---------------------------------------------------------------------------
# Sector preferences
# ---------------------------------------------------------------------------

KNOWN_SECTORS = {
    "tech": ["tech", "technology", "software", "ai", "cloud", "semiconductor"],
    "healthcare": ["health", "healthcare", "biotech", "pharma", "medical"],
    "energy": ["energy", "oil", "gas", "renewable", "solar", "wind"],
    "finance": ["finance", "financial", "banking", "bank", "fintech", "insurance"],
    "consumer": ["consumer", "retail", "goods", "cpg"],
    "real_estate": ["real estate", "reit", "property"],
    "industrials": ["industrial", "manufacturing", "aerospace", "defense"],
    "utilities": ["util"],
    "materials": ["material", "mining", "chemical"],
    "communication": ["communication", "telecom", "media"],
}


def extract_sectors(text: str) -> list:
    t = text.lower()
    if re.search(r"\bno\s+prefer|don.t\s+care|any|broad|all|none|skip", t):
        return []
    found = []
    for sector, keywords in KNOWN_SECTORS.items():
        for kw in keywords:
            if kw in t:
                found.append(sector)
                break
    return list(dict.fromkeys(found))  # deduplicated, order-preserving


def extract_preference(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\bmeme|social|buzz|reddit|hype", t):
        return "meme"
    if re.search(r"\bstandard|etf|spy|qqq|s&p|index|safer", t):
        return "standard"
    if re.search(r"\bno\s*preference|either|no\s*prefer|mixed|both|any", t):
        return "no_preference"
    return None


def extract_loss_comfort(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\blarge\s*swings|big\s*swings|volatile|high\s*drawdown|can\s*tolerate\s*large", t):
        return "large_swings"
    if re.search(r"\bsmall\s*loss|minor\s*loss|can\s*tolerate\s*small", t):
        return "small_losses"
    if re.search(r"\bprefer\s*safe|safer\s*choice|low\s*loss|capital\s*preservation", t):
        return "safer_choice"
    return None


def extract_diversification(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\bsingle|one\s*trend|concentrated|one\s*pick", t):
        return "single_trend_pick"
    if re.search(r"\bbasket|diversif|multiple", t):
        return "basket"
    if re.search(r"\betf|index|etf[-\s]*heavy|mostly\s*etf", t):
        return "etf_heavy"
    return None


# ---------------------------------------------------------------------------
# Aggregate extractor — applies all rules to a message
# ---------------------------------------------------------------------------

def extract_all_fields(text: str) -> dict:
    return {
        "time_horizon": extract_time_horizon(text),
        "risk_tolerance": extract_risk_tolerance(text),
        "objective": extract_objective(text),
        "preference": extract_preference(text),
        "loss_comfort": extract_loss_comfort(text),
        "diversification": extract_diversification(text),
        "priority": extract_priority(text),
        "sector_preferences": extract_sectors(text) or None,
    }
