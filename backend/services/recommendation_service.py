"""
Recommendation engine backed by selected-stocks pipeline outputs.

Primary path:
  - Build deterministic insights from CSV outputs
  - Optionally ask a Qwen-compatible OpenAI endpoint for a user-facing summary

Fallback path:
  - Fully deterministic recommendation if Qwen is not configured/available
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd

try:
    from openai import OpenAI as _OpenAI
except Exception:  # pragma: no cover
    _OpenAI = None


BASE_DIR = Path(__file__).resolve().parents[2]
SUMMARY_CSV = BASE_DIR / "selected_stocks_model_summary.csv"
ROWS_CSV = BASE_DIR / "selected_stocks_walkforward_results.csv"

RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "recommended_bucket": {
            "type": "string",
            "enum": ["social_buzz", "magnificent_7", "mixed"],
        },
        "reasoning": {"type": "string"},
        "top_stocks": {"type": "array", "items": {"type": "string"}, "minItems": 3},
        "meme_stocks": {"type": "array", "items": {"type": "string"}, "minItems": 3},
        "standard_stocks": {"type": "array", "items": {"type": "string"}, "minItems": 3},
        "investor_tip": {"type": "string"},
        "risk_score": {"type": "number"},
        "expected_return": {"type": "string"},
    },
    "required": [
        "recommended_bucket",
        "reasoning",
        "top_stocks",
        "meme_stocks",
        "standard_stocks",
        "investor_tip",
        "risk_score",
        "expected_return",
    ],
}


def _safe_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "time_horizon": profile.get("time_horizon"),
        "risk_tolerance": profile.get("risk_tolerance"),
        "objective": profile.get("objective"),
        "preference": profile.get("preference"),
        "loss_comfort": profile.get("loss_comfort"),
        "diversification": profile.get("diversification"),
        "sector_preferences": profile.get("sector_preferences") or [],
        "priority": profile.get("priority"),
        "extra_notes": profile.get("extra_notes", ""),
    }


def _load_insights() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_CSV}")
    if not ROWS_CSV.exists():
        raise FileNotFoundError(f"Missing rows file: {ROWS_CSV}")

    summary = pd.read_csv(SUMMARY_CSV)
    rows = pd.read_csv(ROWS_CSV)
    for col in ["accuracy", "macro_f1", "macro_recall"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return summary, rows


def _bucket_from_profile(profile: dict[str, Any]) -> str:
    rt = (profile.get("risk_tolerance") or "").lower()
    obj = (profile.get("objective") or "").lower()
    pri = (profile.get("priority") or "").lower()
    pref = (profile.get("preference") or "").lower()
    horizon = (profile.get("time_horizon") or "").upper()

    short_horizon = horizon in {"1W", "1M", "3M"}
    long_horizon = horizon in {"6M+", "1Y", "2Y"}

    if pref == "standard":
        return "magnificent_7" if rt != "high" else "mixed"

    if pref == "meme" and rt == "high" and short_horizon:
        return "social_buzz"

    if rt == "low" or obj == "stability" or pri == "lower_volatility":
        return "magnificent_7"

    if long_horizon and pref != "meme":
        return "magnificent_7"

    if rt == "high" or obj == "growth" or pri == "max_return":
        return "social_buzz" if pref in {"meme", "no_preference", ""} else "mixed"

    return "mixed"


def _score_summary(summary: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    scored = summary.copy()
    # Base reliability score from held-out metrics.
    scored["base_score"] = (
        scored["accuracy"].fillna(0) * 0.45
        + scored["macro_f1"].fillna(0) * 0.35
        + scored["macro_recall"].fillna(0) * 0.20
    )

    # Light risk alignment penalty/boost by objective.
    objective = (profile.get("objective") or "").lower()
    if objective == "stability":
        scored["base_score"] = scored["base_score"] + 0.03 * scored["macro_recall"].fillna(0)
    elif objective == "growth":
        scored["base_score"] = scored["base_score"] + 0.03 * scored["accuracy"].fillna(0)

    return scored.sort_values("base_score", ascending=False).reset_index(drop=True)


def _expected_return_text(profile: dict[str, Any]) -> str:
    rt = (profile.get("risk_tolerance") or "").lower()
    if rt == "high":
        return "12-28%"
    if rt == "low":
        return "4-10%"
    return "7-16%"


def _split_stock_buckets(
    profile: dict[str, Any], summary: pd.DataFrame, rows: pd.DataFrame
) -> tuple[list[str], list[str]]:
    ranked = _score_summary(summary, profile).copy()
    if ranked.empty:
        return [], []

    rows_work = rows.copy()
    if "DATE" in rows_work.columns:
        rows_work["DATE"] = pd.to_datetime(rows_work["DATE"], errors="coerce")
    if "NEXT_RETURN" in rows_work.columns:
        rows_work["NEXT_RETURN"] = pd.to_numeric(rows_work["NEXT_RETURN"], errors="coerce")
    if "CORRECT" in rows_work.columns:
        rows_work["CORRECT"] = rows_work["CORRECT"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )

    if "ticker" in ranked.columns:
        key = "ticker"
    else:
        key = "stock"
        ranked[key] = ranked[key].astype(str).str.upper()
        if "TICKER" in rows_work.columns:
            rows_work["TICKER"] = rows_work["TICKER"].astype(str).str.upper()

    group_key = "TICKER" if "TICKER" in rows_work.columns else "STOCK"
    agg_rows = []
    for k, g in rows_work.groupby(group_key):
        g = g.sort_values("DATE") if "DATE" in g.columns else g
        recent = g.tail(10)
        vol = recent["NEXT_RETURN"].abs().mean() if "NEXT_RETURN" in recent.columns else 0.0
        recent_ret = recent["NEXT_RETURN"].mean() if "NEXT_RETURN" in recent.columns else 0.0
        recent_acc = recent["CORRECT"].mean() if "CORRECT" in recent.columns else 0.0
        buy_ratio = (
            recent["PREDICTED_LABEL"].astype(str).str.lower().isin(["must buy", "maybe buy"]).mean()
            if "PREDICTED_LABEL" in recent.columns
            else 0.0
        )
        agg_rows.append(
            {
                "key": str(k).upper(),
                "recent_vol": float(vol) if pd.notna(vol) else 0.0,
                "recent_ret": float(recent_ret) if pd.notna(recent_ret) else 0.0,
                "recent_acc": float(recent_acc) if pd.notna(recent_acc) else 0.0,
                "buy_ratio": float(buy_ratio) if pd.notna(buy_ratio) else 0.0,
            }
        )

    agg = pd.DataFrame(agg_rows)
    ranked["join_key"] = ranked[key].astype(str).str.upper()
    if not agg.empty:
        ranked = ranked.merge(agg, how="left", left_on="join_key", right_on="key")
    for c in ["recent_vol", "recent_ret", "recent_acc", "buy_ratio"]:
        if c not in ranked.columns:
            ranked[c] = 0.0
        ranked[c] = ranked[c].fillna(0.0)

    ranked["standard_score"] = (
        ranked["base_score"] * 0.65 + ranked["recent_acc"] * 0.25 - ranked["recent_vol"] * 0.10
    )
    ranked["meme_score"] = (
        ranked["recent_vol"] * 0.45
        + ranked["buy_ratio"] * 0.25
        + ranked["recent_ret"] * 0.20
        + ranked["base_score"] * 0.10
    )

    standard = ranked.sort_values("standard_score", ascending=False)["join_key"].head(5).tolist()
    meme = (
        ranked[~ranked["join_key"].isin(standard)]
        .sort_values("meme_score", ascending=False)["join_key"]
        .head(5)
        .tolist()
    )
    if len(meme) < 5:
        meme = (
            ranked.sort_values("meme_score", ascending=False)["join_key"].head(5).tolist()
        )
    return meme, standard


def _deterministic_recommendation(
    profile: dict[str, Any], summary: pd.DataFrame, rows: pd.DataFrame
) -> dict[str, Any]:
    ranked = _score_summary(summary, profile)
    top = ranked.head(5)
    top_stocks = top["ticker"].astype(str).tolist() if "ticker" in top.columns else []
    if not top_stocks and "stock" in top.columns:
        top_stocks = top["stock"].astype(str).str.upper().tolist()

    bucket = _bucket_from_profile(profile)
    horizon = profile.get("time_horizon") or "your horizon"
    risk = profile.get("risk_tolerance") or "medium"
    objective = profile.get("objective") or "balanced"
    avg_acc = float(top["accuracy"].mean()) if len(top) else 0.0

    reasoning = (
        f"For a {horizon} horizon with {risk} risk tolerance and {objective} objective, "
        f"these picks are prioritized by historical out-of-sample reliability "
        f"(top-5 avg accuracy: {avg_acc:.1%})."
    )

    risk_score = 8 if risk == "high" else 3 if risk == "low" else 5

    meme_stocks, standard_stocks = _split_stock_buckets(profile, summary, rows)
    if not standard_stocks:
        standard_stocks = top_stocks[:5]
    if not meme_stocks:
        meme_stocks = list(reversed(top_stocks))[:5]

    if bucket == "social_buzz":
        tip = "Your profile leans higher risk, so consider a heavier meme-stock tilt with disciplined position sizing."
    elif bucket == "magnificent_7":
        tip = "Your profile favors stability; prioritize standard/SPY-style names and keep meme exposure small."
    else:
        tip = "A blended approach fits your profile: use standard names as core and add selective meme exposure for upside."

    return {
        "recommended_bucket": bucket,
        "reasoning": reasoning,
        "top_stocks": top_stocks[:5],
        "meme_stocks": meme_stocks,
        "standard_stocks": standard_stocks,
        "investor_tip": tip,
        "risk_score": risk_score,
        "expected_return": _expected_return_text(profile),
    }


def _qwen_available() -> bool:
    return bool(_OpenAI and os.environ.get("QWEN_API_KEY") and os.environ.get("QWEN_MODEL"))


def _ollama_available() -> bool:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        req = urlrequest.Request(f"{base_url}/api/tags", method="GET")
        with urlrequest.urlopen(req, timeout=1.0) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _build_llm_payload(profile: dict[str, Any], summary: pd.DataFrame, fallback: dict[str, Any]) -> dict[str, Any]:
    ranked = _score_summary(summary, profile).head(10)
    ranked_cols = [c for c in ["stock", "ticker", "accuracy", "macro_f1", "macro_recall"] if c in ranked.columns]
    ranked_payload = ranked[ranked_cols].to_dict(orient="records")
    return {
        "profile": profile,
        "top_ranked_stocks": ranked_payload,
        "fallback": fallback,
        "instruction": (
            "Return JSON with keys: recommended_bucket, reasoning, top_stocks, meme_stocks, standard_stocks, investor_tip, risk_score, expected_return. "
            "recommended_bucket must be one of social_buzz, magnificent_7, mixed. "
            "Keep reasoning concise and practical. "
            "meme_stocks should represent hotter/high-volatility picks; standard_stocks should represent SPY-like stable picks."
        ),
    }


def _validate_schema(parsed: dict[str, Any]) -> None:
    for key in RECOMMENDATION_SCHEMA["required"]:
        if key not in parsed:
            raise ValueError(f"LLM response missing key: {key}")


def _pick_list(llm_result: dict[str, Any], key: str, fallback: list[str], min_len: int = 3) -> list[str]:
    val = llm_result.get(key)
    if not isinstance(val, list):
        return fallback[:5]
    cleaned = [str(x).strip().upper() for x in val if str(x).strip()]
    # Deduplicate while preserving order
    cleaned = list(dict.fromkeys(cleaned))
    if len(cleaned) < min_len:
        return fallback[:5]
    return cleaned[:5]


def _call_qwen(profile: dict[str, Any], summary: pd.DataFrame, fallback: dict[str, Any]) -> dict[str, Any]:
    client = _OpenAI(
        api_key=os.environ["QWEN_API_KEY"],
        base_url=os.environ.get("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )
    model = os.environ["QWEN_MODEL"]

    system = (
        "You are a portfolio recommendation assistant. Use ONLY provided metrics and profile. "
        "Do not invent numbers. Return only strict JSON."
    )
    user = _build_llm_payload(profile, summary, fallback)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.2,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    content = (response.choices[0].message.content or "").strip()
    parsed = json.loads(content)
    _validate_schema(parsed)
    return parsed


def _call_ollama(profile: dict[str, Any], summary: pd.DataFrame, fallback: dict[str, Any]) -> dict[str, Any]:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct")
    timeout_sec = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "30"))

    system = (
        "You are a portfolio recommendation assistant. Use ONLY provided metrics and profile. "
        "Do not invent numbers. Return only strict JSON."
    )
    user = _build_llm_payload(profile, summary, fallback)
    req_payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        "options": {"temperature": 0.2},
    }
    req = urlrequest.Request(
        f"{base_url}/api/chat",
        data=json.dumps(req_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urlerror.URLError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    content = ((body.get("message") or {}).get("content") or "").strip()
    parsed = json.loads(content)
    _validate_schema(parsed)
    return parsed


def generate_recommendation(profile: dict[str, Any]) -> dict[str, Any]:
    clean_profile = _safe_profile(profile)
    summary, rows = _load_insights()
    deterministic = _deterministic_recommendation(clean_profile, summary, rows)
    provider = os.environ.get("LLM_PROVIDER", "auto").strip().lower()

    if provider == "fallback":
        return deterministic

    candidates: list[tuple[str, Any]] = []
    if provider == "qwen":
        candidates = [("qwen", _call_qwen)] if _qwen_available() else []
    elif provider == "ollama":
        candidates = [("ollama", _call_ollama)] if _ollama_available() else []
    else:
        # auto: prefer Qwen when configured, otherwise local Ollama, then deterministic.
        if _qwen_available():
            candidates.append(("qwen", _call_qwen))
        if _ollama_available():
            candidates.append(("ollama", _call_ollama))

    for _name, caller in candidates:
        try:
            llm_result = caller(clean_profile, summary, deterministic)
        except Exception:
            continue

        # Merge to keep deterministic safety for missing optionals.
        meme = _pick_list(llm_result, "meme_stocks", deterministic["meme_stocks"])
        standard = _pick_list(
            llm_result, "standard_stocks", deterministic["standard_stocks"]
        )
        top = _pick_list(llm_result, "top_stocks", deterministic["top_stocks"])
        return {
            "recommended_bucket": llm_result.get(
                "recommended_bucket", deterministic["recommended_bucket"]
            ),
            "reasoning": llm_result.get("reasoning", deterministic["reasoning"]),
            "top_stocks": top,
            "meme_stocks": meme,
            "standard_stocks": standard,
            "investor_tip": llm_result.get("investor_tip", deterministic["investor_tip"]),
            "risk_score": llm_result.get("risk_score", deterministic["risk_score"]),
            "expected_return": llm_result.get(
                "expected_return", deterministic["expected_return"]
            ),
        }

    return deterministic
