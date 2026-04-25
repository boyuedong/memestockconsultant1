from __future__ import annotations

import json
from pathlib import Path
from urllib import parse as urlparse
from urllib import request as urlrequest

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = BASE_DIR / "selected_stocks_model_summary.csv"
RESULTS_CSV = BASE_DIR / "selected_stocks_walkforward_results.csv"

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SUMMARY_CSV)


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS_CSV)


def _api_get(url: str, timeout: float = 10.0) -> dict:
    req = urlrequest.Request(url, method="GET")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _api_post(url: str, payload: dict, timeout: float = 20.0) -> dict:
    req = urlrequest.Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ensure_session_state() -> None:
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = DEFAULT_BACKEND_URL
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile" not in st.session_state:
        st.session_state.profile = {}
    if "current_step" not in st.session_state:
        st.session_state.current_step = "time_horizon"
    if "current_options" not in st.session_state:
        st.session_state.current_options = []
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "total_questions" not in st.session_state:
        st.session_state.total_questions = 6
    if "is_complete" not in st.session_state:
        st.session_state.is_complete = False
    if "recommendation" not in st.session_state:
        st.session_state.recommendation = None
    if "error" not in st.session_state:
        st.session_state.error = None


def _start_chat() -> None:
    base = st.session_state.backend_url.rstrip("/")
    session_id = st.session_state.session_id
    query = ""
    if session_id:
        query = "?" + urlparse.urlencode({"session_id": session_id})

    data = _api_post(f"{base}/api/chat/start{query}", {})
    st.session_state.session_id = data.get("session_id")
    st.session_state.profile = data.get("profile", {})
    st.session_state.messages = [
        {"role": "assistant", "content": data.get("welcome", "Welcome to Stock Assistant.")}
    ]
    st.session_state.current_step = data.get("current_step", "time_horizon")
    st.session_state.current_options = data.get("options", [])
    st.session_state.question_count = int(data.get("question_count", 0))
    st.session_state.total_questions = int(data.get("total_questions", 6))
    st.session_state.is_complete = False
    st.session_state.recommendation = None
    st.session_state.error = None


def _reset_chat() -> None:
    base = st.session_state.backend_url.rstrip("/")
    sid = st.session_state.session_id
    if sid:
        _api_post(f"{base}/api/chat/reset", {"session_id": sid})
    _start_chat()


def _send_message(text: str) -> None:
    base = st.session_state.backend_url.rstrip("/")
    sid = st.session_state.session_id
    if not sid:
        _start_chat()
        sid = st.session_state.session_id

    st.session_state.messages.append({"role": "user", "content": text})
    data = _api_post(
        f"{base}/api/chat/message",
        {"message": text, "session_id": sid},
    )
    st.session_state.profile = data.get("profile", {})
    st.session_state.current_step = data.get("current_step", st.session_state.current_step)
    st.session_state.current_options = data.get("options", st.session_state.current_options)
    st.session_state.question_count = int(data.get("question_count", st.session_state.question_count))
    st.session_state.total_questions = int(data.get("total_questions", st.session_state.total_questions))
    st.session_state.is_complete = bool(data.get("is_complete", False))
    st.session_state.messages.append({"role": "assistant", "content": data.get("reply", "")})

    if st.session_state.is_complete:
        st.session_state.recommendation = _api_post(
            f"{base}/api/chat/recommend", {"profile": st.session_state.profile}
        )


def _render_recommendation_panel() -> None:
    st.subheader("Recommendation")
    rec = st.session_state.recommendation
    if not rec:
        st.info("Complete the chat profile on the left to generate recommendations.")
        return

    st.success(f"Bucket: {rec.get('recommended_bucket', '-')}")
    st.write(rec.get("reasoning", ""))
    c_meme, c_std = st.columns(2)
    c_meme.markdown("**Hottest Meme Stocks**")
    c_meme.write(", ".join(rec.get("meme_stocks", []) or ["-"]))
    c_std.markdown("**Standard SPY-Style Picks**")
    c_std.write(", ".join(rec.get("standard_stocks", []) or ["-"]))

    tip = rec.get("investor_tip")
    if tip:
        st.info(f"Tip: {tip}")

    c1, c2 = st.columns(2)
    c1.metric("Risk score", rec.get("risk_score", "-"))
    c2.metric("Expected return", rec.get("expected_return", "-"))


def app() -> None:
    st.set_page_config(page_title="Stock Recommendation Chatbot", layout="wide")
    _ensure_session_state()

    st.title("Stock Recommendation Chatbot (Streamlit)")
    st.caption(
        "Chat-driven profile extraction -> backend recommendation using XGBoost/LightGBM outputs "
        "with LLM narrative generation."
    )

    with st.sidebar:
        st.header("Backend Connection")
        backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)
        st.session_state.backend_url = backend_url.strip() or DEFAULT_BACKEND_URL

        if st.button("Reconnect / Start Session", width="stretch"):
            try:
                _start_chat()
                st.success("Connected.")
            except Exception as e:
                st.session_state.error = f"Could not connect to backend: {e}"

        if st.button("Reset Conversation", width="stretch"):
            try:
                _reset_chat()
                st.success("Conversation reset.")
            except Exception as e:
                st.session_state.error = f"Could not reset conversation: {e}"

        try:
            health = _api_get(st.session_state.backend_url.rstrip("/") + "/health")
            st.caption(f"Backend health: {health.get('status', 'unknown')}")
        except Exception:
            st.caption("Backend health: unreachable")

    if not st.session_state.messages:
        try:
            _start_chat()
        except Exception as e:
            st.error(
                "Backend unreachable. Start backend on port 8000 and reconnect.\n\n"
                f"Details: {e}"
            )
            return

    if st.session_state.error:
        st.error(st.session_state.error)

    col_chat, col_right = st.columns([1.5, 1.0])

    with col_chat:
        st.subheader("Chat")
        if not st.session_state.is_complete:
            st.caption(
                f"Question {st.session_state.question_count + 1}/{st.session_state.total_questions}"
            )
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.current_options and not st.session_state.is_complete:
            st.write("Quick options:")
            opt_cols = st.columns(min(3, max(1, len(st.session_state.current_options))))
            for idx, opt in enumerate(st.session_state.current_options):
                col = opt_cols[idx % len(opt_cols)]
                if col.button(opt, key=f"opt-{st.session_state.question_count}-{idx}"):
                    try:
                        _send_message(opt)
                    except Exception as e:
                        st.session_state.error = f"Message failed: {e}"
                    st.rerun()

        user_text = st.chat_input("Type your answer...")
        if user_text:
            try:
                _send_message(user_text)
            except Exception as e:
                st.session_state.error = f"Message failed: {e}"
            st.rerun()

    with col_right:
        st.subheader("Extracted Profile")
        profile = st.session_state.profile or {}
        st.json(profile)
        st.caption(f"Current step: `{st.session_state.current_step}`")
        st.caption(f"Profile complete: `{st.session_state.is_complete}`")

        _render_recommendation_panel()

        st.divider()
        st.subheader("Model Snapshot")
        summary_df = load_summary()
        results_df = load_results()
        if summary_df.empty:
            st.warning("No model summary found. Run `python3 train_selected_stocks.py` first.")
        else:
            st.metric("Stocks in run", int(summary_df["stock"].nunique()))
            st.metric("Mean accuracy", f"{summary_df['accuracy'].mean():.1%}")
            st.metric("Median accuracy", f"{summary_df['accuracy'].median():.1%}")
            if not results_df.empty and "CORRECT" in results_df.columns:
                overall = pd.to_numeric(results_df["CORRECT"], errors="coerce").mean()
                st.metric("Row-level accuracy", f"{overall:.1%}")

            show_cols = [
                c
                for c in ["stock", "ticker", "accuracy", "macro_f1", "macro_recall"]
                if c in summary_df.columns
            ]
            st.dataframe(
                summary_df.sort_values("accuracy", ascending=False)[show_cols].head(8),
                width="stretch",
                hide_index=True,
            )


if __name__ == "__main__":
    app()
