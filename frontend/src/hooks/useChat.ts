/**
 * useChat — manages all chatbot state.
 *
 * Handles:
 *  - session lifecycle (start / reset)
 *  - sending messages and updating history
 *  - loading / error states
 *  - triggering the recommendation engine when the profile is complete
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import type { ChatState, InvestorProfile, Message } from "../types/chat";
import { fetchRecommendation, resetChat, sendMessage, startChat } from "../services/chatApi";

const INITIAL_PROFILE: InvestorProfile = {
  time_horizon: null,
  risk_tolerance: null,
  objective: null,
  sector_preferences: [],
  priority: null,
  extra_notes: "",
};

const INITIAL_STATE: ChatState = {
  messages: [],
  profile: INITIAL_PROFILE,
  currentStep: "time_horizon",
  isComplete: false,
  missingFields: ["time_horizon", "risk_tolerance", "objective"],
  isLoading: false,
  sessionId: null,
  error: null,
};

function makeMessage(role: Message["role"], content: string): Message {
  return { id: uuidv4(), role, content, timestamp: new Date() };
}

export function useChat() {
  const [state, setState] = useState<ChatState>(INITIAL_STATE);
  const [recommendation, setRecommendation] = useState<unknown | null>(null);
  const sessionIdRef = useRef<string>(uuidv4());

  // ── Bootstrap session on mount ─────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      setState((s) => ({ ...s, isLoading: true, error: null }));
      try {
        const res = await startChat(sessionIdRef.current);
        sessionIdRef.current = res.session_id;
        setState((s) => ({
          ...s,
          sessionId: res.session_id,
          messages: [makeMessage("assistant", res.welcome)],
          isLoading: false,
        }));
      } catch {
        // Backend not running — show welcome from fallback
        setState((s) => ({
          ...s,
          sessionId: sessionIdRef.current,
          messages: [
            makeMessage(
              "assistant",
              "Hi! I can help recommend whether social buzz stocks or the Magnificent 7 " +
                "better fit your goals.\n\nFirst, what's your investment time horizon? " +
                "For example: 1 month, 3 months, 6 months, or 1 year?"
            ),
          ],
          isLoading: false,
        }));
      }
    })();
  }, []);

  // ── Send a user message ────────────────────────────────────────────────────
  const send = useCallback(async (text: string) => {
    if (!text.trim()) return;

    const userMsg = makeMessage("user", text);
    setState((s) => ({
      ...s,
      messages: [...s.messages, userMsg],
      isLoading: true,
      error: null,
    }));

    try {
      const res = await sendMessage(text, sessionIdRef.current);

      const assistantMsg = makeMessage("assistant", res.reply);
      setState((s) => ({
        ...s,
        messages: [...s.messages, assistantMsg],
        profile: res.profile,
        currentStep: res.current_step,
        isComplete: res.is_complete,
        missingFields: res.missing_fields,
        isLoading: false,
      }));

      // Trigger recommendation engine once profile is complete
      if (res.is_complete) {
        const rec = await fetchRecommendation(res.profile);
        setRecommendation(rec);
      }
    } catch (err) {
      setState((s) => ({
        ...s,
        isLoading: false,
        error: "Could not reach the server. Is the backend running on port 8000?",
      }));
    }
  }, []);

  // ── Reset conversation ─────────────────────────────────────────────────────
  const reset = useCallback(async () => {
    setRecommendation(null);
    setState((s) => ({ ...s, isLoading: true, error: null }));
    try {
      await resetChat(sessionIdRef.current);
      const res = await startChat(sessionIdRef.current);
      setState({
        ...INITIAL_STATE,
        sessionId: sessionIdRef.current,
        messages: [makeMessage("assistant", res.welcome)],
      });
    } catch {
      setState({
        ...INITIAL_STATE,
        sessionId: sessionIdRef.current,
        messages: [
          makeMessage(
            "assistant",
            "Hi! Let's start fresh. What's your investment time horizon?"
          ),
        ],
      });
    }
  }, []);

  return { state, recommendation, send, reset };
}
