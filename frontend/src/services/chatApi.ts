/**
 * API client for the chatbot backend.
 * All requests go through the Vite proxy → http://localhost:8000
 */

import type {
  SendMessageResponse,
  StartResponse,
  InvestorProfile,
  Recommendation,
} from "../types/chat";

const BASE = "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json() as Promise<T>;
}

// ── Chat endpoints ─────────────────────────────────────────────────────────────

/** Create (or reuse) a session and get the welcome message. */
export async function startChat(sessionId?: string): Promise<StartResponse> {
  return post<StartResponse>("/chat/start", { session_id: sessionId });
}

/** Send a user message and get the assistant reply. */
export async function sendMessage(
  message: string,
  sessionId: string
): Promise<SendMessageResponse> {
  return post<SendMessageResponse>("/chat/message", {
    message,
    session_id: sessionId,
  });
}

/** Reset the session to start fresh. */
export async function resetChat(sessionId: string): Promise<void> {
  await post("/chat/reset", { session_id: sessionId });
}

// ── Recommendation endpoint ────────────────────────────────────────────────────
// TODO: replace with real recommendation engine call

export async function fetchRecommendation(
  profile: InvestorProfile
): Promise<Recommendation> {
  return post<Recommendation>("/chat/recommend", { profile });
}
