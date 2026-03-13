/**
 * API client for the chatbot backend.
 * All requests go through the Vite proxy → http://localhost:8000
 */

import type {
  SendMessageResponse,
  StartResponse,
  InvestorProfile,
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
): Promise<unknown> {
  // Connect your actual recommendation/comparison API here.
  // For now, returns a mock result after a short delay.
  await new Promise((r) => setTimeout(r, 800));
  return buildMockRecommendation(profile);
}

function buildMockRecommendation(profile: InvestorProfile) {
  const { risk_tolerance, objective, priority } = profile;

  let bucket: "social_buzz" | "magnificent_7" | "mixed" = "mixed";
  if (risk_tolerance === "high" || objective === "growth" || priority === "max_return") {
    bucket = "social_buzz";
  } else if (risk_tolerance === "low" || objective === "stability") {
    bucket = "magnificent_7";
  }

  const stockMap = {
    social_buzz: ["GME", "AMC", "RIVN", "LCID", "BBBY"],
    magnificent_7: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    mixed: ["AAPL", "MSFT", "GME", "AMZN", "NVDA"],
  };

  return {
    recommended_bucket: bucket,
    reasoning: `Based on your ${risk_tolerance ?? "moderate"} risk tolerance and ${
      objective ?? "balanced"
    } objective, ${
      bucket === "social_buzz"
        ? "social buzz stocks offer higher upside potential."
        : bucket === "magnificent_7"
        ? "the Magnificent 7 provide stability and consistent growth."
        : "a mixed portfolio balances growth and stability."
    }`,
    top_stocks: stockMap[bucket],
    risk_score: risk_tolerance === "high" ? 8 : risk_tolerance === "low" ? 3 : 5,
    expected_return: risk_tolerance === "high" ? "15–30%" : risk_tolerance === "low" ? "5–10%" : "8–15%",
  };
}
