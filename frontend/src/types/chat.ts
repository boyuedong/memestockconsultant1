export type MessageRole = "user" | "assistant";

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
}

export interface InvestorProfile {
  time_horizon: string | null;
  risk_tolerance: string | null;
  objective: string | null;
  sector_preferences: string[];
  priority: string | null;
  extra_notes: string;
}

export type ConversationStep =
  | "time_horizon"
  | "risk_tolerance"
  | "objective"
  | "sector_preferences"
  | "complete";

export interface ChatState {
  messages: Message[];
  profile: InvestorProfile;
  currentStep: ConversationStep;
  isComplete: boolean;
  missingFields: string[];
  isLoading: boolean;
  sessionId: string | null;
  error: string | null;
}

export interface SendMessageResponse {
  reply: string;
  profile: InvestorProfile;
  current_step: ConversationStep;
  is_complete: boolean;
  missing_fields: string[];
  session_id: string;
}

export interface StartResponse {
  welcome: string;
  session_id: string;
  profile: InvestorProfile;
}

// Returned from the recommendation engine after profile is complete
export interface Recommendation {
  recommended_bucket: "social_buzz" | "magnificent_7" | "mixed";
  reasoning: string;
  top_stocks: string[];
  risk_score: number;
  expected_return: string;
}
