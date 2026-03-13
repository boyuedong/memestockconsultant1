import type { InvestorProfile } from "../types/chat";

interface Props {
  profile: InvestorProfile;
  missingFields: string[];
}

const FIELD_LABELS: Record<string, string> = {
  time_horizon: "Time Horizon",
  risk_tolerance: "Risk Tolerance",
  objective: "Objective",
  priority: "Priority",
  sector_preferences: "Sectors",
};

const FIELD_ORDER = ["time_horizon", "risk_tolerance", "objective", "priority", "sector_preferences"];

/** Compact profile progress strip shown above the chat. */
export function ProfileProgress({ profile, missingFields }: Props) {
  return (
    <div className="flex gap-2 flex-wrap px-3 py-2 border-b border-slate-700/50">
      {FIELD_ORDER.filter((f) => f !== "sector_preferences").map((field) => {
        const value = profile[field as keyof InvestorProfile];
        const filled = value !== null && value !== undefined && value !== "";
        return (
          <div
            key={field}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all ${
              filled
                ? "bg-indigo-600/30 text-indigo-300 border border-indigo-500/40"
                : "bg-slate-700/40 text-slate-500 border border-slate-600/30"
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${filled ? "bg-indigo-400" : "bg-slate-600"}`} />
            {FIELD_LABELS[field]}
            {filled && <span className="opacity-70">: {String(value)}</span>}
          </div>
        );
      })}
    </div>
  );
}
