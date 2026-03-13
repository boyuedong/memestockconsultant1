/**
 * Right-hand panel — shows placeholder while chatting, then the recommendation.
 * TODO: Replace mock data display with real recommendation engine output.
 */

import type { InvestorProfile } from "../types/chat";

interface Props {
  recommendation: unknown | null;
  profile: InvestorProfile;
  isComplete: boolean;
}

interface MockRec {
  recommended_bucket: string;
  reasoning: string;
  top_stocks: string[];
  risk_score: number;
  expected_return: string;
}

const BUCKET_LABEL: Record<string, string> = {
  social_buzz: "Social Buzz Stocks",
  magnificent_7: "Magnificent 7",
  mixed: "Mixed Portfolio",
};

const BUCKET_COLOR: Record<string, string> = {
  social_buzz: "from-orange-500/20 to-red-500/10 border-orange-500/30",
  magnificent_7: "from-indigo-500/20 to-blue-500/10 border-indigo-500/30",
  mixed: "from-emerald-500/20 to-teal-500/10 border-emerald-500/30",
};

export function RecommendationPanel({ recommendation, profile, isComplete }: Props) {
  if (!isComplete || !recommendation) {
    return <EmptyState profile={profile} />;
  }

  const rec = recommendation as MockRec;
  const colorClass = BUCKET_COLOR[rec.recommended_bucket] ?? BUCKET_COLOR.mixed;

  return (
    <div className="flex flex-col gap-4 p-6 overflow-y-auto h-full">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl">🎯</span>
        <h2 className="text-lg font-semibold text-white">Your Recommendation</h2>
      </div>

      {/* Main recommendation card */}
      <div className={`rounded-2xl border bg-gradient-to-br p-5 ${colorClass}`}>
        <p className="text-xs text-slate-400 uppercase tracking-widest mb-1">Recommended</p>
        <h3 className="text-2xl font-bold text-white mb-3">
          {BUCKET_LABEL[rec.recommended_bucket]}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed">{rec.reasoning}</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Risk Score" value={`${rec.risk_score}/10`} icon="⚡" />
        <StatCard label="Expected Return" value={rec.expected_return} icon="📈" />
        <StatCard label="Time Horizon" value={profile.time_horizon ?? "—"} icon="🗓️" />
        <StatCard
          label="Risk Level"
          value={capitalize(profile.risk_tolerance ?? "—")}
          icon="🛡️"
        />
      </div>

      {/* Top stocks */}
      <div>
        <p className="text-xs text-slate-400 uppercase tracking-widest mb-2">Top Picks</p>
        <div className="flex flex-wrap gap-2">
          {rec.top_stocks.map((ticker) => (
            <span
              key={ticker}
              className="px-3 py-1.5 rounded-lg bg-slate-700/60 border border-slate-600/40 text-sm font-mono font-semibold text-indigo-300"
            >
              {ticker}
            </span>
          ))}
        </div>
      </div>

      {/* Investor profile summary */}
      <div className="rounded-xl border border-slate-700/50 bg-slate-800/40 p-4">
        <p className="text-xs text-slate-400 uppercase tracking-widest mb-3">Investor Profile</p>
        <div className="space-y-1.5">
          <ProfileRow label="Time Horizon" value={profile.time_horizon} />
          <ProfileRow label="Risk Tolerance" value={profile.risk_tolerance} />
          <ProfileRow label="Objective" value={profile.objective} />
          <ProfileRow label="Priority" value={profile.priority} />
          {profile.sector_preferences.length > 0 && (
            <ProfileRow label="Sectors" value={profile.sector_preferences.join(", ")} />
          )}
        </div>
      </div>

      {/* TODO: connect to real comparison logic */}
      <p className="text-xs text-slate-600 text-center mt-2">
        * Mock data — connect{" "}
        <code className="bg-slate-800 px-1 rounded">fetchRecommendation()</code> in{" "}
        <code className="bg-slate-800 px-1 rounded">chatApi.ts</code> to your real engine
      </p>
    </div>
  );
}

function StatCard({ label, value, icon }: { label: string; value: string; icon: string }) {
  return (
    <div className="rounded-xl border border-slate-700/50 bg-slate-800/40 p-3 flex items-center gap-3">
      <span className="text-xl">{icon}</span>
      <div>
        <p className="text-xs text-slate-500">{label}</p>
        <p className="text-sm font-semibold text-white">{value}</p>
      </div>
    </div>
  );
}

function ProfileRow({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-200 font-medium capitalize">{value ?? "—"}</span>
    </div>
  );
}

function EmptyState({ profile }: { profile: InvestorProfile }) {
  const filledCount = [
    profile.time_horizon,
    profile.risk_tolerance,
    profile.objective,
  ].filter(Boolean).length;

  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 p-8 text-center">
      <div className="w-20 h-20 rounded-full bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
        <span className="text-4xl">📊</span>
      </div>
      <div>
        <h2 className="text-lg font-semibold text-white mb-2">
          Your Recommendation
        </h2>
        <p className="text-sm text-slate-400 max-w-xs leading-relaxed">
          Answer the chatbot's questions on the left to get a personalized stock
          portfolio recommendation.
        </p>
      </div>

      {/* Progress indicator */}
      <div className="w-full max-w-xs">
        <div className="flex justify-between text-xs text-slate-500 mb-2">
          <span>Profile completion</span>
          <span>{filledCount}/3 required fields</span>
        </div>
        <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-indigo-500 rounded-full transition-all duration-500"
            style={{ width: `${(filledCount / 3) * 100}%` }}
          />
        </div>
      </div>

      <div className="flex flex-col gap-2 text-xs text-slate-600">
        {[
          ["🗓️", "Investment time horizon"],
          ["⚡", "Risk tolerance"],
          ["🎯", "Return objective"],
        ].map(([icon, label]) => (
          <div key={label} className="flex items-center gap-2">
            <span>{icon}</span>
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
