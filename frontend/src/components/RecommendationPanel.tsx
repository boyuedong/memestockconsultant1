/**
 * Right-hand panel — shows placeholder while chatting, then the recommendation.
 */

import type { InvestorProfile, Recommendation } from "../types/chat";

interface Props {
  recommendation: Recommendation | null;
  profile: InvestorProfile;
  isComplete: boolean;
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

  const rec = recommendation;
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

      {/* Two-column stock picks */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <StockListCard
          title="Hottest Meme Stocks"
          tickers={rec.meme_stocks}
          accentClass="text-orange-300"
        />
        <StockListCard
          title="Standard SPY-Style Picks"
          tickers={rec.standard_stocks}
          accentClass="text-indigo-300"
        />
      </div>

      {/* Investor tip */}
      <div className="rounded-xl border border-slate-700/50 bg-slate-800/40 p-4">
        <p className="text-xs text-slate-400 uppercase tracking-widest mb-2">Quick Tip</p>
        <p className="text-sm text-slate-200">{rec.investor_tip}</p>
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

      <p className="text-xs text-slate-600 text-center mt-2">
        * Recommendation generated from pipeline insights.
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

function StockListCard({
  title,
  tickers,
  accentClass,
}: {
  title: string;
  tickers: string[];
  accentClass: string;
}) {
  return (
    <div className="rounded-xl border border-slate-700/50 bg-slate-800/40 p-4">
      <p className="text-xs text-slate-400 uppercase tracking-widest mb-2">{title}</p>
      <div className="flex flex-wrap gap-2">
        {tickers.map((ticker) => (
          <span
            key={`${title}-${ticker}`}
            className={`px-3 py-1.5 rounded-lg bg-slate-700/60 border border-slate-600/40 text-sm font-mono font-semibold ${accentClass}`}
          >
            {ticker}
          </span>
        ))}
      </div>
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
