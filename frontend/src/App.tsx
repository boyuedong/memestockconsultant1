/**
 * Root layout — split-panel:
 *   Left  (40%)  ChatWindow
 *   Right (60%)  RecommendationPanel
 */

import { ChatWindow } from "./components/ChatWindow";
import { RecommendationPanel } from "./components/RecommendationPanel";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { state, recommendation, send, reset } = useChat();

  return (
    <div className="min-h-screen bg-surface-950 text-slate-100 flex flex-col">
      {/* Top nav */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-surface-900/80 backdrop-blur-sm z-10">
        <div className="flex items-center gap-2">
          <span className="text-xl">📈</span>
          <span className="font-bold text-white tracking-tight">StockAdvisor</span>
          <span className="text-xs text-slate-500 ml-1">AI Portfolio Assistant</span>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span className="px-2 py-1 rounded-full bg-slate-800 border border-slate-700">
            {state.isComplete ? "✅ Profile complete" : `Step: ${state.currentStep}`}
          </span>
        </div>
      </header>

      {/* Main split panel */}
      <main className="flex flex-1 overflow-hidden p-4 gap-4">
        {/* Left — chat */}
        <div className="w-2/5 min-w-[320px] flex flex-col">
          <ChatWindow state={state} onSend={send} onReset={reset} />
        </div>

        {/* Right — recommendation */}
        <div className="flex-1 rounded-2xl border border-slate-700/50 bg-slate-800/20 overflow-hidden">
          <RecommendationPanel
            recommendation={recommendation}
            profile={state.profile}
            isComplete={state.isComplete}
          />
        </div>
      </main>
    </div>
  );
}
