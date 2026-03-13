/**
 * ChatWindow — the left-side chat panel.
 *
 * Contains:
 *  - Profile progress strip (filled fields)
 *  - Scrollable message list
 *  - Typing indicator
 *  - ChatInput bar
 *  - "Start Over" button
 */

import { useEffect, useRef } from "react";
import type { ChatState } from "../types/chat";
import { ChatInput } from "./ChatInput";
import { MessageBubble } from "./MessageBubble";
import { ProfileProgress } from "./ProfileSidebar";
import { TypingIndicator } from "./TypingIndicator";

interface Props {
  state: ChatState;
  onSend: (text: string) => void;
  onReset: () => void;
}

export function ChatWindow({ state, onSend, onReset }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [state.messages, state.isLoading]);

  return (
    <div className="flex flex-col h-full bg-slate-800/30 border-r border-slate-700/50 rounded-l-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/50 bg-slate-800/50">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <h1 className="text-sm font-semibold text-white">Investment Advisor</h1>
        </div>
        <button
          onClick={onReset}
          className="text-xs text-slate-400 hover:text-slate-200 transition-colors px-2 py-1 rounded-lg hover:bg-slate-700/50"
          title="Start over"
        >
          ↺ Start Over
        </button>
      </div>

      {/* Profile progress strip */}
      <ProfileProgress profile={state.profile} missingFields={state.missingFields} />

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-3 space-y-1 scroll-smooth">
        {state.messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {state.isLoading && <TypingIndicator />}

        {state.error && (
          <div className="mx-4 my-2 px-4 py-2.5 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs">
            {state.error}
          </div>
        )}

        {/* Completion banner */}
        {state.isComplete && (
          <div className="mx-4 my-2 px-4 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs text-center">
            ✅ Profile complete — recommendation generated!
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <ChatInput onSend={onSend} disabled={state.isLoading || state.isComplete} />

      {state.isComplete && (
        <div className="px-3 pb-3">
          <button
            onClick={onReset}
            className="w-full py-2 rounded-xl text-sm font-medium bg-slate-700/60 hover:bg-slate-700 text-slate-300 transition-all"
          >
            ↺ Start a new conversation
          </button>
        </div>
      )}
    </div>
  );
}
