import type { Message } from "../types/chat";

interface Props {
  message: Message;
}

/** Renders a single chat message bubble (user or assistant). */
export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"} px-4 py-1`}>
      {/* Avatar for assistant */}
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-indigo-500/20 border border-indigo-500/40 flex items-center justify-center shrink-0 mr-2 mt-1">
          <span className="text-xs">📈</span>
        </div>
      )}

      <div
        className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? "bg-indigo-600 text-white rounded-br-sm"
            : "bg-slate-700/70 text-slate-100 rounded-bl-sm"
        }`}
      >
        {message.content}
      </div>

      {/* Avatar for user */}
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-indigo-600/30 border border-indigo-500/40 flex items-center justify-center shrink-0 ml-2 mt-1">
          <span className="text-xs">👤</span>
        </div>
      )}
    </div>
  );
}
