import { type KeyboardEvent, useRef, useState } from "react";

interface Props {
  onSend: (text: string) => void;
  options?: string[];
  disabled?: boolean;
}

/** Text input + send button at the bottom of the chat. */
export function ChatInput({ onSend, options = [], disabled = false }: Props) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="p-3 border-t border-slate-700/50 bg-slate-800/50">
      {options.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {options.map((opt) => (
            <button
              key={opt}
              type="button"
              disabled={disabled}
              onClick={() => onSend(opt)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-700/70 border border-slate-600/40 text-slate-200 hover:bg-slate-700 disabled:opacity-40"
            >
              {opt}
            </button>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
          placeholder="Type your answer..."
          className="flex-1 resize-none rounded-xl bg-slate-700/60 border border-slate-600/50
            text-slate-100 placeholder-slate-500 text-sm px-4 py-2.5
            focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50
            disabled:opacity-50 disabled:cursor-not-allowed transition-all
            max-h-32 overflow-y-auto"
          style={{ minHeight: "42px" }}
        />
        <button
          onClick={handleSend}
          disabled={disabled || !text.trim()}
          className="w-10 h-10 rounded-xl bg-indigo-600 hover:bg-indigo-500
            disabled:opacity-40 disabled:cursor-not-allowed
            flex items-center justify-center transition-all active:scale-95 shrink-0"
          title="Send (Enter)"
        >
          <SendIcon />
        </button>
      </div>
    </div>
  );
}

function SendIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      className="w-4 h-4 text-white"
    >
      <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
    </svg>
  );
}
