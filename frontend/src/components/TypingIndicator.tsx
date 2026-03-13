/** Animated three-dot typing indicator shown while the assistant is thinking. */
export function TypingIndicator() {
  return (
    <div className="flex items-end gap-1 px-4 py-3">
      <div className="flex items-center gap-1 bg-slate-700/60 rounded-2xl rounded-bl-sm px-4 py-3">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-2 h-2 rounded-full bg-slate-400 animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    </div>
  );
}
