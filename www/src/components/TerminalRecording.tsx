import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";

interface Line {
  text: string;
  delay: number; // ms before this line appears
  typing?: boolean; // simulate character-by-character typing
  dim?: boolean;
}

const lines: Line[] = [
  { text: "$ devops run sre review", delay: 0, typing: true },
  { text: "", delay: 600 },
  { text: "(use /command for task commands, Ctrl+C to detach)", delay: 800, dim: true },
  { text: "  [14:23:45] Task started", delay: 1200 },
  { text: "  [14:23:46] Starting ops review (model: claude-sonnet-4-20250514)", delay: 1800 },
  { text: "  [14:23:47] Generating mock NewRelic metrics (2 weeks, 5-min granularity)...", delay: 2400 },
  { text: "  [14:24:01] Computing week-over-week statistics...", delay: 3800 },
  { text: "  [14:24:02] Summarizing findings (1 LLM call)...", delay: 4600 },
  { text: "  [14:24:12] Task completed", delay: 6200 },
  { text: "", delay: 6400 },
  { text: "  === Weekly Ops Review ===", delay: 6600 },
  { text: "", delay: 6700 },
  { text: "  Executive Summary:", delay: 6800 },
  { text: "  Traffic volume increased 12% week-over-week across all endpoints.", delay: 6900 },
  { text: "  Two anomalies detected in /api/payments (elevated 5xx rate, p99", delay: 7000 },
  { text: "  latency spike during Thursday 14:00-16:00 window).", delay: 7100 },
  { text: "", delay: 7200 },
  { text: "  Anomalies:", delay: 7300 },
  { text: "    [HIGH] 5xx_rate /api/payments: 3.2% → 8.1% (+153%)", delay: 7400 },
  { text: "    [MEDIUM] p99_latency /api/payments: 800ms → 1240ms", delay: 7500 },
  { text: "", delay: 7600 },
  { text: "  Improvements:", delay: 7700 },
  { text: "    - /api/users error rate decreased 0.8% → 0.3%", delay: 7800 },
  { text: "    - Overall p50 latency down 8% across all endpoints", delay: 7900 },
  { text: "", delay: 8000 },
  { text: "  Follow-up:", delay: 8100 },
  { text: "    - Investigate /api/payments 5xx spike (correlate with deploy log)", delay: 8200 },
  { text: "    - Review connection pool settings for payments service", delay: 8300 },
  { text: "", delay: 8400 },
  { text: "Task completed.", delay: 8500 },
];

function TerminalRecording() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  const [visibleLines, setVisibleLines] = useState<{ text: string; dim?: boolean }[]>([]);
  const [typingText, setTypingText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [finished, setFinished] = useState(false);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const [runKey, setRunKey] = useState(0);

  const startAnimation = () => {
    setVisibleLines([]);
    setTypingText("");
    setIsTyping(false);
    setFinished(false);
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];

    lines.forEach((line, i) => {
      const timer = setTimeout(() => {
        if (i === 0 && line.typing) {
          setIsTyping(true);
          let charIdx = 0;
          const typeInterval = setInterval(() => {
            charIdx++;
            setTypingText(line.text.slice(0, charIdx));
            if (charIdx >= line.text.length) {
              clearInterval(typeInterval);
              setIsTyping(false);
              setVisibleLines((prev) => [...prev, { text: line.text, dim: line.dim }]);
              setTypingText("");
            }
          }, 40);
        } else if (!line.typing) {
          setVisibleLines((prev) => [...prev, { text: line.text, dim: line.dim }]);
        }
        if (i === lines.length - 1) {
          setFinished(true);
        }
      }, line.delay);
      timersRef.current.push(timer);
    });
  };

  useEffect(() => {
    if (!isInView) return;
    startAnimation();
    return () => {
      timersRef.current.forEach(clearTimeout);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isInView, runKey]);

  return (
    <section ref={ref} className="mx-auto max-w-4xl px-6 py-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        className="mb-8 text-center"
      >
        <h2 className="mb-4 text-3xl font-bold text-gray-100 sm:text-4xl">
          See it run
        </h2>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="overflow-hidden rounded-xl border border-surface-border shadow-2xl"
      >
        {/* Title bar */}
        <div className="flex items-center gap-2 border-b border-surface-border bg-surface-light px-4 py-2.5">
          <div className="flex gap-1.5">
            <div className="h-3 w-3 rounded-full bg-red-500/70" />
            <div className="h-3 w-3 rounded-full bg-yellow-500/70" />
            <div className="h-3 w-3 rounded-full bg-green-500/70" />
          </div>
          <span className="ml-3 text-xs font-mono text-gray-500">~</span>
        </div>

        {/* Terminal content */}
        <div className="min-h-[420px] bg-[#0c0c14] p-4 font-mono text-sm leading-relaxed">
          {isTyping && (
            <div className="text-gray-200">
              {typingText}
              <span className="animate-pulse text-cyan-400">▌</span>
            </div>
          )}
          {visibleLines.map((line, i) => (
            <div
              key={i}
              className={line.dim ? "text-gray-600" : "text-gray-300"}
            >
              {line.text || " "}
            </div>
          ))}
          {visibleLines.length > 0 && visibleLines.length < lines.length && !isTyping && (
            <span className="animate-pulse text-cyan-400">▌</span>
          )}
          {finished && (
            <button
              onClick={() => setRunKey((k) => k + 1)}
              className="mt-4 flex items-center gap-1.5 rounded border border-cyan-500/30 px-2.5 py-1 text-xs text-cyan-400 transition-colors hover:border-cyan-500/60 hover:text-cyan-300"
            >
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M1 8a7 7 0 0 1 13-3.5M15 8a7 7 0 0 1-13 3.5" />
                <path d="M14 1v4h-4M2 15v-4h4" />
              </svg>
              Replay
            </button>
          )}
        </div>
      </motion.div>
    </section>
  );
}

export default TerminalRecording;
