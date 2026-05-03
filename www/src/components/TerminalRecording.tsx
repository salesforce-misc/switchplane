import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";

interface Tab {
  slot: number;
  label: string;
  status?: "pending" | "running" | "completed";
  focused: boolean;
}

interface EventLine {
  timestamp?: string;
  text: string;
  style: "info" | "dim" | "progress" | "success" | "error" | "warn" | "result" | "header";
}

const STATUS_ICON: Record<string, [string, string]> = {
  pending:   ["○", "text-[#666688]"],
  running:   ["●", "text-[#00ff88]"],
  completed: ["✓", "text-[#00ff88]"],
};

const STYLE_CLASS: Record<EventLine["style"], string> = {
  info:     "text-[#aaaacc]",
  dim:      "text-[#888899]",
  progress: "text-[#aaaacc]",
  success:  "text-[#00ff88]",
  error:    "text-[#ff5555] font-bold",
  warn:     "text-[#ffaa00]",
  result:   "text-gray-300",
  header:   "text-gray-200 font-bold",
};

function TerminalRecording() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const paneRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(sectionRef, { once: true, margin: "-100px" });
  const [runKey, setRunKey] = useState(0);

  const [tabs, setTabs] = useState<Tab[]>([
    { slot: 0, label: "system", focused: true },
  ]);
  const [focusedSlot, setFocusedSlot] = useState(0);
  const [events, setEvents] = useState<Record<number, EventLine[]>>({ 0: [] });
  const [inputText, setInputText] = useState("");
  const [prompt, setPrompt] = useState("system");
  const [statusBar, setStatusBar] = useState(
    "Tab switch · :help for commands · Ctrl+C quit",
  );
  const [finished, setFinished] = useState(false);

  const visibleEvents = events[focusedSlot] || [];

  useEffect(() => {
    const el = paneRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [visibleEvents.length]);

  useEffect(() => {
    if (!isInView) return;

    const ctrl = new AbortController();
    const { signal } = ctrl;

    const wait = (ms: number) =>
      new Promise<void>((resolve, reject) => {
        const id = setTimeout(resolve, ms);
        signal.addEventListener(
          "abort",
          () => {
            clearTimeout(id);
            reject();
          },
          { once: true },
        );
      });

    const typeInto = (text: string) =>
      new Promise<void>((resolve, reject) => {
        let i = 0;
        const iv = setInterval(() => {
          if (signal.aborted) {
            clearInterval(iv);
            reject();
            return;
          }
          setInputText(text.slice(0, ++i));
          if (i >= text.length) {
            clearInterval(iv);
            resolve();
          }
        }, 40);
        signal.addEventListener("abort", () => clearInterval(iv), {
          once: true,
        });
      });

    const emit = (slot: number, line: EventLine) =>
      setEvents((prev) => ({
        ...prev,
        [slot]: [...(prev[slot] || []), line],
      }));

    const setTabStatus = (slot: number, status: NonNullable<Tab["status"]>) =>
      setTabs((prev) =>
        prev.map((t) => (t.slot === slot ? { ...t, status } : t)),
      );

    const focus = (slot: number) => {
      setFocusedSlot(slot);
      setTabs((prev) =>
        prev.map((t) => ({ ...t, focused: t.slot === slot })),
      );
    };

    (async () => {
      try {
        // Reset for replay
        setTabs([{ slot: 0, label: "system", focused: true }]);
        setEvents({ 0: [] });
        setFocusedSlot(0);
        setInputText("");
        setPrompt("system");
        setStatusBar(
          "Tab switch · :help for commands · Ctrl+C quit",
        );
        setFinished(false);

        await wait(600);
        await typeInto(":run sre review");
        await wait(400);

        // Submit
        setInputText("");
        await wait(200);
        emit(0, {
          timestamp: "14:23:44",
          text: "Submitted task: sre/review → abc1234",
          style: "info",
        });
        await wait(400);

        // Task tab appears
        setTabs((prev) => [
          ...prev,
          { slot: 1, label: "sre/review", status: "pending", focused: false },
        ]);
        setEvents((prev) => ({ ...prev, 1: [] }));
        await wait(500);

        // Auto-focus task tab
        focus(1);
        setPrompt("sre/review");
        await wait(200);
        setTabStatus(1, "running");
        setStatusBar(
          "sre/review [running] abc1234   Tab switch · /cmd task · Ctrl+C quit",
        );

        // Stream events
        await wait(400);
        emit(1, {
          timestamp: "14:23:45",
          text: "Task started",
          style: "info",
        });
        await wait(800);
        emit(1, {
          timestamp: "14:23:46",
          text: "Starting ops review (model: claude-sonnet-4-20250514)",
          style: "progress",
        });
        await wait(800);
        emit(1, {
          timestamp: "14:23:47",
          text: "Generating mock NewRelic metrics (2 weeks, 5-min granularity)...",
          style: "progress",
        });
        await wait(1400);
        emit(1, {
          timestamp: "14:24:01",
          text: "Computing week-over-week statistics...",
          style: "progress",
        });
        await wait(800);
        emit(1, {
          timestamp: "14:24:02",
          text: "Summarizing findings (1 LLM call)...",
          style: "progress",
        });
        await wait(1600);
        emit(1, {
          timestamp: "14:24:12",
          text: "Task completed",
          style: "success",
        });

        // Terminal state
        await wait(200);
        setTabStatus(1, "completed");
        setStatusBar(
          "sre/review [completed] abc1234   Tab switch · Ctrl+C quit",
        );

        // Result block
        await wait(300);
        const results: [string, EventLine["style"]][] = [
          ["", "info"],
          ["=== Weekly Ops Review ===", "header"],
          ["", "info"],
          ["Executive Summary:", "header"],
          [
            "Traffic volume increased 12% week-over-week across all endpoints.",
            "result",
          ],
          [
            "Two anomalies detected in /api/payments (elevated 5xx rate, p99",
            "result",
          ],
          ["latency spike during Thursday 14:00-16:00 window).", "result"],
          ["", "info"],
          ["Anomalies:", "header"],
          [
            "  [HIGH] 5xx_rate /api/payments: 3.2% → 8.1% (+153%)",
            "error",
          ],
          [
            "  [MEDIUM] p99_latency /api/payments: 800ms → 1240ms",
            "warn",
          ],
          ["", "info"],
          ["Improvements:", "header"],
          [
            "  - /api/users error rate decreased 0.8% → 0.3%",
            "success",
          ],
          [
            "  - Overall p50 latency down 8% across all endpoints",
            "success",
          ],
          ["", "info"],
          ["Follow-up:", "header"],
          [
            "  - Investigate /api/payments 5xx spike (correlate with deploy log)",
            "result",
          ],
          [
            "  - Review connection pool settings for payments service",
            "result",
          ],
        ];

        for (const [text, style] of results) {
          emit(1, { text, style });
          await wait(80);
        }

        await wait(500);
        setFinished(true);
      } catch {
        // aborted (cleanup or replay)
      }
    })();

    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isInView, runKey]);

  return (
    <section ref={sectionRef} className="mx-auto max-w-4xl px-6 py-20">
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
        {/* macOS title bar */}
        <div className="flex items-center gap-2 border-b border-surface-border bg-surface-light px-4 py-2.5">
          <div className="flex gap-1.5">
            <div className="h-3 w-3 rounded-full bg-red-500/70" />
            <div className="h-3 w-3 rounded-full bg-yellow-500/70" />
            <div className="h-3 w-3 rounded-full bg-green-500/70" />
          </div>
          <span className="ml-3 font-mono text-xs text-gray-500">devops</span>
        </div>

        {/* Tab bar */}
        <div className="flex items-center gap-4 border-b border-surface-border bg-[#1a1a2e] px-4 py-1.5 font-mono text-xs">
          {tabs.map((tab) => (
            <span
              key={tab.slot}
              className={
                tab.focused
                  ? "font-bold text-[#00ff88]"
                  : "text-[#666688]"
              }
            >
              [{tab.slot}] {tab.label}
              {tab.status && (
                <span className={`ml-1.5 ${STATUS_ICON[tab.status][1]}`}>
                  {STATUS_ICON[tab.status][0]}
                </span>
              )}
            </span>
          ))}
        </div>

        {/* Event pane */}
        <div
          ref={paneRef}
          className="h-[360px] overflow-y-auto bg-[#0c0c14] px-4 py-3 font-mono text-sm leading-relaxed"
        >
          {visibleEvents.map((line, i) => (
            <div key={i} className="flex whitespace-pre">
              {line.timestamp ? (
                <>
                  <span className="shrink-0 select-none text-[#555577]">
                    [{line.timestamp}]{" "}
                  </span>
                  <span className={STYLE_CLASS[line.style]}>{line.text}</span>
                </>
              ) : (
                <span className={STYLE_CLASS[line.style]}>
                  {line.text || " "}
                </span>
              )}
            </div>
          ))}
          {finished && (
            <button
              aria-label="Replay animation"
              onClick={() => setRunKey((k) => k + 1)}
              className="mt-4 flex items-center gap-1.5 rounded border border-cyan-500/30 px-2.5 py-1 text-xs text-cyan-400 transition-colors hover:border-cyan-500/60 hover:text-cyan-300"
            >
              <svg
                aria-hidden="true"
                width="12"
                height="12"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M1 8a7 7 0 0 1 13-3.5M15 8a7 7 0 0 1-13 3.5" />
                <path d="M14 1v4h-4M2 15v-4h4" />
              </svg>
              Replay
            </button>
          )}
        </div>

        {/* Status bar */}
        <div className="truncate border-t border-surface-border bg-[#2d2d4e] px-4 py-1 font-mono text-xs text-[#888899]">
          {statusBar}
        </div>

        {/* Input bar */}
        <div className="flex items-center border-t border-surface-border bg-[#0c0c14] px-4 py-1.5 font-mono text-sm">
          <span className="font-bold text-[#aaaaff]">[{prompt}]</span>
          <span className="mx-1 text-[#666688]">&gt;</span>
          <span className="text-gray-200">{inputText}</span>
          <span className="animate-pulse text-cyan-400">&#x2588;</span>
        </div>
      </motion.div>
    </section>
  );
}

export default TerminalRecording;
