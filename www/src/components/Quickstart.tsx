import { motion } from "framer-motion";

const taskCode = `from langgraph.graph import END, START, StateGraph
from switchplane import Task
from switchplane.agent_runtime import AgentContext

class ReviewTask(Task):
    name = "review"
    description = "Weekly ops review"

    async def run(self, ctx: AgentContext) -> None:
        llm = build_llm(ctx.config["llm"]["model"], ...)

        graph = StateGraph(ReviewState)
        graph.add_node("fetch_metrics", fetch_metrics)
        graph.add_node("analyze_metrics", analyze_metrics)
        graph.add_node("summarize", summarize)      # LLM call
        graph.add_node("compile_report", compile_report)

        graph.add_edge(START, "fetch_metrics")
        graph.add_edge("fetch_metrics", "analyze_metrics")
        graph.add_edge("analyze_metrics", "summarize")
        graph.add_edge("summarize", "compile_report")
        graph.add_edge("compile_report", END)

        result = await graph.compile().ainvoke(initial_state)
        ctx.complete(result["report"])`;

const cliOutput = `$ pip install switchplane
$ devops run sre review

▶ Starting ops review (model: claude-sonnet-4-20250514)
▶ Generating mock NewRelic metrics (2 weeks, 5-min granularity)...
▶ Computing week-over-week statistics...
▶ Summarizing findings (1 LLM call)...
▶ Compiling report...
✓ Task completed

=== Weekly Ops Review ===

Executive Summary:
Traffic volume increased 12% week-over-week across all endpoints.
Two anomalies detected in /api/payments (elevated 5xx rate, p99
latency spike during Thursday 14:00-16:00 window).

Anomalies:
  [HIGH] 5xx_rate /api/payments: 3.2% → 8.1% (+153%)
  [MEDIUM] p99_latency /api/payments: 800ms → 1240ms`;

function Quickstart() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        className="mb-14 text-center"
      >
        <h2 className="mb-4 text-3xl font-bold text-gray-100 sm:text-4xl">
          What you get
        </h2>
        <p className="mx-auto max-w-2xl text-gray-400">
          Define a task as a LangGraph StateGraph. Switchplane handles the rest:
          daemonization, subprocess isolation, IPC, persistence, and a CLI to run it.
        </p>
      </motion.div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Task definition */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="overflow-hidden rounded-xl border border-surface-border bg-surface-light/60"
        >
          <div className="flex items-center gap-2 border-b border-surface-border px-4 py-2">
            <div className="h-2.5 w-2.5 rounded-full bg-cyan-500/50" />
            <span className="text-xs font-mono text-gray-500">tasks/review.py</span>
          </div>
          <pre className="overflow-x-auto p-4 text-xs leading-relaxed text-gray-300 font-mono">
            <code>{taskCode}</code>
          </pre>
        </motion.div>

        {/* CLI output */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="overflow-hidden rounded-xl border border-surface-border bg-surface-light/60"
        >
          <div className="flex items-center gap-2 border-b border-surface-border px-4 py-2">
            <div className="h-2.5 w-2.5 rounded-full bg-amber-500/50" />
            <span className="text-xs font-mono text-gray-500">terminal</span>
          </div>
          <pre className="overflow-x-auto p-4 text-xs leading-relaxed text-gray-300 font-mono">
            <code>{cliOutput}</code>
          </pre>
        </motion.div>
      </div>
    </section>
  );
}

export default Quickstart;
