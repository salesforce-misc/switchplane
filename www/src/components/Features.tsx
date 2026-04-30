import { motion } from "framer-motion";

interface Feature {
  title: string;
  description: string;
  icon: JSX.Element;
}

const features: Feature[] = [
  {
    title: "Cross-task coordination",
    description:
      "Tasks can spawn child tasks, wait for completion, and send notifications to sibling tasks. Build multi-agent workflows where tasks collaborate through a structured messaging system.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <circle cx="6" cy="6" r="3" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="18" cy="6" r="3" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="12" cy="18" r="3" stroke="currentColor" strokeWidth="1.5" />
        <path d="M8.5 7.5L10.5 16M15.5 7.5L13.5 16M9 6h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    title: "CLI and TUI",
    description:
      "Every app is a standalone CLI with streaming event output. A full-screen terminal UI with tab-based task navigation launches automatically for interactive sessions.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <rect x="2" y="4" width="20" height="16" rx="3" stroke="currentColor" strokeWidth="1.5" />
        <path d="M6 9l3 3-3 3M12 15h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    title: "Subprocess isolation",
    description:
      "User code never runs in the control plane. Each agent is a separate process communicating over Unix socketpairs. Crash one agent, the rest keep running.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <rect x="3" y="3" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    ),
  },
  {
    title: "Checkpoint and resume",
    description:
      "LangGraph-native checkpointing backed by SQLite. Interrupt any task and resume from exactly where it left off. State is saved after each graph node.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <path d="M12 2v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M12 8l4 4-4 4-4-4z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
        <path d="M12 16v6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M5 11H2M22 11h-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    title: "LangGraph StateGraphs",
    description:
      "Tasks are defined as LangGraph StateGraph graphs. No proprietary workflow abstraction. Use the full power of LangGraph's nodes, edges, and conditional branching.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <circle cx="5" cy="12" r="2.5" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="19" cy="6" r="2.5" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="19" cy="18" r="2.5" stroke="currentColor" strokeWidth="1.5" />
        <path d="M7.5 11L16.5 7M7.5 13L16.5 17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    title: "MCP integration",
    description:
      "Connect to Model Context Protocol servers for tool access. Declare which servers each task needs. Supports both stdio and HTTP transports.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <path d="M8 12h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <rect x="2" y="8" width="6" height="8" rx="2" stroke="currentColor" strokeWidth="1.5" />
        <rect x="16" y="8" width="6" height="8" rx="2" stroke="currentColor" strokeWidth="1.5" />
        <path d="M5 8V5a2 2 0 012-2h10a2 2 0 012 2v3M5 16v3a2 2 0 002 2h10a2 2 0 002-2v-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    title: "Explicit boundaries",
    description:
      "Each task operates with its own scoped context. No implicit shared state, no bleeding conversation history between agents. Communication is structured, typed message-passing. You can test each task in complete isolation because it genuinely is isolated.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 2" />
        <rect x="7" y="7" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="12" cy="12" r="2" fill="currentColor" />
      </svg>
    ),
  },
];

const cardVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.5, ease: "easeOut" },
  }),
};

function Features() {
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
          Features
        </h2>
      </motion.div>

      <div className="flex flex-wrap justify-center gap-6">
        {features.map((f, i) => (
          <motion.div
            key={f.title}
            custom={i}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-40px" }}
            variants={cardVariants}
            className="group w-full rounded-xl border border-surface-border bg-surface-light/40 p-6 transition-colors hover:border-cyan-500/30 sm:w-[calc(50%-0.75rem)] lg:w-[calc(33.333%-1rem)]"
          >
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500/10 text-cyan-400 transition-colors group-hover:bg-cyan-500/20">
                {f.icon}
              </div>
              <h3 className="text-lg font-semibold text-gray-200">
                {f.title}
              </h3>
            </div>
            <p className="leading-relaxed text-gray-400">
              {f.description}
            </p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

export default Features;
