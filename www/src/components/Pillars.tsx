import { motion } from "framer-motion";

interface Pillar {
  title: string;
  description: string;
  icon: JSX.Element;
}

const pillars: Pillar[] = [
  {
    title: "Deterministic topology",
    description:
      "The graph structure is fixed: what runs, when, and in what order is defined at build time. Within that topology, individual nodes can be as non-deterministic as you need. A node might be a simple function, a full LLM reasoning loop, or a multi-model pipeline. You control where non-determinism lives.",
    icon: (
      <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <path d="M4 4h6v6H4zM14 4h6v6h-6zM9 14h6v6H9z" stroke="currentColor" strokeWidth="1.5" rx="1" />
        <path d="M7 10v1.5a2 2 0 002 2h0M17 10v1.5a2 2 0 01-2 2h0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    title: "Cost",
    description:
      "Reserve expensive model calls for decisions requiring judgment. Deterministic tasks execute as code: microsecond performance, zero marginal cost, no token budget wasted on work a for-loop can do.",
    icon: (
      <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <path d="M12 2v4m0 12v4M2 12h4m12 0h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <circle cx="12" cy="12" r="6" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="12" cy="12" r="2" fill="currentColor" />
      </svg>
    ),
  },
  {
    title: "Auditability and testability",
    description:
      "Agent workflows are observable and testable like standard software. Visible logs, unit-testable nodes, clear failure points. No opaque prompt-based black boxes you can only evaluate by running end-to-end.",
    icon: (
      <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" strokeWidth="1.5" />
        <path d="M8 12l2.5 2.5L16 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    title: "Vendor independence",
    description:
      "Swap between commercial APIs, open-source models, and local execution without rewriting pipeline logic. Different nodes can use different models. Model routing is configuration, not architecture. No provider lock-in baked into your workflow definitions.",
    icon: (
      <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" className="h-6 w-6">
        <rect x="2" y="4" width="8" height="7" rx="2" stroke="currentColor" strokeWidth="1.5" />
        <rect x="14" y="4" width="8" height="7" rx="2" stroke="currentColor" strokeWidth="1.5" />
        <rect x="8" y="14" width="8" height="7" rx="2" stroke="currentColor" strokeWidth="1.5" />
        <path d="M6 11v3h6m6-3v3h-6m0 0v0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
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

function Pillars() {
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
          Design principles
        </h2>
        <p className="mx-auto max-w-lg text-gray-400">
          Four ideas that shape every decision in the codebase.
        </p>
      </motion.div>

      <div className="grid gap-6 sm:grid-cols-2">
        {pillars.map((p, i) => (
          <motion.div
            key={p.title}
            custom={i}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-40px" }}
            variants={cardVariants}
            className="group rounded-xl border border-surface-border bg-surface-light/40 p-6 transition-colors hover:border-cyan-500/30"
          >
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500/10 text-cyan-400 transition-colors group-hover:bg-cyan-500/20">
                {p.icon}
              </div>
              <h3 className="text-lg font-semibold text-gray-200">
                {p.title}
              </h3>
            </div>
            <p className="leading-relaxed text-gray-400">
              {p.description}
            </p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

export default Pillars;
