import { motion } from "framer-motion";

function Hero() {
  return (
    <section className="relative flex min-h-[60vh] flex-col items-center justify-center overflow-hidden px-6 py-24">
      {/* Subtle gradient backdrop */}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-cyan-500/5 via-transparent to-transparent" />

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="relative z-10 text-center"
      >
        <h1 className="mb-6 text-6xl font-bold tracking-tight sm:text-7xl">
          <span className="bg-gradient-to-r from-cyan-400 to-cyan-600 bg-clip-text text-transparent">
            Switchplane
          </span>
        </h1>

        <p className="mx-auto max-w-2xl text-xl leading-relaxed text-gray-400 sm:text-2xl">
          Deterministic control over agent execution.{" "}
          <span className="text-gray-300">
            The graph defines operational boundaries. LLMs operate within them.
          </span>
        </p>

        <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-gray-500">
          LangGraph provides the graph primitives. Switchplane makes them operational:
          process isolation, bidirectional IPC, lifecycle management, and a real control plane.
        </p>

        <p className="mx-auto mt-4 max-w-xl text-sm leading-relaxed text-gray-500">
          For teams building long-running ops automation, internal tooling, and infrastructure
          agents that need to run locally without cloud dependencies.
        </p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="mt-10 flex items-center justify-center gap-2 text-gray-500"
        >
          <span className="inline-block h-px w-8 bg-gray-700" />
          <span className="font-mono">Local-first runtime for deterministic, LLM-augmented task execution</span>
          <span className="inline-block h-px w-8 bg-gray-700" />
        </motion.div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7, duration: 0.8 }}
          className="mt-4 flex items-center justify-center gap-3 text-xs text-gray-500"
        >
          <span className="rounded border border-gray-700 px-2 py-0.5 font-mono">LangGraph-native</span>
          <span className="rounded border border-gray-700 px-2 py-0.5 font-mono">Python 3.12+</span>
          <a href="https://github.com/salesforce-misc/switchplane" target="_blank" rel="noopener noreferrer" className="rounded border border-gray-700 px-2 py-0.5 font-mono transition-colors hover:border-cyan-500/50 hover:text-cyan-400">GitHub</a>
          <a href="https://pypi.org/project/switchplane/" target="_blank" rel="noopener noreferrer" className="rounded border border-gray-700 px-2 py-0.5 font-mono transition-colors hover:border-cyan-500/50 hover:text-cyan-400">PyPI</a>
        </motion.div>
      </motion.div>

      {/* Scroll hint */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.4 }}
        transition={{ delay: 1.2, duration: 0.8 }}
        className="absolute bottom-8"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          className="flex flex-col items-center gap-1 text-xs text-gray-500"
        >
          <span>scroll</span>
          <svg
            aria-hidden="true"
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            className="text-gray-500"
          >
            <path
              d="M4 6l4 4 4-4"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </motion.div>
      </motion.div>
    </section>
  );
}

export default Hero;
