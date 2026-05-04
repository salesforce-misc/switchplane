import { motion } from "framer-motion";

function LocalFirst() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        className="rounded-xl border border-cyan-500/20 bg-gradient-to-br from-cyan-500/5 to-transparent p-8 sm:p-12"
      >
        <h2 className="mb-6 text-2xl font-bold text-gray-100 sm:text-3xl">
          Local-first. No cloud required.
        </h2>

        <div className="grid gap-8 sm:grid-cols-2">
          <div>
            <p className="leading-relaxed text-gray-400">
              Most agent frameworks push you toward cloud infrastructure before you've written your first task.
              Switchplane runs as a local daemon on your machine. Unix sockets for IPC. SQLite
              for persistence. No external dependencies, no API gateway, no deployment pipeline
              to get started.
            </p>
            <p className="mt-4 leading-relaxed text-gray-400">
              For teams doing ops automation, working with sensitive infrastructure, or building
              internal tooling that can't phone home, this isn't a nice-to-have. It's the whole point.
            </p>
          </div>

          <div className="space-y-4">
            <div className="rounded-lg border border-surface-border bg-surface-light/40 p-4">
              <h3 className="mb-1 text-sm font-semibold text-cyan-400">Your machine, your data</h3>
              <p className="text-sm text-gray-400">
                State lives in ~/.myapp/state.db. Logs stay local. No telemetry, no cloud sync.
              </p>
            </div>
            <div className="rounded-lg border border-surface-border bg-surface-light/40 p-4">
              <h3 className="mb-1 text-sm font-semibold text-cyan-400">Zero infrastructure</h3>
              <p className="text-sm text-gray-400">
                No containers, no orchestrators, no managed services. pip install and run.
              </p>
            </div>
            <div className="rounded-lg border border-surface-border bg-surface-light/40 p-4">
              <h3 className="mb-1 text-sm font-semibold text-cyan-400">Works offline</h3>
              <p className="text-sm text-gray-400">
                Deterministic nodes run without network access. Only LLM nodes need an API key.
              </p>
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  );
}

export default LocalFirst;
