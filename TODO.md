# TODO

## High Priority

- [ ] Create custom exception hierarchy (`SwitchplaneError` base + `ConfigError`, `TaskExecutionError`, `McpError`)
- [ ] Add protocol version field to all IPC message types (`CliRequest`, `CliResponse`, `AgentEvent`, `AgentCommand`) — must happen before v1 or adding it later is a breaking change
- [ ] Cap dependency major versions (`click>=8.3,<9`, `pydantic>=2.0,<3`, `langgraph>=1.0,<2`, etc.)

## Medium Priority

- [ ] Fix PID file TOCTOU race condition — use file locking instead of read-then-signal
- [ ] Sanitize agent subprocess environment — whitelist only necessary env vars instead of inheriting all
- [ ] Warn on startup if runtime directory or config file has overly permissive permissions
- [ ] Add `py.typed` marker file for downstream type checker support
- [ ] Add CHANGELOG.md
- [ ] Improve discovery error reporting — import failures and bad task classes are silently swallowed as warnings. Accumulate errors and surface them in CLI output (optionally raise in strict mode).
- [ ] Add oauth.py token refresh retry limit — if refresh fails and server returns 401, it retries with no bound
- [ ] Validate `_util.py` zero-length frames — `readexactly(0)` returns empty bytes without blocking; reject `length == 0`
- [ ] Add defensive checks for daemon response structure in TUI (`check.result["task"]` assumes shape without validation)
- [ ] Validate config DEFAULT_MODEL against llm.py MODELS registry at load time

## Examples

Candidate examples that demonstrate the Switchplane thesis (deterministic code for deterministic problems, LLM only for judgment).

- [ ] **Compliance auditor** — Scan a codebase for license compliance issues. Deterministic: extract dependencies and licenses (`pip licenses`, `package.json`), cross-reference against a policy file (no GPL in commercial, AGPL requires legal review). LLM: explain business risk of each violation in plain English for legal. High stakes — non-deterministic compliance checking is a liability.
- [ ] **Incident postmortem generator** — Feed in PagerDuty timeline, deploy logs (git log between tags), and metrics from the incident window. Deterministic: stitch into chronological timeline, correlate deploys with incident start (git diff + file-to-service mapping). LLM: write the postmortem narrative. Sells to management — postmortems are universally hated busywork.
- [ ] **Cost analyzer** — Analyze LLM API spend. Deterministic: compute cost per workflow, cost per call, tokens wasted on non-judgment tasks, trend lines, projected monthly spend. LLM: generate optimization recommendations. Recursively proves the thesis — the report literally shows money spent asking LLMs to do things code could do.
- [ ] **PR review triager** — Poll GitHub for open PRs. Deterministic: files changed, lines added/removed, packages touched, test coverage, CI status, time open, review comments. Apply rules (auth/ changes → security review, >500 lines + no tests → flagged). LLM: one-paragraph summary of what the PR does.
- [ ] **Database migration risk assessor** — Parse a SQL migration file. Deterministic: extract operations (ALTER, ADD INDEX, DROP COLUMN), apply known risk rules (index on >1M row table → lock, column drop → irreversible). LLM: generate plain-English migration plan with rollback strategy and estimated downtime.

## Nice to Have

- [ ] Per-example README files in `examples/`
- [ ] Coverage thresholds in CI
- [ ] Socket peer credential verification (`SO_PEERCRED` / `LOCAL_PEERCRED`)
- [ ] Rate limiting on socket connections
- [ ] Audit logging
- [ ] Pin example dependencies in `examples/*/pyproject.toml`
- [ ] Make `MAX_MESSAGE_SIZE` configurable (currently hardcoded at 64 MB in `_util.py`)
- [ ] Add task status transition validation in persistence.py (currently accepts any transition, e.g. completed → running)
- [ ] Add `ON DELETE CASCADE` to events FK in persistence.py (orphaned events if tasks deleted outside `clear_terminal_tasks()`)
- [ ] Shell.py hardening for non-local use cases: symlink escape in `validate_path()`, command allowlist resolution, env var filtering
- [ ] Refactor daemon.py signal handler from tuple-expression lambda to proper function
- [ ] Verify `os.setsid()` succeeds in daemon.py double-fork
