# TODO

## High Priority

- [ ] Add GitHub Actions CI pipeline (run `make test` + linter on PR/push)
- [ ] Add linting/formatting config (ruff, mypy)
- [ ] Add docstrings to `Task` class, `@command` decorator, and incomplete `AgentContext` methods (`emit`, `config`, `is_cancelled`, `log`, `command_result`)
- [ ] Create custom exception hierarchy (`SwitchplaneError` base + `ConfigError`, `TaskExecutionError`, `McpError`)
- [ ] Add CONTRIBUTING.md (dev setup, running tests, PR expectations)
- [ ] Add protocol version field to all IPC message types (`CliRequest`, `CliResponse`, `AgentEvent`, `AgentCommand`) — must happen before v1 or adding it later is a breaking change
- [ ] Fix checkpoint.py correctness bugs:
  - `alist()` accepts `config: RunnableConfig | None` but dereferences `config["configurable"]` without None check
  - Metadata deserialization may use wrong type tag (checkpoint type vs metadata type) — silent data corruption risk
  - `aput_writes()` has no transaction wrapper — partial writes on interruption
- [ ] Fix error response ID in transport.py — `CliResponse(id="error", ...)` hardcodes `"error"` instead of echoing request ID, breaking request-response correlation
- [ ] Complete `__init__.py` public API — missing re-exports: `AgentSpec`, `OAuthConfig`, `McpServerConfig`, `TaskStatus`, `AgentStatus`
- [ ] Add pyproject.toml URL metadata (`homepage`, `repository`, `documentation`)
- [ ] Cap dependency major versions (`click>=8.3,<9`, `pydantic>=2.0,<3`, `langgraph>=1.0,<2`, etc.)

## Medium Priority

- [ ] Fix PID file TOCTOU race condition — use file locking instead of read-then-signal
- [ ] Sanitize agent subprocess environment — whitelist only necessary env vars instead of inheriting all
- [ ] Warn on startup if runtime directory or config file has overly permissive permissions
- [ ] Add TUI smoke tests (tab operations, event rendering, scrolling) — currently 0% coverage on largest module
- [ ] Add `py.typed` marker file for downstream type checker support
- [ ] Add CHANGELOG.md
- [ ] Fix resource leaks on error paths:
  - `subprocess_manager.py`: log file handle and `cp_sock` leak if `create_subprocess_exec()` fails
  - `agent_runtime.py`: `NameError` on `_writer` if `asyncio.open_connection()` fails, socket leaks
  - `mcp.py`: `httpx.AsyncClient` created outside context manager, leaks on early exception
  - `persistence.py`: DB connection in unknown state if PRAGMA/CREATE TABLE fails after connect
- [ ] Fix silent command listener death in agent_runtime.py — single malformed JSON message kills the listener loop, making the agent uncancellable. Should log and continue.
- [ ] Wrap event_callback in subprocess_manager.py in try/except — unhandled callback exception kills the event reader task, silently losing all subsequent events
- [ ] Improve discovery error reporting — import failures and bad task classes are silently swallowed as warnings. Accumulate errors and surface them in CLI output (optionally raise in strict mode).
- [ ] Add oauth.py token refresh retry limit — if refresh fails and server returns 401, it retries with no bound
- [ ] Validate `_util.py` zero-length frames — `readexactly(0)` returns empty bytes without blocking; reject `length == 0`
- [ ] Fix `fmt.py` timestamp parsing — `raw_ts.split("T")[1]` crashes with `IndexError` on non-ISO input. Use `datetime.fromisoformat()` with fallback.
- [ ] Fix `tui.py:574` — lowercase `any` type annotation should be `Any`
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
- [ ] CODE_OF_CONDUCT.md
- [ ] Python version test matrix in CI (3.12, 3.13, 3.14)
- [ ] Coverage thresholds in CI
- [ ] Convert `mode` from magic string to enum/Literal
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
