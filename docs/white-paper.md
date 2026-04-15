# Switchplane: A Runtime Control Plane for Agent-Based Task Execution

**Demian Brecht** — Salesforce, Inc.

---

## Abstract

LLM-powered agent systems have a supervision problem. The models themselves are non-deterministic — that's the feature you're paying for — but the infrastructure around them should not be. Task lifecycle, process isolation, event persistence, cancellation semantics, and operational visibility are deterministic problems with known solutions. Mixing them into the same abstraction layer as LLM reasoning is where systems get fragile.

Switchplane is a Python runtime control plane that separates these concerns. It provides a daemonized supervisor, process-isolated agent subprocesses, durable task state in SQLite, bidirectional IPC for mid-flight interaction, and an auto-generated CLI — all without touching the agent's domain logic. Tasks are defined as LangGraph StateGraph graphs. Each application built with Switchplane becomes a standalone command-line tool with its own isolated runtime.

This paper describes the design principles, architecture, and implementation of Switchplane.

---

## 1. The Problem

Most agent frameworks conflate two distinct responsibilities: *what the agent does* and *how the agent runs*. Prompt construction, tool selection, and reasoning strategy live in the same codebase — often in the same abstraction — as task scheduling, state persistence, error recovery, and process lifecycle. The result is that operational concerns are coupled to model behavior, and neither can be reasoned about independently.

This matters because the two have fundamentally different correctness criteria.

Agent reasoning is non-deterministic by design. Given the same input, an LLM may choose different tools, produce different output, or follow a different reasoning path. That's acceptable — it's the source of the system's value. But the infrastructure around it should exhibit *deterministic properties*: guarantees about behavior that hold regardless of the execution path.

There is a useful distinction here. **Deterministic logic** means the same input always produces the same output. **Deterministic properties** are invariants that hold regardless of the execution path: "every state transition is recorded," "a cancelled task will not continue executing," "a failed task can be resumed from its last checkpoint." These are the properties that make systems operable, debuggable, and trustworthy.

When infrastructure logic is entangled with agent logic, these properties are difficult to guarantee. A framework that asks an LLM to manage its own task state, or that relies on in-process execution for isolation, is choosing convenience over correctness.

Switchplane takes a different position: **if it's deterministic, write it in code; if it requires judgment, call the LLM.**

---

## 2. Design Principles

### 2.1 The Runtime Owns the Runtime

The control plane manages task lifecycle, event persistence, process supervision, and IPC. Agents manage reasoning, tool use, and domain logic. These responsibilities never cross boundaries. The control plane never executes user code. Agents never manage their own lifecycle.

This separation is enforced structurally, not by convention. The control plane and agents run in different processes, communicate over IPC, and share no mutable state except the SQLite database (where access patterns are partitioned: the control plane owns task and event tables; agents write only checkpoint data through a separate WAL-mode connection).

### 2.2 Tasks Are First-Class Runtime Entities

Agents are execution hosts. Tasks are the entities that matter. Every task gets:

- A unique identifier
- A persisted lifecycle status (pending → running → completed | failed | cancelled)
- An immutable event history
- Validated input parameters
- Stored results (or structured error information)

This makes tasks independently observable. You can list them, inspect them, cancel them, and resume them without knowing anything about the agent that executed them or the LLM that powered them.

### 2.3 Process Isolation, Not Thread Isolation

Each task runs in a dedicated agent subprocess. This provides real isolation:

- A misbehaving task cannot corrupt the control plane
- Memory leaks are contained to the subprocess
- CPU-bound work doesn't starve the event loop
- Cancellation is reliable — the control plane can terminate a subprocess if graceful cancellation fails

The cost is subprocess overhead. This is a deliberate tradeoff. For the workloads Switchplane targets — LLM-powered agents making API calls, running tools, processing data — the subprocess startup time is negligible compared to task execution time.

### 2.4 LangGraph-Native, Not Framework-Agnostic

Switchplane does not abstract over workflow engines. Tasks are LangGraph StateGraph graphs, and that coupling is intentional.

LangGraph provides graph-based execution, conditional branching, and node-level checkpointing. Switchplane provides the process model, daemon lifecycle, persistence, and CLI operability around it. Trying to abstract over multiple workflow engines would mean lowest-common-denominator support for all of them. Instead, Switchplane commits to LangGraph's model and builds on it directly.

The tradeoff is real: you cannot use Switchplane without LangGraph, and LangGraph API changes become your problem. For the current state of the ecosystem, that bet is worth making.

---

## 3. Architecture

```
CLI ─── Unix socket ───► Control Plane (daemon)
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                 Agent₁    Agent₂    Agent₃
              (subprocess) (subprocess) (subprocess)
                    │         │         │
                  Task₁     Task₂     Task₃
              (StateGraph) (StateGraph) (StateGraph)
```

### 3.1 Per-Application Isolation

Each application built with Switchplane gets its own:

- **Daemon process** — a control plane that manages agents for this application only
- **Runtime directory** (`~/.{app_name}/`) — containing the SQLite database, Unix socket, PID file, logs, and configuration
- **CLI** — auto-generated from the application's registered agents and tasks

There is no shared global runtime. Applications are fully isolated from each other. This means two Switchplane apps running on the same machine cannot interfere with each other's state, processes, or configuration.

### 3.2 The Control Plane

The control plane is a single-process asyncio server. It:

1. **Accepts CLI requests** over a Unix domain socket using 4-byte big-endian length-prefixed JSON framing
2. **Manages agent subprocesses** — spawning, monitoring, and cleaning up
3. **Persists all state** in SQLite (WAL mode) — tasks, events, agent records
4. **Routes commands** to running agents over per-agent IPC channels
5. **Streams events** to connected clients via persistent push connections
6. **Auto-shuts down** after a configurable idle period (default: 5 minutes) when no tasks are running and no clients are connected

The control plane is daemonized via the standard Unix double-fork pattern. It writes a PID file and listens on a Unix socket, both in the application's runtime directory.

### 3.3 Agent Subprocesses and IPC

When a task is submitted, the control plane spawns an agent subprocess. The IPC mechanism is a Unix socketpair:

1. The control plane creates a `socketpair(AF_UNIX, SOCK_STREAM)`
2. One end is passed to the child process via `--ipc-fd` and `pass_fds`
3. Both ends use 4-byte big-endian length-prefixed JSON framing

This gives bidirectional communication:

- **Control plane → Agent**: `execute_task`, `cancel`, `shutdown`, `user_command`
- **Agent → Control plane**: `task.started`, `task.progress`, `task.completed`, `task.failed`, `task.cancelled`, `task.command_result`, `log`

The socketpair approach has several advantages over alternatives (stdin/stdout, named pipes, shared memory):

- **Bidirectional on a single channel** — no need to manage separate read and write pipes
- **File descriptor inheritance** — no filesystem path to manage or race on
- **Clean separation** — stdout and stderr are freed for normal logging, not consumed by the IPC protocol
- **Concurrent command delivery** — the agent runs a command listener as a concurrent asyncio task alongside the main task execution, so cancel and user commands are delivered without blocking

### 3.4 Event Streaming

When a CLI client subscribes to a task, the control plane:

1. Replays all stored events for that task from the database
2. Upgrades the connection to a persistent push stream
3. Forwards new events the moment the agent emits them

This means LLM token output and progress messages appear immediately in the terminal, not in batched polls. The same Unix socket used for request/response handles streaming connections — the server detects `subscribe_task` requests and holds the connection open until the task reaches a terminal state.

### 3.5 Persistence

All task state lives in a single SQLite database per application (`~/.{app_name}/state.db`), running in WAL mode for concurrent read access. The schema includes:

- **tasks** — task records with status, input, result, error, timestamps
- **events** — append-only event log per task (type, payload, timestamp)
- **agents** — agent subprocess records (PID, status, heartbeat)
- **checkpoints** / **checkpoint_writes** — LangGraph checkpoint data for resumable workflows

The control plane owns the tasks, events, and agents tables. Agent subprocesses open their own WAL-mode connection to write checkpoint data. This partitioning avoids lock contention between the control plane's frequent event writes and the agent's checkpoint writes.

---

## 4. The Task Model

### 4.1 Task Definition

A task is a Python class that subclasses `Task`, declares parameters using Pydantic `Field()`, and implements an async `run()` method:

```python
class MyTask(Task):
    name = "process"
    description = "Process input data"
    mode = "ephemeral"  # or "long_running"

    value: str = Field(description="Input value")

    async def run(self, ctx: AgentContext) -> None:
        graph = build_graph().compile()
        result = await graph.ainvoke({"input": self.value})
        ctx.complete(result)
```

Parameters are introspected at discovery time to generate CLI flags. At execution time, they're validated by a dynamically-constructed Pydantic model before the task runs. The task instance receives validated, typed values as instance attributes.

### 4.2 Task Lifecycle

```
PENDING ──► RUNNING ──┬──► COMPLETED
                       ├──► FAILED ────► (resume) ──► PENDING
                       └──► CANCELLED ─► (resume) ──► PENDING
```

Every transition is recorded as an event. Terminal states (completed, failed, cancelled) are immutable — a completed task cannot be re-run, only resumed (which creates a new execution cycle with the same task ID).

### 4.3 Task Commands

Long-running tasks can expose commands — methods decorated with `@command` that can be invoked while the task is running:

```python
@command
def update_config(self, ctx: AgentContext, interval: int | None = None):
    if interval is not None:
        self.poll_interval = interval
    ctx.progress(f"Interval set to {self.poll_interval}s")
```

Commands are delivered over the IPC socketpair. The agent subprocess runs a command listener concurrently with the main task execution. When a command arrives, it's queued and dispatched to the matching handler. Parameters are automatically coerced from string values using Pydantic.

This makes tasks interactive. A long-running monitoring task can have its parameters adjusted mid-flight without restarting. A data pipeline can receive instructions to skip a step or change its target. The task is not a black box — it's a process you can interact with.

### 4.4 Checkpoint and Resume

Tasks opt into checkpointing by passing `ctx.checkpointer` to LangGraph's `graph.compile()`:

```python
graph = build_graph().compile(checkpointer=ctx.checkpointer)
config = {"configurable": {"thread_id": ctx.task_id}}
result = await graph.ainvoke(initial_state, config)
```

LangGraph saves state after each graph node. If the task fails at step 3 of 5, the checkpoint persists in SQLite. Resuming re-uses the same task ID as the thread ID, so LangGraph picks up from the last completed node.

The checkpointer is a custom `SqliteCheckpointSaver` that implements LangGraph's `BaseCheckpointSaver` interface, backed by the application's SQLite database. Each agent subprocess opens its own connection in WAL mode, avoiding lock contention with the control plane.

---

## 5. The Shell: Guardrailed Subprocess Execution

Agent tasks that need to run external commands (git, grep, curl, compilers) face a tension: the LLM needs to invoke tools, but unrestricted subprocess execution is a security risk.

Switchplane's `Shell` class provides a middle ground:

```python
shell = Shell(
    allowed_paths=[Path("/home/user/project")],
    allowed_commands=["git", "rg", "gh"],
)
```

Every invocation is validated against the allowlists before execution. Commands not in the list raise `PermissionError`. Paths passed as working directories or arguments are resolved and checked against `allowed_paths`. The execution uses `asyncio.create_subprocess_exec` — no shell interpretation, so arguments are never passed through `sh -c`.

`Shell.as_tool()` turns a command template into a LangChain `StructuredTool` that an LLM can invoke directly. Template placeholders become typed parameters. Path parameters are validated against the allowlist before execution. This lets you give an LLM access to `git log` or `rg` without giving it access to `rm` or arbitrary file paths.

---

## 6. Configuration

Switchplane uses a two-layer cascading TOML configuration model:

1. **App defaults** — bundled with the application, checked into version control. Contains model names, provider settings, agent-specific defaults.
2. **User overrides** — at `~/.{app_name}/config.toml`, never checked in. Contains API keys, personal preferences, endpoint overrides.

User config is deep-merged onto app defaults; user values win on conflict. Per-agent sections (`[agents.<name>]`) are deep-merged onto the global config before delivery to that agent, allowing agent-specific model or parameter overrides.

This design separates *what the app ships* from *what the user provides*. An app can ship with a model name, provider, and base URL. The user only needs to add their API key. If they want to override the model, they add one line — they don't need to redeclare the entire configuration.

---

## 7. MCP Integration

Switchplane includes optional support for the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). MCP servers are registered at the application level:

```python
app.register_mcp_server(McpServerConfig(
    name="my-tools",
    command=["python", "my_mcp_server.py"],  # stdio transport
))
```

Transport is inferred from the configuration: `command` implies stdio (Switchplane spawns and manages the process), `url` implies HTTP (Switchplane connects to an existing server).

Agents declare which MCP servers they need. The agent runtime manages the client lifecycle and exposes tools via `ctx.mcp_tools()` (LangChain `StructuredTool` wrappers) or `ctx.mcp` (raw MCP sessions).

MCP integration is an optional dependency (`pip install switchplane[mcp]`), so applications that don't use it pay no cost.

---

## 8. CLI and TUI

Every Switchplane application gets an auto-generated CLI. The CLI structure mirrors the runtime's capabilities:

```
<app> run <agent> <task> [--param value ...] [-d]
<app> task list | show | cancel | follow | resume | clear
<app> agent list
<app> runtime start | stop | status
```

Task parameters are introspected from `Field()` declarations and become CLI flags with types, defaults, and descriptions.

Running a task streams events inline. `Ctrl+C` detaches from the stream without cancelling the task — the daemon continues execution. The `--detach` flag starts a task in the background. `task follow` reattaches to a running task's event stream.

Invoking the app with no subcommand opens a full-screen TUI built on prompt_toolkit. The TUI provides tab-based navigation across running tasks, real-time event streaming, command input, and keyboard shortcuts for common operations. A persistent system tab receives daemon command output.

The TUI is only launched when stdout is a TTY. Piped or scripted invocations always get plain text.

---

## 9. What Switchplane Is Not

**Not a hosted platform.** There is no cloud component, no account, no dashboard. It is a Python library that turns agent code into a CLI.

**Not a prompt engineering framework.** It has no opinion on prompting strategies, retrieval patterns, or memory architectures. It handles how your task *runs*. You handle what your task *does*.

**Not a replacement for LangGraph.** It is a host for LangGraph graphs. LangGraph provides graph execution and checkpointing. Switchplane provides the process model, daemon lifecycle, and operational surface around it.

**Not a multi-tenant server.** Each application gets its own isolated daemon. Switchplane is designed for local, developer-operated agent workflows — not for serving multiple users from a shared backend.

---

## 10. Related Work

**LangGraph Platform** provides hosted graph execution with persistence and streaming. Switchplane occupies a similar architectural role but runs entirely locally, targets CLI-first workflows, and provides process isolation via Unix subprocesses rather than container orchestration.

**CrewAI** and **AutoGen** focus on multi-agent orchestration patterns — role assignment, delegation, conversation. Switchplane does not model agent-to-agent interaction. Its agents are independent execution hosts for tasks. Multi-agent coordination, if needed, happens within the task's LangGraph graph.

**Prefect** and **Airflow** are workflow orchestrators for data pipelines. They provide scheduling, dependency resolution, and distributed execution. Switchplane is narrower: it manages the runtime lifecycle of individual agent tasks, not DAGs of dependent jobs. A Prefect flow could invoke a Switchplane task, but they solve different problems.

**Supervisor** and **systemd** manage long-running processes. Switchplane's daemon is conceptually similar but application-aware: it understands task semantics, persists structured events, and provides bidirectional IPC to running tasks rather than just process lifecycle.

---

## 11. Conclusion

Agent systems need infrastructure that is boring in all the right ways. Task lifecycle management, event persistence, process supervision, and cancellation semantics are solved problems. The value of an agent system comes from what the LLM does inside the task, not from how the task is managed.

Switchplane enforces this separation. The control plane provides deterministic properties — every state transition is recorded, cancelled tasks stop executing, failed tasks can resume from checkpoints, running tasks can receive commands mid-flight. The agent provides non-deterministic reasoning — tool selection, content generation, decision-making. Neither crosses into the other's domain.

The result is a system where the interesting parts (what the agent does) can be as creative and unpredictable as the LLM allows, while the operational parts (how the agent runs) are reliable, observable, and controllable from a terminal.

---

## License

Copyright 2025 Salesforce, Inc. Licensed under the Apache License, Version 2.0.
