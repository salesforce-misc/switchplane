# Switchplane

Most agent frameworks hand everything to the LLM and hope for the best. Switchplane takes a different position:

> **If it's deterministic, write it in code. If it requires judgment, call the LLM.**

Here's what that looks like — a weekly ops review built with Switchplane has 4 graph nodes:

```
fetch_metrics → analyze → summarize → compile_report
(deterministic)  (deterministic)  (LLM)     (deterministic)
```

Three nodes are pure Python: pandas for statistical analysis, z-score spike detection, formatted report compilation. One node calls an LLM to interpret the pre-computed statistics into an executive summary. Total LLM cost: **~$0.02**. The deterministic nodes find the anomalies, compute the week-over-week deltas, and format the output. The LLM provides judgment on what the numbers mean. ([Full example below.](#devops-ops-review--the-switchplane-thesis-in-action))

Switchplane is a **runtime control plane** for LangGraph-native agent workflows. It is not a task library, prompt framework, or LLM wrapper. It's a daemonized supervisor that manages agent subprocesses, persists task state in SQLite, and generates a CLI for your application. Each app you build with Switchplane becomes a standalone command-line tool with its own isolated runtime.

> **Early-stage, actively developed.** APIs, IPC protocols, and storage formats may change without notice.

## Why Switchplane?

The industry trend is to lump everything into markdown files and hope things work when thrown at an LLM. Four problems with that:

- **Determinism.** LangGraph graphs execute the flow you defined. Variance occurs where you expect it — when interacting with humans or LLMs — but the overarching execution is guaranteed. Deterministic steps are authored as code, not handed off to an LLM for interpretation.
- **Auditability.** Every task has persistent event history, queryable after the fact. Graph nodes are unit-testable. You can trace exactly what happened and where.
- **Vendor independence.** You control what model you use for what purpose. Swap providers, mix models within a workflow, or run locally — your task logic is a LangGraph graph, not a provider-specific format.
- **Cost.** LLMs are used when judgment is actually required. The rest executes as code — microseconds instead of API calls, at zero marginal cost.

Language models are fundamentally non-deterministic. That's not a bug — it's the feature you're paying for. The better approach: let the LLM be non-deterministic where it's useful, and enforce deterministic properties around it. Your task graph can branch unpredictably. The runtime's behavior should not.

Switchplane enforces those properties:

- **Resumable, multi-step workflows** that survive process restarts
- **Persistent event history** for every task, queryable after the fact
- **Process isolation** via supervised subprocesses, not inline execution
- **Bidirectional IPC** to running tasks: send commands and receive events mid-flight
- **Operational control** from a CLI: start, stop, inspect, cancel, resume

The runtime is deterministic code solving deterministic problems, so the LLM can focus on the judgment calls it's actually good at.

## Architecture

```
<app> CLI → Control Plane (daemon) → Agent (subprocess) → Task (LangGraph StateGraph)
```

Each application built with Switchplane becomes its own CLI with an isolated daemon and runtime directory. There is no shared global runtime. Each app manages its own state.

**Tasks are first-class runtime entities.** Each task has a unique ID, persisted state, event history, lifecycle status, and stored results. Agents exist as execution hosts for tasks.

| Layer | Responsibility |
|---|---|
| **CLI** | Auto-generated from your `Application` object. Submit tasks, stream events, operator commands. |
| **Control Plane** | Per-app daemonized supervisor. Manages agents, routes tasks, persists state. Communicates with CLI over a Unix domain socket. |
| **Agent** | Subprocess that hosts task execution. Bidirectional IPC with control plane over a dedicated Unix socketpair. |
| **Task** | A `Task` subclass with a LangGraph workflow, executed inside an agent. Discovered automatically from the agent's `tasks/` package. |

### Key constraints

- The control plane owns task/event persistence in SQLite; agents write only checkpoint data (via a separate WAL-mode connection)
- The control plane never runs domain logic
- Agent IPC is bidirectional over a per-agent Unix socketpair (length-prefixed JSON)
- Each app gets its own runtime directory at `~/.{app_name}/`
- Auto-shutdown after 5 minutes idle (no tasks or connections)

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Install example apps
uv pip install -e examples/hello
uv pip install -e examples/devops   # ops review: pandas analysis + LLM summary
uv pip install -e examples/weather
uv pip install -e examples/chatbot  # interactive LLM chat
```

## Quick start

### Create a new project

```bash
switchplane init myapp
cd myapp
uv venv .venv && source .venv/bin/activate
uv pip install -e .
myapp agent list
myapp run default hello
```

This generates a complete project with a hello-world task, ready to run. See [Writing an application](#writing-an-application) for details on the generated structure.

### Run an example

```bash
# Run a task — opens the interactive TUI (daemon auto-starts if needed)
hello run example hello --user-name Alice

# Detached: start the task and return immediately, no TUI
hello run example hello --user-name Alice -d

# Run without --user-name to use system username
hello run example hello
```

When running interactively, task events stream to the terminal and you can type commands to the running task. For tasks that pause for user input (status: `interrupted`), you can type freeform text directly. See [CLI reference](#cli-reference) below. To enter the full-screen TUI dashboard, run the app with no subcommand (e.g. just `hello`). See [Interactive TUI](#interactive-tui).

Piped or scripted invocations (`hello run ... | ...`, `hello run ... > file`) work identically — plain text to stdout, no TUI.

## Interactive TUI

Invoking the app with no subcommand (e.g. just `weather`) opens a full-screen terminal UI built on [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/). The TUI auto-discovers running tasks from the daemon.

```
┌────────────────────────────────────────────────────────────┐
│ [0] system  [1] weather/watch ●  [2] chatbot/chat ⏸         │  Tab bar
├────────────────────────────────────────────────────────────┤
│  [14:23:01] Task started                                   │
│  [14:23:35] Temp: 11°C, cloudy                             │  Event pane
│  [14:24:05] Temp: 11°C (no change)                         │
├────────────────────────────────────────────────────────────┤
│ weather/watch [running] a1b2c3d4e5f6  [Tab] switch …       │  Status bar
│ [weather/watch] > _                                        │  Input bar
└────────────────────────────────────────────────────────────┘
```

Tab `[0] system` is always present and receives daemon command output. Task tabs start at `[1]`. Events arrive in real time via a persistent push connection — no polling lag.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Tab` / `Shift+Tab` | Cycle between tabs |
| `0` | Jump to system tab |
| `1`–`9` | Jump to task slot |
| `PgUp` / `PgDn` | Scroll task event pane |
| Mouse wheel | Scroll task event pane |
| `Ctrl+X` | Cancel focused task |
| `Ctrl+D` | Detach focused task from view (task keeps running) |
| `Ctrl+C` | Quit TUI (tasks keep running) |
| `↑` / `↓` | Cycle command history |
| `Enter` | Submit command |

### Input model

The TUI uses a three-tier input prefix scheme:

**Daemon commands** (prefix with `:`) mirror the CLI command structure:

| Command | Description |
|---|---|
| `:run <agent> <task> [--key value …]` | Start a new task |
| `:task follow <task_id>` | Follow an existing task |
| `:task cancel [<task_id>]` | Cancel focused or specified task |
| `:task list [--status <s>]` | List all tasks (optionally filter by status) |
| `:task show <task_id>` | Show task details |
| `:task retry <task_id>` | Retry a failed/cancelled task from last checkpoint |
| `:task clear` | Delete all completed/failed/cancelled tasks |
| `:runtime status` | Show daemon status |
| `:agent list` | List agents and their tasks |
| `:help` | Print all available commands |

**Task commands** (prefix with `/`) are sent to the focused task (for tasks that support `@command`-decorated methods):

```
[weather/watch] > /coordinates --lat 51.5074 --lon -0.1278
```

**Plain text** is sent as freeform input to the focused task when it is waiting for user input (status: `interrupted`). If the task is not waiting, a hint is shown.

### Attaching and detaching

`Ctrl+D` removes the focused task from the TUI view without touching the underlying task. The task keeps running in the daemon. Re-attach later with `:task follow <task_id>` (use `:task list` to get the full task ID). The system tab cannot be detached.

`Ctrl+C` quits the TUI entirely. All tasks keep running — the daemon is unaffected.

## Configuration

Two-layer cascading config: **app defaults** bundled with your application, deep-merged with **user overrides** at `~/.{app_name}/config.toml`.

### App defaults

Apps ship sensible defaults via a TOML file referenced in the Application constructor:

```python
app = Application(name="myapp", default_config=Path(__file__).parent / "config.toml")
```

```toml
# Bundled with the app (checked into VCS)
[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
base_url = "https://corp-proxy.internal/v1"

[agents.bot]
system_prompt = "You are a helpful assistant."
```

### User overrides

Users provide personal config at `~/.{app_name}/config.toml`. This is deep-merged onto app defaults; user values win on conflict:

```toml
# ~/.myapp/config.toml (personal, never checked in)
[llm]
api_key = "sk-ant-..."

# Per-agent overrides (deep-merged onto global config)
[agents.bot.llm]
model = "claude-haiku-4-5-20251001"
```

Global config is available to all agents via `ctx.config`. Per-agent sections under `[agents.<name>]` are deep-merged onto the global config before delivery, so `agents.bot.llm.model` overrides `llm.model` for the bot agent only.

### Custom CA certificates

If your LLM endpoint uses a corporate proxy or internal CA, Python's default trust store won't have the certificate. Place a PEM bundle at `~/.{app_name}/ca-bundle.pem` and the daemon will set `SSL_CERT_FILE` automatically for all agent subprocesses.

To create the bundle (macOS, exports system keychain certs and combines with Python's defaults):

```bash
security find-certificate -a -p /Library/Keychains/System.keychain \
  /System/Library/Keychains/SystemRootCertificates.keychain > /tmp/system_certs.pem
cat "$(python3 -m certifi)" /tmp/system_certs.pem > ~/.myapp/ca-bundle.pem
```

## CLI reference

Every Switchplane app gets the same CLI structure. Replace `<app>` with your app's command name.

### Task execution

```bash
<app> run <agent> <task> [--param value ...] [-d]
```

### Agent discovery

```bash
<app> agent list          # List agents, tasks, parameters, and commands
```

### Runtime management

```bash
<app> runtime start       # Start the control plane daemon
<app> runtime stop        # Graceful shutdown
<app> runtime status      # Show active agents, running tasks, connections
```

### Task inspection

```bash
<app> task list [--status pending|running|interrupted|completed|failed|cancelled]
<app> task show <task_id>
<app> task cancel <task_id>
<app> task follow <task_id>    # Stream events from a running task
<app> task retry <task_id>     # Retry a failed/cancelled task from last checkpoint
<app> task clear               # Purge completed, failed, and cancelled task history
```

### Authentication

Manage OAuth tokens for MCP servers that require authentication. These commands do not require the daemon to be running.

```bash
<app> auth login <server_name>    # Run OAuth flow (opens browser), store tokens
<app> auth status                 # Show token status for all OAuth-enabled servers
<app> auth logout <server_name>   # Remove stored tokens for a server
```

`auth login` handles both MCP-spec OAuth (auto-discovery) and Direct OIDC (explicit endpoints), depending on how the server is configured in your app. After a successful login, tokens are stored in `~/.{app_name}/oauth/<server_name>/` and used automatically for all subsequent MCP connections to that server.

### Task commands

Send commands to running tasks that support them:

```bash
<app> task <task_id> <command> [--key value ...]
```

### Long-running tasks

Events stream to the terminal in real time. `Ctrl+C` detaches without killing the task.

```bash
# Events stream inline. Ctrl+C to detach (task keeps running).
weather run weather watch

# Reattach from the CLI, or from the TUI with :task follow <task_id>:
weather task follow <task_id>

# Change coordinates on a running watch (from TUI or CLI)
weather task <task_id> coordinates --lat 51.5074 --lon -0.1278

# Cancel from anywhere
weather task cancel <task_id>

# Fire-and-forget — no TUI, returns immediately
weather run weather watch -d
```

That `coordinates` command sends a typed, validated command to a *running* task over the bidirectional IPC socketpair between the control plane and the agent subprocess. The task receives it, updates its internal state, and continues executing. No restart, no resubmission. Tasks are not fire-and-forget black boxes; they're processes you can interact with mid-flight.

## Writing an application

The fastest way to start is `switchplane init`:

```bash
switchplane init myapp
```

This generates the following project structure:

### Project structure

```
myapp/
├── pyproject.toml
└── myapp/
    ├── app.py
    └── agents/
        └── default/
            ├── agent.py
            └── tasks/
                └── hello.py
```

### Application object

```python
# myapp/app.py
from switchplane import Application

app = Application(name="myapp")
app.discover_agents("myapp.agents")

def main():
    app.run()
```

`app.run()` discovers agents, builds the CLI, and starts it. The `name` determines the runtime directory (`~/.myapp/`).

### Agent definition

```python
# myapp/agents/myagent/agent.py
from switchplane.agent import AgentSpec

agent_spec = AgentSpec(
    agent_name="myagent",
)
```

Tasks are discovered automatically from the `tasks/` subpackage. No need to declare them in the agent spec.

### Task definition (LangGraph graph)

Tasks are defined as `Task` subclasses with declarative parameters using Pydantic `Field()`. Parameters are validated before execution and available as instance attributes in `run()`.

```python
# myapp/agents/myagent/tasks/mytask.py
from typing import TypedDict
from langgraph.graph import END, StateGraph

from switchplane import Field, Task
from switchplane.agent_runtime import AgentContext


class MyState(TypedDict):
    input_value: str
    result: str | None

def step_one(state: MyState) -> MyState:
    return {**state, "result": f"processed: {state['input_value']}"}

def build_graph() -> StateGraph:
    g = StateGraph(MyState)
    g.add_node("step_one", step_one)
    g.set_entry_point("step_one")
    g.add_edge("step_one", END)
    return g

class MyTask(Task):
    name = "mytask"
    description = "Does something useful"

    value: str = Field(default="", description="Input value to process")

    async def run(self, ctx: AgentContext) -> None:
        graph = build_graph().compile()
        result = await graph.ainvoke({"input_value": self.value, "result": None})
        ctx.complete({"result": result["result"]})
```

Tasks declare their lifecycle mode: `"ephemeral"` (default, runs once) or `"long_running"` (polls/loops until cancelled).

### Task commands

Long-running tasks can expose commands using the `@command` decorator. Commands receive typed parameters that are automatically coerced from CLI string values:

```python
from switchplane import Field, Task, command
from switchplane.agent_runtime import AgentContext

class MyWatcher(Task):
    name = "watch"
    mode = "long_running"

    latitude: float = Field(default=0.0)

    @command
    def set_location(self, ctx: AgentContext, lat: float | None = None):
        if lat is not None:
            self.latitude = lat
        ctx.progress(f"Location updated to {self.latitude}")
        return {"latitude": self.latitude}

    async def run(self, ctx: AgentContext) -> None:
        while not ctx.is_cancelled:
            await self.process_commands(ctx)
            # ... do work using self.latitude ...
```

Commands are invoked from the CLI: `<app> task <task_id> set_location --lat 51.5074`

### Interactive input (LLM chat loops)

Tasks can pause and wait for freeform user input using `ctx.wait_for_input()`. This emits a `task.interrupted` event, blocks until the user types a response, then emits `task.resumed` and returns the text. The task's status changes to `interrupted` while waiting, which enables plain text input in both the TUI and CLI.

This requires a checkpointer (compile your graph with `checkpointer=ctx.checkpointer`).

```python
class ChatTask(Task):
    name = "chat"
    mode = "long_running"

    async def run(self, ctx: AgentContext) -> None:
        # ... build and compile graph with ctx.checkpointer ...

        while not ctx.is_cancelled:
            user_input = await ctx.wait_for_input("You: ")
            if not user_input:
                break
            result = await graph.ainvoke(Command(resume=user_input), config)
            ctx.progress(f"Assistant: {result['messages'][-1].content}")

        ctx.complete({"status": "done"})
```

The `prompt` argument to `wait_for_input()` is displayed to the user as a hint. In the TUI, interrupted tasks show a ⏸ status indicator.

### MCP server integration

Agents can use tools from [MCP](https://modelcontextprotocol.io/) servers. Register servers at the app level, then declare which servers each agent needs. Switchplane manages the MCP client lifecycle (spawning stdio processes or connecting to HTTP endpoints) and exposes tools to your task via `ctx`.

**Register MCP servers in your app:**

```python
# myapp/app.py
from switchplane import Application
from switchplane.app import McpServerConfig, OAuthConfig

app = Application(name="myapp")

# stdio: Switchplane spawns and manages the process
app.register_mcp_server(McpServerConfig(
    name="my-tools",
    command=["python", "my_mcp_server.py"],
))

# HTTP: Switchplane connects to an already-running server
app.register_mcp_server(McpServerConfig(
    name="remote-tools",
    url="http://localhost:8080/mcp",
))

# HTTP with MCP-spec OAuth (auto-discovers endpoints from the server)
app.register_mcp_server(McpServerConfig(
    name="slack",
    url="https://mcp.slack.com/sse",
    oauth=OAuthConfig(client_id="your-client-id", scopes="channels:read"),
))

# HTTP with Direct OIDC (explicit auth/token URLs — for Keycloak etc.)
app.register_mcp_server(McpServerConfig(
    name="internal-tools",
    url="https://internal.corp/mcp",
    oauth=OAuthConfig(
        client_id="your-client-id",
        auth_url="https://sso.corp/auth",
        token_url="https://sso.corp/token",
        scopes="tools:read",
    ),
))

app.discover_agents("myapp.agents")
```

Transport is inferred: provide `command` for stdio, `url` for HTTP. No `transport` field needed.

For HTTP servers that require a fully custom `httpx.AsyncClient` (e.g. mutual TLS), set `http_transport` to a dotted path pointing at a factory function that accepts an `McpServerConfig` and returns an `httpx.AsyncClient`. This is an escape hatch for cases not covered by the built-in OAuth support.

**Declare MCP servers on the agent:**

```python
# myapp/agents/myagent/agent.py
from switchplane.agent import AgentSpec

agent_spec = AgentSpec(
    agent_name="myagent",
    mcp_servers=["my-tools"],
)
```

**Declare MCP servers on a task (optional):**

Tasks can override the agent-level default by declaring which specific servers they need. Only declared servers are started for that task:

```python
class MyTask(Task):
    name = "analyze"
    mcp_servers = ["my-tools"]  # Only start this server, not all agent servers

    async def run(self, ctx: AgentContext) -> None:
        tools = await ctx.mcp_tools()  # Only tools from "my-tools"
```

**Use MCP tools in your task:**

```python
async def run(self, ctx: AgentContext) -> None:
    # Get all MCP tools as LangChain tools, ready for bind_tools()
    tools = await ctx.mcp_tools()
    llm_with_tools = llm.bind_tools(tools)

    # Or access raw MCP sessions directly
    result = await ctx.mcp["my-tools"].call_tool("whoami")
```

MCP support requires the optional `mcp` dependency: `pip install switchplane[mcp]`

#### OAuth authentication for MCP servers

Two modes are supported, both using PKCE:

**MCP-spec OAuth** (leave `auth_url`/`token_url` unset): The MCP SDK's `OAuthClientProvider` discovers authorization endpoints from the server's protected-resource metadata automatically. This works with servers like Slack that implement the MCP OAuth spec.

**Direct OIDC** (set `auth_url` and `token_url`): Switchplane runs the PKCE authorization-code flow directly against the identity provider. Use this for external IdPs like Keycloak that are not discoverable via MCP server metadata.

Both modes use the same interactive login flow — a browser opens for user consent and the resulting tokens are stored locally. The `auth login` command initiates this flow (see [CLI reference](#cli-reference)). Tokens are refreshed automatically on expiry and are stored at `~/.{app_name}/oauth/<server_name>/`.

Agents don't need to do anything special for OAuth-enabled servers. Switchplane injects the authentication into the HTTP transport transparently before the agent connects.

### LLM integration

Switchplane includes an optional LLM module that instantiates LangChain chat models from config. It routes to the correct adapter based on model name prefix — no provider-specific code in your tasks.

```python
from switchplane.llm import build_llm

llm = build_llm("claude-sonnet-4-20250514", api_key="sk-ant-...", base_url=None)
```

In practice, you pull these values from the task's config:

```python
async def run(self, ctx: AgentContext) -> None:
    cfg = ctx.config.get("llm", {})
    llm = build_llm(cfg.get("model"), cfg.get("api_key"), cfg.get("base_url"))
    llm_with_tools = llm.bind_tools(tools)
```

Routing rules:

| Prefix | Adapter | Package |
|---|---|---|
| `claude-*` | `ChatAnthropic` | `langchain-anthropic` |
| `gemini-*` | `ChatGoogleGenerativeAI` | `langchain-google-genai` |
| `gpt-*` | `ChatOpenAI` | `langchain-openai` |

Adapter packages are imported lazily — install only what you need. The module also exports a `MODELS` registry of well-known public models with context window sizes, and a `context_window(model)` helper.

```bash
# Install the LLM module (just langchain-core)
pip install switchplane[llm]

# Then install the adapter for your provider
pip install langchain-anthropic
```

Apps that need custom routing (e.g. through a corporate API gateway) can provide their own `build_llm` and import it instead. Switchplane's version is a sensible default, not a requirement.

### Shell: sandboxed subprocess execution

The `Shell` class provides a guardrailed way for agents to run external commands. You declare which binaries and directories are allowed upfront, and all invocations are validated before execution.

```python
from pathlib import Path
from switchplane import Shell

shell = Shell(
    allowed_paths=[Path("/home/user/project")],
    allowed_commands=["git", "rg", "gh"],
)

# In a task's run():
stdout = await shell.run(["git", "log", "--oneline", "-5"], cwd=repo_path)
ok = await shell.run_ok(["git", "diff", "--quiet"], cwd=repo_path)
```

Commands not in the allowlist raise `PermissionError`. Paths passed as `cwd` are validated against `allowed_paths`. Each invocation has a configurable timeout (default 30s).

**Creating LangChain tools from shell commands:**

`Shell.as_tool()` turns a command template into a `StructuredTool` that an LLM can invoke. Template placeholders become tool parameters. Use `path_params` to declare which placeholders represent filesystem paths — these are validated against the shell's allowed directories before execution:

```python
grep_tool = shell.as_tool(
    name="grep_files",
    cmd_template=["rg", "--no-heading", "-n", "{pattern}", "{directory}"],
    description="Search file contents for a regex pattern.",
    path_params={"directory"},
)

# grep_tool is a LangChain StructuredTool, ready for bind_tools()
tools = [grep_tool] + await ctx.mcp_tools()
llm_with_tools = llm.bind_tools(tools)
```

`Shell` uses `asyncio.create_subprocess_exec` (no shell interpretation), so arguments are never passed through a shell. The allowlist and path validation add defense-in-depth when LLM-generated values flow into command arguments.

For a general-purpose shell tool, `bash_tool()` returns a single `StructuredTool` that parses commands with `shlex` and validates against the allowlist. Working directory is locked to `allowed_paths[0]`, output is truncated to `max_output_chars` (default 30,000 characters). `agent_tools()` returns a minimal coding-focused set: bash + write_file + edit_file.

```python
# General-purpose bash tool
bash = shell.bash_tool()

# Minimal set for coding agents: bash + write_file + edit_file
tools = shell.agent_tools()
```

### Cross-task coordination

Tasks can spawn child tasks, wait for their completion, and send notifications to sibling tasks. Child tasks are linked via `parent_task_id`. Requests travel over the existing agent-CP socketpair as `AgentRequest`/`AgentResponse` messages.

```python
async def run(self, ctx: AgentContext) -> None:
    # Spawn a child task (returns immediately with task_id)
    child_id = await ctx.submit_task("worker", "process", {"chunk": 1})

    # Wait for it to reach a terminal state
    result = await ctx.wait_for_task(child_id)

    # Or spawn multiple and wait in parallel
    ids = [
        await ctx.submit_task("worker", "process", {"chunk": i})
        for i in range(3)
    ]
    results = await ctx.wait_for_tasks(ids)

    # Send a notification to another running task
    await ctx.notify_task(other_task_id, {"status": "ready"})

    # Block until a notification arrives (or timeout)
    notification = await ctx.wait_for_notification(timeout=60.0)
```

`wait_for_task` polls until the child reaches a terminal state (completed, failed, cancelled). `wait_for_notification` wakes immediately when a notification arrives -- useful for event-driven coordination between long-running tasks.

### Checkpoint and resume

Tasks can opt into checkpointing so that failed or cancelled runs can be resumed from the last completed graph node. Switchplane provides a LangGraph-compatible checkpoint saver backed by the app's SQLite database. Pass it to `graph.compile()` and use `ctx.task_id` as the thread ID:

```python
class MyTask(Task):
    name = "pipeline"
    description = "Multi-step data pipeline"

    async def run(self, ctx: AgentContext) -> None:
        graph = build_graph().compile(checkpointer=ctx.checkpointer)
        config = {"configurable": {"thread_id": ctx.task_id}}

        result = await graph.ainvoke(initial_state, config)
        ctx.complete(result)
```

LangGraph saves state after each node execution. If the task fails halfway through, the checkpoint persists in SQLite. Resuming re-uses the same task ID as the thread ID, so LangGraph picks up from the last completed node:

```bash
# Run a multi-step task
myapp run myagent pipeline
# Task fails at step 3 of 5...

# Retry from last checkpoint (step 3)
myapp task retry <task_id>

# Or retry detached
myapp task retry <task_id> -d
```

Only tasks in a terminal state (failed, cancelled, or completed) can be retried. Tasks that don't use `ctx.checkpointer` run without checkpointing; retry will re-execute from the beginning.

### CLI entry point

In your `pyproject.toml`:

```toml
[project.scripts]
myapp = "myapp.app:main"
```

Install in editable mode and your app is available as a CLI command.

## Debugging agents

Agents run as detached subprocesses with `stdin` redirected to `/dev/null`, so `pdb.set_trace()` is unusable. Switchplane ships an opt-in [debugpy](https://github.com/microsoft/debugpy) listener you can attach to from VS Code or any debugpy-compatible client.

Install the optional `debug` extra:

```bash
uv pip install -e '.[debug]'
```

Set `SWITCHPLANE_DEBUG_AGENT` in the environment that launches the daemon (i.e. before you run your app). Each agent subprocess will host a debugpy listener on `127.0.0.1`, emit a `progress` event announcing the bound port, and block until a client attaches.

| Value | Behavior |
|---|---|
| unset / empty | No-op (default) |
| `1` or `true` | Listen on the default debugpy port `5678` |
| `auto` | Bind an ephemeral free port — useful when running multiple agents |
| any other integer | Listen on that specific port |

The bound host and port are logged to the control plane log and surfaced to the running task via a `progress` event, e.g.:

```
execution paused: debugpy listening on 127.0.0.1:5678, waiting for client to attach
```

You'll see this line in the CLI / TUI event stream as soon as the task starts, before any task code runs.

VS Code `launch.json`:

```json
{
  "name": "Attach: switchplane agent",
  "type": "debugpy",
  "request": "attach",
  "connect": { "host": "127.0.0.1", "port": 5678 },
  "justMyCode": false
}
```

Once attached, set breakpoints anywhere in your task code or LangGraph nodes — execution resumes from `agent_main` and stops at the first hit. The listener is bound to loopback only; debugpy permits arbitrary code execution by any client that can reach the port, so binding `127.0.0.1` keeps it confined to the local machine.

## Examples

### devops: Ops review — the Switchplane thesis in action

A weekly ops review that fetches service metrics, runs statistical analysis, and produces an executive summary. This is the example that demonstrates *why* Switchplane exists: out of 4 graph nodes, only 1 calls an LLM. The rest is deterministic code — pandas for analysis, z-score spike detection, formatted report compilation.

The graph:

```
fetch_metrics → analyze → summarize → compile_report
(deterministic)  (deterministic)  (LLM)     (deterministic)
```

Uses mock NewRelic-style data (request rates by endpoint/status code, response time percentiles) with injected anomalies so the analysis has something real to find. In production, `fetch_metrics` would be an API call — everything else stays the same.

```bash
uv pip install -e examples/devops

# Set your API key (the only user config needed)
mkdir -p ~/.devops && echo -e '[llm]\napi_key = "sk-ant-..."' > ~/.devops/config.toml

devops run sre review
```

**What the analysis finds (deterministically, zero LLM cost):**
- Payment endpoint 500s spiked Wednesday 14:00–16:59 UTC (z-scores 6.8–7.7)
- 5xx error rate for `/api/payments` up from 1.50% → 1.95% WoW
- Order endpoint p99 latency peaked at 1949ms (prev week: 742ms)
- Global HTTP 500/503 volume up ~7% WoW

The LLM's only job: interpret these pre-computed statistics into an executive summary with anomaly classification. One API call, ~5K input tokens, ~\$0.02.

### hello: Simple LangGraph graph

Two-node graph (`get_user` -> `say_hello`). Good starting point for understanding the project structure.

```bash
uv pip install -e examples/hello
hello run example hello --user-name Alice
```

### chatbot: Interactive LLM chat

A conversational chatbot that demonstrates interactive tasks with freeform text input. The task uses LangGraph's `interrupt()` to pause the graph and wait for user input via `ctx.wait_for_input()`. Each user message resumes the graph, the LLM responds, and the graph interrupts again — a standard chat loop built on checkpoint-backed graph execution.

```bash
uv pip install -e examples/chatbot

# Set your API key
mkdir -p ~/.chatbot && echo -e '[llm]\napi_key = "sk-ant-..."' > ~/.chatbot/config.toml

# Start chatting
chatbot run bot chat
```

In the TUI, plain text typed while the task is in `interrupted` state is sent directly as user input. In CLI attached mode (`run`/`follow`), the same applies — just type and press Enter. Use `/end` to finish the session.

### weather: Long-running polling task

Watches weather conditions using the Open-Meteo API. Polls on an interval, detects changes, and streams progress events. Demonstrates long-running tasks, cancellation, task commands, checkpoint/resume, and config usage.

```bash
uv pip install -e examples/weather
weather run weather watch
# Events stream inline. Ctrl+C to detach (task keeps running).

# Check on it — from the TUI use :task list and :task follow, or from the CLI:
weather task list
weather task follow <task_id>

# Change coordinates on a running watch (from TUI input or CLI)
weather task <task_id> coordinates --lat 51.5074 --lon -0.1278

# Cancel and resume (picks up with last known weather state)
weather task cancel <task_id>
weather task retry <task_id>
```

## Runtime directory

Each app gets its own runtime directory at `~/.{app_name}/`:

```
~/.myapp/
├── config.toml      # Application configuration
├── state.db         # SQLite database (WAL mode)
├── runtime.sock     # Unix domain socket
├── runtime.pid      # Daemon PID file
├── ca-bundle.pem    # Optional custom CA certificates
├── oauth/
│   └── <server_name>/
│       ├── tokens.json       # Stored OAuth tokens
│       └── client_info.json  # OAuth client registration
└── logs/
    └── control_plane.log
```

## What this is not

Switchplane is not a hosted platform. There's no cloud component, no account to create, no dashboard. It's a Python library that turns your code into a CLI.

It is not a prompt engineering framework. It has no opinion on prompting strategies, retrieval patterns, or memory architectures. It does include LLM provider config, MCP integration, and LangChain tool wrappers, so it makes opinionated choices about the infrastructure around your LLM calls. The line it draws: Switchplane handles how your task *runs*. You handle what your task *does*.

It is not a replacement for LangGraph. It's a host for LangGraph graphs, and that coupling is deliberate. LangGraph provides checkpointing and graph execution. Switchplane provides the process model, daemon lifecycle, and CLI operability around it. The tradeoff is real: you can't use Switchplane without LangGraph, and LangGraph's API changes become your problem. For now, that bet is worth making.

## Technology

- **Python 3.12+** with asyncio
- **Click** for CLI generation
- **prompt_toolkit** for the interactive TUI
- **Pydantic v2** for models and serialization
- **SQLite** (via aiosqlite) for persistence with WAL mode
- **LangGraph** for task workflow execution
- **MCP** (optional) for Model Context Protocol client and tool integration

### Event streaming

The TUI receives events via a persistent push connection, not polling. When you subscribe to a task, the control plane replays all stored events for that task and then pushes new events the moment the agent emits them. This means LLM token output and progress messages appear immediately rather than arriving in batches. The same Unix socket used for regular CLI requests handles streaming connections; the server upgrades the connection on a `subscribe_task` request and holds it open until the task reaches a terminal state. Interactive input (freeform text and `/` commands) flows back through the same connection.
