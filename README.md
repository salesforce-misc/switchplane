# Switchplane

Switchplane is a local runtime control plane for LangGraph tasks. It gives each application an isolated daemon, generated CLI, subprocess execution, persistent task history, and operational controls.

> **If it is deterministic, write it in code. If it requires judgment, call the model.**

```text
fetch -> analyze -> judge -> act
 code      code      LLM     code
```

LangGraph defines what your task does. Switchplane handles how it runs: process isolation, persistence, event delivery, checkpoint-backed retry, interactive input, and control from a CLI or terminal UI.

> **Early-stage, actively developed.** Public APIs, IPC protocols, and storage formats may change without notice.

## Why Switchplane?

Agent workflows need nondeterminism in specific places, not throughout the runtime around them. Switchplane keeps the execution model explicit while leaving model choice and task logic in your code.

- **Deterministic orchestration.** LangGraph executes the flow you define. Ordinary Python handles validation, analysis, branching, and output formatting.
- **Operational control.** Start, inspect, follow, command, cancel, and retry tasks from a generated CLI or full-screen TUI.
- **Persistent history.** Task status, durable events, structured results, errors, and checkpoints live in a per-application SQLite database.
- **Process isolation.** User code runs in supervised subprocesses rather than inside the control plane.
- **Model independence.** Select providers at runtime, mix models in one graph, or omit LLMs entirely.

Switchplane is not a hosted platform, prompt framework, or replacement for LangGraph. It is the local process and operations layer around LangGraph applications.

## Quick Start

Switchplane requires Python 3.12+ and currently targets POSIX systems such as macOS and Linux.

From a repository checkout:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Create and run a standalone application:

```bash
switchplane init myapp
cd myapp
uv pip install -e .

myapp agent list
myapp run default hello --user-name Alice
```

The daemon starts automatically. The task runs in an agent subprocess, and its result and events are stored under `~/.myapp/`.

```bash
myapp task list                    # Find task IDs
myapp task show <task_id>          # Inspect the result and event history
myapp run default hello -d         # Submit and return immediately
myapp                              # Open the full-screen TUI in a terminal
```

Attached `run` output is plain text and works in pipelines. A bare application command opens the TUI only when standard input is a TTY.

The generated hello task reports progress during the run and stores its greeting as the task result. Use `task show` to inspect it.

## Mental Model

```text
myapp CLI
    |
    v
per-app control plane daemon
    |
    +-- SQLite: tasks, events, results, checkpoints
    |
    v
agent subprocess
    |
    v
Task.run(ctx) -> LangGraph StateGraph
```

Each application has its own CLI, daemon, Unix socket, database, logs, and configuration. There is no shared global Switchplane runtime.

| Layer | Responsibility |
|---|---|
| **Application** | Registers discovery roots, configuration, and optional MCP servers, then builds the CLI. |
| **Control plane** | Owns task lifecycle and persistence, supervises subprocesses, and routes messages. |
| **Agent** | Provides the subprocess environment in which a task executes. |
| **Task** | Defines typed parameters and runs a LangGraph graph or other task-specific control loop. |

Tasks are the first-class runtime entities. Each task has an ID, lifecycle status, input, durable event history, and stored result. Agents are execution hosts for those tasks.

The control plane never runs domain logic. CLI traffic uses a Unix domain socket; every launched agent receives a dedicated bidirectional Unix socketpair.

## Build a Task

`switchplane init myapp` generates this structure:

```text
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

The application discovers agents and exposes the generated CLI:

```python
# myapp/app.py
from switchplane import Application

app = Application(name="myapp")
app.discover_agents("myapp.agents")


def main():
    app.run()
```

An agent is an execution host. Tasks are discovered from its `tasks` package:

```python
# myapp/agents/default/agent.py
from switchplane.agent import AgentSpec

agent_spec = AgentSpec(agent_name="default")
```

A task combines typed parameters with a LangGraph workflow:

```python
# myapp/agents/default/tasks/greet.py
from typing import TypedDict

from langgraph.graph import END, StateGraph

from switchplane import Field, Task
from switchplane.agent_runtime import AgentContext


class GreetState(TypedDict):
    name: str
    greeting: str


def greet(state: GreetState) -> dict:
    return {"greeting": f"Hello, {state['name']}!"}


def build_graph() -> StateGraph:
    graph = StateGraph(GreetState)
    graph.add_node("greet", greet)
    graph.set_entry_point("greet")
    graph.add_edge("greet", END)
    return graph


class GreetTask(Task):
    name = "greet"
    description = "Greet someone by name"

    recipient: str = Field(description="Who to greet")

    async def run(self, ctx: AgentContext) -> None:
        graph = build_graph().compile()
        result = await graph.ainvoke(
            {"name": self.recipient, "greeting": ""}
        )
        ctx.stream_flush(result["greeting"])
        ctx.complete(result)
```

Run it with:

```bash
myapp run default greet --recipient Alice
```

`Field()` attributes become validated CLI parameters. `ctx.stream_flush()` emits user-facing, replayable text; `ctx.complete()` stores the structured result for later inspection with `task show`.

Keep one public `Task` subclass in each task module, with the module filename matching `Task.name` (`greet.py` and `name = "greet"`).

### Checkpointing

Pass Switchplane's LangGraph checkpointer when a task should retry from saved graph state:

```python
graph = build_graph().compile(checkpointer=ctx.checkpointer)
config = {"configurable": {"thread_id": ctx.task_id}}
checkpoint = await graph.aget_state(config)
input_state = None if checkpoint and checkpoint.values else initial_state
result = await graph.ainvoke(input_state, config)
ctx.complete(result)
```

LangGraph saves state after each node. `task retry` relaunches the same task ID, but task code still decides how to resume: invoke with `None` from an existing checkpoint, resume an interrupt with `Command`, or start with fresh input when no checkpoint exists. A task without a checkpointer runs again from the beginning.

## Operate Tasks

Every Switchplane application gets the same command structure. Replace `<app>` with the application's command name.

| Purpose | Command |
|---|---|
| Open the TUI | `<app>` |
| Run attached | `<app> run <agent> <task> [--key value ...]` |
| Run detached | `<app> run <agent> <task> [--key value ...] -d` |
| Discover agents and tasks | `<app> agent list` |
| List tasks | `<app> task list [--status <status>]` |
| Inspect a task | `<app> task show <task_id>` |
| Follow task events | `<app> task follow <task_id>` |
| Cancel a task | `<app> task cancel <task_id>` |
| Retry a task | `<app> task retry <task_id> [-d]` |
| Send a task command | `<app> task <task_id> <command> [--key value ...]` |
| Hide terminal tasks | `<app> task clear` |
| Delete terminal task data | `<app> task purge [-y]` |
| Manage the daemon | `<app> runtime start\|status\|stop` |
| Log in to an MCP server | `<app> auth login <server_name>` |
| Inspect MCP authentication | `<app> auth status` |
| Log out of an MCP server | `<app> auth logout <server_name>` |

`Ctrl+C` detaches from attached output without cancelling the task. Use `task cancel` when the task itself should stop.

Task lifecycle statuses are:

```text
pending -> running -> completed
              |-----> failed
              |-----> cancelled
              |-----> interrupted -> running

completed / failed / cancelled -> cleared
completed / failed / cancelled -> retry -> pending
```

`task clear` hides terminal tasks from the default listing but preserves their data. `task purge` permanently removes terminal task records, events, checkpoints, task files, and task logs.

### Runtime Commands

Long-running tasks can expose typed commands with `@command`:

```python
from switchplane import Task, command


class WatchTask(Task):
    name = "watch"
    mode = "long_running"

    latitude = 49.2827
    longitude = -123.1207

    @command
    def coordinates(self, ctx, lat: float, lon: float):
        self.latitude = lat
        self.longitude = lon
        return {"latitude": lat, "longitude": lon}

    async def run(self, ctx):
        while not ctx.is_cancelled:
            await self.process_commands(ctx)
            ctx.progress(
                f"Watching ({self.latitude}, {self.longitude})"
            )
            if not await ctx.sleep(60):
                break
```

```bash
weather task <task_id> coordinates --lat 51.5074 --lon -0.1278
```

`mode = "long_running"` describes the task; your code still owns its loop, cancellation checks, sleeps, and command processing.

### Interactive TUI

Run an application with no subcommand to open its terminal UI. It shows a system tab plus one tab per attached task and receives new events over a persistent connection.

The input model has three forms:

```text
:task list                      daemon command
/coordinates --lat 49.2        command for the focused task
hello                           freeform input while the task is interrupted
```

Use `Tab` and `Shift+Tab` to switch tasks, `Ctrl+D` to detach the focused tab without changing its task, `Ctrl+X` to cancel the focused task, and `Ctrl+C` to leave the TUI.

## Capabilities

| Capability | What Switchplane provides |
|---|---|
| **Persistence** | SQLite-backed task records, durable events, results, errors, and LangGraph checkpoints. |
| **Event output** | Durable lifecycle, progress, and flush events plus ephemeral streaming chunks. |
| **Checkpoint retry** | Operator-triggered retry using the original task ID and any saved graph state. |
| **Interactive input** | `ctx.wait_for_input()` for checkpoint-backed graphs that pause for freeform input. |
| **Configuration** | App TOML defaults deep-merged with user overrides at `~/.<app>/config.toml`. |
| **LLM routing** | `ctx.llm()` and named `[llm.providers.<name>]` entries selected at runtime. |
| **MCP** | Managed stdio or streamable HTTP sessions and LangChain-compatible tools. |
| **Shell tools** | Allowlists for executable commands and filesystem paths, without shell interpretation. |
| **Coordination** | Child-task submission, status queries, parallel waits, parent-cascade cancellation, and notifications. |
| **Task files** | Per-task data directories accessed through path-restricted `Task.open()`. |

### Configuration and Integrations

Applications can bundle TOML defaults while users keep credentials and overrides in `~/.<app>/config.toml`. Named entries under `[llm.providers.<name>]` let a task choose models at runtime:

```python
default_llm = ctx.llm()
fast_llm = ctx.llm("fast")
```

Model adapters are loaded lazily, so applications install only the provider packages they use.

MCP servers are registered on the application, allowed by the agent, and selected by the task. `ctx.mcp_tools()` returns a name-to-tool mapping that can be bound to an LLM:

```python
tool_map = await ctx.mcp_tools()
llm = ctx.llm().bind_tools(list(tool_map.values()))
```

Install MCP support with `pip install 'switchplane[mcp]'`. Stdio and streamable HTTP servers are supported, including OAuth for HTTP servers.

Tasks can also submit child tasks, wait for one or many results, and exchange notifications through `ctx`. Child relationships are persisted, and cancelling a parent cascades to non-terminal children.

## Examples

Install all example packages from a repository checkout with `make install-examples`, or install one with `uv pip install -e examples/<name>`. LLM-backed examples also require provider credentials and the model adapter they use.

| Example | Demonstrates | Setup | Run |
|---|---|---|---|
| [`hello`](examples/hello) | Minimal two-node LangGraph task | None | `hello run example hello --user-name Alice` |
| [`weather`](examples/weather) | Long-running polling, commands, cancellation, and checkpoint retry | Network access | `weather run weather watch` |
| [`chatbot`](examples/chatbot) | LangGraph interrupts and freeform input | Anthropic API key | `chatbot run bot chat` |
| [`devops`](examples/devops) | Deterministic metrics analysis with model-based triage and summarization | API key plus model adapter | `devops run sre review` |
| [`quality`](examples/quality) | Multi-model review fan-out, tool use, synthesis, and GitHub or local output | Provider keys and GitHub access | `quality run pr review --pr <url> --local` |

The devops example captures the core Switchplane thesis: ordinary Python fetches and analyzes metrics, model calls make bounded judgment calls, and deterministic code compiles the report. The quality example demonstrates a larger graph with concurrent provider and review-domain branches. See its [example README](examples/quality/README.md) for the full architecture and security model.

To run devops with its default model, install `langchain-anthropic`:

```bash
uv pip install langchain-anthropic
mkdir -p ~/.devops
```

Then add the key to `~/.devops/config.toml` without replacing any existing settings:

```toml
[llm]
api_key = "sk-ant-..."
```

Run the review with `devops run sre review`, then inspect its stored report with `devops task show <task_id>`.

## Project Status

Switchplane is alpha software. It currently provides a local, POSIX-only runtime built on Unix sockets and process forking.

- Python 3.12+
- LangGraph-native task execution
- Per-application daemon and runtime directory
- No hosted service, account, or browser dashboard
- Apache-2.0 license

Switchplane deliberately handles how a task runs, not what it does. Your application owns prompts, tools, retrieval, domain rules, and graph design. Switchplane owns the process model and operational controls around them.
