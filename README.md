# Switchplane

**Switchplane turns LangGraph workflows into locally operable applications, with a generated CLI, supervised task processes, persistent task history, and checkpoint recovery.**

> If it is deterministic, write it in code. If it requires judgment, call the model.

Most agent harnesses begin with a model-directed loop: give an LLM tools and let it decide what happens next. Switchplane begins with an application-defined workflow. You write the nodes, edges, state, and failure boundaries in Python with [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview); LLM calls are just nodes where nondeterminism is useful.

```text
fetch_metrics -> analyze -> summarize -> compile_report
    Python        Python        LLM           Python
```

That distinction matters when the workflow must be testable, inspectable, cost-conscious, and operable after the first prompt. Switchplane adds a per-application control plane around the graph: validated task inputs, supervised subprocesses, persisted task records and events, live commands, child tasks, and opt-in LangGraph checkpoints.

Switchplane is not a coding agent or a replacement for LangGraph. It is the local runtime that turns your workflows into an operational CLI.

> **Alpha software:** APIs, IPC protocols, and storage formats may change.

## Quickstart

Requires a Unix-like OS and Python 3.12+. The commands below use [uv](https://docs.astral.sh/uv/); equivalent `venv` and pip commands also work.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install 'switchplane==0.10.2'

switchplane init myapp
cd myapp
uv pip install -e .

myapp agent list
myapp run default hello --user-name Alice
```

`switchplane init` generates a complete package with an application, agent, LangGraph task, and CLI entry point:

```text
myapp/
├── pyproject.toml
└── myapp/
    ├── app.py
    └── agents/default/
        ├── agent.py
        └── tasks/hello.py
```

The daemon starts on demand. The attached command streams events inline; `Ctrl+C` detaches without stopping the task. Use the printed task ID with `myapp task show <task-id>` to inspect the structured greeting. Run `myapp` with no subcommand to open the full-screen TUI.

## Runtime model

```text
myapp CLI/TUI
      |
      | Unix socket
      v
per-app control plane ----> SQLite task/event store
      |
      | dedicated socketpair
      v
supervised task subprocess ----> LangGraph workflow
```

Each application gets its own runtime under `~/.<app-name>/`. The control plane owns task and event persistence; task execution runs in a child process. A task has a stable ID, validated input, lifecycle status, structured result or error, and durable application events. Streaming token chunks are live-only.

The process boundary protects the control plane from task crashes and permits bidirectional commands. It is **not** a security sandbox: task processes retain the current user's filesystem, environment, and network access.

## How it differs

[OpenCode](https://opencode.ai/docs/) and [Pi](https://pi.dev/docs/latest) are capable, extensible coding-agent harnesses. Their primary abstraction is an interactive session driven by an LLM/tool loop. Switchplane's primary abstraction is an application task whose control flow is defined in code.

| | OpenCode | Pi | Switchplane |
|---|---|---|---|
| Primary use | Coding agent | Coding-agent harness | Runtime for agentic applications |
| Default control flow | Model-directed tool loop | Model-directed tool loop | Application-defined graph |
| Durable unit | Session and messages | Session tree | Task, events, result, optional graph state |
| Composition | Agents, subagents, plugins | Extensions, skills, subprocess patterns | Graph nodes and linked child tasks |
| Recovery | Continue a session | Continue a session | Explicitly retry from an opt-in LangGraph checkpoint |
| Task execution | Session/server runtime | In-process by default; isolation is opt-in | Supervised subprocess per task |
| Operator surface | TUI, desktop, web, server, SDK | TUI, print, RPC, SDK | Generated per-app CLI and TUI |

This is a difference in defaults, not capability. OpenCode plugins and Pi extensions can implement deterministic orchestration; Switchplane makes application-authored workflow topology and task operations the starting point. OpenCode and Pi are stronger choices for interactive coding, broad client surfaces, and more extensive documented agent customization. Switchplane is for workflows where the model should participate in the process, not own it.

## Write a workflow

The generated application discovers tasks and becomes the installed CLI:

```python
# myapp/app.py
from switchplane import Application

app = Application(name="myapp")
app.discover_agents("myapp.agents")


def main():
    app.run()
```

Tasks declare Pydantic-validated inputs and execute ordinary async Python. Switchplane is designed to host LangGraph graphs, but it does not force every task to use one.

```python
# myapp/agents/default/tasks/report.py
from typing import TypedDict

from langgraph.graph import END, StateGraph
from switchplane import Field, Task
from switchplane.agent_runtime import AgentContext


class State(TypedDict):
    source: str
    result: str | None


def analyze(state: State) -> State:
    return {**state, "result": f"analyzed {state['source']}"}


class Report(Task):
    name = "report"
    description = "Analyze a source"

    source: str = Field(description="Input to analyze")

    async def run(self, ctx: AgentContext) -> None:
        graph = StateGraph(State)
        graph.add_node("analyze", analyze)
        graph.set_entry_point("analyze")
        graph.add_edge("analyze", END)

        result = await graph.compile().ainvoke(
            {"source": self.source, "result": None}
        )
        ctx.complete(result)
```

```bash
myapp run default report --source metrics.json
```

Keep deterministic work in ordinary nodes. Put model calls only in nodes that need classification, synthesis, planning, or other judgment. The runtime enforces execution mechanics, not whether your node is deterministic.

## Operate tasks

Every Switchplane app exposes the same command structure:

```bash
# Execute
<app> run <agent> <task> [--param value ...]
<app> run <agent> <task> [--param value ...] --detach

# Inspect and control
<app> agent list
<app> task list [--status running]
<app> task show <task-id>
<app> task follow <task-id>
<app> task cancel <task-id>
<app> task retry <task-id> [-d]
<app> task clear
<app> task purge

# Manage the daemon
<app> runtime start
<app> runtime status
<app> runtime stop
```

`task clear` hides terminal tasks by marking them `cleared`; data remains until `task purge`. Purging is refused while tasks are running.

### Interactive tasks

Long-running tasks can receive typed commands without restarting:

```python
from switchplane import Task, command
from switchplane.agent_runtime import AgentContext


class Watch(Task):
    name = "watch"
    mode = "long_running"

    @command
    def coordinates(self, ctx: AgentContext, lat: float, lon: float):
        self.lat, self.lon = lat, lon
        return {"lat": lat, "lon": lon}

    async def run(self, ctx: AgentContext) -> None:
        while not ctx.is_cancelled:
            await self.process_commands(ctx)
            await ctx.sleep(60)
```

```bash
<app> task <task-id> coordinates --lat 51.5074 --lon -0.1278
```

`mode = "long_running"` is descriptive metadata; the task implements its own loop. With `ctx.checkpointer` configured, a task can also call `ctx.wait_for_input()` to enter the `interrupted` state and accept freeform text.

In attached CLI mode, type `/command`; `Ctrl+C` detaches. In the TUI:

| Input | Action |
|---|---|
| `:run ...`, `:task ...`, `:runtime ...` | Control-plane command |
| `/command ...` | Command for the focused task |
| Plain text | Input for an interrupted task |
| `Tab`, `0`-`9` | Switch tabs |
| `Ctrl+X`, `Ctrl+D`, `Ctrl+C` | Cancel, detach, quit |

The TUI receives pushed events over a persistent subscription. Attached CLI mode currently polls for incremental events.

### Checkpoint and retry

Opt a LangGraph task into persisted graph state:

```python
async def run(self, ctx: AgentContext) -> None:
    graph = build_graph().compile(checkpointer=ctx.checkpointer)
    config = {"configurable": {"thread_id": ctx.task_id}}
    checkpoint = await graph.aget_state(config)
    graph_input = None if checkpoint.values else initial_state
    result = await graph.ainvoke(graph_input, config)
    ctx.complete(result)
```

```bash
myapp task retry <task-id>
```

Checkpoints survive agent and daemon exits. Recovery is explicit, not automatic: startup marks orphaned live tasks failed, and `task retry` relaunches the same task ID. The task must detect the saved state and resume LangGraph with `None`, as above; passing `initial_state` starts again at the entry point. Without `ctx.checkpointer`, retry reruns the task from the beginning.

### Child tasks

Tasks can submit linked work through the control plane:

```python
child_id = await ctx.submit_task("worker", "process", {"chunk": 1})
child = await ctx.wait_for_task(child_id)

await ctx.notify_task(other_task_id, {"status": "ready"})
message = await ctx.wait_for_notification(timeout=60)
```

`wait_for_task()` polls durable task state. Notifications are transient and require the target task to be running. Cancelling a parent cascades to its children.

## Configuration

Applications can bundle defaults and users override them at `~/.<app-name>/config.toml`. Values are deep-merged, and each agent receives global settings merged with its own overrides.

```python
from pathlib import Path
from switchplane import Application

app = Application(
    name="myapp",
    default_config=Path(__file__).parent / "config.toml",
)
```

```toml
# Bundled defaults
[llm]
model = "claude-sonnet-4-20250514"

[agents.reviewer]
system_prompt = "Review the supplied evidence."
```

```toml
# ~/.myapp/config.toml
[llm]
api_key = "..."

[agents.reviewer.llm]
model = "claude-haiku-4-5-20251001"
```

Configuration is available as `ctx.config`. The optional `switchplane.llm` helper routes `claude-*`, `gemini-*`, and `gpt-*` models to their LangChain adapters. Applications can supply their own factory for other providers or local models.

## MCP

Register stdio or streamable-HTTP [MCP](https://modelcontextprotocol.io/) servers on the application, make them available to an agent, and select the servers required by each task:

```python
# app.py
from switchplane.app import McpServerConfig

app.register_mcp_server(McpServerConfig(
    name="tools",
    command=["python", "tools_server.py"],
))
```

```python
# agent.py
from switchplane.agent import AgentSpec

agent_spec = AgentSpec(agent_name="default", mcp_servers=["tools"])
```

```python
class Analyze(Task):
    name = "analyze"
    mcp_servers = ["tools"]

    async def run(self, ctx: AgentContext) -> None:
        tool_map = await ctx.mcp_tools()
        tools = list(tool_map.values())
```

Install with `pip install 'switchplane[mcp]'`. HTTP servers may use built-in PKCE authentication (MCP OAuth discovery or explicitly configured OIDC):

```bash
<app> auth login <server>
<app> auth status
<app> auth logout <server>
```

## Guardrailed shell

`Shell` executes argument arrays without shell interpretation and can restrict executable basenames, working directories, time, and explicitly declared path parameters.

```python
from pathlib import Path
from switchplane import Shell

shell = Shell(
    allowed_paths=[Path.cwd()],
    allowed_commands=["git", "rg"],
)

output = await shell.run(["git", "status", "--short"], cwd=Path.cwd())
```

These checks are defense in depth, not an OS sandbox. Ordinary command arguments are not treated as paths unless the tool declares them as path parameters.

## Examples

| Example | Demonstrates |
|---|---|
| [`hello`](examples/hello/) | Minimal two-node graph |
| [`devops`](examples/devops/) | Three deterministic analysis/report nodes around one LLM summary node |
| [`weather`](examples/weather/) | Long-running task, commands, cancellation, checkpoints |
| [`chatbot`](examples/chatbot/) | LangGraph interrupts and freeform input |

Run the examples from a source checkout:

```bash
uv pip install -e . -e examples/hello -e examples/weather
hello run example hello --user-name Alice
weather run weather watch
```

The `devops` and `chatbot` examples require a matching LangChain provider adapter and API key; see their package and config files before running them.

## Scope and limits

- Local, Unix-only runtime; no hosted service or account.
- LangGraph is a deliberate dependency, not an abstraction hidden behind Switchplane.
- Separate task processes are not security sandboxes and have no built-in resource quotas.
- Durable lifecycle and application events do not include ephemeral token chunks or automatic graph-node tracing.
- Checkpoint recovery is opt-in and operator-triggered.
- The runtime handles how a task runs. Prompting, retrieval, memory, and domain logic remain application concerns.

## Development

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e '.[test]'
make test
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the [code of conduct](CODE_OF_CONDUCT.md). Licensed under [Apache 2.0](LICENSE).
