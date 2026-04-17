"""Project scaffolding for Switchplane applications."""

import keyword
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Templates
#
# Each template uses a plain {project_name} placeholder, substituted via
# str.replace().  This avoids the brace-escaping headaches that .format()
# would cause inside Python source code.
# ---------------------------------------------------------------------------

PYPROJECT_TOML = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{project_name}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "switchplane",
]

[project.scripts]
{project_name} = "{project_name}.app:main"
"""

APP_PY = """\
from switchplane import Application

app = Application(name="{project_name}")
app.discover_agents("{project_name}.agents")


def main():
    app.run()
"""

AGENT_PY = """\
from switchplane.agent import AgentSpec

agent_spec = AgentSpec(
    agent_name="default",
)
"""

HELLO_TASK_PY = """\
from typing import TypedDict

from langgraph.graph import END, StateGraph

from switchplane import Field, Task
from switchplane.agent_runtime import AgentContext


class HelloState(TypedDict):
    name: str | None
    greeting: str | None


def greet(state: HelloState) -> HelloState:
    name = state["name"] or "World"
    return {"name": name, "greeting": f"Hello, {name}! Welcome to Switchplane."}


def build_graph() -> StateGraph:
    graph = StateGraph(HelloState)
    graph.add_node("greet", greet)
    graph.set_entry_point("greet")
    graph.add_edge("greet", END)
    return graph


class HelloTask(Task):
    name = "hello"
    description = "Greet a user by name"

    user_name: str | None = Field(default=None, description="Name to greet")

    async def run(self, ctx: AgentContext) -> None:
        ctx.progress("Running hello workflow")
        app = build_graph().compile()
        result = await app.ainvoke({"name": self.user_name, "greeting": None})
        ctx.complete({"greeting": result["greeting"]})
"""

# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

# (relative_path_template, content_or_None)
# Paths containing {project_name} are substituted at generation time.
# A content value of None means an empty file (used for __init__.py).

_FILES: list[tuple[str, str | None]] = [
    ("pyproject.toml", PYPROJECT_TOML),
    ("{project_name}/__init__.py", None),
    ("{project_name}/app.py", APP_PY),
    ("{project_name}/agents/__init__.py", None),
    ("{project_name}/agents/default/__init__.py", None),
    ("{project_name}/agents/default/agent.py", AGENT_PY),
    ("{project_name}/agents/default/tasks/__init__.py", None),
    ("{project_name}/agents/default/tasks/hello.py", HELLO_TASK_PY),
]


def _validate_project_name(name: str) -> str | None:
    """Return an error message if *name* is not a valid Python identifier, else None."""
    if not name.isidentifier():
        return f"'{name}' is not a valid Python identifier."
    if keyword.iskeyword(name):
        return f"'{name}' is a reserved Python keyword."
    return None


def generate_project(project_name: str, parent: Path) -> Path:
    """Create a new Switchplane project directory under *parent*.

    Returns the path to the created project root.
    """
    project_dir = parent / project_name

    if project_dir.exists():
        raise click.ClickException(f"Directory '{project_name}' already exists.")

    for rel_path_template, content_template in _FILES:
        rel_path = rel_path_template.replace("{project_name}", project_name)
        file_path = project_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if content_template is None:
            file_path.touch()
        else:
            file_path.write_text(content_template.replace("{project_name}", project_name))

    return project_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Switchplane project tools."""


@cli.command()
@click.argument("project_name")
def init(project_name: str) -> None:
    """Create a new Switchplane project."""
    error = _validate_project_name(project_name)
    if error:
        raise click.ClickException(error)

    project_dir = generate_project(project_name, Path.cwd())

    click.echo(f"\nCreated project '{project_name}' at {project_dir}.\n")
    click.echo("Next steps:")
    click.echo(f"  cd {project_name}")
    click.echo("  uv venv .venv && source .venv/bin/activate")
    click.echo("  uv pip install -e .")
    click.echo(f"  {project_name} agent list")
