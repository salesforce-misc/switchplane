"""Hello task - LangGraph graph with get_user and say_hello nodes."""

import getpass
from typing import TypedDict

from langgraph.graph import END, StateGraph

from switchplane import Field, Task
from switchplane.agent_runtime import AgentContext

# -- Graph state --


class HelloState(TypedDict):
    name: str | None
    greeting: str | None


# -- Framework tool --


def get_current_user_name() -> str:
    """Framework-native tool: get the current system username."""
    return getpass.getuser()


# -- Graph nodes --


def get_user(state: HelloState) -> HelloState:
    """Resolve the user name, falling back to the system username."""
    name = state.get("name")
    if not name:
        name = get_current_user_name()
    return {"name": name, "greeting": state.get("greeting")}


def say_hello(state: HelloState) -> HelloState:
    """Generate the greeting."""
    name = state["name"]
    return {"name": name, "greeting": f"Hello, {name}! Welcome to Switchplane."}


# -- Build the graph --


def build_graph() -> StateGraph:
    graph = StateGraph(HelloState)
    graph.add_node("get_user", get_user)
    graph.add_node("say_hello", say_hello)
    graph.set_entry_point("get_user")
    graph.add_edge("get_user", "say_hello")
    graph.add_edge("say_hello", END)
    return graph


# -- Task implementation --


class HelloTask(Task):
    name = "hello"
    description = "Greet a user by name"

    user_name: str | None = Field(default=None, description="Name to greet")

    async def run(self, ctx: AgentContext) -> None:
        """Execute the hello task as a LangGraph graph."""
        ctx.progress("Building LangGraph workflow")

        app = build_graph().compile()

        initial_state: HelloState = {
            "name": self.user_name,
            "greeting": None,
        }

        ctx.progress("Executing: get_user -> say_hello")
        result = await app.ainvoke(initial_state)

        ctx.complete({"greeting": result["greeting"]})
