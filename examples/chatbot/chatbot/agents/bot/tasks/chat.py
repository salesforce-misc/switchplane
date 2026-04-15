"""Interactive chat task -- demonstrates freeform input with LangGraph interrupt."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from switchplane import Task, command
from switchplane.agent_runtime import AgentContext
from switchplane.llm import build_llm


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(llm):
    async def respond(state: ChatState) -> ChatState:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    def wait_for_user(state: ChatState) -> ChatState:
        user_text = interrupt("You: ")
        return {"messages": [HumanMessage(content=user_text)]}

    graph = StateGraph(ChatState)
    graph.add_node("respond", respond)
    graph.add_node("wait_for_user", wait_for_user)
    graph.set_entry_point("respond")
    graph.add_edge("respond", "wait_for_user")
    graph.add_edge("wait_for_user", "respond")
    return graph


class ChatTask(Task):
    name = "chat"
    description = "Interactive chat with an LLM"
    mode = "long_running"

    _ending: bool = False

    @command
    def end(self, ctx: AgentContext):
        """End the chat session."""
        self._ending = True
        # Unblock wait_for_input by injecting a sentinel input
        ctx._command_queue.put_nowait({"action": "__input__", "params": {"text": ""}})
        return {"status": "ending"}

    async def run(self, ctx: AgentContext) -> None:
        llm_config = ctx.config.get("llm", {})
        model = llm_config.get("model", "claude-sonnet-4-20250514")
        llm = build_llm(model, llm_config.get("api_key"), llm_config.get("base_url"))

        graph = build_graph(llm).compile(checkpointer=ctx.checkpointer)
        config = {"configurable": {"thread_id": ctx.task_id}}

        initial_state: ChatState = {
            "messages": [
                SystemMessage(content="You are a helpful assistant. Be concise."),
                HumanMessage(content="Hello!"),
            ],
        }

        # First turn: LLM responds to "Hello!", then graph interrupts at wait_for_user
        result = await graph.ainvoke(initial_state, config)
        last_msg = result["messages"][-1]
        ctx.progress(f"Assistant: {last_msg.content}")

        # Chat loop
        while not self._ending:
            user_input = await ctx.wait_for_input("You: ")
            if self._ending or not user_input:
                break

            # Resume the graph: wait_for_user returns with user text, then respond runs
            result = await graph.ainvoke(Command(resume=user_input), config)
            last_msg = result["messages"][-1]
            ctx.progress(f"Assistant: {last_msg.content}")

        ctx.complete({"turns": len(result.get("messages", [])) // 2})
