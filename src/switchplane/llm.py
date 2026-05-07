"""LLM integration for Switchplane applications.

Provides a default build_llm that uses native LangChain adapters.
Apps can override this with their own implementation (e.g. for
API gateways).

Requires one or more optional LangChain adapter packages:
  - langchain-anthropic (Claude models)
  - langchain-google-genai (Gemini models)
  - langchain-openai (OpenAI models)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from switchplane.agent_runtime import AgentContext

__all__ = [
    "MODELS",
    "ModelInfo",
    "Tool",
    "build_llm",
    "context_window",
    "extract_response_text",
    "run_tool_loop",
]

from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from switchplane.config import DEFAULT_MODEL


class Tool:
    """Wrapper associating a LangChain tool with an optional render function.

    When used in a tool_map, the render_fn is called on each invocation to
    emit progress info. If None, the invocation is silent (useful for tools
    that emit their own events, e.g. file_edit).

    Backwards-compatible: run_tool_loop also accepts bare StructuredTool
    instances in the tool_map and falls back to default rendering.

    Note: not a BaseTool subclass — LangChain's bind_tools requires BaseTool
    instances, so callers must pass [t.tool for t in tools] to bind_tools.
    """

    def __init__(self, tool: Any, render_fn: Callable[["AgentContext", str, dict], None] | None = None):
        self.tool = tool
        self.name = tool.name
        self.render_fn = render_fn

    async def ainvoke(self, args: dict) -> Any:
        return await self.tool.ainvoke(args)


class ModelInfo(NamedTuple):
    name: str
    context_window: int


# Well-known public models — apps can use these immediately
MODELS: dict[str, ModelInfo] = {
    # Anthropic — 200k context
    "claude-sonnet-4-20250514": ModelInfo("claude-sonnet-4-20250514", 200_000),
    "claude-sonnet-4-5-20250929": ModelInfo("claude-sonnet-4-5-20250929", 200_000),
    "claude-sonnet-4-6": ModelInfo("claude-sonnet-4-6", 200_000),
    "claude-opus-4-20250514": ModelInfo("claude-opus-4-20250514", 200_000),
    "claude-opus-4-6-v1": ModelInfo("claude-opus-4-6-v1", 200_000),
    "claude-haiku-4-5-20251001": ModelInfo("claude-haiku-4-5-20251001", 200_000),
    # Google — 1M context
    "gemini-2.0-flash": ModelInfo("gemini-2.0-flash", 1_000_000),
    "gemini-2.5-pro": ModelInfo("gemini-2.5-pro", 1_000_000),
    "gemini-2.5-flash": ModelInfo("gemini-2.5-flash", 1_000_000),
    # OpenAI — 128k context
    "gpt-4o": ModelInfo("gpt-4o", 128_000),
    "gpt-4o-mini": ModelInfo("gpt-4o-mini", 128_000),
}


def context_window(model: str) -> int:
    """Return the context window size for a model."""
    info = MODELS.get(model)
    return info.context_window if info else 200_000


def build_llm(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Instantiate a LangChain chat model.

    Routes to the appropriate LangChain adapter based on model name prefix:
      - claude-* -> ChatAnthropic (requires langchain-anthropic)
      - gemini-* -> ChatGoogleGenerativeAI (requires langchain-google-genai)
      - gpt-*    -> ChatOpenAI (requires langchain-openai)
    """
    if model.startswith("claude"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Claude models require langchain-anthropic: pip install langchain-anthropic") from None
        kwargs = {"model": model}
        if api_key:
            kwargs["anthropic_api_key"] = api_key
        if base_url:
            kwargs["anthropic_api_url"] = base_url
        return ChatAnthropic(**kwargs)
    elif model.startswith("gemini"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "Gemini models require langchain-google-genai: pip install langchain-google-genai"
            ) from None
        kwargs = {"model": model}
        if api_key:
            kwargs["google_api_key"] = api_key
        return ChatGoogleGenerativeAI(**kwargs)
    elif model.startswith("gpt"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("OpenAI models require langchain-openai: pip install langchain-openai") from None
        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    else:
        raise ValueError(f"Unknown model prefix: {model}. Expected claude-*, gemini-*, or gpt-*")


# ---------------------------------------------------------------------------
# Tool-loop utilities
# ---------------------------------------------------------------------------

_MAX_TOOL_RESULT_CHARS = 8_000
_MAX_REPEAT_CALLS = 3


def extract_response_text(content) -> str:
    """Extract concatenated text blocks from an LLM response content field.

    Handles three forms:
      - A list of content blocks (standard structured response).
      - A string that looks like a serialized list of dicts (checkpoint
        deserialization artifact) — parsed via ast.literal_eval.
      - A plain string (returned as-is).
    """
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    if isinstance(content, str) and content.startswith("[{"):
        import ast

        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, list):
                return "\n".join(
                    block.get("text", "") for block in parsed if isinstance(block, dict) and block.get("type") == "text"
                )
        except (ValueError, SyntaxError):
            pass
    return content


async def run_tool_loop(
    llm_with_tools,
    messages: list,
    tool_map: dict,
    ctx: AgentContext,
    model_name: str,
    *,
    label: str = "Working",
    max_retries: int = 1,
    truncate_results: bool = True,
    progress_every: int | None = None,
):
    """Drive an LLM tool-calling loop until the model produces a final answer.

    Parameters
    ----------
    llm_with_tools:
        A LangChain chat model already bound with tools.
    messages:
        The conversation message list (mutated in place).
    tool_map:
        Mapping of tool name -> langchain tool instance.
    ctx:
        Agent context providing stream_flush, progress, and tool_invoke.
    model_name:
        Model identifier used to determine context window for trimming.
    label:
        Human-readable label for progress messages.
    max_retries:
        Number of attempts per tool call on transient failures (timeout /
        cancellation). Set to 1 for no retries.
    truncate_results:
        Whether to truncate long tool results to _MAX_TOOL_RESULT_CHARS.
    progress_every:
        If set, emit a progress message every N turns. None disables.
    """
    from langchain_core.messages.utils import trim_messages

    turn = 0
    last_sig: str | None = None
    repeat_count = 0

    while True:
        trimmed = trim_messages(
            messages,
            max_tokens=context_window(model_name),
            token_counter="approximate",
            strategy="last",
            include_system=True,
            start_on="human",
        )
        messages.clear()
        messages.extend(trimmed)

        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            text = extract_response_text(response.content)
            if text.strip():
                ctx.stream_flush(text.strip())
            return response

        turn += 1

        # Detect repeated identical tool-call sequences.
        sig = json.dumps(
            [(tc["name"], tc["args"]) for tc in response.tool_calls],
            sort_keys=True,
        )
        if sig == last_sig:
            repeat_count += 1
            if repeat_count >= _MAX_REPEAT_CALLS:
                ctx.progress(f"{label}: detected repeated tool calls, stopping.")
                return response
        else:
            last_sig = sig
            repeat_count = 1

        if progress_every and turn % progress_every == 0:
            msg = f"Still {label.lower()} ({turn} turns)..."
            if hasattr(ctx, "task_id") and ctx.task_id:
                msg += f" Cancel with: task cancel {ctx.task_id}"
            ctx.progress(msg)

        text = extract_response_text(response.content)
        if text.strip():
            ctx.stream_flush(text.strip())

        for tc in response.tool_calls:
            entry = tool_map.get(tc["name"])
            if not entry:
                messages.append(
                    {"role": "tool", "content": f"Error: unknown tool '{tc['name']}'", "tool_call_id": tc["id"]}
                )
                continue

            if isinstance(entry, Tool):
                if entry.render_fn is not None:
                    entry.render_fn(ctx, tc["name"], tc["args"])
                tool = entry.tool
            else:
                args_summary = " ".join((str(v).splitlines() or [""])[0][:80] for v in tc["args"].values())
                ctx.tool_invoke(tc["name"], args_summary)
                tool = entry

            result: str | None = None
            for attempt in range(max_retries):
                try:
                    result = str(await tool.ainvoke(tc["args"]))
                    if truncate_results and len(result) > _MAX_TOOL_RESULT_CHARS:
                        result = result[:_MAX_TOOL_RESULT_CHARS] + f"\n...[truncated, {len(result)} chars total]"
                    break
                except TimeoutError:
                    if attempt < max_retries - 1:
                        ctx.progress(f"Tool call timed out, retrying ({attempt + 2}/{max_retries})...")
                    else:
                        result = f"Error: tool call timed out after {max_retries} attempts"
                except Exception as e:
                    result = f"Error: {e}"
                    break

            messages.append({"role": "tool", "content": result, "tool_call_id": tc["id"]})
