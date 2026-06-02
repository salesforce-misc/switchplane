"""Tests for switchplane.llm — model registry and build_llm factory."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from switchplane.llm import (
    _MAX_REPEAT_CALLS,
    _MAX_TOOL_RESULT_CHARS,
    DEFAULT_MODEL,
    MODELS,
    ModelInfo,
    build_llm,
    context_window,
    extract_response_text,
    run_tool_loop,
)

# ---------------------------------------------------------------------------
# ModelInfo / MODELS registry
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_is_named_tuple(self):
        info = ModelInfo(name="my-model", context_window=100_000)
        assert info.name == "my-model"
        assert info.context_window == 100_000

    def test_tuple_unpacking(self):
        name, window = ModelInfo("x", 42)
        assert name == "x"
        assert window == 42


class TestModelsRegistry:
    def test_default_model_present(self):
        assert DEFAULT_MODEL in MODELS

    def test_all_anthropic_models_have_200k_context(self):
        anthropic = [k for k in MODELS if k.startswith("claude")]
        assert anthropic, "No Anthropic models in registry"
        for key in anthropic:
            assert MODELS[key].context_window == 200_000, key

    def test_all_gemini_models_have_1m_context(self):
        gemini = [k for k in MODELS if k.startswith("gemini")]
        assert gemini, "No Gemini models in registry"
        for key in gemini:
            assert MODELS[key].context_window == 1_000_000, key

    def test_all_openai_models_have_128k_context(self):
        openai = [k for k in MODELS if k.startswith("gpt")]
        assert openai, "No OpenAI models in registry"
        for key in openai:
            assert MODELS[key].context_window == 128_000, key

    def test_model_name_matches_key(self):
        for key, info in MODELS.items():
            assert info.name == key


# ---------------------------------------------------------------------------
# context_window
# ---------------------------------------------------------------------------


class TestContextWindow:
    def test_known_anthropic_model(self):
        assert context_window("claude-sonnet-4-20250514") == 200_000

    def test_known_gemini_model(self):
        assert context_window("gemini-2.0-flash") == 1_000_000

    def test_known_openai_model(self):
        assert context_window("gpt-4o") == 128_000

    def test_unknown_model_falls_back_to_200k(self):
        assert context_window("unknown-future-model") == 200_000

    def test_empty_string_falls_back(self):
        assert context_window("") == 200_000

    def test_returns_int(self):
        assert isinstance(context_window(DEFAULT_MODEL), int)


# ---------------------------------------------------------------------------
# build_llm — helpers
# ---------------------------------------------------------------------------


def _mock_module(cls_name: str) -> tuple[MagicMock, MagicMock]:
    """Return (module_mock, class_mock) with class_mock set as cls_name attr."""
    cls_mock = MagicMock()
    mod_mock = MagicMock()
    setattr(mod_mock, cls_name, cls_mock)
    return mod_mock, cls_mock


# ---------------------------------------------------------------------------
# build_llm — Anthropic (claude-*)
# ---------------------------------------------------------------------------


class TestBuildLlmAnthropic:
    def test_returns_chat_anthropic_instance(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            result = build_llm("claude-sonnet-4-20250514")
        assert result is cls.return_value

    def test_passes_model_name(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            build_llm("claude-opus-4-20250514")
        cls.assert_called_once_with(model="claude-opus-4-20250514")

    def test_passes_api_key_when_provided(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            build_llm("claude-sonnet-4-20250514", api_key="sk-ant-test")
        _, kwargs = cls.call_args
        assert kwargs.get("anthropic_api_key") == "sk-ant-test"

    def test_passes_base_url_when_provided(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            build_llm("claude-sonnet-4-20250514", base_url="https://my-proxy/")
        _, kwargs = cls.call_args
        assert kwargs.get("anthropic_api_url") == "https://my-proxy/"

    def test_omits_api_key_when_none(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            build_llm("claude-sonnet-4-20250514")
        _, kwargs = cls.call_args
        assert "anthropic_api_key" not in kwargs

    def test_omits_base_url_when_none(self):
        mod, cls = _mock_module("ChatAnthropic")
        with patch.dict(sys.modules, {"langchain_anthropic": mod}):
            build_llm("claude-sonnet-4-20250514")
        _, kwargs = cls.call_args
        assert "anthropic_api_url" not in kwargs

    def test_import_error_raises_with_hint(self):
        with (
            patch.dict(sys.modules, {"langchain_anthropic": None}),
            pytest.raises(ImportError, match="langchain-anthropic"),
        ):
            build_llm("claude-sonnet-4-20250514")


# ---------------------------------------------------------------------------
# build_llm — Google (gemini-*)
# ---------------------------------------------------------------------------


class TestBuildLlmGemini:
    def test_returns_chat_google_instance(self):
        mod, cls = _mock_module("ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": mod}):
            result = build_llm("gemini-2.0-flash")
        assert result is cls.return_value

    def test_passes_model_name(self):
        mod, cls = _mock_module("ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": mod}):
            build_llm("gemini-2.5-pro")
        cls.assert_called_once_with(model="gemini-2.5-pro")

    def test_passes_api_key_when_provided(self):
        mod, cls = _mock_module("ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": mod}):
            build_llm("gemini-2.0-flash", api_key="gai-key")
        _, kwargs = cls.call_args
        assert kwargs.get("google_api_key") == "gai-key"

    def test_omits_api_key_when_none(self):
        mod, cls = _mock_module("ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": mod}):
            build_llm("gemini-2.0-flash")
        _, kwargs = cls.call_args
        assert "google_api_key" not in kwargs

    def test_base_url_not_forwarded(self):
        """Gemini adapter has no base_url kwarg; build_llm must not pass it."""
        mod, cls = _mock_module("ChatGoogleGenerativeAI")
        with patch.dict(sys.modules, {"langchain_google_genai": mod}):
            build_llm("gemini-2.0-flash", base_url="https://proxy/")
        _, kwargs = cls.call_args
        assert "base_url" not in kwargs

    def test_import_error_raises_with_hint(self):
        with (
            patch.dict(sys.modules, {"langchain_google_genai": None}),
            pytest.raises(ImportError, match="langchain-google-genai"),
        ):
            build_llm("gemini-2.0-flash")

    def test_import_error_when_not_installed(self):
        # langchain_google_genai is not installed in this environment —
        # removing any cached mock ensures the real absence is exercised.
        saved = sys.modules.pop("langchain_google_genai", None)
        try:
            with pytest.raises(ImportError, match="langchain-google-genai"):
                build_llm("gemini-2.0-flash")
        finally:
            if saved is not None:
                sys.modules["langchain_google_genai"] = saved


# ---------------------------------------------------------------------------
# build_llm — OpenAI (gpt-*)
# ---------------------------------------------------------------------------


class TestBuildLlmOpenAI:
    def test_returns_chat_openai_instance(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            result = build_llm("gpt-4o")
        assert result is cls.return_value

    def test_passes_model_name(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            build_llm("gpt-4o-mini")
        cls.assert_called_once_with(model="gpt-4o-mini")

    def test_passes_api_key_when_provided(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            build_llm("gpt-4o", api_key="sk-openai")
        _, kwargs = cls.call_args
        assert kwargs.get("api_key") == "sk-openai"

    def test_passes_base_url_when_provided(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            build_llm("gpt-4o", base_url="https://oai-proxy/v1")
        _, kwargs = cls.call_args
        assert kwargs.get("base_url") == "https://oai-proxy/v1"

    def test_omits_api_key_when_none(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            build_llm("gpt-4o")
        _, kwargs = cls.call_args
        assert "api_key" not in kwargs

    def test_omits_base_url_when_none(self):
        mod, cls = _mock_module("ChatOpenAI")
        with patch.dict(sys.modules, {"langchain_openai": mod}):
            build_llm("gpt-4o")
        _, kwargs = cls.call_args
        assert "base_url" not in kwargs

    def test_import_error_raises_with_hint(self):
        with patch.dict(sys.modules, {"langchain_openai": None}), pytest.raises(ImportError, match="langchain-openai"):
            build_llm("gpt-4o")

    def test_import_error_when_not_installed(self):
        saved = sys.modules.pop("langchain_openai", None)
        try:
            with (
                patch.dict(sys.modules, {"langchain_openai": None}),
                pytest.raises(ImportError, match="langchain-openai"),
            ):
                build_llm("gpt-4o")
        finally:
            if saved is not None:
                sys.modules["langchain_openai"] = saved


# ---------------------------------------------------------------------------
# build_llm — unknown prefix
# ---------------------------------------------------------------------------


class TestBuildLlmUnknownPrefix:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model prefix"):
            build_llm("llama-3-70b")

    def test_error_message_includes_model(self):
        with pytest.raises(ValueError, match="llama-3-70b"):
            build_llm("llama-3-70b")

    def test_error_message_lists_known_prefixes(self):
        with pytest.raises(ValueError, match=r"claude-\*"):
            build_llm("mistral-large")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Unknown model prefix"):
            build_llm("")


# ---------------------------------------------------------------------------
# extract_response_text
# ---------------------------------------------------------------------------


class TestExtractResponseText:
    def test_list_single_text_block(self):
        content = [{"type": "text", "text": "hello"}]
        assert extract_response_text(content) == "hello"

    def test_list_multiple_text_blocks_joined_with_newline(self):
        content = [
            {"type": "text", "text": "line1"},
            {"type": "text", "text": "line2"},
        ]
        assert extract_response_text(content) == "line1\nline2"

    def test_list_mixed_block_types_only_text_extracted(self):
        content = [
            {"type": "text", "text": "answer"},
            {"type": "tool_use", "name": "search", "input": {}},
            {"type": "text", "text": "more"},
        ]
        assert extract_response_text(content) == "answer\nmore"

    def test_empty_list_returns_empty_string(self):
        assert extract_response_text([]) == ""

    def test_serialized_string_parsed_successfully(self):
        content = "[{'type': 'text', 'text': 'hello'}]"
        assert extract_response_text(content) == "hello"

    def test_serialized_string_parse_failure_returns_as_is(self):
        content = "[{broken"
        assert extract_response_text(content) == "[{broken"

    def test_plain_string_returned_as_is(self):
        assert extract_response_text("just a string") == "just a string"

    def test_empty_string_returned_as_is(self):
        assert extract_response_text("") == ""

    def test_list_with_text_block_missing_text_key(self):
        # A text-type block without a 'text' key should produce empty string for that block
        content = [{"type": "text"}, {"type": "text", "text": "ok"}]
        assert extract_response_text(content) == "\nok"

    def test_list_with_non_dict_items_ignored(self):
        # Non-dict items in the list are skipped by the isinstance check
        content = [{"type": "text", "text": "yes"}, "stray string", 42]
        assert extract_response_text(content) == "yes"


# ---------------------------------------------------------------------------
# run_tool_loop — helpers
# ---------------------------------------------------------------------------


def _make_response(content="done", tool_calls=None):
    """Create a mock LLM response with .content and .tool_calls."""
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = tool_calls or []
    return resp


def _make_ctx(task_id="task-123"):
    """Create a mock AgentContext with the methods run_tool_loop uses."""
    ctx = MagicMock()
    ctx.task_id = task_id
    ctx.stream_flush = MagicMock()
    ctx.progress = MagicMock()
    ctx.tool_invoke = MagicMock()
    return ctx


def _make_tool(name: str, result: str = "tool result"):
    """Create a mock tool with an async ainvoke."""
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value=result)
    return tool


# ---------------------------------------------------------------------------
# run_tool_loop
# ---------------------------------------------------------------------------


class TestRunToolLoop:
    """Tests for the async tool-calling loop."""

    @pytest.fixture(autouse=True)
    def _patch_trim_messages(self):
        """Patch trim_messages to be a no-op (returns a copy of messages)."""
        with patch(
            "langchain_core.messages.utils.trim_messages",
            side_effect=lambda msgs, **kwargs: list(msgs),
        ):
            yield

    @pytest.mark.asyncio
    async def test_immediate_answer_no_tool_calls(self):
        """LLM responds without tool_calls — returns immediately."""
        response = _make_response(content="final answer")
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=response)
        ctx = _make_ctx()
        messages = [{"role": "human", "content": "hi"}]

        result = await run_tool_loop(llm, messages, {}, ctx, "claude-sonnet-4-20250514")

        assert result is response
        ctx.stream_flush.assert_called_once_with("final answer")

    @pytest.mark.asyncio
    async def test_single_tool_call_then_answer(self):
        """LLM calls a tool once, then produces final answer."""
        tool_response = _make_response(
            content="thinking",
            tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "tc1"}],
        )
        final_response = _make_response(content="the answer")

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])

        tool = _make_tool("search", result="search results")
        tool_map = {"search": tool}
        ctx = _make_ctx()
        messages = [{"role": "human", "content": "query"}]

        result = await run_tool_loop(llm, messages, tool_map, ctx, "claude-sonnet-4-20250514")

        assert result is final_response
        tool.ainvoke.assert_called_once_with({"q": "test"})
        ctx.tool_invoke.assert_called_once_with("search", "q=test")
        # Tool message appended to messages
        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "search results"
        assert tool_msgs[0]["tool_call_id"] == "tc1"

    @pytest.mark.asyncio
    async def test_unknown_tool_appends_error(self):
        """Tool not in tool_map results in error message with correct tool_call_id."""
        tool_response = _make_response(
            content="",
            tool_calls=[{"name": "nonexistent", "args": {}, "id": "tc-bad"}],
        )
        final_response = _make_response(content="ok")

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])

        ctx = _make_ctx()
        messages = [{"role": "human", "content": "q"}]

        await run_tool_loop(llm, messages, {}, ctx, "claude-sonnet-4-20250514")

        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "unknown tool" in tool_msgs[0]["content"]
        assert tool_msgs[0]["tool_call_id"] == "tc-bad"

    @pytest.mark.asyncio
    async def test_result_truncation_when_enabled(self):
        """Long tool results are truncated when truncate_results=True."""
        long_result = "x" * (_MAX_TOOL_RESULT_CHARS + 500)
        tool_response = _make_response(
            content="",
            tool_calls=[{"name": "big", "args": {}, "id": "tc2"}],
        )
        final_response = _make_response(content="done")

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])

        tool = _make_tool("big", result=long_result)
        ctx = _make_ctx()
        messages = [{"role": "human", "content": "q"}]

        await run_tool_loop(llm, messages, {"big": tool}, ctx, "claude-sonnet-4-20250514", truncate_results=True)

        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        content = tool_msgs[0]["content"]
        assert len(content) < len(long_result)
        assert "truncated" in content
        assert str(len(long_result)) in content

    @pytest.mark.asyncio
    async def test_no_truncation_when_disabled(self):
        """Long tool results are preserved when truncate_results=False."""
        long_result = "y" * (_MAX_TOOL_RESULT_CHARS + 1000)
        tool_response = _make_response(
            content="",
            tool_calls=[{"name": "big", "args": {}, "id": "tc3"}],
        )
        final_response = _make_response(content="done")

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])

        tool = _make_tool("big", result=long_result)
        ctx = _make_ctx()
        messages = [{"role": "human", "content": "q"}]

        await run_tool_loop(llm, messages, {"big": tool}, ctx, "claude-sonnet-4-20250514", truncate_results=False)

        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == long_result

    @pytest.mark.asyncio
    async def test_repeat_detection_stops_after_max_repeats(self):
        """Same tool calls repeated _MAX_REPEAT_CALLS times triggers early exit."""
        # Need _MAX_REPEAT_CALLS responses with identical tool calls.
        # First sets sig with repeat_count=1, subsequent increments.
        # Exits when repeat_count >= _MAX_REPEAT_CALLS (3rd iteration).
        responses = []
        for i in range(_MAX_REPEAT_CALLS + 1):
            responses.append(
                _make_response(content="", tool_calls=[{"name": "fetch", "args": {"url": "http://x"}, "id": f"tc-{i}"}])
            )

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=responses)

        tool = _make_tool("fetch", result="same")
        ctx = _make_ctx()
        messages = [{"role": "human", "content": "q"}]

        await run_tool_loop(llm, messages, {"fetch": tool}, ctx, "claude-sonnet-4-20250514")

        ctx.progress.assert_any_call("Working: detected repeated tool calls, stopping.")
        assert llm.ainvoke.call_count == _MAX_REPEAT_CALLS

    @pytest.mark.asyncio
    async def test_timeout_retry(self):
        """Tool raises TimeoutError, retried up to max_retries."""
        tool_response = _make_response(
            content="",
            tool_calls=[{"name": "slow", "args": {}, "id": "tc-to"}],
        )
        final_response = _make_response(content="done")

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])

        tool = MagicMock()
        # First call times out, second succeeds
        tool.ainvoke = AsyncMock(side_effect=[TimeoutError("timed out"), "ok result"])
        ctx = _make_ctx()
        messages = [{"role": "human", "content": "q"}]

        await run_tool_loop(llm, messages, {"slow": tool}, ctx, "claude-sonnet-4-20250514", max_retries=2)

        assert tool.ainvoke.call_count == 2
        ctx.progress.assert_any_call("Tool call timed out, retrying (2/2)...")
        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "ok result"

    @pytest.mark.asyncio
    async def test_timeout_exhausted(self):
        """All retries exhausted on TimeoutError produces error message."""
        tool_response = _make_response(
            content="",
            tool_calls=[{"name": "slow", "args": {}, "id": "tc-to2"}],
        )
        final_response = _make_response(content="done")

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])

        tool = MagicMock()
        tool.ainvoke = AsyncMock(side_effect=TimeoutError("timed out"))
        ctx = _make_ctx()
        messages = [{"role": "human", "content": "q"}]

        await run_tool_loop(llm, messages, {"slow": tool}, ctx, "claude-sonnet-4-20250514", max_retries=2)

        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert "timed out after 2 attempts" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_tool_generic_exception_captured(self):
        """Tool raises a generic Exception, error string is captured."""
        tool_response = _make_response(
            content="",
            tool_calls=[{"name": "broken", "args": {}, "id": "tc-err"}],
        )
        final_response = _make_response(content="done")

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[tool_response, final_response])

        tool = MagicMock()
        tool.ainvoke = AsyncMock(side_effect=RuntimeError("something broke"))
        ctx = _make_ctx()
        messages = [{"role": "human", "content": "q"}]

        await run_tool_loop(llm, messages, {"broken": tool}, ctx, "claude-sonnet-4-20250514")

        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "Error: something broke"

    @pytest.mark.asyncio
    async def test_progress_every_n_turns(self):
        """progress_every=2 emits progress on turn 2."""
        # We need 2 tool-call turns then a final answer (3 ainvoke calls total)
        tc1 = _make_response(
            content="",
            tool_calls=[{"name": "a", "args": {"x": "1"}, "id": "t1"}],
        )
        tc2 = _make_response(
            content="",
            tool_calls=[{"name": "b", "args": {"y": "2"}, "id": "t2"}],
        )
        final = _make_response(content="result")

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[tc1, tc2, final])

        tool_a = _make_tool("a", "r1")
        tool_b = _make_tool("b", "r2")
        ctx = _make_ctx(task_id="task-456")
        messages = [{"role": "human", "content": "q"}]

        await run_tool_loop(
            llm,
            messages,
            {"a": tool_a, "b": tool_b},
            ctx,
            "claude-sonnet-4-20250514",
            progress_every=2,
        )

        # Turn 1: no progress. Turn 2: progress emitted.
        progress_calls = [call.args[0] for call in ctx.progress.call_args_list]
        assert any("2 turns" in msg for msg in progress_calls)
        assert any("task cancel task-456" in msg for msg in progress_calls)

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """asyncio.CancelledError from tool is NOT caught — it propagates."""
        tool_response = _make_response(
            content="",
            tool_calls=[{"name": "cancel_me", "args": {}, "id": "tc-c"}],
        )

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=tool_response)

        tool = MagicMock()
        tool.ainvoke = AsyncMock(side_effect=asyncio.CancelledError())
        ctx = _make_ctx()
        messages = [{"role": "human", "content": "q"}]

        with pytest.raises(asyncio.CancelledError):
            await run_tool_loop(llm, messages, {"cancel_me": tool}, ctx, "claude-sonnet-4-20250514")

    @pytest.mark.asyncio
    async def test_no_stream_flush_on_empty_content(self):
        """stream_flush is not called when response content is empty/whitespace."""
        response = _make_response(content="   ")
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=response)
        ctx = _make_ctx()
        messages = [{"role": "human", "content": "hi"}]

        await run_tool_loop(llm, messages, {}, ctx, "claude-sonnet-4-20250514")

        ctx.stream_flush.assert_not_called()
