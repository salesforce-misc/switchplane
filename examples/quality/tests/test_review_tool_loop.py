"""Adversarial tests for the review_branch tool-binding and tool-loop contract.

Every existing branch test replaces ``ratelimit.with_rate_limit_retry`` with a stub
that reads ``runnable.bind_tools.call_args`` off a ``Mock`` and then drives the
recording tools by hand. That harness never asks whether the *real* collaborators
would accept what production passes them, and it supplies the tool loop that
production is missing. Three defects hide in that gap:

1. ``Shell`` is constructed without ls/find/grep, so the real ``fs_tools()`` raises.
2. ``fs_tools()`` returns ``switchplane.llm.Tool`` wrappers, not ``BaseTool``, so
   the real ``bind_tools`` rejects them.
3. ``review_branch`` awaits a single ``ainvoke`` and never dispatches the returned
   tool calls, so nothing is ever recorded against a real model.

All imports are function-scoped to match the suite convention (see conftest.py).
"""

from __future__ import annotations

import pytest


class TestRealShellToolSurface:
    """The Shell that run() builds must actually be able to produce fs_tools()."""

    def test_task_shell_allowlist_supports_fs_tools(self, monkeypatch, tmp_path):
        """The allowlist ``ReviewTask.run`` passes must satisfy the real ``fs_tools()``.

        ``Shell.fs_tools()`` (src/switchplane/shell.py:273-275) hard-requires ls, find
        and grep in ``allowed_commands`` and raises ValueError otherwise. If the two
        drift apart, every branch dies in its own try/except, records a failed note, and
        the run reports "all reviewer branches failed" — the reviewer never reads a file.

        conftest's ``FakeShell.fs_tools()`` returns ``[]``, so no other test couples the
        two. This one reads the allowlist out of production (rather than restating a
        literal) and feeds it to the real ``Shell``, so editing either side alone fails.
        """
        from quality.agents.pr.tasks import review as review_module

        from switchplane.shell import Shell

        captured: dict[str, list[str]] = {}

        class RecordingShell(Shell):
            def __init__(self, *args, **kwargs):
                captured["allowed_commands"] = list(kwargs.get("allowed_commands", []))
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(review_module, "Shell", RecordingShell, raising=True)
        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: _StopAfterShell(), raising=True)

        from conftest import FakeAgentContext

        ctx = FakeAgentContext(
            config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "model-a"}}}},
            runtime_dir_path=tmp_path,
        )

        task = review_module.ReviewTask()
        task.pr = "https://github.com/org/repo/pull/1"
        task.local = True

        import asyncio

        asyncio.run(task.run(ctx))

        allowed = captured.get("allowed_commands")
        assert allowed, "ReviewTask.run must construct a Shell with an explicit allowlist"

        # Verify the captured allowlist matches production's constant
        from quality.agents.pr.tasks.review import _SHELL_ALLOWED_COMMANDS

        assert allowed == _SHELL_ALLOWED_COMMANDS, (
            f"Shell allowlist must match _SHELL_ALLOWED_COMMANDS. Got {allowed}, expected {_SHELL_ALLOWED_COMMANDS}"
        )

        shell = Shell(allowed_paths=[tmp_path], allowed_commands=allowed, timeout=300.0)
        tools = shell.fs_tools()

        names = sorted(getattr(t, "name", "") for t in tools)
        assert names, f"fs_tools() must return the filesystem tools, got {names}"

    @pytest.mark.asyncio
    async def test_fs_tools_are_bindable_by_langchain(self, monkeypatch, tmp_path):
        """Whatever ``review_branch`` hands ``bind_tools`` must be convertible by LangChain.

        ``Shell.fs_tools()`` returns ``switchplane.llm.Tool`` wrappers, and that class
        documents itself (src/switchplane/llm.py:49-51):

            "not a BaseTool subclass — LangChain's bind_tools requires BaseTool
            instances, so callers must pass [t.tool for t in tools] to bind_tools."

        Every existing branch test hides a regression here because ``bind_tools`` is a
        ``Mock`` that accepts anything. This test captures the real argument production
        passes and runs it through ``convert_to_openai_tool`` — the converter every real
        chat model's ``bind_tools`` calls internally.
        """
        from langchain_core.messages import AIMessage
        from langchain_core.utils.function_calling import convert_to_openai_tool
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr import prompts as prompts_module
        from quality.agents.pr.tasks.review import review_branch

        from quality import ratelimit as ratelimit_module
        from switchplane.shell import Shell

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None, raising=True)
        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt", raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda r: r, raising=True)

        bound: list = []

        class CapturingLLM:
            def bind_tools(self, tools):
                bound.extend(tools)
                return self

            async def ainvoke(self, messages):
                return AIMessage(content="done")

        from conftest import FakeAgentContext

        ctx = FakeAgentContext(
            config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "model-a"}}}},
            runtime_dir_path=tmp_path,
        )
        ctx.llm = lambda name=None: CapturingLLM()

        # The real Shell, with the allowlist production uses, so real Tool wrappers flow.
        from quality.agents.pr.tasks.review import _SHELL_ALLOWED_COMMANDS

        shell = Shell(
            allowed_paths=[tmp_path],
            allowed_commands=_SHELL_ALLOWED_COMMANDS,
            timeout=300.0,
        )

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": str(tmp_path),
        }

        result = await review_branch(ctx, shell, state)

        assert not any(n.get("failed") for n in result["notes"]), (
            f"branch must not fail before reaching bind_tools: {result['notes']}"
        )
        assert bound, "review_branch must bind tools to the model"

        for t in bound:
            convert_to_openai_tool(t)


class _StopAfterShell:
    """Minimal build_graph stand-in: compiles, then returns an empty result.

    Lets ``ReviewTask.run`` proceed past Shell construction without needing the whole
    graph, while keeping ``run``'s own control flow (including the cleanup finally).
    """

    def compile(self, checkpointer=None):
        return self

    async def aget_state(self, config):
        from types import SimpleNamespace

        return SimpleNamespace(values={})

    async def ainvoke(self, state, config=None):
        return {}


class TestBranchDrivesToolCalls:
    """review_branch must execute the tool calls the model requests."""

    @pytest.mark.asyncio
    async def test_branch_executes_model_requested_tool_calls(self, monkeypatch, tmp_path):
        """A model that asks for record_finding must produce a recorded finding.

        review.py:407-408 is the entire "tool loop":

            messages = [HumanMessage(content=prompt_text)]
            await llm.ainvoke(messages)

        One ainvoke, return value discarded, no dispatch of ``response.tool_calls``,
        no tool-result turn, no iteration. A real model answers ``ainvoke`` with an
        AIMessage carrying ``tool_calls``; nothing in production ever invokes them.
        So ``findings_list`` stays empty and every real review returns zero findings.

        The suite's branch tests all pass because their fake ``ainvoke`` reaches into
        ``bind_tools.call_args`` and calls ``record_finding.ainvoke(...)`` itself —
        the test harness supplies the missing loop. This test instead returns a
        normal AIMessage with tool_calls, exactly as a real model does, and asserts
        production dispatched it.
        """
        from langchain_core.messages import AIMessage
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr import prompts as prompts_module
        from quality.agents.pr.tasks.review import review_branch

        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None, raising=True)
        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt", raising=True)

        ainvoke_calls: list[list] = []

        class ToolCallingLLM:
            """Answers the first turn with a tool call, then stops — a real model's shape."""

            def __init__(self):
                self.turn = 0

            def bind_tools(self, tools):
                self.bound_tools = tools
                return self

            async def ainvoke(self, messages):
                ainvoke_calls.append(messages)
                self.turn += 1
                if self.turn == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "record_finding",
                                "args": {
                                    "path": "auth.py",
                                    "line": 42,
                                    "severity": "high",
                                    "body": "Missing authorization check",
                                },
                                "id": "call_1",
                            }
                        ],
                    )
                return AIMessage(content="Review complete.")

        llm = ToolCallingLLM()

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda r: r, raising=True)

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(
            config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "model-a"}}}},
            runtime_dir_path=tmp_path,
        )
        ctx.llm = lambda name=None: llm

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": str(tmp_path),
        }

        result = await review_branch(ctx, FakeShell(), state)

        assert not any(n.get("failed") for n in result["notes"]), f"Branch must not fail: {result['notes']}"
        assert len(result["findings"]) == 1, (
            "review_branch must dispatch the tool calls the model returned. "
            f"Got findings={result['findings']} after {len(ainvoke_calls)} ainvoke call(s) — "
            "production awaits ainvoke once and discards the response, so the model's "
            "record_finding request is never executed."
        )
        assert result["findings"][0]["path"] == "auth.py"
        assert result["findings"][0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_branch_feeds_tool_results_back_to_model(self, monkeypatch, tmp_path):
        """After running a tool, the branch must give the model the result and let it continue.

        A reviewer that reads one file and is never asked again cannot review anything.
        The loop must (a) append the tool result and (b) call ainvoke a second time, so
        the model can read, reason, then record.

        Production calls ainvoke exactly once (review.py:408), so the second turn — the
        one where a real reviewer would record findings after reading — never happens.
        """
        from langchain_core.messages import AIMessage
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr import prompts as prompts_module
        from quality.agents.pr.tasks.review import review_branch

        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None, raising=True)
        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt", raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda r: r, raising=True)

        turns: list[int] = []

        class TwoTurnLLM:
            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                turns.append(len(messages))
                if len(turns) == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{"name": "record_note", "args": {"body": "read the file"}, "id": "c1"}],
                    )
                return AIMessage(content="done")

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(
            config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "model-a"}}}},
            runtime_dir_path=tmp_path,
        )
        ctx.llm = lambda name=None: TwoTurnLLM()

        state = {
            "domain": "security",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": str(tmp_path),
        }

        await review_branch(ctx, FakeShell(), state)

        assert len(turns) >= 2, (
            f"Expected at least 2 ainvoke turns (model call, then continuation after the "
            f"tool result), got {len(turns)}. review.py:408 awaits ainvoke once, so the "
            "model never sees its own tool results and can never record after reading."
        )
        assert turns[1] > turns[0], (
            f"The second turn must carry the appended tool result: message count should grow, got {turns}."
        )
