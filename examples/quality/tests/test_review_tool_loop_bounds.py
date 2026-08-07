"""Adversarial tests for tool-loop turn bounds.

Now that ``run_tool_loop`` actually drives the review (#56), the loop's termination
conditions are live code for the first time. ``switchplane/llm.py:238`` is a bare
``while True`` whose only defenses are:

- ``if not response.tool_calls: return`` — the model volunteering to stop
- ``_MAX_REPEAT_CALLS`` (llm.py:266-270) — three *byte-identical* tool-call sequences

There is no turn cap and no wall-clock budget. A model that alternates between two
different tool calls never trips the repeat detector, so the loop runs until the model
happens to stop. Every turn is a paid API call against attacker-influenced input: the
PR diff is what steers the model, so a crafted diff is the input that decides how many
turns the reviewer buys.

The blast radius compounds downstream. Each turn can append to ``findings_list``, and
``synthesize_and_post`` interpolates the whole list into one prompt with no cap
(review.py:868, ``json.dumps(findings, indent=2)``), so an unbounded branch also
produces an unbounded synthesis prompt.

All imports are function-scoped to match the suite convention (see conftest.py).
"""

from __future__ import annotations

import pytest


class TestToolLoopTurnBudget:
    """A reviewer branch must not run an unbounded number of model turns."""

    @pytest.mark.asyncio
    async def test_alternating_tool_calls_hit_a_turn_cap(self, monkeypatch, tmp_path):
        """A model alternating two tool calls must be stopped by a turn budget.

        ``_MAX_REPEAT_CALLS`` compares ``json.dumps`` of the tool-call list against the
        previous turn's (llm.py:262-273), so it only fires on byte-identical repeats.
        Alternating ``record_note`` and ``record_finding`` resets ``last_sig`` every
        turn, ``repeat_count`` never reaches 3, and the loop spins.

        This test lets a cooperative model stop at 500 turns so the test terminates
        either way — but 500 turns already means 500 paid API calls for one branch, on
        one of (domains x providers) branches. A bounded loop must stop well before that.
        """
        from langchain_core.messages import AIMessage
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr import prompts as prompts_module
        from quality.agents.pr.tasks.review import review_branch

        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None, raising=True)
        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt", raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda r: r, raising=True)

        # Generous ceiling: high enough that tripping it proves "unbounded", low enough
        # that the test always terminates.
        hard_stop = 500
        turns = {"n": 0}

        class AlternatingLLM:
            """Never repeats a tool-call sequence, so the repeat detector never fires."""

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                turns["n"] += 1
                if turns["n"] > hard_stop:
                    return AIMessage(content="stopping")
                if turns["n"] % 2:
                    call = {"name": "record_note", "args": {"body": f"note {turns['n']}"}}
                else:
                    call = {
                        "name": "record_finding",
                        "args": {
                            "path": "a.py",
                            "line": turns["n"],
                            "severity": "low",
                            "body": "issue",
                        },
                    }
                call["id"] = f"call_{turns['n']}"
                return AIMessage(content="", tool_calls=[call])

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(
            config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "model-a"}}}},
            runtime_dir_path=tmp_path,
        )
        ctx.llm = lambda name=None: AlternatingLLM()

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "claude-sonnet-4-20250514",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": str(tmp_path),
        }

        await review_branch(ctx, FakeShell(), state)

        assert turns["n"] < hard_stop, (
            f"The tool loop ran {turns['n']} model turns without stopping. "
            "run_tool_loop (switchplane/llm.py:238) is a bare `while True` whose only "
            "bound is _MAX_REPEAT_CALLS, which requires byte-identical consecutive "
            "tool-call sequences. A model that alternates two calls never trips it, so "
            "one branch can issue unbounded paid API calls on attacker-influenced input. "
            "The loop needs a turn cap."
        )

    @pytest.mark.asyncio
    async def test_long_running_branch_emits_progress(self, monkeypatch, tmp_path):
        """A branch grinding through many turns must tell the operator it is alive.

        ``run_tool_loop`` supports ``progress_every`` (llm.py:206, 275-279) and even
        renders a cancel hint with the task id — but review.py:419-425 never passes it,
        so the parameter defaults to ``None`` and the whole feature is dead code at this
        call site.

        Consequence: a branch that spins for hundreds of turns emits no progress at all.
        The operator sees a task that looks hung, with no signal distinguishing "thinking"
        from "wedged" and no surfaced cancel instruction.
        """
        from langchain_core.messages import AIMessage
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr import prompts as prompts_module
        from quality.agents.pr.tasks.review import review_branch

        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None, raising=True)
        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt", raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda r: r, raising=True)

        turns = {"n": 0}
        total_turns = 40

        class ChattyLLM:
            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                turns["n"] += 1
                if turns["n"] > total_turns:
                    return AIMessage(content="done")
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "record_note",
                            "args": {"body": f"note {turns['n']}"},
                            "id": f"call_{turns['n']}",
                        }
                    ],
                )

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(
            config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "model-a"}}}},
            runtime_dir_path=tmp_path,
        )
        ctx.llm = lambda name=None: ChattyLLM()

        state = {
            "domain": "security",
            "provider": "alpha",
            "model": "claude-sonnet-4-20250514",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": str(tmp_path),
        }

        await review_branch(ctx, FakeShell(), state)

        # Path-reached guard: if the loop short-circuited, the absence of progress
        # messages would be meaningless.
        assert turns["n"] > 10, f"loop must actually have run many turns, got {turns['n']}"

        assert ctx.progress_calls, (
            f"A branch that ran {turns['n']} turns emitted zero progress messages. "
            "run_tool_loop accepts progress_every (switchplane/llm.py:206) but "
            "review.py:419-425 never passes it, so long branches look hung and the "
            "cancel hint is never surfaced."
        )
