"""Adversarial tests for context trimming inside the reviewer's tool loop.

``run_tool_loop`` trims the conversation at the top of every turn and *replaces*
the caller's list with the result (``switchplane/llm.py:244-254``)::

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

``start_on="human"`` tells ``trim_messages`` that the kept window must begin at a
``HumanMessage``. The reviewer's conversation has exactly one human message — the
initial prompt (review.py:446) — and everything after it is an
``AIMessage``/``ToolMessage`` alternation. So once the accumulated tool results push
the total past the context window, the only trailing window that satisfies
``start_on="human"`` is the empty one, and ``trim_messages`` returns ``[]``.

There is no guard on that result. ``messages.clear()`` then destroys the caller's
conversation *in place* and the model is invoked with zero messages. Nothing recovers:
each subsequent turn re-trims a list that is empty-or-one-message, so the branch spins
out its remaining turn budget with no prompt and no history.

This is reachable inside production's own budget. review.py:452-460 calls the loop with
``max_turns=100``, and results are truncated to ``_MAX_TOOL_RESULT_CHARS`` = 8,000 chars
(llm.py:165) — so ~100 turns of file reads is ~800k chars, several times a 200k window
and over six times gpt-4o's 128k. The diff steers which files the model reads, making
this attacker-influenceable: a PR touching many large files walks the reviewer into it.

Both tests assert on what the *model receives*, which is the only place the bug is
observable — the branch does not raise, it silently reviews nothing.

All imports are function-scoped to match the suite convention (see conftest.py).
"""

from __future__ import annotations

import pytest


class TestTrimNeverEmptiesTheConversation:
    """The model must never be invoked with an empty message list."""

    @pytest.mark.asyncio
    async def test_long_tool_history_does_not_wipe_the_conversation(self, monkeypatch, tmp_path):
        """A reviewer reading many large files must keep its prompt.

        Drives the real ``review_branch`` -> ``run_tool_loop`` with production's real
        ``max_turns=100``, a real ``read_file`` tool returning oversized content, and a
        model in ``MODELS`` (gpt-4o, 128k — llm.py:82). The fake LLM only records what it
        was handed; it makes no claim about trimming, so a failure here is production's.

        The assertion is the invariant the loop's own next line depends on: ``ainvoke``
        must never be called with zero messages. Real providers reject an empty message
        list outright, so in production this is either a hard API error or — worse, as
        here — a model asked to review with no instructions and no history.
        """
        from langchain_core.messages import AIMessage
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr import prompts as prompts_module
        from quality.agents.pr.tasks.review import review_branch

        from quality import ratelimit as ratelimit_module
        from switchplane.llm import _MAX_TOOL_RESULT_CHARS

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None, raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda r: r, raising=True)

        # A marker inside the real prompt, so "did the model still have its
        # instructions" is checked against production's prompt, not a stand-in.
        marker = "ADVERSARIAL-PROMPT-MARKER"
        real_initial = prompts_module.initial_prompt
        monkeypatch.setattr(
            prompts_module,
            "initial_prompt",
            lambda *a, **k: marker + " " + real_initial(*a, **k),
            raising=True,
        )

        # Real files in the worktree, so the real Shell-backed read_file returns
        # oversized content through the real sandbox rather than a stubbed tool.
        big = "y" * (_MAX_TOOL_RESULT_CHARS * 2)
        for i in range(120):
            (tmp_path / f"f{i}.py").write_text(big)

        seen_counts: list[int] = []
        seen_marker: list[bool] = []

        class FileReadingLLM:
            """Reads one large file per turn, then stops. Records nothing else."""

            def __init__(self):
                self.turn = 0

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                self.turn += 1
                seen_counts.append(len(messages))
                seen_marker.append(any(marker in str(getattr(m, "content", m)) for m in messages))
                if self.turn > 110:
                    return AIMessage(content="review complete")
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"file_path": str(tmp_path / f"f{self.turn}.py")},
                            "id": f"call_{self.turn}",
                        }
                    ],
                )

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(
            config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "gpt-4o"}}}},
            runtime_dir_path=tmp_path,
        )
        ctx.llm = lambda name=None: FileReadingLLM()

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "gpt-4o",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": str(tmp_path),
        }

        await review_branch(ctx, FakeShell(), state)

        # Path-reached guard: if the loop stopped early, an absence of empty
        # invocations below would prove nothing.
        assert len(seen_counts) > 70, (
            f"the loop must actually have run deep into its budget, got {len(seen_counts)} turns"
        )

        empty_turns = [i + 1 for i, n in enumerate(seen_counts) if n == 0]
        assert not empty_turns, (
            f"The model was invoked with an EMPTY message list on {len(empty_turns)} of "
            f"{len(seen_counts)} turns, starting at turn {empty_turns[0]}. "
            "trim_messages(start_on='human') (switchplane/llm.py:245-252) can only return a "
            "window beginning at a HumanMessage; the reviewer has exactly one, so once tool "
            "results exceed the context window the only valid window is the empty one. "
            "llm.py:253-254 then does messages.clear(); messages.extend([]) and invokes the "
            "model with nothing. Real providers reject an empty message list."
        )

    @pytest.mark.asyncio
    async def test_trimming_preserves_the_review_instructions(self, monkeypatch, tmp_path):
        """Trimming must never drop the prompt that defines the review task.

        Distinct failure from the test above: even a *non-empty* trimmed window is
        worthless if it lost the initial prompt, because the prompt is what carries the
        domain, the diff, and the "record findings via record_finding" contract
        (prompts.py). A model that keeps only tool output has no idea what it is doing,
        and its output is still fed to synthesis and posted to the PR as a review.

        ``include_system=True`` does not help: review.py:446 builds the conversation as
        ``[HumanMessage(prompt_text)]`` with no SystemMessage, so the instructions live in
        the one message ``start_on="human"`` is anchored to and ``strategy="last"`` will
        discard.
        """
        from langchain_core.messages import AIMessage
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr import prompts as prompts_module
        from quality.agents.pr.tasks.review import review_branch

        from quality import ratelimit as ratelimit_module
        from switchplane.llm import _MAX_TOOL_RESULT_CHARS

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None, raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda r: r, raising=True)

        marker = "ADVERSARIAL-PROMPT-MARKER"
        real_initial = prompts_module.initial_prompt
        monkeypatch.setattr(
            prompts_module,
            "initial_prompt",
            lambda *a, **k: marker + " " + real_initial(*a, **k),
            raising=True,
        )

        big = "y" * (_MAX_TOOL_RESULT_CHARS * 2)
        for i in range(120):
            (tmp_path / f"f{i}.py").write_text(big)

        seen_marker: list[bool] = []

        class FileReadingLLM:
            def __init__(self):
                self.turn = 0

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                self.turn += 1
                seen_marker.append(any(marker in str(getattr(m, "content", m)) for m in messages))
                if self.turn > 110:
                    return AIMessage(content="review complete")
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"file_path": str(tmp_path / f"f{self.turn}.py")},
                            "id": f"call_{self.turn}",
                        }
                    ],
                )

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(
            config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "gpt-4o"}}}},
            runtime_dir_path=tmp_path,
        )
        ctx.llm = lambda name=None: FileReadingLLM()

        state = {
            "domain": "security",
            "provider": "alpha",
            "model": "gpt-4o",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": str(tmp_path),
        }

        await review_branch(ctx, FakeShell(), state)

        assert len(seen_marker) > 70, (
            f"the loop must actually have run deep into its budget, got {len(seen_marker)} turns"
        )

        lost = [i + 1 for i, ok in enumerate(seen_marker) if not ok]
        assert not lost, (
            f"The review instructions were missing from the model's context on "
            f"{len(lost)} of {len(seen_marker)} turns, starting at turn {lost[0]}. "
            "The reviewer's only HumanMessage is the initial prompt (review.py:446); "
            "strategy='last' discards it once tool results fill the window, and "
            "include_system=True cannot compensate because no SystemMessage is used. "
            "The branch then keeps calling the model with no task definition, and whatever "
            "it produces is still synthesized and posted to the PR."
        )


class TestAnchorPlusTailStaysWithinWindow:
    """The #71 fix pins the anchor UNCONDITIONALLY — the anchor can dwarf the window.

    #71 changed the trim to ``[anchor, *trim_messages(messages[1:], ...)]`` where
    ``anchor = messages[0]``. That keeps the prompt (good) but the anchor is never
    itself measured against the window. The reviewer's anchor is the initial prompt,
    which embeds the PR diff verbatim (prompts.py:172-173 -> ```diff\\n{diff}```).
    ``gh.get_pr_diff`` (gh.py:401-417) returns the raw diff with no size bound, and
    the diff is attacker-controlled. A PR whose diff alone exceeds the model's window
    makes ``run_tool_loop`` invoke the model with anchor+tail far over the limit on the
    very first turn — the provider rejects it (or bills/latency explodes), so the
    branch reviews nothing.

    This probes #71 item (ii): "does the anchor-plus-tail ever still exceed the window
    pathologically?" It does — the fix bounds the tail but not the anchor.
    """

    @pytest.mark.asyncio
    async def test_oversized_diff_prompt_is_not_sent_over_window(self):
        """A prompt larger than the model window must not be handed to the model whole.

        Drives the real ``run_tool_loop`` with a real model name (gpt-4o, 128k window —
        llm.py:82) and a single oversized HumanMessage standing in for the initial
        prompt with a huge embedded diff. The invariant: the message list handed to the
        model on every turn must fit the model's context window. The current fix pins the
        anchor without measuring it, so the very first invoke is several times over.
        """
        from langchain_core.messages import AIMessage, HumanMessage

        from switchplane.llm import context_window, run_tool_loop

        model_name = "gpt-4o"
        window = context_window(model_name)

        # Approximate token count the same way trim_messages("approximate") does:
        # ~4 chars per token.
        def approx_tokens(messages) -> int:
            return sum(len(str(getattr(m, "content", m))) for m in messages) // 4

        seen_tokens: list[int] = []

        class RecordingLLM:
            """Records the size of each invocation, then finishes immediately."""

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                seen_tokens.append(approx_tokens(messages))
                return AIMessage(content="done", tool_calls=[])

        class Ctx:
            task_id = "t"

            def progress(self, *a, **k):
                pass

            def stream_flush(self, *a, **k):
                pass

            def tool_invoke(self, *a, **k):
                pass

        # An attacker-controlled diff that alone dwarfs the 128k window, embedded in the
        # prompt exactly the way prompts.initial_prompt does.
        big_diff = "+ leak this\n" * 200_000
        prompt = f"Review this PR for the quality domain.\n```diff\n{big_diff}\n```"
        messages = [HumanMessage(content=prompt)]

        await run_tool_loop(
            RecordingLLM(),
            messages,
            {},
            Ctx(),
            model_name,
            label="quality/alpha",
            max_turns=5,
        )

        assert seen_tokens, "run_tool_loop must have invoked the model at least once"
        over = [n for n in seen_tokens if n > window]
        assert not over, (
            f"The model ({model_name}, window={window} tokens) was invoked with "
            f"{over[0]} approx tokens — {over[0] / window:.1f}x the window. "
            "run_tool_loop pins messages[0] as the anchor without measuring it "
            "(llm.py:250, 259), so an oversized attacker-controlled diff embedded in the "
            "prompt (gh.get_pr_diff is unbounded) is sent to the provider whole. Providers "
            "reject an over-window request, so the branch reviews nothing."
        )
