"""Tests for the PR review LangGraph graph — state, fan-out, branches, in-process tools.

Scope: LangGraph StateGraph construction, Send-based fan-out cross-product with the
DOMAINS tuple, branch node execution logic (prompt selection, model construction,
tool binding, tool loop running), and the record_finding/record_note in-process tools.

Synthesis and posting are task #7 (a separate file).

All imports of quality.* are inside test functions to avoid collection-time import
failures before the path-injection fixture runs (see conftest.py).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from conftest import FakeAgentContext as BaseAgentContext
from conftest import FakeShell as BaseShell


class FakeAgentContext(BaseAgentContext):
    """Test-specific subclass that records llm() calls for assertions.

    Extends the shared FakeAgentContext from conftest.py with recording state
    for llm() and progress() calls, so tests can assert which provider, model,
    and prompt variant each branch used.
    """

    def __init__(self, providers: list[str], config: dict, runtime_dir_path=None):
        super().__init__(config=config, runtime_dir_path=runtime_dir_path)
        self._providers_override = providers
        self.is_cancelled = False

        # Recording state
        self.llm_calls: list[tuple[str | None, str | None]] = []  # (provider_name, model_override)
        self._llms: dict[str | None, Mock] = {}  # provider -> mock llm

    @property
    def providers(self) -> list[str]:
        """Override providers property to return test-specific list."""
        return self._providers_override

    def llm(self, name: str | None = None, *, model: str | None = None):
        """Record llm() call and return a mock LLM for that provider."""
        self.llm_calls.append((name, model))
        if name not in self._llms:
            mock_llm = Mock()
            mock_llm.bind_tools = Mock(return_value=mock_llm)
            self._llms[name] = mock_llm
        return self._llms[name]

    async def check_cancelled(self) -> None:
        import asyncio

        if self.is_cancelled:
            raise asyncio.CancelledError("Cancelled for testing")


class FakeShell(BaseShell):
    """Test-specific subclass that records fs_tools() calls.

    Extends the shared FakeShell from conftest.py with call recording and
    returns mock tools for test assertions.
    """

    def __init__(self):
        super().__init__()
        self.fs_tools_calls = 0

    def fs_tools(self) -> list[Mock]:
        """Return a list of mock filesystem tools and record the call."""
        self.fs_tools_calls += 1
        # Create fresh tool mocks per call (each branch should get its own)
        read_mock = Mock()
        read_mock.name = "read_file"
        grep_mock = Mock()
        grep_mock.name = "grep_code"
        ls_mock = Mock()
        ls_mock.name = "ls_dir"
        return [read_mock, grep_mock, ls_mock]


class TestReviewState:
    """Tests for ReviewState Pydantic BaseModel and concurrent fan-in behavior."""

    def test_concurrent_branch_writes_with_reducer_succeed(self):
        """Concurrent branch writes to reducer fields must succeed.

        LangGraph raises InvalidUpdateError when concurrent nodes write the same field
        without a reducer. ReviewState must be a Pydantic BaseModel with Annotated[list, operator.add]
        fields for `findings` and `notes` so all branches can write concurrently.
        """
        # Import LangGraph types to test the annotation
        import operator
        from typing import Annotated, get_args, get_origin, get_type_hints

        from pydantic import BaseModel
        from quality.agents.pr.tasks.review import ReviewState

        # Assert ReviewState is a Pydantic BaseModel
        assert issubclass(ReviewState, BaseModel), "ReviewState must be a Pydantic BaseModel (not TypedDict)"

        hints = get_type_hints(ReviewState, include_extras=True)

        # Assert specific reducer fields exist: findings and notes
        reducer_fields = []
        for name, hint in hints.items():
            if get_origin(hint) is Annotated:
                args = get_args(hint)
                if len(args) > 1 and args[1] is operator.add:
                    reducer_fields.append(name)

        assert "findings" in reducer_fields, "ReviewState must have a 'findings' field with operator.add reducer"
        assert "notes" in reducer_fields, "ReviewState must have a 'notes' field with operator.add reducer"


class TestRouting:
    """Tests for route_to_branches conditional routing logic."""

    def test_state_error_routes_to_end(self):
        """When state.error is set, route_to_branches must return END.

        Per ava semantics: only state.error routes to END. Empty matrix routes to synthesis.
        This pins the error-path short-circuit that skips review when setup fails.
        """
        from langgraph.graph import END
        from quality.agents.pr.tasks.review import route_to_branches

        state = {"error": "URL parsing failed", "matrix": [("alpha", "model-a")], "diff": "diff"}
        result = route_to_branches(state)

        assert result == END, "state.error must route to END, skipping review and synthesis"

    def test_empty_diff_routes_to_synthesis(self):
        """When diff is empty, route_to_branches must return "synthesize_and_post".

        An empty diff means no changes to review. Synthesis produces APPROVE with no findings.
        """
        from quality.agents.pr.tasks.review import route_to_branches

        state = {"error": None, "matrix": [("alpha", "model-a")], "diff": ""}
        result = route_to_branches(state)

        assert result == "synthesize_and_post", "Empty diff must route to synthesis (APPROVE with no findings)"


class TestFanOutDispatch:
    """Tests for the dispatch node that returns Send(...) objects."""

    def test_two_providers_two_domains_yields_four_sends(self):
        """With 2 providers and 2 domains, the cross-product yields exactly 4 Send objects.

        CRITICAL: This test hardcodes the expected domain strings ("quality", "security")
        rather than importing DOMAINS, because comparing the implementation to itself is
        tautological. A count-only assertion of `== 4` would pass if the implementation
        sent four identical branches — the exact mutant this test kills.

        Fan-out lives in the conditional-edge router (route_to_branches), not in a node.
        """
        from quality.agents.pr.tasks.review import route_to_branches

        # Pass matrix directly in state (setup node populates this in production)
        state = {
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "matrix": [("alpha", "model-a-1"), ("beta", "model-b-2")],
            "error": None,
        }
        sends = route_to_branches(state)

        # Assert 4 Sends with the specific cross-product tuples
        assert isinstance(sends, list), "route_to_branches must return a list of Send objects"
        assert len(sends) == 4, f"Expected 4 Sends (2 providers × 2 domains), got {len(sends)}"

        # Extract (domain, provider, model) tuples from Send objects
        # Send payloads are ReviewState instances (Pydantic BaseModel), so use attribute access
        send_tuples = [(s.node, s.arg.cur_domain, s.arg.cur_provider, s.arg.cur_model) for s in sends]

        # Pin the exact expected cross-product (hardcoded domains to kill tautology)
        expected = [
            ("review_branch", "quality", "alpha", "model-a-1"),
            ("review_branch", "quality", "beta", "model-b-2"),
            ("review_branch", "security", "alpha", "model-a-1"),
            ("review_branch", "security", "beta", "model-b-2"),
        ]

        # Sort both for stable comparison (dispatch order may vary)
        send_tuples_sorted = sorted(send_tuples, key=lambda t: (t[1], t[2]))
        expected_sorted = sorted(expected, key=lambda t: (t[1], t[2]))

        assert send_tuples_sorted == expected_sorted, (
            f"Cross-product mismatch.\nGot: {send_tuples_sorted}\nExpected: {expected_sorted}"
        )

    def test_empty_matrix_routes_to_synthesis(self):
        """When matrix is [], route_to_branches returns "synthesize_and_post".

        Per ava semantics: empty matrix routes to synthesis, NOT to END. Only state.error
        routes to END.

        NOTE: This test only checks routing. The claim that synthesis "makes the empty case
        explicit" rather than "looking like approval" is contradicted by production behavior
        — see test_stock_config_with_no_api_keys_must_not_post_false_clean_review (bug #53).
        """
        from quality.agents.pr.tasks.review import route_to_branches

        state = {"repo": "github.com/org/repo", "number": 1, "diff": "diff", "matrix": [], "error": None}
        result = route_to_branches(state)

        assert result == "synthesize_and_post", "Empty matrix must route to synthesis, not END."


class TestBranchExecution:
    """Tests for the review_branch node — prompt selection, model construction, tool loop."""

    @pytest.mark.asyncio
    async def test_branch_uses_initial_prompt_when_no_baseline(self, monkeypatch):
        """A branch without prior findings must use initial_prompt, not followup_prompt.

        This pins the branch logic that checks for baseline presence before choosing
        which prompt variant to invoke.
        """
        # Stub load_baseline to return None (no prior findings)
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)

        # Stub prompt builders to return identifiable strings
        from quality.agents.pr import prompts as prompts_module

        initial_called = {"value": False}
        followup_called = {"value": False}

        def fake_initial(*args, **kwargs):
            initial_called["value"] = True
            return "INITIAL_PROMPT"

        def fake_followup(*args, **kwargs):
            followup_called["value"] = True
            return "FOLLOWUP_PROMPT"

        monkeypatch.setattr(prompts_module, "initial_prompt", fake_initial)
        monkeypatch.setattr(prompts_module, "followup_prompt", fake_followup)

        # Stub with_rate_limit_retry to return a mock LLM that short-circuits the tool loop
        from quality import ratelimit as ratelimit_module

        def fake_with_rate_limit_retry(runnable):
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=Mock(content="Done", tool_calls=[]))
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        from pathlib import Path

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=Path("/fake/runtime"),
        )
        shell = FakeShell()

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }

        await review_branch(ctx, shell, state)

        assert initial_called["value"], "Branch must call initial_prompt when no baseline exists"
        assert not followup_called["value"], "Branch must NOT call followup_prompt when no baseline exists"

    @pytest.mark.asyncio
    async def test_branch_uses_followup_prompt_when_baseline_has_prior_findings(self, monkeypatch):
        """A branch with prior findings for its domain must use followup_prompt.

        This pins the baseline-presence check and the prompt variant selection logic.
        """
        # Stub load_baseline to return a baseline with findings for the domain
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        baseline = {
            "head_sha": "abc123",
            "findings": [{"domain": "quality", "severity": "low", "path": "foo.py", "line": 10, "title": "test"}],
        }
        monkeypatch.setattr(memory_module, "load_baseline", lambda path: baseline)

        # Stub prompt builders
        from quality.agents.pr import prompts as prompts_module

        initial_called = {"value": False}
        followup_called = {"value": False}

        def fake_initial(*args, **kwargs):
            initial_called["value"] = True
            return "INITIAL_PROMPT"

        def fake_followup(*args, **kwargs):
            followup_called["value"] = True
            return "FOLLOWUP_PROMPT"

        monkeypatch.setattr(prompts_module, "initial_prompt", fake_initial)
        monkeypatch.setattr(prompts_module, "followup_prompt", fake_followup)

        # Stub with_rate_limit_retry
        from quality import ratelimit as ratelimit_module

        def fake_with_rate_limit_retry(runnable):
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=Mock(content="Done", tool_calls=[]))
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        from pathlib import Path

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=Path("/fake/runtime"),
        )
        shell = FakeShell()

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }

        await review_branch(ctx, shell, state)

        assert followup_called["value"], "Branch must call followup_prompt when baseline has prior findings"
        assert not initial_called["value"], "Branch must NOT call initial_prompt when prior findings exist"

    @pytest.mark.asyncio
    async def test_followup_requires_both_is_followup_and_cur_prior(self, monkeypatch):
        """Branch uses followup_prompt only when BOTH is_followup AND cur_prior are truthy.

        Per ava semantics: condition is `if state.is_followup and state.cur_prior:` — both, not either.
        A follow-up run whose baseline has no findings *for this domain* still gets initial_prompt.
        This is easy to get wrong by checking only `is_followup`.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        # Baseline has findings for SECURITY but not QUALITY
        baseline = {
            "head_sha": "abc123",
            "findings": [{"domain": "security", "severity": "high", "path": "auth.py", "line": 20, "title": "XSS"}],
        }
        monkeypatch.setattr(memory_module, "load_baseline", lambda path: baseline)

        # Stub _format_prior to return empty string for quality (no prior for this domain)
        from quality.agents.pr import prompts as prompts_module

        def fake_format_prior(baseline_arg, domain):
            return "Prior: XSS finding" if domain == "security" else ""

        monkeypatch.setattr(prompts_module, "_format_prior", fake_format_prior)

        initial_called = {"value": False}
        followup_called = {"value": False}

        def fake_initial(*args, **kwargs):
            initial_called["value"] = True
            return "INITIAL_PROMPT"

        def fake_followup(*args, **kwargs):
            followup_called["value"] = True
            return "FOLLOWUP_PROMPT"

        monkeypatch.setattr(prompts_module, "initial_prompt", fake_initial)
        monkeypatch.setattr(prompts_module, "followup_prompt", fake_followup)

        from quality import ratelimit as ratelimit_module

        def fake_with_rate_limit_retry(runnable):
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=Mock(content="Done", tool_calls=[]))
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        from pathlib import Path

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=Path("/fake/runtime"),
        )
        shell = FakeShell()

        # State: is_followup=True but cur_prior="" (no prior for quality domain)
        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
            "is_followup": True,
            "cur_prior": "",
        }

        await review_branch(ctx, shell, state)

        # Must use initial_prompt because cur_prior is empty
        assert initial_called["value"], (
            "Branch must call initial_prompt when is_followup=True but cur_prior='' (no prior findings for THIS domain)"
        )
        assert not followup_called["value"], "Branch must NOT call followup_prompt when cur_prior is empty"

    @pytest.mark.asyncio
    async def test_branch_constructs_llm_with_correct_provider_and_model(self, monkeypatch):
        """Each branch must call ctx.llm(provider_name) to construct its LLM.

        This test uses unguessable provider/model values to ensure the branch
        genuinely resolves from its Send arguments rather than hardcoding.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)

        from quality.agents.pr import prompts as prompts_module

        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt")

        from quality import ratelimit as ratelimit_module

        captured_runnable = {"value": None}

        def fake_with_rate_limit_retry(runnable):
            captured_runnable["value"] = runnable
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=Mock(content="Done", tool_calls=[]))
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        ctx = FakeAgentContext(
            providers=["gamma"], config={"llm": {"providers": {"gamma": {"api_key": "key-g", "model": "model-g-3"}}}}
        )
        shell = FakeShell()

        state = {
            "domain": "security",
            "provider": "gamma",
            "model": "model-g-3",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }

        await review_branch(ctx, shell, state)

        # Assert ctx.llm was called with the branch's provider
        assert ("gamma", None) in ctx.llm_calls, f"Branch must call ctx.llm('gamma'), got: {ctx.llm_calls}"

        # Assert the returned LLM was passed to with_rate_limit_retry
        assert captured_runnable["value"] is ctx._llms["gamma"], (
            "Branch must pass the provider's LLM to with_rate_limit_retry"
        )

    @pytest.mark.asyncio
    async def test_branch_binds_fs_tools_plus_recording_tools(self, monkeypatch):
        """Each branch must bind both the fs_tools and the two recording tools.

        This pins the tool set available to the LLM during its review loop.

        Security property (#65 Part B): the Shell backing the model's fs_tools is
        constructed INSIDE ``review_branch`` and scoped to the branch's own
        ``worktree_path`` — NOT the outer task-level Shell (which reaches ``repos/``)
        and NOT the runtime dir. A branch reviewing one PR must not be able to read a
        sibling repo's checkout or the operator's config. We capture the Shell
        production actually constructs (via a recording subclass of the real Shell)
        and assert on its ``allowed_paths``, so a future refactor that re-points the
        model's fs_tools at a wider Shell fails here instead of silently regressing.
        """
        from pathlib import Path

        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks import review as review_module
        from quality.agents.pr.tasks.review import review_branch

        from switchplane.shell import Shell

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)

        from quality.agents.pr import prompts as prompts_module

        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt")

        # Capture the Shell review_branch constructs for the model's fs_tools.
        # RecordingShell is a real Shell subclass, so fs_tools() and validate_path
        # behave exactly as in production — only the allowed_paths is observed.
        captured_shells: list[Shell] = []

        class RecordingShell(Shell):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured_shells.append(self)

        monkeypatch.setattr(review_module, "Shell", RecordingShell, raising=True)

        from quality import ratelimit as ratelimit_module

        def fake_with_rate_limit_retry(runnable):
            # Production order: with_rate_limit_retry(llm.bind_tools(tools))
            # So bind_tools was already called; runnable is the bound LLM.
            # Read tools from the recorded call instead of monkeypatching after the fact.
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=Mock(content="Done", tool_calls=[]))
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        runtime_dir = Path("/fake/runtime")
        worktree = "/fake/runtime/repos/repo/wt-quality"

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=runtime_dir,
        )
        # Outer task-level shell is deliberately scoped WIDE (repos/) to prove the
        # branch does not reuse it for the model's fs_tools.
        shell = FakeShell()

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": worktree,
        }

        await review_branch(ctx, shell, state)

        # Extract tools from the mock LLM's bind_tools call
        # ctx.llm("alpha") was called and returned a mock from ctx._llms
        mock_llm_from_ctx = ctx._llms["alpha"]
        assert mock_llm_from_ctx.bind_tools.called, "Branch must call bind_tools with a tool list"
        tools = mock_llm_from_ctx.bind_tools.call_args[0][0]

        # Extract tool names
        tool_names = {getattr(t, "name", str(t)) for t in tools}

        # The branch must construct exactly one worktree-scoped Shell for fs_tools.
        assert len(captured_shells) == 1, "Branch must construct exactly one Shell (the worktree-scoped fs_tools Shell)"
        fs_shell = captured_shells[0]

        # The security property: fs_tools' Shell is scoped to the branch worktree,
        # NOT the runtime dir and NOT the parent repos/ dir. Assert on the real
        # allowed_paths production computed, resolved the way Shell resolves them.
        expected = Path(worktree).resolve()
        assert fs_shell.allowed_paths == [expected], (
            f"fs_tools Shell must be scoped to the branch worktree_path, got {fs_shell.allowed_paths}"
        )
        # Explicit negatives: the two wider scopes #65 Part B moved off of.
        assert runtime_dir.resolve() not in fs_shell.allowed_paths, (
            "fs_tools Shell must not be scoped to the runtime dir"
        )
        assert (runtime_dir / "repos").resolve() not in fs_shell.allowed_paths, (
            "fs_tools Shell must not be scoped to the shared repos/ dir"
        )
        # The model's fs_tools must come from this worktree-scoped Shell, not the
        # outer task-level shell.
        assert shell.fs_tools_calls == 0, "Branch must NOT source fs_tools from the outer task-level shell"

        # Assert recording tools are present
        assert "record_finding" in tool_names, "Branch must bind record_finding tool"
        assert "record_note" in tool_names, "Branch must bind record_note tool"


class TestRecordingTools:
    """Tests for the in-process record_finding and record_note tools."""

    @pytest.mark.asyncio
    async def test_record_finding_captures_path_line_severity_body(self, monkeypatch):
        """record_finding must capture path, line, severity, and body into the branch's findings list.

        Each branch closure must collect into its own list — this test runs one branch
        to pin the basic capture. Cross-contamination is tested separately.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)

        from quality.agents.pr import prompts as prompts_module

        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt")

        # Stub the LLM to invoke record_finding once, then stop
        from quality import ratelimit as ratelimit_module

        def fake_with_rate_limit_retry(runnable):
            # Production order: with_rate_limit_retry(llm.bind_tools(tools))
            # Read tools from runnable.bind_tools.call_args (already called before we see it)
            async def fake_ainvoke(prompt):
                # Extract tools from the bound LLM's recorded call
                tools = runnable.bind_tools.call_args[0][0]
                record_finding_tool = next((t for t in tools if getattr(t, "name", None) == "record_finding"), None)
                assert record_finding_tool is not None, "record_finding tool not found in bound tools"

                # Invoke it via ainvoke (tools may be async, and .func may be None or wrapped)
                await record_finding_tool.ainvoke(
                    {"path": "foo.py", "line": 42, "severity": "high", "body": "Test finding"}
                )

                # Return a message signaling stop (tool_calls=[] so run_tool_loop terminates)
                msg = Mock()
                msg.content = "Done"
                msg.tool_calls = []
                return msg

            mock_llm = Mock()
            mock_llm.ainvoke = fake_ainvoke
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        from pathlib import Path

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=Path("/fake/runtime"),
        )
        shell = FakeShell()

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }

        result = await review_branch(ctx, shell, state)

        # Assert the finding was captured in the branch result
        assert "findings" in result, "Branch result must include a 'findings' list"
        findings = result["findings"]
        assert len(findings) == 1, f"Expected 1 finding, got {len(findings)}"
        finding = findings[0]
        assert finding["path"] == "foo.py", f"path mismatch: {finding}"
        assert finding["line"] == 42, f"line mismatch: {finding}"
        assert finding["severity"] == "high", f"severity mismatch: {finding}"
        assert finding["body"] == "Test finding", f"body mismatch: {finding}"

    @pytest.mark.asyncio
    async def test_record_finding_coerces_invalid_severity_to_medium(self, monkeypatch):
        """record_finding must coerce invalid severity to "medium", not raise.

        Per ava semantics: the caller is an LLM, so rejecting a tool call loses a real
        finding over a formatting slip. Invalid severity → "medium" + success message.
        Also pins .strip().lower() normalization: "  HIGH  " → "high" (not coerced).
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)

        from quality.agents.pr import prompts as prompts_module

        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt")

        from quality import ratelimit as ratelimit_module

        def fake_with_rate_limit_retry(runnable):
            # Production order: with_rate_limit_retry(llm.bind_tools(tools))
            async def fake_ainvoke(prompt):
                tools = runnable.bind_tools.call_args[0][0]
                record_finding_tool = next((t for t in tools if getattr(t, "name", None) == "record_finding"), None)

                # Call with invalid severity and with whitespace-padded valid severity
                await record_finding_tool.ainvoke(
                    {"path": "foo.py", "line": 1, "severity": "URGENT", "body": "finding 1"}
                )
                await record_finding_tool.ainvoke(
                    {"path": "bar.py", "line": 2, "severity": "  HIGH  ", "body": "finding 2"}
                )

                msg = Mock()
                msg.content = "Done"
                msg.tool_calls = []
                return msg

            mock_llm = Mock()
            mock_llm.ainvoke = fake_ainvoke
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        from pathlib import Path

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=Path("/fake/runtime"),
        )
        shell = FakeShell()

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }

        result = await review_branch(ctx, shell, state)

        # Assert both findings were recorded
        assert len(result["findings"]) == 2, f"Expected 2 findings, got {len(result['findings'])}"

        finding1 = result["findings"][0]
        finding2 = result["findings"][1]

        # Invalid "URGENT" must coerce to "medium"
        assert finding1["severity"] == "medium", (
            f"Invalid severity 'URGENT' must coerce to 'medium', got {finding1['severity']}"
        )

        # "  HIGH  " must normalize to "high" (not coerce)
        assert finding2["severity"] == "high", (
            f"Severity '  HIGH  ' must normalize to 'high' via .strip().lower(), got {finding2['severity']}"
        )

    @pytest.mark.asyncio
    async def test_two_branches_findings_do_not_cross_contaminate(self, monkeypatch):
        """Two concurrent branches must record into separate findings lists.

        A shared mutable default would cause findings to bleed across branches — this
        test pins that each branch closure captures into its own list.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)

        from quality.agents.pr import prompts as prompts_module

        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt")

        from quality import ratelimit as ratelimit_module

        call_counter = {"count": 0}

        def fake_with_rate_limit_retry(runnable):
            # Production order: with_rate_limit_retry(llm.bind_tools(tools))
            # Use a counter to differentiate sequential branch calls
            call_id = call_counter["count"]
            call_counter["count"] += 1

            async def fake_ainvoke(prompt):
                tools = runnable.bind_tools.call_args[0][0]
                record_finding_tool = next((t for t in tools if getattr(t, "name", None) == "record_finding"), None)

                # Each branch call gets a unique call_id, so findings will be distinct
                await record_finding_tool.ainvoke(
                    {"path": f"file-{call_id}.py", "line": 1, "severity": "info", "body": f"finding-{call_id}"}
                )

                msg = Mock()
                msg.content = "Done"
                msg.tool_calls = []
                return msg

            mock_llm = Mock()
            mock_llm.ainvoke = fake_ainvoke
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        from pathlib import Path

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=Path("/fake/runtime"),
        )
        shell = FakeShell()

        # Run two branches with different domains
        state1 = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }
        result1 = await review_branch(ctx, shell, state1)

        state2 = {
            "domain": "security",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }
        result2 = await review_branch(ctx, shell, state2)

        # Assert each branch has exactly one finding and they are different
        assert len(result1["findings"]) == 1, "Branch 1 must have 1 finding"
        assert len(result2["findings"]) == 1, "Branch 2 must have 1 finding"

        finding1 = result1["findings"][0]
        finding2 = result2["findings"][0]

        # They must be distinct (different path or body)
        assert finding1["path"] != finding2["path"] or finding1["body"] != finding2["body"], (
            "Two branches recorded the same finding — findings list is shared (mutable default bug)"
        )


class TestCancellation:
    """Tests for cancellation mid-fan-out."""

    @pytest.mark.asyncio
    async def test_branch_respects_cancellation(self, monkeypatch):
        """A branch must check ctx.is_cancelled and raise CancelledError.

        This pins the cancellation check that prevents a long-running review from
        continuing after the user cancels the task.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)

        from quality.agents.pr import prompts as prompts_module

        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt")

        from quality import ratelimit as ratelimit_module

        def fake_with_rate_limit_retry(runnable):
            # Return a mock LLM that never completes (would hang if not cancelled)
            async def fake_ainvoke(prompt):
                import asyncio

                await asyncio.sleep(100)  # Would hang
                return Mock(content="Done", tool_calls=[])

            mock_llm = Mock()
            mock_llm.ainvoke = fake_ainvoke
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        from pathlib import Path

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=Path("/fake/runtime"),
        )
        ctx.is_cancelled = True  # Signal cancellation
        shell = FakeShell()

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }

        import asyncio

        with pytest.raises(asyncio.CancelledError):
            await review_branch(ctx, shell, state)


class TestEdgeCases:
    """Edge cases and error paths."""

    @pytest.mark.asyncio
    async def test_branch_that_records_nothing_is_valid(self, monkeypatch):
        """A branch that records no findings (clean code) must be a valid outcome, not an error.

        This pins the empty-findings case as a success, not a failure.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)

        from quality.agents.pr import prompts as prompts_module

        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt")

        from quality import ratelimit as ratelimit_module

        def fake_with_rate_limit_retry(runnable):
            async def fake_ainvoke(prompt):
                # LLM stops without recording anything
                return Mock(content="Code looks good, no findings.", tool_calls=[])

            mock_llm = Mock()
            mock_llm.ainvoke = fake_ainvoke
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        from pathlib import Path

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=Path("/fake/runtime"),
        )
        shell = FakeShell()

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }

        result = await review_branch(ctx, shell, state)

        # Assert branch completed successfully with empty findings
        assert "findings" in result, "Branch result must include a 'findings' key"
        assert result["findings"] == [], "Branch with no findings must return empty list"

        # Assert branch returns ONLY reducer fields (not domain/model/cur_*)
        # Per ava semantics and InvalidUpdateError prevention: returning non-reducer fields
        # from concurrent branches raises InvalidUpdateError. Attribution must ride inside
        # each finding/note dict, not at the top level of the return value.
        assert set(result.keys()) == {"findings", "notes"}, (
            f"Branch must return ONLY reducer fields {{findings, notes}}, got: {set(result.keys())}"
        )

    @pytest.mark.asyncio
    async def test_branch_failure_preserves_partial_findings_and_adds_failed_note(self, monkeypatch):
        """When a branch raises, it must preserve partial findings and add a failed=True note.

        Per ava semantics: branch failure is isolated. The failed branch returns:
        - `findings`: partial findings recorded before the exception (NOT [])
        - `notes`: a note with `failed: True`, domain, model, and error type in body
        - ctx.progress call to inform the user

        This ensures: (1) one gateway outage doesn't void other branches' work,
        (2) partial work before the exception is preserved, (3) synthesis sees the
        branch happened rather than assuming the model found nothing.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)

        from quality.agents.pr import prompts as prompts_module

        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt")

        from quality import ratelimit as ratelimit_module

        def fake_with_rate_limit_retry(runnable):
            # Production order: with_rate_limit_retry(llm.bind_tools(tools))
            async def fake_ainvoke(prompt):
                tools = runnable.bind_tools.call_args[0][0]
                record_finding_tool = next((t for t in tools if getattr(t, "name", None) == "record_finding"), None)

                # Record one finding, then raise
                await record_finding_tool.ainvoke(
                    {"path": "partial.py", "line": 5, "severity": "low", "body": "Partial finding before crash"}
                )
                raise RuntimeError("Simulated gateway failure")

            mock_llm = Mock()
            mock_llm.ainvoke = fake_ainvoke
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        from pathlib import Path

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=Path("/fake/runtime"),
        )
        shell = FakeShell()

        state = {
            "domain": "security",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }

        result = await review_branch(ctx, shell, state)

        # Assert partial finding is preserved (NOT discarded)
        assert "findings" in result, "Branch result must include 'findings'"
        assert len(result["findings"]) == 1, "Partial findings before exception must be preserved"
        assert result["findings"][0]["path"] == "partial.py", "Partial finding content must be intact"

        # Assert a failed=True note was added
        assert "notes" in result, "Branch result must include 'notes'"
        assert len(result["notes"]) >= 1, "Branch must add a failed note on exception"

        failed_note = next((n for n in result["notes"] if n.get("failed")), None)
        assert failed_note is not None, "Branch must add a note with failed=True"
        assert failed_note["domain"] == "security", "Failed note must include domain"
        assert failed_note["model"] == "model-a", "Failed note must include model"
        assert "RuntimeError" in failed_note["body"], "Failed note body must mention exception type"

        # Assert ctx.progress was called to inform the user
        assert any("failed" in msg.lower() for msg in ctx.progress_calls), (
            "Branch must call ctx.progress to report failure"
        )

    @pytest.mark.asyncio
    async def test_branch_failure_excludes_exception_message_from_note(self, monkeypatch):
        """Branch exception message with secret must NOT leak into failed note.

        Production (review.py:428) includes only type(exc).__name__ in the note body, not
        str(exc). The exception message is the actual leak vector: note bodies feed the
        synthesis prompt (review.py:795) and are published to GitHub, so str(exc) must
        never reach them.

        This test verifies the NEGATIVE half of the #50 security property: the exception
        TYPE appears in the note (existing test), but the exception MESSAGE does not.

        Coverage, not a fix: production is correct today (verified by execution with a
        realistic credential in the exception message — note body had only the type,
        progress had the redacted traceback).
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import review_branch

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)

        from quality.agents.pr import prompts as prompts_module

        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt")

        from quality import ratelimit as ratelimit_module

        # Realistic credential shape: GitHub PAT
        SECRET = "ghp_" + "a" * 32  # 36-char GitHub PAT

        def fake_with_rate_limit_retry(runnable):
            async def fake_ainvoke(prompt):
                # Raise exception with secret in message (simulating gateway auth failure)
                raise RuntimeError(f"gateway auth failed for token {SECRET}")

            mock_llm = Mock()
            mock_llm.ainvoke = fake_ainvoke
            return mock_llm

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

        from pathlib import Path

        ctx = FakeAgentContext(
            providers=["alpha"],
            config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
            runtime_dir_path=Path("/fake/runtime"),
        )
        shell = FakeShell()

        state = {
            "domain": "security",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": "/wt",
        }

        result = await review_branch(ctx, shell, state)

        # Assert a failed=True note was added
        assert "notes" in result, "Branch result must include 'notes'"
        failed_note = next((n for n in result["notes"] if n.get("failed")), None)
        assert failed_note is not None, "Branch must add a note with failed=True"

        # POSITIVE: exception type IS present (keeps #50 property)
        assert "RuntimeError" in failed_note["body"], "Failed note body must mention exception type"

        # NEGATIVE: raw secret is NOT in ANY string field of the note
        # Iterate all fields to prevent future field additions from reintroducing the leak
        for field_name, field_value in failed_note.items():
            if isinstance(field_value, str):
                assert SECRET not in field_value, (
                    f"Failed note field '{field_name}' must not contain raw secret. "
                    f"Note bodies feed synthesis prompt and are published to GitHub. "
                    f"Production must use type(exc).__name__ only, not str(exc)."
                )

        # Assert secret is not in ctx.progress calls (redaction should work there too)
        for progress_msg in ctx.progress_calls:
            assert SECRET not in progress_msg, (
                "ctx.progress must not contain raw secret. "
                "Production redacts via redact_secrets(traceback.format_exc()) at review.py:422."
            )

    @pytest.mark.asyncio
    async def test_stock_config_with_no_api_keys_must_not_post_false_clean_review(
        self, monkeypatch, tmp_path, stub_setup_seams
    ):
        """Stock install with no api_keys must NOT post "No issues found" — zero reviewers ran.

        Bug #53: A stock install posts a GitHub review saying "No quality or security issues
        found." when zero models actually reviewed anything. The shipped config declares
        opus+gpt with no api_key by design (users add keys in ~/.quality/config.toml), so
        _resolve_matrix returns [] and route_to_branches sends it straight to synthesis.
        The total-outage guard at review.py:745 requires `notes` to be non-empty, but an
        empty matrix yields no notes and no findings — so it lands in the clean-PR branch
        at :750. "Zero reviewers ran" and "two reviewers ran and found nothing" are currently
        the same state.

        This test pins the NEGATIVE property: with the real shipped config, _resolve_matrix
        returns [] and driving the graph with matrix=[] must NOT submit any review, NOT
        write a baseline, and must surface an error mentioning api_key or provider config.

        Covers both GitHub and local mode (the local branch at :751-759 writes an artifact
        with the same false text).

        Uses ``stub_setup_seams`` rather than patching out the ``setup`` node: ``_resolve_matrix``
        is called INSIDE setup (review.py:236), so replacing the node would delete the code
        under test and the empty matrix would come from the initial state instead of from the
        shipped config. Letting the real setup body run makes this an end-to-end assertion about
        a stock install, and keeps the link between "config has no api_key" and "nothing posted".
        """
        import tomllib
        from pathlib import Path

        from quality.agents.pr.tasks.review import ReviewState, _resolve_matrix, build_graph

        # Load the REAL shipped config — the point is that the *shipped* default has this property
        shipped_config_path = Path(__file__).parent.parent / "quality" / "config.toml"
        with open(shipped_config_path, "rb") as f:
            shipped_config = tomllib.load(f)

        # Verify the shipped config has no api_key in the default [llm] block
        assert "api_key" not in shipped_config.get("llm", {}), (
            "This test assumes the shipped config has no api_key in [llm] by design. "
            "If that changed, this test's premise is invalid."
        )

        # Verify the shipped config has providers without api_keys
        providers_config = shipped_config.get("llm", {}).get("providers", {})
        assert len(providers_config) > 0, "Shipped config must declare providers"
        for name, cfg in providers_config.items():
            assert "api_key" not in cfg, (
                f"Provider '{name}' has an api_key in shipped config — "
                "this test assumes providers lack api_keys by design."
            )

        # Use shared fake from conftest
        from conftest import FakeAgentContext, FakeShell

        # Custom context that raises if llm() is called (proves no model was invoked)
        class CtxThatRejectsLLMCalls(FakeAgentContext):
            def __init__(self):
                super().__init__(config=shipped_config, runtime_dir_path=tmp_path)
                self.task_id = "test-stock-config"

            def llm(self, name=None):
                raise AssertionError(
                    f"llm(name={name!r}) was called, but with matrix=[] no model should run. "
                    "This means the empty-matrix guard is not working."
                )

        ctx = CtxThatRejectsLLMCalls()

        # 1. Verify _resolve_matrix returns [] with the shipped config
        matrix = _resolve_matrix(ctx)
        assert matrix == [], f"_resolve_matrix with shipped config (no api_keys) must return [], got {matrix}"

        # Stub GitHub seams
        from quality import gh as gh_module

        submitted_reviews = []
        posted_comments = []

        async def fake_submit_pr_review(shell, repo, number, event, body):
            submitted_reviews.append((event, body))

        async def fake_create_pr_review_comment(shell, repo, number, body, path, line, commit_id=None):
            posted_comments.append({"path": path, "line": line, "body": body})

        async def fake_list_review_comments(shell, repo, number):
            return []

        def fake_commentable_lines(diff):
            return {"test.py": {10}}

        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", fake_create_pr_review_comment, raising=True)
        monkeypatch.setattr(gh_module, "list_review_comments", fake_list_review_comments, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", fake_commentable_lines, raising=True)

        # Stub memory seams
        from quality.agents.pr import memory as memory_module

        saved_baselines = []

        def fake_save_baseline(root, **kwargs):
            saved_baselines.append(kwargs)
            return root / "baseline.json"

        monkeypatch.setattr(memory_module, "save_baseline", fake_save_baseline, raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "baseline.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: {"findings": []}, raising=True)

        # Setup seams are stubbed by the stub_setup_seams fixture, which leaves the real
        # setup body — and therefore _resolve_matrix — executing.

        # 2. Drive the graph in GitHub mode. matrix is deliberately NOT set here: the real
        # setup node resolves it from the shipped config, which is the property under test.
        shell = FakeShell()

        initial_state_github = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path="",  # setup will populate
            head_sha="",
            error=None,
            is_followup=False,
            findings=[],
            notes=[],
            local=False,
        )

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        result_github = await compiled.ainvoke(initial_state_github)

        # Assert the real setup node RAN and resolved the matrix from the shipped config,
        # rather than short-circuiting on a seam error. Without this, a setup failure would
        # satisfy the error assertion below for entirely the wrong reason.
        assert result_github.get("head_sha") == "stub-head-sha", (
            f"setup must have run (head_sha from stub_setup_seams), got {result_github.get('head_sha')!r}. "
            "If this is empty, setup raised and every assertion below is testing the wrong failure."
        )
        assert any("no api_key configured" in m for m in ctx.progress_calls), (
            "setup must have resolved the matrix from the shipped config and skipped both "
            f"key-less providers. progress_calls={ctx.progress_calls}"
        )
        assert result_github.get("matrix") == [], (
            f"matrix must be resolved to [] by setup, got {result_github.get('matrix')!r}"
        )

        # 3. Assert NO review was submitted (GitHub mode)
        assert len(submitted_reviews) == 0, (
            f"Expected 0 reviews submitted with matrix=[], got {len(submitted_reviews)}. "
            f"Submitted: {submitted_reviews}. "
            "Bug #53: stock install posts false 'No issues found' when zero reviewers ran."
        )

        # 4. Assert NO baseline was written (GitHub mode)
        assert len(saved_baselines) == 0, (
            f"Expected 0 baselines saved with matrix=[], got {len(saved_baselines)}. "
            "An empty matrix should not persist a baseline claiming a clean review."
        )

        # 5. Assert error is present and mentions api_key or provider config
        assert result_github.get("error"), (
            "result['error'] must be truthy with matrix=[]. "
            "Setup failure (e.g. 'FakeShell' object has no attribute 'run') is a lying failure — "
            "assert error mentions api_key or provider config, not a setup defect."
        )
        error_text = result_github["error"].lower()
        assert "api_key" in error_text or "provider" in error_text or "config" in error_text, (
            f"Error must mention api_key or provider config, got: {result_github['error']}"
        )

        # 6. Cover local mode — reset recording state
        submitted_reviews.clear()
        saved_baselines.clear()

        initial_state_local = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path="",
            head_sha="",
            error=None,
            is_followup=False,
            findings=[],
            notes=[],
            local=True,  # Local mode
        )

        result_local = await compiled.ainvoke(initial_state_local)

        # 7. Assert NO artifact was written (local mode)
        artifact_dir = tmp_path / "reviews" / "github.com/org/repo"
        artifact_files = list(artifact_dir.glob("*.md")) if artifact_dir.exists() else []
        assert len(artifact_files) == 0, (
            f"Expected 0 artifacts written with matrix=[] in local mode, got {len(artifact_files)}. "
            "Bug #53: local branch at review.py:751-759 writes artifact with false clean text."
        )

        # 8. Assert NO baseline was written (local mode)
        assert len(saved_baselines) == 0, (
            f"Expected 0 baselines saved with matrix=[] in local mode, got {len(saved_baselines)}"
        )

        # 9. Assert error is present (local mode)
        assert result_local.get("error"), "result['error'] must be truthy with matrix=[] in local mode"
        error_text_local = result_local["error"].lower()
        assert "api_key" in error_text_local or "provider" in error_text_local or "config" in error_text_local, (
            f"Error must mention api_key or provider config (local mode), got: {result_local['error']}"
        )
