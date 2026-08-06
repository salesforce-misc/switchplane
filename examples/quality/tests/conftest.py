"""Test configuration and shared fixtures for examples/quality.

The import-path seam: no example package is installed in .venv, so we inject
the example root onto sys.path at session scope. This works only because the
example has no compiled dependencies (numpy, pandas, etc.).
"""

from pathlib import Path

import pytest

# Resolve the quality root (one level up from tests/)
QUALITY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _inject_quality_on_path():
    """Inject the quality example root onto sys.path so imports resolve.

    Guard: skip if langchain_core is absent (indicates switchplane[llm] not installed).
    """
    pytest.importorskip("langchain_core")
    import sys

    sys.path.insert(0, str(QUALITY_ROOT))
    yield
    sys.path.remove(str(QUALITY_ROOT))


class FakeAgentContext:
    """Shared fake AgentContext that mirrors the real class.

    Mirrored from src/switchplane/agent_runtime.py:202-266.
    Surface must match: config (attribute), runtime_dir (property),
    providers (property), checkpointer (attribute).

    Used across all test files to avoid drift between fakes and production.
    """

    def __init__(self, config: dict | None = None, runtime_dir_path: Path | None = None):
        """Initialize with config dict and optional runtime_dir override.

        Args:
            config: Configuration dict (matches self.config in real AgentContext)
            runtime_dir_path: Path to return from runtime_dir property (for tmp_path usage)
        """
        # Plain attribute, not a method (agent_runtime.py:240)
        self.config = config or {"llm": {"api_key": "test-key", "providers": {}}}

        # Store runtime dir path for property
        self._runtime_dir_path = runtime_dir_path or Path("/tmp/fake-runtime")

        # Other common attributes
        self.task_id = "test-task-123"
        self.checkpointer = None  # Set to None or Mock() by tests

        # Recording state for test assertions
        self.progress_calls: list[str] = []
        self._completed = False
        self._failed = False
        self.completion_payload = None
        self.failure_message = None

    @property
    def runtime_dir(self) -> Path:
        """Return runtime directory path (property, not method).

        Real AgentContext: agent_runtime.py:252-256.
        """
        return self._runtime_dir_path

    @property
    def providers(self) -> list[str]:
        """Return sorted list of provider name strings (property, not method).

        Real AgentContext: agent_runtime.py:258-266.
        Derives from self.config, so fake and real stay synchronized.
        """
        return sorted(self.config.get("llm", {}).get("providers", {}))

    async def check_cancelled(self) -> None:
        """Check if task was cancelled (async coroutine).

        Real AgentContext: agent_runtime.py awaits this at review.py:302.
        This fake implementation is a no-op (never raises).
        """
        pass

    def progress(self, message: str, **kwargs) -> None:
        """Record progress calls for test assertions."""
        self.progress_calls.append(message)

    def stream_flush(self, text: str) -> None:
        """Stream output (no-op in fake, for run_tool_loop compatibility)."""
        pass

    def tool_invoke(self, name: str, args_summary: str) -> None:
        """Record tool invocation (no-op in fake, for run_tool_loop compatibility)."""
        pass

    def complete(self, payload: dict) -> None:
        """Record completion for test assertions."""
        self._completed = True
        self.completion_payload = payload

    def fail(self, error: str, traceback_str: str | None = None) -> None:
        """Record failure for test assertions.

        Real AgentContext: agent_runtime.py:403-408.
        CRITICAL: This fake must NOT raise. A raising fail() hides missing-return
        defects in production — code that calls ctx.fail() without an explicit
        return/raise passes under test (the exception unwinds) but in production
        falls through and keeps executing after a fatal error.
        """
        self._failed = True
        self.failure_message = error
        self.failure_traceback = traceback_str

    def llm(self, name: str | None = None, *, model: str | None = None):
        """Return a fake LLM for tests.

        Override this method in test-specific subclasses if you need
        structured output or tool binding behavior.
        """
        from unittest.mock import Mock

        return Mock()


class FakeShell:
    """Shared fake Shell that mirrors switchplane.shell.Shell.

    Must provide fs_tools() method used by review.py:395.
    """

    def fs_tools(self) -> list:
        """Return empty list of filesystem tools (sync method).

        Real Shell: switchplane.shell.Shell.fs_tools() returns LangChain tools.
        This fake returns an empty list since tests don't exercise the tools.
        """
        return []


@pytest.fixture
def fake_ctx(tmp_path):
    """Fixture providing a FakeAgentContext with tmp_path as runtime_dir.

    Usage:
        def test_something(fake_ctx):
            fake_ctx.config["llm"]["api_key"] = "sk-test"
            assert fake_ctx.runtime_dir == tmp_path
    """
    return FakeAgentContext(runtime_dir_path=tmp_path)


@pytest.fixture
def stub_setup_seams(monkeypatch, tmp_path):
    """Stub the setup-node seams so the real graph can run offline.

    The setup node (review.py) calls into ``quality.gh`` for clone, worktree, diff,
    and PR metadata. A test that compiles the real graph without these stubs does
    NOT fail loudly: setup's broad ``except Exception`` converts the error into
    ``{"error": ...}``, ``route_to_branches`` short-circuits the fan-out to END, and
    every output key keeps its default. ``local_artifact_path`` stays ``""`` — and
    since ``Path("") == Path(".")``, downstream assertions silently target the CWD.

    This fixture is opt-in, NOT autouse: ``test_gh.py`` and ``test_gh_worktree_git.py``
    exist to test these very functions, and stubbing them globally would gut those
    suites. Request it explicitly in tests that drive the graph.

    All stubs are async, matching the real coroutine functions — a sync stub would
    make the ``await`` in setup raise TypeError. ``raising=True`` catches renames.

    Returns the worktree Path, so tests can assert against the checkout location.
    """
    pytest.importorskip("langchain_core")
    from quality import gh as gh_module

    worktree = tmp_path / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)

    async def _clone_or_update_repo(shell, repo, cache_root):
        clone = Path(cache_root) / repo
        clone.mkdir(parents=True, exist_ok=True)
        return clone

    async def _create_pr_worktree(shell, repo_path, pr_number, task_id):
        return worktree, "stub-head-sha"

    async def _get_pr_diff(shell, repo, pr_number):
        return "diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n@@ -1 +1,2 @@\n x = 1\n+y = 2\n"

    async def _get_pr_head_sha(shell, repo, pr_number):
        return "stub-head-sha"

    # Distinct from the authenticated user, so is_self_review stays False by default.
    async def _get_pr_author(shell, repo, pr_number):
        return "pr-author"

    async def _get_authenticated_user(shell, repo):
        return "authed-user"

    for name, stub in (
        ("clone_or_update_repo", _clone_or_update_repo),
        ("create_pr_worktree", _create_pr_worktree),
        ("get_pr_diff", _get_pr_diff),
        ("get_pr_head_sha", _get_pr_head_sha),
        ("get_pr_author", _get_pr_author),
        ("get_authenticated_user", _get_authenticated_user),
    ):
        monkeypatch.setattr(gh_module, name, stub, raising=True)

    return worktree


class FakeLLMForReview:
    """Shared fake LLM that serves both review branch (tool-calling) and synthesis.

    Production has two distinct call sites with different contracts:
    1. Review branch: bind_tools → run_tool_loop, needs AIMessage-like response with .tool_calls
    2. Synthesis: with_structured_output → returns SynthResult

    One shape cannot satisfy both. This fake provides both paths.

    Usage:
        ctx.llm = lambda name=None: FakeLLMForReview(synth_result=MockSynthResult())
    """

    def __init__(self, synth_result=None):
        """Initialize with optional synthesis result override.

        Args:
            synth_result: Result to return from with_structured_output path.
                         Defaults to a minimal MockSynthResult.
        """
        self._synth_result = synth_result
        self._bound_for_tools = False

    def bind_tools(self, tools):
        """Tool-calling path: mark this as bound for tools, return self."""
        self._bound_for_tools = True
        return self

    def with_structured_output(self, schema):
        """Synthesis path: return self (ainvoke will return synth_result)."""
        return self

    async def ainvoke(self, messages):
        """Return appropriate response based on whether we're in tool-calling or synthesis mode.

        Tool-calling mode (after bind_tools): returns AIMessage-like with empty tool_calls
        to terminate run_tool_loop immediately without executing tools.

        Synthesis mode (after with_structured_output): returns the configured synth_result.
        """
        if self._bound_for_tools:
            # Tool-calling path: return AIMessage-like response with empty tool_calls
            # This terminates run_tool_loop (switchplane/llm.py:253) without executing tools
            from langchain_core.messages import AIMessage

            return AIMessage(content="No findings", tool_calls=[])
        else:
            # Synthesis path: return the configured result
            if self._synth_result is None:
                # Default minimal synthesis result
                from pydantic import BaseModel as PydanticBaseModel

                class MockSynthComment(PydanticBaseModel):
                    path: str = "test.py"
                    line: int | None = 10
                    severity: str = "medium"
                    body: str = "Test finding"
                    models: list[str] = ["test-model"]

                class MockSynthResult(PydanticBaseModel):
                    summary: str = "Review complete"
                    event: str = "COMMENT"
                    comments: list[MockSynthComment] = []

                return MockSynthResult()
            return self._synth_result
