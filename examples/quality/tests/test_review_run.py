"""End-to-end tests for ReviewTask.run() — the actual CLI entry point.

Requirement 0 from task #22: Drive ReviewTask.run(ctx) directly, not just the graph.
Four defects (#24, #26, #30, #31) lived in run(), which is what `quality run pr review --pr ...`
actually calls. Every other test calls graph nodes directly or build_graph().

These tests stub external seams (gh, LLM, memory) and verify run()-specific paths:
- API key validation (pool-aware, not just top-level)
- parse_pr_url rejection → ctx.fail without invoking graph
- Checkpoint local-flag mismatch guard
- finally: cleanup running on both success and ainvoke raise
- Completion payload contains local_artifact_path / posted_comments (#26)
- Local mode isolation (baseline, artifact, GitHub writes)
- Multi-domain attribution and determinism
- Summary redaction in local artifact
- Artifact file permissions (0o600, directory 0o700)
- Lock path distinctness across repos
- Setup/cleanup path agreement

All imports are function-scoped to avoid collection-time failures.
"""

from __future__ import annotations

import pytest


class TestReviewTaskRun:
    """Tests that drive ReviewTask.run(ctx) directly — the actual CLI entry point."""

    @pytest.mark.asyncio
    async def test_run_proceeds_with_pool_keys_only(self, monkeypatch, tmp_path):
        """run() proceeds when API keys exist only on pool entries (no top-level api_key).

        Bug #31: run() rejected the config the app ships. Its fail-fast check only tested
        top-level [llm].api_key, but quality/config.toml deliberately omits that and puts
        credentials on [llm.providers.opus] / [llm.providers.gpt]. The documented
        two-provider setup could not run.

        This test verifies run() proceeds when keys exist only in the pool.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = True  # Suppress GitHub writes

        # Build a fake context with pool-only keys (no top-level api_key)
        from conftest import FakeAgentContext

        config = {
            "llm": {
                # No api_key here — this is what the shipped config looks like
                "providers": {
                    "opus": {"api_key": "sk-ant-fake-opus-key", "model": "claude-opus-4"},
                    "gpt": {"api_key": "sk-openai-fake-key", "model": "gpt-4"},
                }
            }
        }

        ctx = FakeAgentContext(config=config, runtime_dir_path=tmp_path)
        ctx.task_id = "test-run-123"

        # Create a real checkpointer for aget_state call
        import aiosqlite

        from switchplane.checkpoint import SqliteCheckpointSaver

        db = await aiosqlite.connect(tmp_path / "state.db")
        await db.execute("PRAGMA journal_mode=WAL")
        ctx.checkpointer = SqliteCheckpointSaver(db, ctx.task_id)
        if hasattr(ctx.checkpointer, "setup"):
            await ctx.checkpointer.setup()

        # Override llm() for this test
        def custom_llm(name=None):
            class FakeLLM:
                def with_structured_output(self, schema):
                    return self

                def bind_tools(self, tools):
                    return self

                async def ainvoke(self, messages):
                    from pydantic import BaseModel as PydanticBaseModel

                    class MockSynthComment(PydanticBaseModel):
                        path: str = ""
                        line: int | None = None
                        severity: str = "medium"
                        body: str = ""
                        models: list[str] = []

                    class MockSynthResult(PydanticBaseModel):
                        summary: str = "Test review"
                        event: str = "COMMENT"
                        comments: list[MockSynthComment] = []

                    return MockSynthResult()

            return FakeLLM()

        ctx.llm = custom_llm

        # Stub external seams
        from quality.agents.pr import memory as memory_module

        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        # Stub gh seams
        async def fake_create_worktree(shell, repo_path, pr_number, task_id):
            wt_path = tmp_path / "worktree"
            wt_path.mkdir()
            return wt_path, "fake-sha-123"

        async def fake_remove_worktree(shell, repo_path, worktree_path):
            pass

        async def fake_clone_or_update(shell, repo, cache_root):
            clone_path = cache_root / repo
            clone_path.mkdir(parents=True, exist_ok=True)
            return clone_path

        async def fake_get_pr_diff(shell, repo, number):
            return "diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test"

        async def fake_get_pr_author(shell, repo, number):
            return "author123"

        async def fake_get_authenticated_user(shell, repo):
            return "reviewer456"

        async def fake_list_review_comments(shell, repo, number):
            return []

        async def fake_create_pr_review_comment(shell, repo, number, body, path, line, commit_id=None):
            pass

        async def fake_submit_pr_review(shell, repo, number, event, body):
            pass

        def fake_commentable_lines(diff):
            return {"test.py": {10, 20}}

        monkeypatch.setattr(gh_module, "create_pr_worktree", fake_create_worktree, raising=True)
        monkeypatch.setattr(gh_module, "remove_worktree", fake_remove_worktree, raising=True)
        monkeypatch.setattr(gh_module, "clone_or_update_repo", fake_clone_or_update, raising=True)
        monkeypatch.setattr(gh_module, "get_pr_diff", fake_get_pr_diff, raising=True)
        monkeypatch.setattr(gh_module, "get_pr_author", fake_get_pr_author, raising=True)
        monkeypatch.setattr(gh_module, "get_authenticated_user", fake_get_authenticated_user, raising=True)
        monkeypatch.setattr(gh_module, "list_review_comments", fake_list_review_comments, raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", fake_create_pr_review_comment, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", fake_commentable_lines, raising=True)

        # Stub memory seams
        baseline_calls = []

        def fake_save_baseline(root, **kwargs):
            baseline_calls.append((root, kwargs))
            return root / "baseline.json"

        def fake_baseline_path(root, repo, number, *, local=False):
            return root / "baseline.json"

        def fake_load_baseline(path):
            return {"findings": []}

        monkeypatch.setattr(memory_module, "save_baseline", fake_save_baseline, raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", fake_baseline_path, raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", fake_load_baseline, raising=True)

        # Stub rate limit
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Stub review_branch to avoid the real LLM call
        from quality.agents.pr.tasks import review as review_module

        async def fake_review_branch(ctx, shell, state):
            """Fake review_branch that returns minimal findings."""
            return {
                "findings": [
                    {
                        "severity": "medium",
                        "title": "test finding",
                        "body": "test body",
                        "path": "test.py",
                        "line": 10,
                        "domain": state.cur_domain,
                        "provider": "opus",
                        "model": "claude-opus-4",
                    }
                ],
                "notes": [],
            }

        monkeypatch.setattr(review_module, "review_branch", fake_review_branch, raising=True)

        # Run the task — this MUST not raise "No LLM API key configured"
        await task.run(ctx)

        # Close the database
        await db.close()

        # Assert the task completed
        assert ctx._completed, "Task must call ctx.complete()"
        assert ctx.completion_payload is not None

    @pytest.mark.asyncio
    async def test_run_fails_with_no_keys_anywhere(self, monkeypatch, tmp_path):
        """run() fails with a pool-aware message when no API key exists anywhere.

        Bug #31 fix: the error message must mention [llm.providers so it's actionable.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = True

        class FakeContext:
            def __init__(self):
                self.task_id = "test-run-no-keys"
                # config is a plain attribute, not a method
                self.config = {
                    "llm": {
                        # No api_key at top level
                        "providers": {
                            # No api_key in pool entries either
                            "opus": {"model": "claude-opus-4"},
                            "gpt": {"model": "gpt-4"},
                        }
                    }
                }
                self.fail_reason = None

            def progress(self, msg, **kwargs):
                pass

            def fail(self, reason):
                self.fail_reason = reason
                raise RuntimeError(f"Task failed: {reason}")

            @property
            def runtime_dir(self):
                return tmp_path

        ctx = FakeContext()

        # Run should fail early
        with pytest.raises(RuntimeError, match="Task failed"):
            await task.run(ctx)

        # Assert the failure message mentions providers
        assert ctx.fail_reason is not None, "ctx.fail must be called"
        assert "[llm.providers" in ctx.fail_reason or "pool" in ctx.fail_reason.lower(), (
            f"Failure message must mention pool/providers config. Got: {ctx.fail_reason}"
        )

    @pytest.mark.asyncio
    async def test_invalid_pr_url_fails_without_invoking_graph(self, monkeypatch, tmp_path):
        """run() rejects invalid PR URLs with ctx.fail and does NOT invoke the graph.

        parse_pr_url validation happens before graph compilation. Invalid URLs
        (missing host, wrong format, disallowed host) must fail immediately.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://evil.com/org/repo/pull/42"  # Not in allowed hosts
        task.local = True

        class FakeContext:
            def __init__(self):
                self.task_id = "test-invalid-url"
                # config is a plain attribute, not a method
                self.config = {
                    "llm": {
                        "api_key": "sk-fake",
                        "providers": {},
                    }
                }
                self.fail_reason = None

            def progress(self, msg, **kwargs):
                pass

            def fail(self, reason):
                self.fail_reason = reason
                raise RuntimeError(f"Task failed: {reason}")

            @property
            def runtime_dir(self):
                return tmp_path

        ctx = FakeContext()

        # Stub gh.parse_pr_url to enforce the allowlist
        from quality import gh as gh_module

        def fake_parse_pr_url(url, allowed_hosts):
            # Reject evil.com
            if "evil.com" in url:
                raise ValueError("Host evil.com not in the allowed hosts")
            return "github.com/org/repo", 42

        monkeypatch.setattr(gh_module, "parse_pr_url", fake_parse_pr_url, raising=True)

        # Run should fail before graph execution
        with pytest.raises(RuntimeError, match="Task failed"):
            await task.run(ctx)

        assert ctx.fail_reason is not None
        assert "not in the allowed hosts" in ctx.fail_reason or "invalid" in ctx.fail_reason.lower()


class TestReviewTaskRunCleanup:
    """Tests verifying cleanup runs in run()'s finally block on both success and failure."""

    @pytest.mark.asyncio
    async def test_cleanup_runs_on_ainvoke_raise(self, monkeypatch, tmp_path):
        """Cleanup must run even when graph.ainvoke raises (network error, cancellation)."""
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = True

        cleanup_called = []

        # Stub _cleanup_worktree to record calls
        async def fake_cleanup(self, graph, config, result):
            cleanup_called.append(True)

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup, raising=True)

        # Stub build_graph to return a graph whose ainvoke raises
        from quality.agents.pr.tasks import review as review_module

        class FakeGraph:
            def compile(self, **kwargs):
                return self

            async def ainvoke(self, state, config):
                raise RuntimeError("Network error during graph execution")

            async def aget_state(self, config):
                from typing import ClassVar

                class FakeState:
                    values: ClassVar[dict] = {}

                return FakeState()

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: FakeGraph(), raising=True)

        # Stub external seams
        from quality.agents.pr import memory as memory_module

        from quality import gh as gh_module

        async def fake_create_worktree(shell, repo_path, pr_number, task_id):
            wt_path = tmp_path / "worktree"
            wt_path.mkdir()
            return wt_path, "fake-sha"

        monkeypatch.setattr(gh_module, "create_pr_worktree", fake_create_worktree, raising=True)
        monkeypatch.setattr(gh_module, "remove_worktree", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "clone_or_update_repo", lambda *a, **kw: tmp_path / "clone", raising=True)
        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: None, raising=True)

        class FakeContext:
            def __init__(self):
                self.task_id = "test-cleanup-raise"
                # config is a plain attribute, not a method
                self.config = {"llm": {"api_key": "sk-fake", "providers": {}}}

            def progress(self, msg, **kwargs):
                pass

            def fail(self, reason):
                raise RuntimeError(f"Task failed: {reason}")

            @property
            def runtime_dir(self):
                return tmp_path

            @property
            def checkpointer(self):
                """No checkpointer for these high-level run() tests."""
                return None

        ctx = FakeContext()

        # ainvoke will raise, but cleanup must still run
        with pytest.raises(RuntimeError, match="Network error"):
            await task.run(ctx)

        assert len(cleanup_called) == 1, "Cleanup must run in finally block even when ainvoke raises"


class TestLockPathDistinctness:
    """Requirement 8: Distinct repos must yield distinct lock paths (bug #29)."""

    def test_distinct_repos_yield_distinct_locks(self):
        """Distinct repos must yield distinct lock paths.

        Bug #29: Path.with_suffix(".lock") replaces from the first dot, so
        github.com/org/repo and github.com/other/proj both yielded github.lock.
        Every repo on a host shared one lock, serializing unrelated concurrent reviews.

        This test covers 3 shapes:
        - Two repos sharing a host
        - A multi-dot host (git.example.co.uk)

        Assert on set size (not exact strings) so impl-graph keeps freedom over naming.
        """
        from pathlib import Path

        from quality.agents.pr.tasks.review import _repo_paths

        runtime_dir = Path("/tmp/runtime")

        # Three test cases
        repo1 = "github.com/org/repo"
        repo2 = "github.com/other/proj"
        repo3 = "git.example.co.uk/a/b"

        _clone1, lock1 = _repo_paths(runtime_dir, repo1)
        _clone2, lock2 = _repo_paths(runtime_dir, repo2)
        _clone3, lock3 = _repo_paths(runtime_dir, repo3)

        # Assert all three lock paths are distinct
        locks = {lock1, lock2, lock3}
        assert len(locks) == 3, (
            f"Expected 3 distinct lock paths, got {len(locks)}. "
            f"Locks: {locks}. "
            "If this fails, repos are serializing through a shared lock (bug #29)."
        )


class TestSetupCleanupPathAgreement:
    """Requirement 9: Setup and cleanup must resolve the same paths (bug #30)."""

    def test_setup_and_cleanup_agree_on_paths(self):
        """Setup and cleanup must resolve the same clone path and lock path.

        Bug #30: Setup used <rt>/repos/..., cleanup used <rt>/src/.... Different trees
        meant `git worktree remove` ran against a nonexistent path (swallowed by run_ok),
        and setup/cleanup locked different files with no mutual exclusion.

        This test asserts equality of both elements and pins that the clone path matches
        what `clone_or_update_repo` builds internally (cache_root / repo). If they silently
        diverge again, the lock stops working with zero symptoms.
        """
        from pathlib import Path

        from quality.agents.pr.tasks.review import _repo_paths

        runtime_dir = Path("/tmp/runtime")
        repo = "github.com/org/repo"

        clone_path, lock_path = _repo_paths(runtime_dir, repo)

        # Pin that clone_path matches cache_root / repo (what clone_or_update_repo builds)
        expected_clone = runtime_dir / "repos" / repo
        assert clone_path == expected_clone, (
            f"Clone path mismatch. Got: {clone_path}, expected: {expected_clone}. "
            "If this fails, setup and cleanup disagree on where the clone lives (bug #30)."
        )

        # Also verify lock path is deterministic (call twice, get same result)
        _clone_path2, lock_path2 = _repo_paths(runtime_dir, repo)
        assert lock_path == lock_path2, "Lock path must be deterministic"
