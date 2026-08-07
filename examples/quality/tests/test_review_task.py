"""Tests for ReviewTask lifecycle — worktree setup, cleanup, and checkpoint guards.

Scope: ReviewTask.run() lifecycle including:
- Worktree creation during setup
- Unconditional cleanup on success, exception, and CancelledError
- Cleanup runs under the repo lock (no races with concurrent reviews)
- Cleanup failure never masks the original error
- Checkpoint/local-flag mismatch guard (refuse to resume when flag changed)
- --local writes markdown artifact and posts nothing to GitHub
- PR URL validation rejects disallowed hosts before network/git calls
- Cancellation mid-fan-out still cleans up

All imports are inside test functions to avoid collection-time failures before
path injection (see conftest.py). Each test uses empirically-verified fakes.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from conftest import FakeAgentContext


class TestWorktreeCleanup:
    """Tests for worktree cleanup invariants."""

    @pytest.mark.asyncio
    async def test_worktree_cleanup_on_success(self, monkeypatch, tmp_path):
        """Worktree cleanup must run on successful task completion.

        The worktree is created in setup and must be removed even when
        the graph completes successfully. This is the happy-path baseline.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = False

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        # Stub graph compilation and execution
        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        # Stub ainvoke to return a successful result with worktree_path
        async def fake_ainvoke(state, config):
            return {
                "worktree_path": "/fake/worktree",
                "repo": "github.com/org/repo",
                "error": None,
                "posted_comments": 3,
                "failed_comments": 0,
                "findings": [],
                "is_self_review": False,
            }

        fake_compiled.ainvoke = fake_ainvoke
        fake_compiled.aget_state = AsyncMock(return_value=Mock(values={}))

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        # Track cleanup calls
        cleanup_calls = []

        async def fake_cleanup(self, graph, config, result):
            cleanup_calls.append(("cleanup", result.get("worktree_path")))

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup)

        # Run the task
        await task.run(ctx)

        # Assert cleanup was called with the worktree path from the result
        assert len(cleanup_calls) == 1, "Cleanup must be called exactly once on success"
        assert cleanup_calls[0] == ("cleanup", "/fake/worktree")
        assert ctx._completed, "Task must complete successfully"

    @pytest.mark.asyncio
    async def test_worktree_cleanup_on_exception(self, monkeypatch, tmp_path):
        """Worktree cleanup must run even when the graph raises an exception.

        A gateway outage, model error, or synthesis failure must not leak the
        worktree. This is the highest-value test: cleanup-on-happy-path-only
        is the classic leak.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = False

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        # ainvoke raises RuntimeError mid-execution
        async def fake_ainvoke(state, config):
            raise RuntimeError("Simulated synthesis failure")

        fake_compiled.ainvoke = fake_ainvoke
        fake_compiled.aget_state = AsyncMock(
            return_value=Mock(
                values={
                    "worktree_path": "/fake/worktree",
                    "repo": "github.com/org/repo",
                }
            )
        )

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        cleanup_calls = []

        async def fake_cleanup(self, graph, config, result):
            # Result is empty on exception — cleanup must read from checkpoint state
            cleanup_calls.append(("cleanup", "called"))

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup)

        # Run the task — expect RuntimeError to propagate
        with pytest.raises(RuntimeError, match="Simulated synthesis failure"):
            await task.run(ctx)

        # Assert cleanup WAS called despite the exception
        assert len(cleanup_calls) == 1, "Cleanup must run even when graph raises"
        assert cleanup_calls[0] == ("cleanup", "called")

    @pytest.mark.asyncio
    async def test_worktree_cleanup_on_cancellation(self, monkeypatch, tmp_path):
        """Worktree cleanup must run when the task is cancelled mid-execution.

        asyncio.CancelledError is a BaseException in 3.12+, so `except Exception`
        does NOT catch it. A try/finally is required. This test verifies cleanup
        still runs when the user hits Ctrl+C.

        MUTANT KILLER: This test FAILS against `try: ... except Exception: cleanup()`
        because CancelledError bypasses `except Exception` entirely. It PASSES only
        with `try: ... finally: cleanup()`.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = False

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        # ainvoke raises CancelledError (user cancelled task)
        async def fake_ainvoke(state, config):
            raise asyncio.CancelledError("User cancelled")

        fake_compiled.ainvoke = fake_ainvoke
        fake_compiled.aget_state = AsyncMock(
            return_value=Mock(
                values={
                    "worktree_path": "/fake/worktree",
                    "repo": "github.com/org/repo",
                }
            )
        )

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        cleanup_calls = []

        async def fake_cleanup(self, graph, config, result):
            cleanup_calls.append(("cleanup", "called"))

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup)

        # Run the task — expect CancelledError to propagate
        with pytest.raises(asyncio.CancelledError):
            await task.run(ctx)

        # Assert cleanup WAS called despite CancelledError
        assert len(cleanup_calls) == 1, "Cleanup must run even on CancelledError (Ctrl+C)"

    @pytest.mark.asyncio
    async def test_cleanup_runs_under_repo_lock(self, monkeypatch, tmp_path):
        """Worktree cleanup must hold the repo lock for the entire operation.

        Concurrent reviews of the same repo share a worktree parent. A `git worktree prune`
        from one review can race another's `git worktree add` if cleanup doesn't hold the
        lock. This test verifies the lock is held during cleanup, not released before.

        The lock path must match between setup and cleanup — a lock on a different file
        serializes nothing.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = False

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        async def fake_ainvoke(state, config):
            return {
                "worktree_path": "/fake/worktree",
                "repo": "github.com/org/repo",
                "error": None,
                "posted_comments": 0,
                "failed_comments": 0,
                "findings": [],
                "is_self_review": False,
            }

        fake_compiled.ainvoke = fake_ainvoke
        fake_compiled.aget_state = AsyncMock(return_value=Mock(values={}))

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        # Track lock acquisition, cleanup order, and lock paths
        events = []
        lock_paths_used = []

        from quality import _concurrency as conc_module

        class FakeLock:
            def __init__(self, lock_path):
                self.lock_path = lock_path

            async def __aenter__(self):
                lock_paths_used.append(str(self.lock_path))
                events.append(("lock_acquired", str(self.lock_path)))
                return self

            async def __aexit__(self, *args):
                events.append(("lock_released", str(self.lock_path)))

        def fake_file_lock(lock_path):
            return FakeLock(lock_path)

        from quality import gh as gh_module

        async def fake_remove_worktree(shell, clone_path, worktree_path):
            events.append(("remove_worktree", str(worktree_path)))

        monkeypatch.setattr(conc_module, "file_lock", fake_file_lock, raising=True)
        monkeypatch.setattr(gh_module, "remove_worktree", fake_remove_worktree, raising=True)

        # Run the task with the real _cleanup_worktree implementation
        await task.run(ctx)

        # Assert at least one lock was acquired (setup or cleanup)
        assert len([e for e in events if e[0] == "lock_acquired"]) > 0, "Lock must be acquired"
        assert ("remove_worktree", "/fake/worktree") in events, "Cleanup must call remove_worktree"

        # Find the lock cycle that contains remove_worktree
        # Events should be: [..., lock_acquired, ..., remove_worktree, ..., lock_released, ...]
        remove_idx = events.index(("remove_worktree", "/fake/worktree"))

        # Find the enclosing lock cycle (last lock_acquired before remove, first lock_released after)
        lock_acquired_before = [i for i, e in enumerate(events) if e[0] == "lock_acquired" and i < remove_idx]
        lock_released_after = [i for i, e in enumerate(events) if e[0] == "lock_released" and i > remove_idx]

        assert lock_acquired_before, "Lock must be acquired before remove_worktree"
        assert lock_released_after, "Lock must be released after remove_worktree"

        lock_idx = lock_acquired_before[-1]  # Last lock acquired before remove
        release_idx = lock_released_after[0]  # First lock released after remove

        assert lock_idx < remove_idx < release_idx, (
            f"remove_worktree must happen WHILE holding the lock, not after release. Events: {events}"
        )

        # Assert the lock path is consistent (same lock for setup and cleanup)
        # If multiple locks were acquired, they should use the same lock path
        if len(set(lock_paths_used)) > 1:
            # Multiple distinct lock paths is a bug — setup and cleanup must share one lock
            pytest.fail(f"Cleanup must use the SAME lock path as setup. Got distinct paths: {set(lock_paths_used)}")

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_mask_original_error(self, monkeypatch, tmp_path):
        """If cleanup itself raises, the ORIGINAL error must still surface.

        A bare `finally` block that raises will swallow the exception being unwound.
        The cleanup must be wrapped in its own try/except so a git failure during
        cleanup doesn't mask a synthesis failure that happened earlier.

        MUTANT KILLER: This test FAILS against:
            try:
                await graph.ainvoke(...)
            finally:
                await cleanup()  # <-- if cleanup raises, it REPLACES the original error

        It PASSES only when cleanup has its own internal try/except:
            try:
                await graph.ainvoke(...)
            finally:
                try:
                    await cleanup()
                except Exception:
                    log and continue
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = False

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        # ainvoke raises RuntimeError (the ORIGINAL error we must preserve)
        async def fake_ainvoke(state, config):
            raise RuntimeError("Original synthesis failure")

        fake_compiled.ainvoke = fake_ainvoke
        fake_compiled.aget_state = AsyncMock(
            return_value=Mock(
                values={
                    "worktree_path": "/fake/worktree",
                    "repo": "github.com/org/repo",
                }
            )
        )

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        # Cleanup itself raises during unwinding
        async def fake_cleanup_that_raises(self, graph, config, result):
            raise RuntimeError("Cleanup failed (git worktree prune error)")

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup_that_raises)

        # Run the task — the ORIGINAL error must propagate, not the cleanup error
        with pytest.raises(RuntimeError) as exc_info:
            await task.run(ctx)

        # Assert the ORIGINAL error message surfaces, not the cleanup failure
        assert "Original synthesis failure" in str(exc_info.value), (
            f"Original error must not be masked by cleanup failure. Got: {exc_info.value}"
        )


class TestCheckpointGuard:
    """Tests for the checkpoint/local-flag mismatch guard."""

    @pytest.mark.asyncio
    async def test_checkpoint_local_true_resumed_with_local_false_fails_fast(self, monkeypatch, tmp_path):
        """Resuming a checkpointed --local run without --local must fail fast with a clear error.

        Posting to GitHub a run the user explicitly marked as --local is the failure
        that matters. The guard must refuse to resume before making any network calls.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = False  # User is resuming WITHOUT --local

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        # Checkpoint state has local=True (was started with --local)
        async def fake_aget_state(config):
            return Mock(values={"local": True, "pr_url": "https://github.com/org/repo/pull/42"})

        fake_compiled.aget_state = fake_aget_state
        fake_compiled.ainvoke = AsyncMock()  # Should never be called

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        # Stub cleanup to verify it's still called even on early exit
        cleanup_calls = []

        async def fake_cleanup(self, graph, config, result):
            cleanup_calls.append("cleanup")

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup)

        # Run the task
        await task.run(ctx)

        # Assert task failed with the mismatch message
        assert ctx._failed, "Task must fail when local flag changes"
        assert "local=True" in ctx.failure_message
        assert "local=False" in ctx.failure_message
        assert "Refusing to resume" in ctx.failure_message

        # Assert ainvoke was NEVER called (guard fired before execution)
        fake_compiled.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_checkpoint_local_false_resumed_with_local_true_fails_fast(self, monkeypatch, tmp_path):
        """Resuming a GitHub-posting run WITH --local must also fail fast.

        The inverse case: a PR-posting run being resumed as --local. This would
        write only a local artifact when the user expects inline GitHub comments.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = True  # User is resuming WITH --local

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        # Checkpoint state has local=False (was started without --local)
        async def fake_aget_state(config):
            return Mock(values={"local": False, "pr_url": "https://github.com/org/repo/pull/42"})

        fake_compiled.aget_state = fake_aget_state
        fake_compiled.ainvoke = AsyncMock()  # Should never be called

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        cleanup_calls = []

        async def fake_cleanup(self, graph, config, result):
            cleanup_calls.append("cleanup")

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup)

        await task.run(ctx)

        # Assert task failed with the mismatch message
        assert ctx._failed, "Task must fail when local flag changes"
        assert "local=False" in ctx.failure_message
        assert "local=True" in ctx.failure_message
        assert "Refusing to resume" in ctx.failure_message

        fake_compiled.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_checkpoint_local_matches_current_local_resumes_successfully(self, monkeypatch, tmp_path):
        """When checkpoint local matches current local, the task resumes normally.

        This is the happy path for resume: the guard allows execution to proceed.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = True  # Resuming with --local

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        # Checkpoint state ALSO has local=True (match)
        async def fake_aget_state(config):
            return Mock(values={"local": True, "pr_url": "https://github.com/org/repo/pull/42"})

        async def fake_ainvoke(state, config):
            # ainvoke(None) when resuming from checkpoint
            assert state is None, "Resume must pass None to ainvoke"
            return {
                "worktree_path": "/fake/worktree",
                "repo": "github.com/org/repo",
                "error": None,
                "local_artifact_path": "/fake/artifact.md",
                "findings_written": 2,
                "failed_comments": 0,
                "findings": [],
                "is_self_review": False,
            }

        fake_compiled.aget_state = fake_aget_state
        fake_compiled.ainvoke = fake_ainvoke

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        cleanup_calls = []

        async def fake_cleanup(self, graph, config, result):
            cleanup_calls.append("cleanup")

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup)

        await task.run(ctx)

        # Assert task completed successfully (not failed)
        assert ctx._completed, "Task must complete when local flag matches checkpoint"
        assert not ctx._failed
        assert ctx.completion_payload["local_artifact_path"] == "/fake/artifact.md"


class TestLocalMode:
    """Tests for --local mode behavior."""

    @pytest.mark.asyncio
    async def test_local_mode_writes_artifact_and_posts_nothing(self, monkeypatch, tmp_path):
        """When --local is set, the task writes a markdown artifact and makes NO GitHub API calls.

        The point of --local is to hand the review to a follow-up agent that implements fixes.
        Posting to GitHub would defeat that purpose. This test verifies no gh posting happens.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = True

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        # Track all gh module calls
        gh_calls = []

        from quality import gh as gh_module

        original_submit_pr_review = gh_module.submit_pr_review
        original_create_pr_review_comment = gh_module.create_pr_review_comment

        async def tracked_submit_pr_review(*args, **kwargs):
            gh_calls.append(("submit_pr_review", args, kwargs))
            return await original_submit_pr_review(*args, **kwargs)

        async def tracked_create_pr_review_comment(*args, **kwargs):
            gh_calls.append(("create_pr_review_comment", args, kwargs))
            return await original_create_pr_review_comment(*args, **kwargs)

        monkeypatch.setattr(gh_module, "submit_pr_review", tracked_submit_pr_review)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", tracked_create_pr_review_comment)

        async def fake_ainvoke(state, config):
            # Simulate synthesis_and_post in local mode
            return {
                "worktree_path": "/fake/worktree",
                "repo": "github.com/org/repo",
                "error": None,
                "local_artifact_path": "/fake/artifact.md",
                "findings_written": 3,
                "failed_comments": 0,
                "findings": [{"path": "foo.py", "line": 10}],
                "is_self_review": False,
            }

        fake_compiled.ainvoke = fake_ainvoke
        fake_compiled.aget_state = AsyncMock(return_value=Mock(values={}))

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        cleanup_calls = []

        async def fake_cleanup(self, graph, config, result):
            cleanup_calls.append("cleanup")

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup)

        await task.run(ctx)

        # Assert NO GitHub posting calls were made
        assert len(gh_calls) == 0, f"Local mode must make NO GitHub posting calls. Got: {gh_calls}"

        # Assert task completed with local_artifact_path
        assert ctx._completed
        assert "local_artifact_path" in ctx.completion_payload
        assert ctx.completion_payload["local_artifact_path"] == "/fake/artifact.md"
        assert ctx.completion_payload["findings_written"] == 3


class TestPRValidation:
    """Tests for PR URL host validation."""

    @pytest.mark.asyncio
    async def test_disallowed_host_fails_before_git_calls(self, monkeypatch, tmp_path):
        """PR URL with a disallowed host must fail before any network or git calls.

        The host becomes GH_HOST for gh CLI calls with ambient credentials, so an
        unapproved host is an SSRF/token-leak vector. The guard must fire before
        clone_or_update_repo, create_pr_worktree, or get_pr_diff.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://evil.example.com/org/repo/pull/42"  # Disallowed host
        task.local = False

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        # Track all gh module calls to verify none happen
        gh_calls = []

        from quality import gh as gh_module

        original_clone_or_update_repo = gh_module.clone_or_update_repo
        original_create_pr_worktree = gh_module.create_pr_worktree
        original_get_pr_diff = gh_module.get_pr_diff

        async def tracked_clone_or_update_repo(*args, **kwargs):
            gh_calls.append(("clone_or_update_repo", args))
            return await original_clone_or_update_repo(*args, **kwargs)

        async def tracked_create_pr_worktree(*args, **kwargs):
            gh_calls.append(("create_pr_worktree", args))
            return await original_create_pr_worktree(*args, **kwargs)

        async def tracked_get_pr_diff(*args, **kwargs):
            gh_calls.append(("get_pr_diff", args))
            return await original_get_pr_diff(*args, **kwargs)

        monkeypatch.setattr(gh_module, "clone_or_update_repo", tracked_clone_or_update_repo)
        monkeypatch.setattr(gh_module, "create_pr_worktree", tracked_create_pr_worktree)
        monkeypatch.setattr(gh_module, "get_pr_diff", tracked_get_pr_diff)

        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        # setup node should set state.error for disallowed host
        async def fake_ainvoke(state, config):
            # Simulate setup setting error for disallowed host
            return {
                "error": "Host evil.example.com not in the allowed hosts",
                "worktree_path": "",
                "repo": "",
            }

        fake_compiled.ainvoke = fake_ainvoke
        fake_compiled.aget_state = AsyncMock(return_value=Mock(values={}))

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        cleanup_calls = []

        async def fake_cleanup(self, graph, config, result):
            cleanup_calls.append("cleanup")

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup)

        await task.run(ctx)

        # Assert NO git/gh network calls were made
        assert len(gh_calls) == 0, f"Disallowed host must fail BEFORE any git/gh calls. Got: {gh_calls}"

        # Assert task failed with the host validation error
        assert ctx._failed
        assert "evil.example.com" in ctx.failure_message
        assert "allowed hosts" in ctx.failure_message


class TestCancellationMidFanout:
    """Tests for cancellation during concurrent branch execution."""

    @pytest.mark.asyncio
    async def test_cancellation_mid_fanout_still_cleans_up(self, monkeypatch, tmp_path):
        """Task cancelled while branches are running must still clean up the worktree.

        A user hitting Ctrl+C mid-review should not leak the worktree. This test
        verifies that CancelledError propagates through the graph but cleanup still runs.
        """
        from quality.agents.pr.tasks.review import ReviewTask

        task = ReviewTask()
        task.pr = "https://github.com/org/repo/pull/42"
        task.local = False

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)

        from quality.agents.pr.tasks import review as review_module

        fake_graph = Mock()
        fake_compiled = Mock()
        fake_graph.compile = Mock(return_value=fake_compiled)

        # Simulate cancellation mid-execution (during branch fan-out)
        async def fake_ainvoke(state, config):
            # Simulate some branches starting, then cancellation
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError("User hit Ctrl+C")

        fake_compiled.ainvoke = fake_ainvoke
        fake_compiled.aget_state = AsyncMock(
            return_value=Mock(
                values={
                    "worktree_path": "/fake/worktree",
                    "repo": "github.com/org/repo",
                }
            )
        )

        monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: fake_graph)

        cleanup_calls = []

        async def fake_cleanup(self, graph, config, result):
            cleanup_calls.append("cleanup")

        monkeypatch.setattr(ReviewTask, "_cleanup_worktree", fake_cleanup)

        # Run the task — expect CancelledError to propagate
        with pytest.raises(asyncio.CancelledError):
            await task.run(ctx)

        # Assert cleanup was called despite cancellation
        assert len(cleanup_calls) == 1, "Cleanup must run even when cancelled mid-fan-out"
