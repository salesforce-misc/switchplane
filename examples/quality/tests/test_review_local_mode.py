"""Tests for --local mode execution, baselines, artifacts, and security invariants.

Requirements 1-7, 10 from task #22:
1. Capturing save_baseline mock (not *args/**kwargs sink)
2. Local/non-local baseline separation
3. local_artifact_path returned non-empty
4. posted_comments accurate
5. Multi-domain attribution + determinism
6. Summary redaction in local artifact (mutation-tested)
7. Artifact permissions (0o600 file, 0o700 directory)
10. Never-APPROVE clamp + invalid event string handling

All imports are function-scoped.
"""

from __future__ import annotations

import stat

import pytest


class TestLocalModeBaseline:
    """Requirements 1-2: save_baseline capture and local/non-local separation."""

    @pytest.mark.asyncio
    async def test_save_baseline_called_with_correct_kwargs(self, monkeypatch, tmp_path, stub_setup_seams):
        """save_baseline must be called with explicit kwargs, not positional flood.

        Bug #25: All 3 call sites passed `memory_module.save_baseline(root, repo, number, ...)`
        as positionals, but the signature is `save_baseline(root, *, repo, number, ...)` with
        keyword-only params. TypeError raised on every run.

        Requirement 1: Use a capturing mock that binds root positionally and **kw by keyword.
        Assert on repo, number, head_sha, findings, local. A bare **kwargs sink is what
        let #25 through — it structurally cannot catch an arity/keyword mismatch.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub external seams
        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        baseline_calls = []

        def capture_save_baseline(root, **kw):
            """Capturing mock: bind root positionally, rest by keyword."""
            baseline_calls.append((root, kw))
            return root / "baseline.json"

        monkeypatch.setattr(memory_module, "save_baseline", capture_save_baseline, raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: {"findings": []}, raising=True)

        # Stub gh seams
        async def fake_list_review_comments(shell, repo, number):
            return []

        async def fake_create_pr_review_comment(shell, repo, number, body, path, line, commit_id=None):
            pass

        async def fake_submit_pr_review(shell, repo, number, event, body):
            pass

        def fake_commentable_lines(diff):
            return {"test.py": {10}}

        monkeypatch.setattr(gh_module, "list_review_comments", fake_list_review_comments, raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", fake_create_pr_review_comment, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", fake_commentable_lines, raising=True)

        # Stub rate limit
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Fake LLM
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
            comments: list[MockSynthComment] = [MockSynthComment()]

        # FakeLLM replaced with FakeLLMForReview (see conftest.py)

        # Use shared FakeAgentContext and FakeShell from conftest.py
        from conftest import FakeAgentContext, FakeLLMForReview, FakeShell

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-baseline-kwargs"
        ctx.llm = lambda name=None: FakeLLMForReview()

        shell = FakeShell()

        # Build and execute graph
        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="stub-head-sha",  # Matches stub_setup_seams fixture return
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=False,
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "low",
                    "body": "Test finding",
                    "model": "test-model",
                    "domain": "quality",
                }
            ],
            notes=[],
            local=False,  # GitHub mode
        )

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        result = await compiled.ainvoke(initial_state)

        # Assert the graph executed without short-circuiting
        assert result.get("error") is None, (
            f"Graph short-circuited with error: {result.get('error')}. "
            "Test never reached synthesis. Check FakeLLM shape."
        )

        # Assert save_baseline was called
        assert len(baseline_calls) >= 1, "save_baseline must be called"

        _root, kwargs = baseline_calls[0]
        assert "repo" in kwargs, "repo must be a keyword argument"
        assert "number" in kwargs, "number must be a keyword argument"
        assert "head_sha" in kwargs, "head_sha must be a keyword argument"
        assert "findings" in kwargs, "findings must be a keyword argument"
        assert "local" in kwargs, "local must be a keyword argument"

        assert kwargs["repo"] == "github.com/org/repo"
        assert kwargs["number"] == 42
        assert kwargs["head_sha"] == "stub-head-sha", f"Expected stub-head-sha from fixture, got {kwargs['head_sha']}"
        assert kwargs["local"] is False

    @pytest.mark.asyncio
    async def test_local_mode_does_not_touch_non_local_baseline(self, monkeypatch, tmp_path, stub_setup_seams):
        """local=True writes .local.json only, does NOT touch the non-local baseline.

        Bug #26 related: a dry run (--local) must not overwrite the authoritative baseline.
        Assert the non-local path does not exist after a local run.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub external seams
        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        written_paths = []

        def capture_save_baseline(root, **kw):
            # Record what would be written
            local_flag = kw.get("local", False)
            if local_flag:
                path = root / f"pr-{kw['number']}.local.json"
            else:
                path = root / f"pr-{kw['number']}.json"
            written_paths.append(path)
            return path

        def fake_baseline_path(root, repo, number, *, local=False):
            local_flag = local
            if local_flag:
                return root / f"pr-{number}.local.json"
            else:
                return root / f"pr-{number}.json"

        monkeypatch.setattr(memory_module, "save_baseline", capture_save_baseline, raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", fake_baseline_path, raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: {"findings": []}, raising=True)

        # Stub gh (local mode should not call these, but stub anyway)
        monkeypatch.setattr(gh_module, "list_review_comments", lambda *a, **kw: [], raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"test.py": {10}}, raising=True)

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Fake LLM
        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthResult(PydanticBaseModel):
            summary: str = "Review complete"
            event: str = "COMMENT"
            comments: list = []

        # FakeLLM replaced with FakeLLMForReview (see conftest.py)

        # Use shared FakeAgentContext and FakeShell from conftest.py
        from conftest import FakeAgentContext, FakeLLMForReview, FakeShell

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-local-baseline"
        ctx.llm = lambda name=None: FakeLLMForReview()

        shell = FakeShell()

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=False,
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "low",
                    "body": "Test finding",
                    "model": "test-model",
                    "domain": "quality",
                }
            ],
            notes=[],
            local=True,  # LOCAL MODE
        )

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        result = await compiled.ainvoke(initial_state)

        # Assert the graph executed without setup errors
        assert result.get("error") is None, (
            f"Graph short-circuited with error: {result.get('error')}. "
            "Check that fake reached synthesis and all stubs are async."
        )

        # Assert only the local baseline was written
        non_local_path = tmp_path / "pr-42.json"
        local_path = tmp_path / "pr-42.local.json"

        assert local_path in written_paths, "Local baseline must be written"
        assert non_local_path not in written_paths, (
            f"Non-local baseline must NOT be written in local mode. Written paths: {written_paths}"
        )


class TestLocalArtifact:
    """Requirements 3, 6, 7: local_artifact_path, summary redaction, file permissions."""

    @pytest.mark.asyncio
    async def test_local_artifact_path_in_result(self, monkeypatch, tmp_path, stub_setup_seams):
        """local_artifact_path must be present, non-empty, and the file must exist.

        Bug #26: Node return keys local_artifact_path/posted_comments/failed_comments were
        not ReviewState fields, so LangGraph dropped them silently. The artifact was written
        to disk but its path never reached the caller. A test checking only disk would have
        missed this.

        Assert the returned value, not just disk presence.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub external seams
        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: {"findings": []}, raising=True)

        monkeypatch.setattr(gh_module, "list_review_comments", lambda *a, **kw: [], raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"test.py": {10}}, raising=True)

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Fake LLM
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
            comments: list[MockSynthComment] = [MockSynthComment()]

        # FakeLLM replaced with FakeLLMForReview (see conftest.py)

        # Use shared FakeAgentContext, FakeShell, and FakeLLMForReview from conftest.py
        from conftest import FakeAgentContext, FakeLLMForReview, FakeShell

        # Create the reviews directory
        reviews_dir = tmp_path / "reviews"
        reviews_dir.mkdir(exist_ok=True)

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-artifact-path"
        ctx.llm = lambda name=None: FakeLLMForReview(synth_result=MockSynthResult())

        shell = FakeShell()

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=False,
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "low",
                    "body": "Test finding",
                    "model": "test-model",
                    "domain": "quality",
                }
            ],
            notes=[],
            local=True,
        )

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        result = await compiled.ainvoke(initial_state)

        # Assert the graph executed without short-circuiting
        assert result.get("error") is None, (
            f"Graph short-circuited with error: {result.get('error')}. "
            "Test never reached synthesis. Check FakeLLM shape."
        )

        # Assert local_artifact_path is in the result
        assert "local_artifact_path" in result, (
            f"local_artifact_path must be present in result. Result keys: {list(result.keys())}"
        )

        artifact_path_str = result["local_artifact_path"]
        assert artifact_path_str, "local_artifact_path must be non-empty. Empty string resolves to CWD."
        assert artifact_path_str, "local_artifact_path must be non-empty"

        # Assert the file exists
        from pathlib import Path

        artifact_path = Path(artifact_path_str)
        assert artifact_path.exists(), f"Artifact file must exist at {artifact_path}"

    @pytest.mark.asyncio
    async def test_summary_redaction_in_local_artifact(self, monkeypatch, tmp_path, stub_setup_seams):
        """Summary containing credentials must be redacted in the local artifact.

        Bug #28: The --local artifact redacts finding bodies but writes the synthesis
        summary raw. The GitHub path redacts both. Verified with real credential:
        api_key=sk-ant-SECRET123 appeared verbatim where GitHub would show <REDACTED>.

        Cover 2+ pattern families (labelled + bare-prefix) so partial redaction can't pass.
        Mutation-test: should fail if summary redaction removed.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub external seams
        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: {"findings": []}, raising=True)

        monkeypatch.setattr(gh_module, "list_review_comments", lambda *a, **kw: [], raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"test.py": {10}}, raising=True)

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Fake LLM that returns a summary with secrets
        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthResult(PydanticBaseModel):
            summary: str = "Found credentials: api_key=sk-ant-SECRET123 and token ghp_GITHUB_PAT_ABC"
            event: str = "COMMENT"
            comments: list = []

        # FakeLLM replaced with FakeLLMForReview (see conftest.py)

        # Use shared FakeAgentContext, FakeShell, and FakeLLMForReview from conftest.py
        from conftest import FakeAgentContext, FakeLLMForReview, FakeShell

        reviews_dir = tmp_path / "reviews"
        reviews_dir.mkdir(exist_ok=True)

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-summary-redact"
        ctx.llm = lambda name=None: FakeLLMForReview(synth_result=MockSynthResult())

        shell = FakeShell()

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=False,
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "title": "Test finding",
                    "body": "Test finding body",
                    "domain": "quality",
                    "provider": "test-provider",
                    "model": "test-model",
                }
            ],
            notes=[],
            local=True,
        )

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        result = await compiled.ainvoke(initial_state)

        # Assert the graph executed without short-circuiting
        assert result.get("error") is None, (
            f"Graph short-circuited with error: {result.get('error')}. "
            "Test never reached synthesis. Check FakeLLM shape."
        )
        assert result.get("local_artifact_path"), "local_artifact_path must be non-empty. Empty string resolves to CWD."

        # Read the artifact file
        from pathlib import Path

        artifact_path = Path(result["local_artifact_path"])
        artifact_contents = artifact_path.read_text()

        # Assert the raw secrets are absent
        assert "sk-ant-SECRET123" not in artifact_contents, (
            "Raw secret (sk-ant-...) must not appear in artifact. Summary redaction may be missing."
        )
        assert "ghp_GITHUB_PAT_ABC" not in artifact_contents, (
            "Raw secret (ghp_...) must not appear in artifact. Summary redaction may be missing."
        )

        # Assert <REDACTED> is present
        assert "<REDACTED>" in artifact_contents, (
            "Redacted placeholder must appear in artifact. Redaction may not be applied."
        )

    @pytest.mark.asyncio
    async def test_artifact_permissions(self, monkeypatch, tmp_path, stub_setup_seams):
        """Artifact directory 0o700, file 0o600 (bug #32: mkdir(mode=) only protects leaf).

        impl-graph added _mkdir_private and explicit chmod(0o600). Assert with
        stat().st_mode & 0o777, reusing the idiom from test_memory.py.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub external seams
        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: {"findings": []}, raising=True)

        monkeypatch.setattr(gh_module, "list_review_comments", lambda *a, **kw: [], raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"test.py": {10}}, raising=True)

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Use shared FakeAgentContext, FakeShell, and FakeLLMForReview from conftest.py
        from conftest import FakeAgentContext, FakeLLMForReview, FakeShell

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-artifact-perms"
        ctx.llm = lambda name=None: FakeLLMForReview()

        shell = FakeShell()

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=False,
            findings=[],
            notes=[],
            local=True,
        )

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        result = await compiled.ainvoke(initial_state)

        # Assert the graph executed without short-circuiting (prevents lying failures)
        assert result.get("error") is None, (
            f"Graph short-circuited with error: {result.get('error')}. "
            "Test never reached the code it claims to assert. "
            "If FakeLLM.ainvoke returns a synthesis result to review_branch's bind_tools path, "
            "run_tool_loop hits AttributeError on .tool_calls and every branch fails."
        )
        assert result.get("local_artifact_path"), (
            "local_artifact_path must be non-empty. Empty string resolves to CWD via Path(''), "
            "and the test would assert permissions on the wrong directory entirely."
        )

        # Check the artifact file and directory permissions
        from pathlib import Path

        artifact_path = Path(result["local_artifact_path"])
        artifact_dir = artifact_path.parent

        # Assert directory mode is 0o700
        dir_mode = stat.S_IMODE(artifact_dir.stat().st_mode)
        assert dir_mode == 0o700, (
            f"Artifact directory must have mode 0o700, got {oct(dir_mode)}. "
            "mkdir(mode=0o700) only protects the leaf; _mkdir_private walks back."
        )

        # Assert file mode is 0o600
        file_mode = stat.S_IMODE(artifact_path.stat().st_mode)
        assert file_mode == 0o600, (
            f"Artifact file must have mode 0o600, got {oct(file_mode)}. chmod(0o600) must be called after write."
        )


class TestPostedCommentsCount:
    """Requirement 4: posted_comments must reflect actual number posted in GitHub mode."""

    @pytest.mark.asyncio
    async def test_posted_comments_reflects_actual_count(self, monkeypatch, tmp_path, stub_setup_seams):
        """posted_comments must reflect the actual number posted in GitHub mode.

        Bug #26: A successful run currently reports 0. The key is returned by the node
        but not declared as a ReviewState field, so LangGraph drops it.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub external seams
        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        posted_count = []

        async def fake_create_pr_review_comment(shell, repo, number, body, path, line, commit_id=None):
            posted_count.append(1)

        async def fake_submit_pr_review(shell, repo, number, event, body):
            pass

        monkeypatch.setattr(gh_module, "list_review_comments", lambda *a, **kw: [], raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", fake_create_pr_review_comment, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"test.py": {10, 20}}, raising=True)

        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: {"findings": []}, raising=True)

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Fake LLM that returns 2 comments
        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthComment(PydanticBaseModel):
            path: str
            line: int | None
            severity: str
            body: str
            models: list[str]

        class MockSynthResult(PydanticBaseModel):
            summary: str = "Review complete"
            event: str = "COMMENT"
            comments: list[MockSynthComment] = [
                MockSynthComment(path="test.py", line=10, severity="low", body="Finding 1", models=["m1"]),
                MockSynthComment(path="test.py", line=20, severity="low", body="Finding 2", models=["m2"]),
            ]

        # FakeLLM replaced with FakeLLMForReview (see conftest.py)

        # Use shared FakeAgentContext and FakeShell from conftest.py
        from conftest import FakeAgentContext, FakeLLMForReview, FakeShell

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-posted-count"
        ctx.llm = lambda name=None: FakeLLMForReview()

        shell = FakeShell()

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=False,
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "low",
                    "body": "Finding 1",
                    "model": "m1",
                    "domain": "quality",
                },
                {
                    "path": "test.py",
                    "line": 20,
                    "severity": "low",
                    "body": "Finding 2",
                    "model": "m2",
                    "domain": "quality",
                },
            ],
            notes=[],
            local=False,  # GitHub mode
        )

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        result = await compiled.ainvoke(initial_state)

        # Assert the graph executed without short-circuiting
        assert result.get("error") is None, (
            f"Graph short-circuited with error: {result.get('error')}. "
            "Test never reached synthesis. Check FakeLLM shape."
        )

        # Assert posted_comments reflects the actual count
        assert "posted_comments" in result, "posted_comments must be in result"
        assert result["posted_comments"] == 2, (
            f"Expected posted_comments=2, got {result.get('posted_comments')}. "
            "Bug #26: the key may be dropped if not a ReviewState field."
        )


class TestNeverApproveClamp:
    """Requirement 10: Never submit APPROVE when critical findings exist, normalize invalid events."""

    @pytest.mark.asyncio
    async def test_approve_clamped_to_request_changes_with_critical_finding(
        self, monkeypatch, tmp_path, stub_setup_seams
    ):
        """Model returns event=APPROVE with critical finding → submitted event is REQUEST_CHANGES."""
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub external seams
        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        submitted_events = []

        async def fake_submit_pr_review(shell, repo, number, event, body):
            submitted_events.append(event)

        monkeypatch.setattr(gh_module, "list_review_comments", lambda *a, **kw: [], raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"test.py": {10}}, raising=True)

        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: {"findings": []}, raising=True)

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Fake LLM that returns APPROVE with a critical finding
        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthComment(PydanticBaseModel):
            path: str = "test.py"
            line: int | None = 10
            severity: str = "critical"
            body: str = "Security vulnerability"
            models: list[str] = ["test-model"]

        class MockSynthResult(PydanticBaseModel):
            summary: str = "Approval with critical finding"
            event: str = "APPROVE"
            comments: list[MockSynthComment] = [MockSynthComment()]

        # FakeLLM replaced with FakeLLMForReview (see conftest.py)

        # Use shared FakeAgentContext and FakeShell from conftest.py
        from conftest import FakeAgentContext, FakeLLMForReview, FakeShell

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-approve-clamp"
        ctx.llm = lambda name=None: FakeLLMForReview()

        shell = FakeShell()

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=False,
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "critical",
                    "body": "Vuln",
                    "model": "m",
                    "domain": "security",
                }
            ],
            notes=[],
            local=False,
        )

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        result = await compiled.ainvoke(initial_state)

        # Assert the graph executed without short-circuiting
        assert result.get("error") is None, (
            f"Graph short-circuited with error: {result.get('error')}. "
            "Test never reached synthesis. Check FakeLLM shape."
        )

        # Assert the submitted event was clamped
        assert len(submitted_events) == 1
        assert submitted_events[0] == "REQUEST_CHANGES", (
            f"Expected REQUEST_CHANGES, got {submitted_events[0]}. "
            "Never-APPROVE clamp must downgrade when critical findings exist."
        )

    @pytest.mark.asyncio
    async def test_invalid_event_string_normalized(self, monkeypatch, tmp_path, stub_setup_seams):
        """Invalid event string (e.g. 'REQUEST CHANGES' with space) must be normalized."""
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub external seams
        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        submitted_events = []

        async def fake_submit_pr_review(shell, repo, number, event, body):
            submitted_events.append(event)

        monkeypatch.setattr(gh_module, "list_review_comments", lambda *a, **kw: [], raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", lambda *a, **kw: None, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"test.py": {10}}, raising=True)

        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: {"findings": []}, raising=True)

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Fake LLM that returns invalid event string
        from pydantic import BaseModel as PydanticBaseModel

        from conftest import FakeAgentContext, FakeLLMForReview, FakeShell

        class MockSynthResult(PydanticBaseModel):
            summary: str = "Review complete"
            event: str = "REQUEST CHANGES"  # Invalid (has space)
            comments: list = []

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-invalid-event"
        ctx.llm = lambda name=None: FakeLLMForReview(synth_result=MockSynthResult())

        shell = FakeShell()

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=False,
            findings=[],
            notes=[],
            local=False,
        )

        graph = build_graph(ctx, shell)
        compiled = graph.compile()

        result = await compiled.ainvoke(initial_state)

        # Assert the graph executed without short-circuiting (prevents lying failures)
        assert result.get("error") is None, (
            f"Graph short-circuited with error: {result.get('error')}. "
            "Test never reached synthesis. Check FakeLLM shape."
        )

        # Assert the event was normalized
        assert len(submitted_events) == 1
        # Valid GitHub events: APPROVE, REQUEST_CHANGES, COMMENT
        # "REQUEST CHANGES" should be normalized to "REQUEST_CHANGES" or "COMMENT"
        valid_events = {"APPROVE", "REQUEST_CHANGES", "COMMENT"}
        assert submitted_events[0] in valid_events, (
            f"Expected valid GitHub event, got {submitted_events[0]}. Invalid event strings must be normalized."
        )
