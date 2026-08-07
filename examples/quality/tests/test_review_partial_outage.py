"""Adversarial tests for partial reviewer-outage disclosure and dedup pagination.

Two gaps that a green suite tolerates because both produce a *plausible-looking*
successful review:

1. **Partial fan-out failure is never disclosed to the PR.** ``synthesize_and_post``
   only errors when *every* branch failed (review.py:745-748). If the whole security
   domain crashed and quality succeeded, the run posts a normal review with no
   deterministic marker that half the matrix never executed. The failed notes are
   handed to the synthesis *model* as prose (review.py:799) — an untrusted summarizer
   is the only thing standing between "security review crashed" and a reader who
   believes the PR was security-reviewed.

2. **Dedup reads only the first page of review comments.** ``list_review_comments``
   (gh.py:492-496) calls ``gh api .../comments`` with no ``--paginate``, so GitHub
   returns at most one page. On a PR that already carries more comments than a page,
   ``_existing_comment_lines`` misses the older ones and the follow-up review
   re-posts findings it already raised — the idempotency guarantee in README.md:86-94.

All imports are function-scoped to match the suite convention (see conftest.py).
"""

from __future__ import annotations

import pytest


class TestPartialOutageDisclosure:
    """A review that only half-ran must say so, deterministically."""

    @pytest.mark.asyncio
    async def test_failed_branch_disclosed_in_posted_review_body(self, monkeypatch, tmp_path, stub_setup_seams):
        """When the security branch fails but quality succeeds, the posted review must say so.

        review.py:745-748 surfaces an error only when ``all()`` notes are failures AND
        there are no findings. A partial outage falls through to the normal path and
        posts a review whose body is whatever the synthesis model returned.

        Here the synthesis model returns a clean, confident summary that does not
        mention the outage (exactly what a model does when it decides the failure note
        isn't worth summarizing). Nothing in production appends the outage
        deterministically, so the PR author reads a review that looks complete while
        the entire security domain never executed.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        submitted: list[tuple[str, str]] = []

        async def fake_submit_pr_review(shell, repo, number, event, body):
            submitted.append((event, body))

        async def fake_create_pr_review_comment(shell, repo, number, body, path, line, commit_id=None):
            pass

        async def fake_list_review_comments(shell, repo, number):
            return []

        monkeypatch.setattr(gh_module, "list_review_comments", fake_list_review_comments, raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", fake_create_pr_review_comment, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"test.py": {10}}, raising=True)

        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: None, raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthComment(PydanticBaseModel):
            path: str = "test.py"
            line: int | None = 10
            severity: str = "low"
            body: str = "Minor naming nit"
            models: list[str] = ["model-a"]

        class MockSynthResult(PydanticBaseModel):
            # A confident, clean summary that omits the outage entirely.
            summary: str = "The change is small and well tested. One minor naming nit."
            event: str = "COMMENT"
            comments: list[MockSynthComment] = [MockSynthComment()]

        class FakeLLM:
            def with_structured_output(self, schema):
                return self

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                return MockSynthResult()

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-partial-outage"
        ctx.llm = lambda name=None: FakeLLM()

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("alpha", "model-a")],
            local=False,
            # Quality succeeded; the whole security domain crashed.
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "low",
                    "body": "Minor naming nit",
                    "model": "model-a",
                    "domain": "quality",
                }
            ],
            notes=[
                {
                    "domain": "security",
                    "provider": "alpha",
                    "model": "model-a",
                    "failed": True,
                    "body": "_(reviewer branch security/alpha failed: RuntimeError)_",
                }
            ],
        )

        compiled = build_graph(ctx, FakeShell()).compile()
        await compiled.ainvoke(initial_state)

        assert len(submitted) == 1, f"expected one review submission, got {submitted}"
        _event, body = submitted[0]

        lowered = body.lower()
        assert "security" in lowered and ("failed" in lowered or "incomplete" in lowered or "did not" in lowered), (
            "The posted review must deterministically disclose that a reviewer branch "
            "failed, so nobody reads a half-run review as a complete one. "
            f"Posted body was:\n{body}\n\n"
            "review.py errors only when EVERY branch fails; a partial outage is passed "
            "to the synthesis model as prose and vanishes if the model omits it."
        )

    @pytest.mark.asyncio
    async def test_partial_outage_surfaced_in_local_artifact(self, monkeypatch, tmp_path, stub_setup_seams):
        """The --local artifact must disclose the outage too.

        Same defect, second published surface. The artifact is built from ``summary``
        and ``comments`` only (review.py:899-916) — ``notes`` are never rendered, so a
        failed branch leaves no trace in the file the user reads.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"test.py": {10}}, raising=True)
        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: None, raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthComment(PydanticBaseModel):
            path: str = "test.py"
            line: int | None = 10
            severity: str = "low"
            body: str = "Minor naming nit"
            models: list[str] = ["model-a"]

        class MockSynthResult(PydanticBaseModel):
            summary: str = "Looks good overall."
            event: str = "COMMENT"
            comments: list[MockSynthComment] = [MockSynthComment()]

        class FakeLLM:
            def with_structured_output(self, schema):
                return self

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                return MockSynthResult()

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-partial-outage-local"
        ctx.llm = lambda name=None: FakeLLM()

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("alpha", "model-a")],
            local=True,
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "low",
                    "body": "Minor naming nit",
                    "model": "model-a",
                    "domain": "quality",
                }
            ],
            notes=[
                {
                    "domain": "security",
                    "provider": "alpha",
                    "model": "model-a",
                    "failed": True,
                    "body": "_(reviewer branch security/alpha failed: RuntimeError)_",
                }
            ],
        )

        compiled = build_graph(ctx, FakeShell()).compile()
        result = await compiled.ainvoke(initial_state)

        artifact = result.get("local_artifact_path", "")
        assert artifact, "local mode must produce an artifact path"

        from pathlib import Path

        text = Path(artifact).read_text()
        lowered = text.lower()
        assert "security" in lowered and ("failed" in lowered or "incomplete" in lowered), (
            "The --local artifact must disclose that the security branch failed. "
            f"Artifact contents:\n{text}\n\n"
            "review.py:899-916 renders only summary and comments; notes are dropped."
        )


class TestDedupPagination:
    """Comment dedup must consider every existing comment, not just the first page."""

    @pytest.mark.asyncio
    async def test_list_review_comments_requests_all_pages(self, tmp_path):
        """``gh api`` without --paginate returns only the first page.

        gh.py:492-496 issues a bare ``gh api repos/<r>/pulls/<n>/comments``. GitHub
        caps a page at 30 items by default, and gh does not follow Link headers unless
        asked. So on any PR that already carries more than a page of review comments,
        ``_existing_comment_lines`` (review.py:682) sees a truncated set, the
        ``(path, line) in already`` check at review.py:992 misses, and the follow-up
        review re-posts comments it made last run.

        That breaks the idempotency the README promises (README.md:86-94) and is
        invisible to test_gh.py::test_list_review_comments_returns_list, which stubs a
        two-element single page.
        """
        from quality.gh import list_review_comments

        class RecordingShell:
            """Mirrors switchplane.shell.Shell.run's signature (shell.py:142-160)."""

            def __init__(self):
                self.commands: list[list[str]] = []

            async def run(self, cmd, cwd=None, env=None, timeout=None):
                self.commands.append(list(cmd))
                return "[]"

            async def run_ok(self, cmd, cwd=None, env=None, timeout=None):
                self.commands.append(list(cmd))
                return True

        shell = RecordingShell()
        await list_review_comments(shell, "github.com/org/repo", 42)

        assert shell.commands, "list_review_comments must invoke gh"
        cmd = shell.commands[0]
        assert "--paginate" in cmd, (
            f"list_review_comments must request every page; got {cmd}. "
            "Without --paginate the dedup set is truncated at one page and the "
            "follow-up review duplicates comments it already posted."
        )
