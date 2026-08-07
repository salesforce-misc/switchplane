"""Multi-domain attribution and determinism tests (requirement 5, bug #27).

Bug #27: Baseline persists only one domain per location. A finding flagged by BOTH
`quality` and `security` at the same path:line must be seen as prior by BOTH on
follow-up run. Today persistence keeps one domain arbitrarily (next(iter(set))) and
_format_prior matches on `domain ==`, so the losing domain re-reports it as new.

Also pin determinism: two runs with identical input must persist an identical domain list.

Do NOT pin current storage shape — impl-graph needs freedom to fix #27 (adding `domains`
list with `domain` back-compat fallback). Assert on behavior: follow-up dedup works for
all domains that contributed.

All imports are function-scoped.
"""

from __future__ import annotations

import pytest


class TestMultiDomainAttribution:
    """Requirement 5: Multi-domain attribution and determinism."""

    @pytest.mark.asyncio
    async def test_finding_seen_as_prior_by_all_domains(self, monkeypatch, tmp_path, stub_setup_seams):
        """A finding flagged by BOTH quality and security must be seen as prior by BOTH.

        Bug #27: Baseline persists only one domain arbitrarily. On follow-up, _format_prior
        matches on `domain ==`, so the losing domain re-reports it as new.

        This test runs the graph twice:
        1. Initial run with findings from both domains at the same location
        2. Follow-up run with the same findings

        Assert that both domains see the finding as prior (not re-reported).
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub external seams
        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        # Storage for baseline persistence
        persisted_baseline = None

        def fake_save_baseline(root, **kwargs):
            nonlocal persisted_baseline
            persisted_baseline = kwargs
            return root / "baseline.json"

        def fake_baseline_path(root, repo, number, *, local=False):
            return root / "baseline.json"

        def fake_load_baseline(path):
            # First run: no baseline
            # Follow-up run: return the persisted baseline
            if persisted_baseline is None:
                return {"findings": []}
            return {"findings": persisted_baseline.get("findings", [])}

        monkeypatch.setattr(memory_module, "save_baseline", fake_save_baseline, raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", fake_baseline_path, raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", fake_load_baseline, raising=True)

        # Stub gh seams with realistic round-trip
        # GitHub API returns comments with nested user dict: {"login": "..."}
        posted_comments_run1 = []
        posted_comments_run2 = []
        current_run_comments = posted_comments_run1
        all_posted_comments = []  # Accumulates across runs for list_review_comments

        async def fake_list_review_comments(shell, repo, number):
            # Return GitHub API shape: nested user dict with login
            return [
                {
                    "path": c["path"],
                    "line": c["line"],
                    "body": c["body"],
                    "user": {"login": "authed-user"},  # Matches get_authenticated_user stub
                }
                for c in all_posted_comments
            ]

        async def fake_create_pr_review_comment(shell, repo, number, body, path, line, commit_id=None):
            comment = {"path": path, "line": line, "body": body}
            current_run_comments.append(comment)
            all_posted_comments.append(comment)

        async def fake_submit_pr_review(shell, repo, number, event, body):
            pass

        def fake_commentable_lines(diff):
            return {"test.py": {10}}

        monkeypatch.setattr(gh_module, "list_review_comments", fake_list_review_comments, raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", fake_create_pr_review_comment, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", fake_commentable_lines, raising=True)

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Fake LLM that returns the same finding
        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthComment(PydanticBaseModel):
            path: str = "test.py"
            line: int | None = 10
            severity: str = "medium"
            body: str = "Repeated finding"
            models: list[str] = ["test-model"]

        class MockSynthResult(PydanticBaseModel):
            summary: str = "Review complete"
            event: str = "COMMENT"
            comments: list[MockSynthComment] = [MockSynthComment()]

        # Use shared fakes from conftest.py
        from conftest import FakeAgentContext, FakeLLMForReview, FakeShell

        class FakeContext(FakeAgentContext):
            """Subclass shared fake to add custom LLM behavior for this test."""

            def __init__(self):
                super().__init__(runtime_dir_path=tmp_path)
                self.task_id = "test-multi-domain"

            def llm(self, name=None):
                return FakeLLMForReview(synth_result=MockSynthResult())

        shell = FakeShell()

        # Run 1: Initial review with findings from both domains at the same location
        initial_state_run1 = ReviewState(
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
                    "body": "Quality finding",
                    "model": "test-model",
                    "domain": "quality",
                },
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Security finding",
                    "model": "test-model",
                    "domain": "security",
                },
            ],
            notes=[],
            local=False,
        )

        graph = build_graph(FakeContext(), shell)
        compiled = graph.compile()

        result_run1 = await compiled.ainvoke(initial_state_run1)

        # Assert the graph executed without setup errors
        assert result_run1.get("error") is None, (
            f"Graph short-circuited with error: {result_run1.get('error')}. "
            "Check that FakeContext has providers property and FakeShell has fs_tools()."
        )

        # Assert baseline was persisted
        assert persisted_baseline is not None, "Baseline must be persisted after run 1"
        assert len(persisted_baseline["findings"]) > 0, "Baseline must contain findings"

        # Check that the persisted finding has both domains
        persisted_finding = persisted_baseline["findings"][0]
        if "domains" in persisted_finding:
            # New format: domains list
            assert "quality" in persisted_finding["domains"], "quality domain must be in domains list"
            assert "security" in persisted_finding["domains"], "security domain must be in domains list"
        elif "domain" in persisted_finding:
            # Old format: single domain (bug #27 — this is what we're testing against)
            # For now, just verify the field exists
            pass

        # Run 2: Follow-up review with the SAME findings at the same location
        current_run_comments = posted_comments_run2

        initial_state_run2 = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/test.py b/test.py\n@@ -1 +1 @@\n+test",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("test-provider", "test-model")],
            error=None,
            is_followup=True,
            findings=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Quality finding",
                    "model": "test-model",
                    "domain": "quality",
                },
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Security finding",
                    "model": "test-model",
                    "domain": "security",
                },
            ],
            notes=[],
            local=False,
        )

        await compiled.ainvoke(initial_state_run2)

        # Assert that NO new comments were posted on run 2
        # Both domains should see the finding as prior
        assert len(posted_comments_run2) == 0, (
            f"Expected 0 comments posted on follow-up (both domains see as prior), "
            f"got {len(posted_comments_run2)}. "
            "Bug #27: one domain may not see the finding as prior because baseline "
            "persisted only one domain arbitrarily."
        )

    @pytest.mark.asyncio
    async def test_domain_list_determinism(self, monkeypatch, tmp_path, stub_setup_seams):
        """Two runs with identical input must persist an identical domain list.

        Bug #27 note: The team-lead's two runs produced `quality` then `security`.
        This is non-deterministic set iteration.

        Assert that two runs with the same findings produce the same persisted domain list.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        # Stub external seams
        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        # Storage for baseline persistence
        persisted_baselines = []

        def fake_save_baseline(root, **kwargs):
            persisted_baselines.append(kwargs)
            return root / "baseline.json"

        monkeypatch.setattr(memory_module, "save_baseline", fake_save_baseline, raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: {"findings": []}, raising=True)

        # Stub gh seams (must be async to match production)
        async def fake_list_review_comments(*a, **kw):
            return []

        async def fake_create_pr_review_comment(*a, **kw):
            pass

        async def fake_submit_pr_review(*a, **kw):
            pass

        monkeypatch.setattr(gh_module, "list_review_comments", fake_list_review_comments, raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", fake_create_pr_review_comment, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)
        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"test.py": {10}}, raising=True)

        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Fake LLM
        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthComment(PydanticBaseModel):
            path: str = "test.py"
            line: int | None = 10
            severity: str = "medium"
            body: str = "Finding"
            models: list[str] = ["test-model"]

        class MockSynthResult(PydanticBaseModel):
            summary: str = "Review complete"
            event: str = "COMMENT"
            comments: list[MockSynthComment] = [MockSynthComment()]

        # Use shared fakes from conftest.py
        from conftest import FakeAgentContext, FakeLLMForReview, FakeShell

        class FakeContext(FakeAgentContext):
            """Subclass shared fake to add custom LLM behavior for this test."""

            def __init__(self):
                super().__init__(runtime_dir_path=tmp_path)
                self.task_id = "test-determinism"

            def llm(self, name=None):
                return FakeLLMForReview(synth_result=MockSynthResult())

        shell = FakeShell()

        # Run the graph twice with IDENTICAL input
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
                    "body": "Quality finding",
                    "model": "test-model",
                    "domain": "quality",
                },
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Security finding",
                    "model": "test-model",
                    "domain": "security",
                },
            ],
            notes=[],
            local=False,
        )

        graph = build_graph(FakeContext(), shell)
        compiled = graph.compile()

        # Run 1
        result_run1 = await compiled.ainvoke(initial_state)

        # Assert the graph executed without setup errors
        assert result_run1.get("error") is None, (
            f"Graph short-circuited with error: {result_run1.get('error')}. "
            "Check that FakeContext has providers property and FakeShell has fs_tools()."
        )

        # Run 2 with identical state
        result_run2 = await compiled.ainvoke(initial_state)

        # Assert the second run also executed without errors
        assert result_run2.get("error") is None, (
            f"Graph short-circuited on run 2 with error: {result_run2.get('error')}"
        )

        # Assert both runs persisted baselines
        assert len(persisted_baselines) == 2, "Two runs must persist two baselines"

        # Extract domain lists from both runs
        findings_run1 = persisted_baselines[0].get("findings", [])
        findings_run2 = persisted_baselines[1].get("findings", [])

        assert len(findings_run1) > 0, "Run 1 must persist findings"
        assert len(findings_run2) > 0, "Run 2 must persist findings"

        # Compare domain lists for the first finding
        finding_run1 = findings_run1[0]
        finding_run2 = findings_run2[0]

        if "domains" in finding_run1 and "domains" in finding_run2:
            # New format: compare domains lists
            domains_run1 = finding_run1["domains"]
            domains_run2 = finding_run2["domains"]

            assert domains_run1 == domains_run2, (
                f"Domain lists must be identical across runs. "
                f"Run 1: {domains_run1}, Run 2: {domains_run2}. "
                "Bug #27: non-deterministic set iteration can produce different orders."
            )
        elif "domain" in finding_run1 and "domain" in finding_run2:
            # Old format: compare single domain (will be non-deterministic with bug #27)
            domain_run1 = finding_run1["domain"]
            domain_run2 = finding_run2["domain"]

            # We can't assert equality here if the bug is present, but we can flag it
            if domain_run1 != domain_run2:
                pytest.fail(
                    f"Domain is non-deterministic: Run 1: {domain_run1}, Run 2: {domain_run2}. "
                    "Bug #27: set iteration picks an arbitrary domain."
                )
