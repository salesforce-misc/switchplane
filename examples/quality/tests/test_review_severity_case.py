"""Adversarial tests for severity normalization in the review gate.

``record_finding`` normalizes severity to lowercase (review.py:339-341), so the
existing tests all feed the gate lowercase strings and pass. But the severities that
reach ``_synth_event`` / ``comment_sort_key`` come from the **synthesis model**, not
from ``record_finding``: ``synthesize_and_post`` reads ``c.get("severity")`` straight
off the synthesis result (review.py:865, 906, 979-982, 1071) with no normalization.

``SynthComment.severity`` is a bare ``str`` with a free-text description, so an
uppercase or padded value is exactly what a model returns some fraction of the time.
Every downstream consumer compares it against lowercase-only tables:

    _BLOCKING_SEVERITIES = {"high", "critical"}        # review.py:446
    _SEVERITY_ORDER = {"info": 0, ... "critical": 4}   # review.py:445

so "CRITICAL" is neither blocking nor high-ranked. That silently relaxes the merge
gate to COMMENT — the exact rubber-stamp outcome the untrusted-model-output rule
exists to prevent (memory: llm-review-gate-untrusted-model-output).

All imports are function-scoped to match the suite convention (see conftest.py).
"""

from __future__ import annotations

import pytest


class TestSeverityNormalizationInGate:
    """Blocking-severity detection must not depend on the model's casing."""

    def test_uppercase_critical_still_blocks(self):
        """severity="CRITICAL" must yield REQUEST_CHANGES, not COMMENT.

        _synth_event (review.py:461) tests ``c.get("severity") in _BLOCKING_SEVERITIES``
        against the lowercase-only set at review.py:446. "CRITICAL" is not in it, so a
        critical finding is treated as non-blocking and the gate relaxes to COMMENT.

        Existing coverage feeds only lowercase, because it stubs findings by hand or
        routes them through record_finding's lowercasing. Synthesis output is not
        lowercased anywhere.
        """
        from quality.agents.pr.tasks.review import _synth_event

        comments = [{"path": "auth.py", "line": 10, "severity": "CRITICAL", "body": "RCE"}]

        assert _synth_event(comments) == "REQUEST_CHANGES", (
            "Uppercase 'CRITICAL' must be recognized as blocking. "
            f"Got {_synth_event(comments)} — _BLOCKING_SEVERITIES is lowercase-only, "
            "so the gate silently relaxes on a model that shouts."
        )

    def test_padded_and_mixed_case_high_still_blocks(self):
        """Whitespace-padded and mixed-case blocking severities must still block.

        _resolve_event already normalizes the model's *event* with
        ``.strip().upper().replace(" ", "_")`` (review.py:488). The model's *severity*
        gets no equivalent treatment, so the same class of formatting noise that the
        event path defends against sails through the severity path.
        """
        from quality.agents.pr.tasks.review import _synth_event

        for sev in (" high ", "High", "HIGH", "Critical"):
            comments = [{"path": "a.py", "line": 1, "severity": sev, "body": "x"}]
            assert _synth_event(comments) == "REQUEST_CHANGES", (
                f"severity={sev!r} must be recognized as blocking, got {_synth_event(comments)}"
            )

    def test_uppercase_critical_cannot_relax_the_resolved_event(self):
        """End-to-end through the gate: a shouted critical must not earn a COMMENT.

        This is the security-relevant composition. The model returns event="COMMENT"
        (a plausible, valid, non-escalating value) alongside a finding it labelled
        "CRITICAL". Because the severity table misses the casing, ``_synth_event``
        returns COMMENT, ``max()`` has nothing stricter to pick, and a critical
        finding ships as a non-blocking comment.
        """
        from quality.agents.pr.tasks.review import _effective_event, _resolve_event

        comments = [{"path": "auth.py", "line": 10, "severity": "CRITICAL", "body": "RCE"}]

        resolved = _effective_event(_resolve_event("COMMENT", comments), False)

        assert resolved == "REQUEST_CHANGES", (
            f"A critical finding must block the merge gate regardless of casing; got {resolved}. "
            "The model controls both the event string and the severity string, so "
            "case-sensitivity here is a one-token gate bypass."
        )


class TestSeverityOrderNormalization:
    """Severity ranking (sort order, merge precedence) must also be case-insensitive."""

    @pytest.mark.asyncio
    async def test_uppercase_critical_is_posted_before_lowercase_high(self, monkeypatch, tmp_path, stub_setup_seams):
        """Comments are posted most-severe-first; casing must not scramble that order.

        ``comment_sort_key`` ranks via ``_SEVERITY_ORDER.get(sev, 2)``, defaulting
        unknown keys to 2 ("medium"). If the severity is not normalized first,
        "CRITICAL" scores the medium default and ranks *below* a lowercase "high" (3),
        so the most serious finding is no longer posted first. Ordering matters because
        GitHub truncates long reviews and the sort is the only prioritization.

        Asserted on the observable post order through the real graph rather than on a
        ``_SEVERITY_ORDER`` lookup: the table is an implementation detail, and
        normalizing at the synthesis boundary (where the untrusted string enters) is a
        valid fix that a raw-lookup assertion would wrongly reject.
        """
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr.tasks.review import ReviewState, build_graph

        from quality import gh as gh_module
        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "baseline_path", lambda *a, **kw: tmp_path / "b.json", raising=True)
        monkeypatch.setattr(memory_module, "load_baseline", lambda p: None, raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)

        # Both lines are commentable, so ordering is decided purely by severity rank.
        monkeypatch.setattr(gh_module, "commentable_lines", lambda d: {"a.py": {10, 20}}, raising=True)

        posted: list[tuple[int, str]] = []

        async def fake_create_pr_review_comment(shell, repo, number, body, path, line, commit_id=None):
            posted.append((line, body))

        async def fake_submit_pr_review(*a, **kw):
            pass

        monkeypatch.setattr(gh_module, "list_review_comments", lambda *a, **kw: [], raising=True)
        monkeypatch.setattr(gh_module, "create_pr_review_comment", fake_create_pr_review_comment, raising=True)
        monkeypatch.setattr(gh_module, "submit_pr_review", fake_submit_pr_review, raising=True)

        from pydantic import BaseModel as PydanticBaseModel

        class MockSynthComment(PydanticBaseModel):
            path: str
            line: int
            severity: str
            body: str
            models: list[str] = ["model-a"]

        class MockSynthResult(PydanticBaseModel):
            summary: str = "Two findings."
            event: str = "COMMENT"
            # The shouted critical is listed SECOND, so a correct sort must reorder it.
            comments: list[MockSynthComment] = [
                MockSynthComment(path="a.py", line=10, severity="high", body="the high one"),
                MockSynthComment(path="a.py", line=20, severity="CRITICAL", body="the critical one"),
            ]

        class FakeLLM:
            def with_structured_output(self, schema):
                return self

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                return MockSynthResult()

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(runtime_dir_path=tmp_path)
        ctx.task_id = "test-severity-sort"
        ctx.llm = lambda name=None: FakeLLM()

        initial_state = ReviewState(
            repo="github.com/org/repo",
            number=42,
            diff="diff --git a/a.py b/a.py\n@@ -1 +1 @@\n+x",
            worktree_path=str(tmp_path / "worktree"),
            head_sha="fake-sha",
            matrix=[("alpha", "model-a")],
            local=False,
            findings=[
                {
                    "path": "a.py",
                    "line": 10,
                    "severity": "high",
                    "body": "the high one",
                    "model": "model-a",
                    "domain": "quality",
                },
                {
                    "path": "a.py",
                    "line": 20,
                    "severity": "CRITICAL",
                    "body": "the critical one",
                    "model": "model-a",
                    "domain": "security",
                },
            ],
            notes=[],
        )

        result = await build_graph(ctx, FakeShell()).compile().ainvoke(initial_state)

        # Path-reached guard: without this, a short-circuit to the outage/error branch
        # posts nothing and the ordering assertion below would blame the sort.
        assert result.get("error") is None, f"graph short-circuited instead of posting: {result.get('error')!r}"
        assert len(posted) == 2, f"both findings must be posted, got {posted}"

        assert posted[0][0] == 20, (
            "The 'CRITICAL' finding (line 20) must be posted before the lowercase 'high' "
            f"one (line 10); got post order {[line for line, _ in posted]}. "
            "Severity is not normalized before the _SEVERITY_ORDER lookup, so 'CRITICAL' "
            "collapses to the medium default and sorts below 'high'."
        )

    def test_merge_keeps_highest_severity_across_casing(self):
        """_comments_from_findings must keep the more severe label when merging.

        review.py:559 compares ``_SEVERITY_ORDER.get(sev, 2) > ...get(existing, 2)``.
        "CRITICAL" scores the medium default (2), which does not beat an incumbent
        "high" (3), so merging drops the critical label. The retained severity is
        what both the merge gate and the persisted baseline read, so a security
        branch's critical finding is downgraded by a quality branch's high one.

        This is the ``_comments_from_findings`` fallback path, which runs whenever
        synthesis returns no structured comments (review.py:877-879) — i.e. exactly
        when raw model severities reach the gate unmediated.
        """
        from quality.agents.pr.tasks.review import _comments_from_findings

        findings = [
            {"path": "a.py", "line": 5, "severity": "high", "body": "slow", "model": "m1", "domain": "quality"},
            {"path": "a.py", "line": 5, "severity": "CRITICAL", "body": "RCE", "model": "m2", "domain": "security"},
        ]

        merged = _comments_from_findings(findings)

        assert len(merged) == 1, f"same path+line must merge, got {merged}"
        assert merged[0]["severity"].lower() == "critical", (
            f"Merge must retain the highest severity; got {merged[0]['severity']!r}. "
            "The critical finding was ranked as 'medium' by the case-sensitive table "
            "and lost to the 'high' incumbent."
        )
