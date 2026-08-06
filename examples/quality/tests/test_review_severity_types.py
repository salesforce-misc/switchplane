"""Regression guards for the type discipline protecting the severity boundary.

Severity is normalized in two different styles, and only one of them is type-safe:

- The *lookup* sites coerce: ``str(c.get("severity", "medium")).strip().lower()``
  (``_synth_event`` review.py:513, ``_comments_from_findings`` review.py:599,
  ``comment_sort_key`` review.py:1082).
- The *boundary* does not: review.py:970 and the fallback at :981 both call
  ``c["severity"].strip().lower()`` on the raw value, which is an ``AttributeError``
  on any non-str.

That boundary runs inside ``synthesize_and_post`` *after* the synthesis retry loop has
already broken out (review.py:942-955), so an exception there is not retried — it would
escape the fan-in node and destroy the whole review after every reviewer branch has
already been paid for.

The uncoerced boundary is safe today only because both of its feeds are typed:

1. Structured synthesis output is validated by ``SynthComment.severity: str``, so
   pydantic rejects a non-string before ``model_dump()`` ever produces one.
2. The ``_comments_from_findings`` fallback coerces with ``str(...)`` at review.py:599,
   and its input comes from ``record_finding``, whose ``@tool`` schema also declares
   ``severity: str``.

These tests pin those two invariants. They are guards, not bug reports: if someone
relaxes either annotation, loosens the model to non-strict, or adds a third feed into
the boundary, review.py:970 becomes a live crash and these fail. The crash itself is
currently unreachable through production — reproducing it requires a fake that bypasses
pydantic, which would prove nothing about the real code — so no failing test is filed.

All imports are function-scoped to match the suite convention (see conftest.py).
"""

from __future__ import annotations

import pytest


class _StopSynthesis(Exception):
    """Aborts synthesis once the schema has been captured."""


def _capture_synth_schema():
    """Return the real ``SynthResult`` schema built inside ``synthesize_and_post``.

    The model is defined in the function body, so the only honest way to assert on it is
    to let the node construct it and intercept the ``with_structured_output`` argument.
    Rebuilding it here by hand would test the copy, not production.
    """
    import asyncio
    import tempfile
    from pathlib import Path

    import pytest as _pytest

    captured = {}

    class SchemaCapturingLLM:
        def with_structured_output(self, schema):
            captured["schema"] = schema
            raise _StopSynthesis()

        async def ainvoke(self, messages):  # pragma: no cover - never reached
            raise AssertionError("ainvoke must not be reached")

    async def _run():
        from quality.agents.pr.tasks.review import synthesize_and_post

        from conftest import FakeAgentContext, FakeShell
        from quality import ratelimit as ratelimit_module

        mp = _pytest.MonkeyPatch()
        try:
            mp.setattr(ratelimit_module, "with_rate_limit_retry", lambda x: x, raising=True)
            ctx = FakeAgentContext(runtime_dir_path=Path(tempfile.mkdtemp()))
            ctx.llm = lambda name=None: SchemaCapturingLLM()
            state = {
                "repo": "github.com/org/repo",
                "number": 1,
                "diff": "diff",
                "head_sha": "sha1",
                "local": True,
                "matrix": [("alpha", "model-a")],
                "findings": [
                    {
                        "path": "a.py",
                        "line": 10,
                        "severity": "high",
                        "body": "an issue",
                        "model": "model-a",
                        "domain": "quality",
                    }
                ],
                "notes": [],
            }
            try:
                await synthesize_and_post(ctx, FakeShell(), state)
            except _StopSynthesis:
                pass
        finally:
            mp.undo()

    asyncio.run(_run())

    assert "schema" in captured, (
        "synthesize_and_post did not call with_structured_output — the schema-capture "
        "helper is out of date with production and must be fixed before trusting it"
    )
    return captured["schema"]


class TestSynthCommentRejectsNonStringSeverity:
    """Pydantic validation is what keeps a non-str away from review.py:970."""

    @pytest.mark.parametrize("bad_severity", [5, 1.5, None, ["high"], {"level": "high"}])
    def test_synth_comment_rejects_non_string_severity(self, bad_severity):
        """``SynthComment`` must refuse a non-string severity.

        The schema is captured from the real node rather than restated here, so this
        guard cannot drift from the model production actually hands the provider.
        """
        import pydantic

        schema = _capture_synth_schema()

        # The nested comment model is what guards the boundary.
        comment_model = schema.model_fields["comments"].annotation.__args__[0]

        with pytest.raises(pydantic.ValidationError) as exc:
            comment_model(path="a.py", line=10, severity=bad_severity, body="an issue")

        assert "severity" in str(exc.value), (
            f"validation must reject severity={bad_severity!r} on the severity field, got: {exc.value}"
        )

    def test_synth_comment_still_accepts_a_normal_severity(self):
        """Negative control: the guard above must not pass because everything fails.

        Without this, an over-tightened model that rejected valid severities too would
        satisfy the parametrized test while breaking every real review.
        """
        schema = _capture_synth_schema()
        comment_model = schema.model_fields["comments"].annotation.__args__[0]

        c = comment_model(path="a.py", line=10, severity="HIGH ", body="an issue")
        assert c.severity == "HIGH ", (
            f"the model must pass severity through unmodified — review.py:970 owns normalization, got {c.severity!r}"
        )


class TestRecordFindingSeverityIsTyped:
    """The other feed into the boundary: record_finding -> _comments_from_findings."""

    @pytest.mark.asyncio
    async def test_non_string_severity_does_not_fail_the_branch(self, monkeypatch, tmp_path):
        """A model sending a non-string severity must not kill the reviewer branch.

        ``record_finding``'s ``@tool`` schema declares ``severity: str`` (review.py:340),
        so a non-string is rejected during dispatch. What matters is the *handling*:
        ``run_tool_loop`` must feed the validation error back as a tool message and let
        the branch continue, not let it escape into the branch's failure path.

        Driven through the real ``review_branch`` so the tool is resolved from
        ``tool_map`` by name exactly as the loop does it.
        """
        from langchain_core.messages import AIMessage
        from quality.agents.pr import memory as memory_module
        from quality.agents.pr import prompts as prompts_module
        from quality.agents.pr.tasks.review import review_branch

        from quality import ratelimit as ratelimit_module

        monkeypatch.setattr(memory_module, "load_baseline", lambda path: None, raising=True)
        monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt", raising=True)
        monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda r: r, raising=True)

        class IntSeverityLLM:
            """Sends severity=5, then a well-formed finding, then stops."""

            def __init__(self):
                self.turn = 0

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                self.turn += 1
                if self.turn == 1:
                    args = {"path": "a.py", "line": 10, "severity": 5, "body": "an issue"}
                elif self.turn == 2:
                    args = {"path": "b.py", "line": 20, "severity": "high", "body": "real issue"}
                else:
                    return AIMessage(content="done")
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "record_finding", "args": args, "id": f"call_{self.turn}"}],
                )

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(
            config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "model-a"}}}},
            runtime_dir_path=tmp_path,
        )
        ctx.llm = lambda name=None: IntSeverityLLM()

        state = {
            "domain": "quality",
            "provider": "alpha",
            "model": "model-a",
            "repo": "github.com/org/repo",
            "number": 1,
            "diff": "diff",
            "worktree_path": str(tmp_path),
        }

        result = await review_branch(ctx, FakeShell(), state)

        assert not any(n.get("failed") for n in result["notes"]), (
            f"a non-string severity must not fail the whole branch: {result['notes']}"
        )

        # The branch must have kept working after the rejection — otherwise "didn't fail"
        # could just mean "stopped quietly", which is the same lost review.
        assert any(f["path"] == "b.py" for f in result["findings"]), (
            f"the branch must continue recording after a rejected tool call, got findings: {result['findings']}"
        )

    def test_comments_from_findings_coerces_severity(self):
        """The fallback feed must coerce, since review.py:981 will call ``.strip()``.

        ``_comments_from_findings`` is the second path into the boundary, taken when
        synthesis returns no structured comments (review.py:971-981). Its findings come
        from ``record_finding``, so the ``str(...)`` coercion at review.py:599 is
        load-bearing: without it, :981 crashes on anything non-str.
        """
        from quality.agents.pr.tasks.review import _comments_from_findings

        findings = [
            {"path": "a.py", "line": 10, "severity": 5, "body": "x", "model": "m", "domain": "quality"},
            {"path": "b.py", "line": 20, "severity": None, "body": "y", "model": "m", "domain": "quality"},
        ]

        comments = _comments_from_findings(findings)

        assert comments, "fallback must produce comments for the given findings"
        for c in comments:
            assert isinstance(c["severity"], str), (
                f"severity must be coerced to str before review.py:981 calls .strip(), "
                f"got {c['severity']!r} ({type(c['severity']).__name__})"
            )
            # Prove the value survives the boundary's own operation.
            c["severity"].strip().lower()
