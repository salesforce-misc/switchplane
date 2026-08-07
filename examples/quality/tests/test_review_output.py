"""Output-contract tests for Quality review summaries and attribution footers."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


class SynthResult:
    def __init__(self, *, summary: str, comments: list[dict], event: str = "COMMENT"):
        self._value = {"summary": summary, "comments": comments, "event": event}

    def model_dump(self):
        return self._value


class SynthLLM:
    def __init__(self, result: SynthResult):
        self.result = result
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        return self.result


def _state(*, findings=None, matrix=None, notes=None, branch_executions=None, local=False):
    return {
        "repo": "github.com/org/repo",
        "number": 42,
        "diff": "diff --git a/test.py b/test.py\n@@ -9,0 +10 @@\n+changed\n",
        "head_sha": "abc123",
        "findings": findings or [],
        "notes": notes or [],
        "branch_executions": branch_executions or [],
        "matrix": matrix or [("alpha", "quality-model"), ("beta", "security-model")],
        "is_self_review": False,
        "authed_user": "reviewer",
        "local": local,
        "error": None,
    }


def _assert_top_level_footer(body: str, *models: str) -> None:
    summary, separator, footer = body.rpartition("---")
    assert separator, "top-level attribution must be rendered as a footer"
    assert "Review summary" in summary
    expected = f"quality/review: [{', '.join(dict.fromkeys(models))}]"
    assert footer.strip() == expected
    assert "Quality reviewer" not in footer
    assert "Security reviewer" not in footer
    assert "Posted by" not in footer


def _assert_inline_footer(body: str, *models: str) -> None:
    expected = f"quality/review: [{', '.join(dict.fromkeys(models))}]"
    non_empty_lines = [line for line in body.splitlines() if line.strip()]
    assert non_empty_lines[-1] == expected
    assert "Validated/recovered contributing models" not in body
    assert "Origin: switchplane-quality" not in body
    assert "**quality/review**" not in body


def test_inline_footer_is_one_deduplicated_final_line():
    from quality.agents.pr.tasks.review import _render_inline_body

    body = _render_inline_body(
        {
            "body": "Finding body.",
            "models": ["model-a", "model-b", "model-a"],
        }
    )

    assert [line for line in body.splitlines() if line.strip()][-1] == "quality/review: [model-a, model-b]"
    assert "Validated/recovered contributing models" not in body
    assert "Origin: switchplane-quality" not in body
    assert "**quality/review**" not in body


def _assert_marker_is_visible(markdown: str, marker: str) -> None:
    before_marker, separator, _after_marker = markdown.partition(marker)
    assert separator, f"missing deterministic provenance marker: {marker}"

    in_comment = False
    for token in re.findall(r"<!--|-->", before_marker):
        if token == "<!--":
            in_comment = True
        elif in_comment:
            in_comment = False
    assert not in_comment, "an open HTML comment can hide deterministic provenance"

    open_fence: tuple[str, int] | None = None
    for line in before_marker.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if not match:
            continue
        fence = match.group(1)
        if open_fence is None:
            open_fence = (fence[0], len(fence))
        elif fence[0] == open_fence[0] and len(fence) >= open_fence[1]:
            open_fence = None
    assert open_fence is None, "an open Markdown fence can absorb deterministic provenance"

    open_details = len(re.findall(r"<details(?:\s|>)", before_marker, flags=re.IGNORECASE))
    close_details = len(re.findall(r"</details\s*>", before_marker, flags=re.IGNORECASE))
    assert open_details <= close_details, "an open HTML details block can hide deterministic provenance"


@pytest.fixture
def output_harness(monkeypatch, tmp_path):
    from quality.agents.pr import memory as memory_module

    from conftest import FakeAgentContext, FakeShell
    from quality import gh as gh_module
    from quality import ratelimit as ratelimit_module

    posted = {"reviews": [], "comments": [], "baselines": []}

    async def submit_pr_review(shell, repo, number, event, body):
        posted["reviews"].append({"event": event, "body": body})

    async def create_pr_review_comment(shell, repo, number, body, path, line, commit_id=None):
        posted["comments"].append({"body": body, "path": path, "line": line})

    async def list_review_comments(shell, repo, number):
        return []

    monkeypatch.setattr(gh_module, "submit_pr_review", submit_pr_review, raising=True)
    monkeypatch.setattr(gh_module, "create_pr_review_comment", create_pr_review_comment, raising=True)
    monkeypatch.setattr(gh_module, "list_review_comments", list_review_comments, raising=True)
    monkeypatch.setattr(gh_module, "commentable_lines", lambda diff: {"test.py": {10}}, raising=True)

    def save_baseline(*args, **kwargs):
        posted["baselines"].append(kwargs)
        return tmp_path / "baseline.json"

    monkeypatch.setattr(memory_module, "save_baseline", save_baseline, raising=True)
    monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", lambda value: value, raising=True)

    ctx = FakeAgentContext(runtime_dir_path=tmp_path)
    return ctx, FakeShell(), posted


@pytest.mark.asyncio
async def test_clean_github_review_has_descriptive_summary_and_unified_footer(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    await synthesize_and_post(ctx, shell, _state())

    assert len(posted["reviews"]) == 1
    body = posted["reviews"][0]["body"]
    assert "No quality or security issues found" in body
    _assert_top_level_footer(body, "quality-model", "security-model")


@pytest.mark.asyncio
@pytest.mark.parametrize("synth_models", [[], ["invented-model"]], ids=["missing", "invented"])
async def test_synthesized_comment_repairs_missing_or_invented_models(output_harness, synth_models):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary=(
                "The change preserves the existing control flow and introduces no material risk. "
                "The focused regression coverage provides high test confidence."
            ),
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Handle the empty input before indexing.",
                    "models": synth_models,
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "medium",
            "body": "Handle the empty input before indexing.",
            "model": "quality-model",
            "provider": "alpha",
            "domain": "quality",
        }
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    assert len(posted["comments"]) == 1
    inline = posted["comments"][0]["body"]
    _assert_inline_footer(inline, "quality-model")
    assert "invented-model" not in inline


@pytest.mark.asyncio
async def test_synthesized_github_review_has_unified_top_level_footer(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary=(
                "The change preserves the existing control flow and introduces no material risk. "
                "The focused regression coverage provides high test confidence."
            ),
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Handle the empty input before indexing.",
                    "models": ["quality-model"],
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "medium",
            "body": "Handle the empty input before indexing.",
            "model": "quality-model",
            "provider": "alpha",
            "domain": "quality",
        }
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    assert len(posted["reviews"]) == 1
    _assert_top_level_footer(posted["reviews"][0]["body"], "quality-model", "security-model")


@pytest.mark.asyncio
async def test_raw_finding_fallback_uses_inline_footer(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    ctx.llm = lambda name=None: SynthLLM(SynthResult(summary="Substantive assessment.", comments=[]))
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "high",
            "body": "The unchecked value can terminate the worker.",
            "model": "quality-model",
            "provider": "alpha",
            "domain": "quality",
        }
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    assert len(posted["comments"]) == 1
    assert "unchecked value" in posted["comments"][0]["body"]
    _assert_inline_footer(posted["comments"][0]["body"], "quality-model")


@pytest.mark.asyncio
async def test_local_artifact_has_top_level_and_per_finding_footers(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, _posted = output_harness
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The changed path has a bounded correctness risk and direct regression coverage.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "low",
                    "body": "Keep the guard adjacent to the access.",
                    "models": ["quality-model"],
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "low",
            "body": "Keep the guard adjacent to the access.",
            "model": "quality-model",
            "provider": "alpha",
            "domain": "quality",
        }
    ]

    result = await synthesize_and_post(ctx, shell, _state(findings=findings, local=True))
    artifact = Path(result["local_artifact_path"]).read_text()

    _assert_top_level_footer(artifact, "quality-model", "security-model")
    assert "Keep the guard adjacent" in artifact
    assert artifact.count("quality-model") >= 2, "model must appear in top-level and per-finding attribution"


@pytest.mark.asyncio
async def test_clean_local_artifact_has_top_level_footer(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, _posted = output_harness
    result = await synthesize_and_post(ctx, shell, _state(local=True))
    artifact = Path(result["local_artifact_path"]).read_text()

    assert "No quality or security issues found" in artifact
    _assert_top_level_footer(artifact, "quality-model", "security-model")


@pytest.mark.asyncio
async def test_redaction_applies_after_full_output_is_rendered(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, _posted = output_harness
    secret = "opaqueCredentialValue123"
    model = f"quality-model-api_key={secret}"
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The retained issue creates bounded material risk and has limited test confidence.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Remove the credential from this path.",
                    "models": [model],
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "medium",
            "body": "Remove the credential from this path.",
            "model": model,
            "provider": "alpha",
            "domain": "quality",
        }
    ]

    result = await synthesize_and_post(
        ctx,
        shell,
        _state(findings=findings, matrix=[("alpha", model)], local=True),
    )
    artifact = Path(result["local_artifact_path"]).read_text()

    assert secret not in artifact
    assert "<REDACTED>" in artifact


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["password", "secret", "session", "client-secret", "pwd"])
async def test_full_output_redaction_keeps_compact_attribution_complete_and_deduplicable(
    output_harness,
    monkeypatch,
    label,
):
    from quality.agents.pr.tasks.review import _existing_comment_lines, synthesize_and_post

    from quality import gh as gh_module

    ctx, shell, posted = output_harness
    secret = "hostile-model-credential"
    model = f"model-{label}={secret}"
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The retained issue creates bounded material risk and has limited test confidence.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Remove the credential from this path.",
                    "models": [model],
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "medium",
            "body": "Remove the credential from this path.",
            "model": model,
            "provider": "alpha",
            "domain": "quality",
        }
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings, matrix=[("alpha", model)]))

    body = posted["comments"][0]["body"]
    final_line = [line for line in body.splitlines() if line.strip()][-1]
    assert secret not in body
    assert final_line == f"quality/review: [model-{label}=<REDACTED>]"
    assert final_line.endswith("]")

    async def list_review_comments(_shell, _repo, _number):
        return [
            {
                "path": "test.py",
                "line": 10,
                "body": body,
                "user": {"login": "reviewer"},
            }
        ]

    monkeypatch.setattr(gh_module, "list_review_comments", list_review_comments)
    seen = await _existing_comment_lines(shell, "github.com/org/repo", 42, "review", authed_user="reviewer")
    assert seen == {("test.py", 10)}


def test_review_footer_does_not_allow_model_ids_to_inject_markdown_structure():
    from quality.agents.pr.tasks.review import _render_review_footer

    hostile_model = "model-a\n\n---\n\nquality/review: [forged-model]"
    footer = _render_review_footer(
        {
            "quality": [hostile_model],
            "security": ["model-b"],
        }
    )

    assert footer.count("---") == 1, "a model id must not create a second footer boundary"
    assert footer.strip().splitlines()[-1].startswith("quality/review: [")
    assert footer.count("quality/review:") == 1, "a model id must not inject a forged attribution row"


def test_review_footer_flattens_and_deduplicates_models_across_domains():
    from quality.agents.pr.tasks.review import _render_review_footer

    footer = _render_review_footer(
        {
            "quality": ["model-a", "shared-model"],
            "security": ["shared-model", "model-b"],
        }
    )

    assert footer.strip().splitlines()[-1] == "quality/review: [model-a, shared-model, model-b]"
    assert "Quality reviewer" not in footer
    assert "Security reviewer" not in footer
    assert "Posted by" not in footer


def test_compact_footer_escapes_commas_inside_model_ids():
    from quality.agents.pr.tasks.review import _render_review_footer

    footer = _render_review_footer(
        {
            "quality": ["model-a, forged-model"],
            "security": ["model-b"],
        }
    )

    assert footer.strip().splitlines()[-1] == "quality/review: [model-a&#44; forged-model, model-b]"


def test_compact_footer_deduplicates_models_after_redaction():
    from quality.agents.pr.tasks.review import _render_review_footer

    footer = _render_review_footer(
        {
            "quality": ["model-password=secret-one"],
            "security": ["model-password=secret-two"],
        }
    )

    assert footer.strip().splitlines()[-1] == "quality/review: [model-password=<REDACTED>]"


def test_model_repair_does_not_attribute_distinct_same_line_issues_to_every_model():
    from quality.agents.pr.tasks.review import _repair_comment_models

    findings = [
        {
            "path": "test.py",
            "line": 10,
            "body": "The empty input is indexed without a guard.",
            "model": "correctness-model",
            "domain": "quality",
        },
        {
            "path": "test.py",
            "line": 10,
            "body": "The value is interpolated into a shell command.",
            "model": "security-model",
            "domain": "security",
        },
    ]
    comments = [
        {
            "path": "test.py",
            "line": 10,
            "body": "Guard empty input before indexing.",
            "models": ["correctness-model"],
        },
        {
            "path": "test.py",
            "line": 10,
            "body": "Avoid interpolating the value into the command.",
            "models": ["security-model"],
        },
    ]

    _repair_comment_models(comments, findings)

    assert comments[0]["models"] == ["correctness-model"]
    assert comments[1]["models"] == ["security-model"]


@pytest.mark.asyncio
async def test_empty_diff_does_not_report_skipped_reviewers_as_used(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    state = _state()
    state["diff"] = ""

    await synthesize_and_post(ctx, shell, state)

    body = posted["reviews"][0]["body"]
    assert "quality-model" not in body
    assert "security-model" not in body
    assert body.rstrip().endswith("quality/review: []")


@pytest.mark.asyncio
async def test_failed_branch_model_is_not_reported_in_unified_footer(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The quality review found one non-blocking issue; security coverage was incomplete.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "low",
                    "body": "Keep the guard adjacent to the access.",
                    "models": ["model-a"],
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "low",
            "body": "Keep the guard adjacent to the access.",
            "model": "model-a",
            "provider": "alpha",
            "domain": "quality",
        }
    ]
    notes = [
        {
            "domain": "security",
            "provider": "beta",
            "model": "model-b",
            "failed": True,
            "body": "_(reviewer branch security/beta failed: RuntimeError)_",
        }
    ]

    await synthesize_and_post(
        ctx,
        shell,
        _state(
            findings=findings,
            notes=notes,
            matrix=[("alpha", "model-a"), ("beta", "model-b")],
            branch_executions=[
                {"domain": "quality", "provider": "alpha", "model": "model-a", "succeeded": True},
                {"domain": "security", "provider": "beta", "model": "model-b", "succeeded": False},
            ],
        ),
    )

    _assert_top_level_footer(posted["reviews"][0]["body"], "model-a")
    assert "model-b" not in posted["reviews"][0]["body"].rpartition("---")[2]


@pytest.mark.asyncio
async def test_legacy_partial_outage_keeps_model_that_succeeded_in_another_domain(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    notes = [
        {
            "domain": "security",
            "provider": "alpha",
            "model": "shared-model",
            "failed": True,
            "body": "_(reviewer branch security/alpha failed: RuntimeError)_",
        }
    ]

    await synthesize_and_post(
        ctx,
        shell,
        _state(
            notes=notes,
            matrix=[("alpha", "shared-model")],
        ),
    )

    _assert_top_level_footer(posted["reviews"][0]["body"], "shared-model")


@pytest.mark.asyncio
async def test_one_failed_clean_branch_is_partial_not_total_outage(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    notes = [
        {
            "domain": "security",
            "provider": "beta",
            "model": "model-b",
            "failed": True,
            "body": "_(reviewer branch security/beta failed: RuntimeError)_",
        }
    ]

    result = await synthesize_and_post(
        ctx,
        shell,
        _state(
            notes=notes,
            matrix=[("alpha", "model-a"), ("beta", "model-b")],
        ),
    )

    assert "error" not in result, "three silent clean branches and one failure are a partial outage"
    assert len(posted["reviews"]) == 1
    body = posted["reviews"][0]["body"].lower()
    assert "security" in body and "failed" in body


@pytest.mark.asyncio
async def test_clean_review_submission_failure_does_not_persist_success(output_harness, monkeypatch):
    from quality.agents.pr import memory as memory_module
    from quality.agents.pr.tasks.review import synthesize_and_post

    from quality import gh as gh_module

    ctx, shell, _posted = output_harness
    persisted = []

    async def fail_submit(*args, **kwargs):
        raise RuntimeError("GitHub unavailable")

    monkeypatch.setattr(gh_module, "submit_pr_review", fail_submit, raising=True)
    monkeypatch.setattr(memory_module, "save_baseline", lambda *a, **kw: persisted.append(kw), raising=True)

    result = await synthesize_and_post(ctx, shell, _state())

    assert result.get("error") == "Failed to submit COMMENT review: GitHub unavailable"
    assert persisted == [], "an unpublished clean verdict must not become the follow-up baseline"


@pytest.mark.asyncio
async def test_generic_synthesis_summary_is_augmented_with_deterministic_assessment(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="Material risk. Test confidence.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "low",
                    "body": "Keep the guard adjacent to the access.",
                    "models": ["quality-model"],
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "low",
            "body": "Keep the guard adjacent to the access.",
            "model": "quality-model",
            "provider": "alpha",
            "domain": "quality",
        }
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    summary = posted["reviews"][0]["body"].lower().split("---", 1)[0]
    assert "material risk. test confidence." not in summary
    assert "reviewer evidence" in summary
    assert "changed code" in summary
    assert "did not execute" in summary


@pytest.mark.asyncio
async def test_synthesized_comment_restores_union_of_models_for_same_issue(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The retained issue creates bounded runtime risk; focused test coverage is absent.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Check that input is non-empty before indexing it.",
                    "models": ["model-a"],
                    "source_ids": ["finding-a"],
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "medium",
            "body": body,
            "model": model,
            "provider": provider,
            "domain": "quality",
            "source_id": source_id,
        }
        for model, provider, source_id, body in (
            ("model-a", "alpha", "finding-a", "Guard empty input before indexing."),
            ("model-b", "beta", "finding-b", "Check that input is non-empty before indexing it."),
        )
    ]

    await synthesize_and_post(
        ctx,
        shell,
        _state(findings=findings, matrix=[("alpha", "model-a"), ("beta", "model-b")]),
    )

    assert len(posted["comments"]) == 1
    _assert_inline_footer(posted["comments"][0]["body"], "model-a", "model-b")


@pytest.mark.asyncio
async def test_clean_partial_outage_reports_clean_successes_and_incomplete_coverage(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    notes = [
        {
            "domain": "security",
            "provider": "beta",
            "model": "model-b",
            "failed": True,
            "body": "_(reviewer branch security/beta failed: RuntimeError)_",
        }
    ]
    executions = [
        {"domain": "quality", "provider": "alpha", "model": "model-a", "succeeded": True},
        {"domain": "security", "provider": "beta", "model": "model-b", "succeeded": False},
    ]

    result = await synthesize_and_post(
        ctx,
        shell,
        _state(
            notes=notes,
            branch_executions=executions,
            matrix=[("alpha", "model-a"), ("beta", "model-b")],
        ),
    )

    assert "error" not in result
    body = posted["reviews"][0]["body"].lower()
    assert "successful reviewer" in body and "found no issues" in body
    assert "coverage was incomplete" in body or "coverage is incomplete" in body
    assert "no quality or security issues" not in body
    assert "reviewer matrix identified no material risk" not in body
    assert len(posted["baselines"]) == 1
    baseline_summary = posted["baselines"][0]["summary"].lower()
    assert "coverage was incomplete" in baseline_summary or "coverage is incomplete" in baseline_summary
    assert "no quality or security issues found" not in baseline_summary


@pytest.mark.asyncio
async def test_authoritative_branch_executions_drive_unified_footer_attribution(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    notes = [
        {
            "domain": "security",
            "provider": "beta",
            "model": "security-model",
            "failed": True,
            "body": "_(reviewer branch security/beta failed: RuntimeError)_",
        }
    ]
    executions = [
        {"domain": "quality", "provider": "alpha", "model": "quality-model", "succeeded": True},
        {"domain": "security", "provider": "beta", "model": "security-model", "succeeded": False},
    ]

    await synthesize_and_post(
        ctx,
        shell,
        _state(
            notes=notes,
            branch_executions=executions,
            matrix=[("alpha", "quality-model"), ("beta", "security-model")],
        ),
    )

    _assert_top_level_footer(posted["reviews"][0]["body"], "quality-model")
    assert "security-model" not in posted["reviews"][0]["body"].rpartition("---")[2]


@pytest.mark.asyncio
async def test_authoritative_failed_executions_are_total_outage_without_failure_notes(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    executions = [
        {"domain": "quality", "provider": "alpha", "model": "model-a", "succeeded": False},
        {"domain": "security", "provider": "beta", "model": "model-b", "succeeded": False},
    ]

    result = await synthesize_and_post(
        ctx,
        shell,
        _state(
            branch_executions=executions,
            matrix=[("alpha", "model-a"), ("beta", "model-b")],
        ),
    )

    assert "All reviewer branches failed" in result.get("error", "")
    assert posted["reviews"] == []


@pytest.mark.asyncio
async def test_baseline_path_internal_type_error_is_not_signature_probed(output_harness, monkeypatch):
    from quality.agents.pr import memory as memory_module
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, _posted = output_harness
    calls = 0

    def broken_baseline_path(root, repo, number, *, local=False):
        nonlocal calls
        calls += 1
        raise TypeError("internal baseline failure")

    monkeypatch.setattr(memory_module, "baseline_path", broken_baseline_path, raising=True)
    state = _state(
        notes=[
            {
                "domain": "security",
                "provider": "beta",
                "model": "security-model",
                "failed": True,
                "body": "reviewer failed",
            }
        ],
        branch_executions=[
            {"domain": "quality", "provider": "alpha", "model": "quality-model", "succeeded": True},
            {"domain": "security", "provider": "beta", "model": "security-model", "succeeded": False},
        ],
    )

    with pytest.raises(TypeError, match="internal baseline failure"):
        await synthesize_and_post(ctx, shell, state)
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("synth_case", ["omitted", "downgraded"])
async def test_raw_blocking_finding_imposes_request_changes(output_harness, synth_case):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    raw_high = {
        "path": "test.py",
        "line": 10,
        "severity": "critical",
        "body": "Untrusted input reaches command execution.",
        "model": "security-model",
        "provider": "beta",
        "domain": "security",
        "source_id": "security-1",
    }
    raw_low = {
        "path": "test.py",
        "line": 10,
        "severity": "low",
        "body": "Keep the validation next to the command.",
        "model": "quality-model",
        "provider": "alpha",
        "domain": "quality",
        "source_id": "quality-1",
    }
    if synth_case == "omitted":
        comment = {
            "path": "test.py",
            "line": 10,
            "severity": "low",
            "body": raw_low["body"],
            "models": ["quality-model"],
            "source_ids": ["quality-1"],
        }
    else:
        comment = {
            "path": "test.py",
            "line": 10,
            "severity": "low",
            "body": raw_high["body"],
            "models": ["security-model"],
            "source_ids": ["security-1"],
        }
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The retained finding carries material risk and lacks direct regression-test confidence.",
            comments=[comment],
            event="COMMENT",
        )
    )

    await synthesize_and_post(ctx, shell, _state(findings=[raw_high, raw_low]))

    assert posted["reviews"][0]["event"] == "REQUEST_CHANGES"


@pytest.mark.asyncio
async def test_synthesized_comment_without_validated_raw_source_is_not_posted(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The raw finding has bounded material risk and no direct test confidence.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Fabricated synthesized issue with no reviewer source.",
                    "models": ["quality-model"],
                    "source_ids": ["invented-source"],
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "medium",
            "body": "Guard empty input before indexing.",
            "model": "quality-model",
            "provider": "alpha",
            "domain": "quality",
            "source_id": "quality-1",
        }
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    published = "\n".join(
        [*(comment["body"] for comment in posted["comments"]), *(review["body"] for review in posted["reviews"])]
    )
    assert "Fabricated synthesized issue" not in published
    assert "Guard empty input before indexing" in published


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "fabricated"),
    [
        ("path", "relocated.py"),
        ("line", 999),
        ("severity", "critical"),
        ("body", "Fabricated replacement issue."),
    ],
)
async def test_valid_source_id_cannot_relocate_or_fabricate_raw_finding_fields(
    output_harness,
    field,
    fabricated,
):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    raw = {
        "path": "test.py",
        "line": 10,
        "severity": "low",
        "body": "Guard empty input before indexing.",
        "model": "quality-model",
        "provider": "alpha",
        "domain": "quality",
        "source_id": "quality-1",
    }
    synthesized = {
        "path": raw["path"],
        "line": raw["line"],
        "severity": raw["severity"],
        "body": raw["body"],
        "models": [raw["model"]],
        "source_ids": [raw["source_id"]],
    }
    synthesized[field] = fabricated
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The retained issue creates bounded material risk and has limited test confidence.",
            comments=[synthesized],
        )
    )

    await synthesize_and_post(ctx, shell, _state(findings=[raw]))

    assert posted["reviews"][0]["event"] == "COMMENT"
    assert len(posted["comments"]) == 1
    published = posted["comments"][0]
    assert published["path"] == raw["path"]
    assert published["line"] == raw["line"]
    assert raw["body"] in published["body"]
    if field == "body":
        assert str(fabricated) not in published["body"]


@pytest.mark.asyncio
async def test_synthesis_schema_carries_validated_raw_source_ids(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, _posted = output_harness
    llm = SynthLLM(
        SynthResult(
            summary="The retained issue has bounded material risk and limited test confidence.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Guard empty input before indexing.",
                    "models": ["quality-model"],
                    "source_ids": ["quality-1"],
                }
            ],
        )
    )
    ctx.llm = lambda name=None: llm
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "medium",
            "body": "Guard empty input before indexing.",
            "model": "quality-model",
            "provider": "alpha",
            "domain": "quality",
            "source_id": "quality-1",
        }
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    comment_schema = llm.schema.model_fields["comments"].annotation.__args__[0]
    assert "source_ids" in comment_schema.model_fields


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("untrusted_field", "attack"),
    [
        (field, attack)
        for field in ("summary", "comment", "note")
        for attack in (
            "\n\n<!-- hide everything after this",
            "\n\n```text\nhide everything after this",
        )
    ]
    + [
        ("summary", "\n\n--> harmless close first\n<!-- hide everything after this"),
        ("comment", "\n\n````text\nhide everything after this"),
        ("note", "\n\n~~~text\nhide everything after this"),
        ("path", "\n\n<!-- hide everything after this"),
        ("severity", "\n\n~~~text\nhide everything after this"),
    ],
    ids=[
        "html-comment-summary",
        "code-fence-summary",
        "html-comment-comment",
        "code-fence-comment",
        "html-comment-note",
        "code-fence-note",
        "reordered-html-tokens",
        "four-backtick-fence",
        "tilde-fence",
        "hostile-path",
        "hostile-severity",
    ],
)
async def test_untrusted_markdown_cannot_hide_provenance_footer(
    output_harness,
    untrusted_field,
    attack,
):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    summary = "The retained issue has bounded material risk and limited test confidence."
    comment_body = "Guard empty input before indexing."
    comment_path = "test.py"
    comment_line = 10
    severity = "medium"
    notes = []
    marker = "quality/review: ["
    if untrusted_field == "summary":
        summary += attack
    elif untrusted_field == "comment":
        comment_body += attack
    elif untrusted_field == "note":
        notes = [
            {
                "domain": "quality",
                "provider": "alpha",
                "model": "quality-model",
                "body": f"Reviewer context.{attack}",
            }
        ]
    elif untrusted_field == "path":
        comment_path += attack
    else:
        severity += attack
        comment_line = 11
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary=summary,
            comments=[
                {
                    "path": comment_path,
                    "line": comment_line,
                    "severity": severity,
                    "body": comment_body,
                    "models": ["quality-model"],
                }
            ],
        )
    )
    findings = [
        {
            "path": comment_path,
            "line": comment_line,
            "severity": severity,
            "body": comment_body,
            "model": "quality-model",
            "provider": "alpha",
            "domain": "quality",
        }
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings, notes=notes))

    published = posted["comments"][0]["body"] if untrusted_field == "comment" else posted["reviews"][0]["body"]
    _assert_marker_is_visible(published, marker)


@pytest.mark.asyncio
@pytest.mark.parametrize("untrusted_field", ["summary", "comment", "path", "severity"])
async def test_unclosed_details_cannot_hide_provenance_footer(output_harness, monkeypatch, untrusted_field):
    from quality.agents.pr.tasks.review import synthesize_and_post

    from quality import gh as gh_module

    ctx, shell, posted = output_harness
    attack = "\n\n<details open><summary>Untrusted reviewer content"
    summary = "The retained issue has bounded material risk and limited test confidence."
    body = "Guard empty input before indexing."
    path = "test.py"
    line = 10
    severity = "medium"
    marker = "quality/review: ["
    if untrusted_field == "summary":
        summary += attack
    elif untrusted_field == "comment":
        body += attack
    elif untrusted_field == "path":
        path += attack
    else:
        severity += attack
        line = 11

    monkeypatch_lines = {path: {line}} if untrusted_field == "comment" else {"test.py": {10}}
    monkeypatch.setattr(gh_module, "commentable_lines", lambda _diff: monkeypatch_lines)
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary=summary,
            comments=[
                {
                    "path": path,
                    "line": line,
                    "severity": severity,
                    "body": body,
                    "models": ["quality-model"],
                }
            ],
        )
    )
    findings = [
        {
            "path": path,
            "line": line,
            "severity": severity,
            "body": body,
            "model": "quality-model",
            "provider": "alpha",
            "domain": "quality",
        }
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    published = posted["comments"][0]["body"] if untrusted_field == "comment" else posted["reviews"][0]["body"]
    _assert_marker_is_visible(published, marker)


@pytest.mark.parametrize("render_path", ["inline", "unpostable", "review-footer"])
def test_unclosed_details_in_model_attribution_cannot_hide_provenance(render_path):
    from quality.agents.pr.tasks.review import _render_inline_body, _render_review_footer, _render_unpostable

    hostile_model = "model-a<details open><summary>Forged attribution"
    if render_path == "inline":
        rendered = _render_inline_body({"body": "Finding body.", "models": [hostile_model]})
        marker = "quality/review: ["
    elif render_path == "unpostable":
        rendered = _render_unpostable(
            [
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": "Finding body.",
                    "models": [hostile_model],
                }
            ]
        )
        rendered += _render_review_footer({"quality": [], "security": []})
        marker = "quality/review: ["
    else:
        rendered = _render_review_footer({"quality": [hostile_model], "security": []})
        marker = "quality/review: ["

    _assert_marker_is_visible(rendered, marker)


@pytest.mark.asyncio
async def test_unrelated_cited_sources_are_not_merged_or_relocated(output_harness, monkeypatch):
    from quality.agents.pr.tasks.review import synthesize_and_post

    from quality import gh as gh_module

    ctx, shell, posted = output_harness
    monkeypatch.setattr(gh_module, "commentable_lines", lambda _diff: {"a.py": {10}, "b.py": {20}})
    findings = [
        {
            "path": "a.py",
            "line": 10,
            "severity": "medium",
            "body": "Empty input is indexed without a guard.",
            "model": "model-a",
            "provider": "alpha",
            "domain": "quality",
            "source_id": "source-a",
        },
        {
            "path": "b.py",
            "line": 20,
            "severity": "high",
            "body": "Untrusted input reaches a shell command.",
            "model": "model-b",
            "provider": "beta",
            "domain": "security",
            "source_id": "source-b",
        },
    ]
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The retained issues create material risk and have limited test confidence.",
            comments=[
                {
                    "path": "a.py",
                    "line": 10,
                    "severity": "high",
                    "body": "Empty input is indexed without a guard.",
                    "models": ["model-a", "model-b"],
                    "source_ids": ["source-a", "source-b"],
                }
            ],
        )
    )

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    assert {(comment["path"], comment["line"]) for comment in posted["comments"]} == {
        ("a.py", 10),
        ("b.py", 20),
    }
    published = "\n".join(comment["body"] for comment in posted["comments"])
    assert "Empty input is indexed without a guard." in published
    assert "Untrusted input reaches a shell command." in published


@pytest.mark.asyncio
async def test_supported_body_with_appended_allegation_is_not_published_verbatim(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    raw_body = "Guard empty input before indexing."
    unsupported = "This also bypasses authorization for every account."
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The retained issue creates bounded material risk and has limited test confidence.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": f"{raw_body} {unsupported}",
                    "models": ["quality-model"],
                    "source_ids": ["quality-1"],
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "medium",
            "body": raw_body,
            "model": "quality-model",
            "provider": "alpha",
            "domain": "quality",
            "source_id": "quality-1",
        }
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    assert len(posted["comments"]) == 1
    assert raw_body in posted["comments"][0]["body"]
    assert unsupported not in posted["comments"][0]["body"]


@pytest.mark.asyncio
async def test_mixed_sourced_and_legacy_findings_preserve_legacy_without_weakening_validation(
    output_harness,
    monkeypatch,
):
    from quality.agents.pr.tasks.review import synthesize_and_post

    from quality import gh as gh_module

    ctx, shell, posted = output_harness
    monkeypatch.setattr(gh_module, "commentable_lines", lambda _diff: {"test.py": {10}, "legacy.py": {20}})
    unsupported = "It also grants administrator access."
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "medium",
            "body": "Guard empty input before indexing.",
            "model": "new-model",
            "provider": "alpha",
            "domain": "quality",
            "source_id": "source-new",
        },
        {
            "path": "legacy.py",
            "line": 20,
            "severity": "low",
            "body": "Close the file handle on the error path.",
            "model": "legacy-model",
            "provider": "beta",
            "domain": "quality",
        },
    ]
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The retained issues create bounded material risk and have limited test confidence.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "medium",
                    "body": f"Guard empty input before indexing. {unsupported}",
                    "models": ["new-model"],
                    "source_ids": ["source-new"],
                },
                {
                    "path": "legacy.py",
                    "line": 20,
                    "severity": "low",
                    "body": "Close the file handle on the error path.",
                    "models": ["legacy-model"],
                    "source_ids": [],
                },
            ],
        )
    )

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    assert {(comment["path"], comment["line"]) for comment in posted["comments"]} == {
        ("test.py", 10),
        ("legacy.py", 20),
    }
    published = "\n".join(comment["body"] for comment in posted["comments"])
    assert "Guard empty input before indexing." in published
    assert "Close the file handle on the error path." in published
    assert unsupported not in published


@pytest.mark.asyncio
@pytest.mark.parametrize("successful_domain_has_finding", [False, True], ids=["clean-success", "finding-success"])
async def test_partial_outage_baseline_preserves_prior_failed_domain_findings(
    output_harness,
    monkeypatch,
    successful_domain_has_finding,
):
    from quality.agents.pr import memory as memory_module
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    prior = {
        "head_sha": "old-sha",
        "summary": "Prior review.",
        "findings": [
            {
                "domains": ["security"],
                "domain": "security",
                "path": "auth.py",
                "line": 30,
                "severity": "high",
                "title": "Prior authorization bypass remains unresolved.",
            }
        ],
    }
    monkeypatch.setattr(memory_module, "load_baseline", lambda _path: prior, raising=True)

    findings = []
    if successful_domain_has_finding:
        findings = [
            {
                "path": "test.py",
                "line": 10,
                "severity": "low",
                "body": "Close the handle on the error path.",
                "model": "quality-model",
                "provider": "alpha",
                "domain": "quality",
                "source_id": "quality-1",
            }
        ]
        ctx.llm = lambda name=None: SynthLLM(
            SynthResult(
                summary="The quality finding creates bounded material risk; security coverage was incomplete.",
                comments=[
                    {
                        "path": "test.py",
                        "line": 10,
                        "severity": "low",
                        "body": "Close the handle on the error path.",
                        "models": ["quality-model"],
                        "source_ids": ["quality-1"],
                    }
                ],
            )
        )

    state = _state(
        findings=findings,
        notes=[
            {
                "domain": "security",
                "provider": "beta",
                "model": "security-model",
                "failed": True,
                "body": "_(reviewer branch security/beta failed: RuntimeError)_",
            }
        ],
        branch_executions=[
            {"domain": "quality", "provider": "alpha", "model": "quality-model", "succeeded": True},
            {"domain": "security", "provider": "beta", "model": "security-model", "succeeded": False},
        ],
        matrix=[("alpha", "quality-model"), ("beta", "security-model")],
    )

    await synthesize_and_post(ctx, shell, state)

    assert len(posted["baselines"]) == 1
    persisted = posted["baselines"][0]["findings"]
    assert any(
        finding.get("path") == "auth.py"
        and finding.get("line") == 30
        and "security" in (finding.get("domains") or [finding.get("domain")])
        for finding in persisted
    ), "a failed domain must retain its prior baseline finding"
    if successful_domain_has_finding:
        assert any(finding.get("path") == "test.py" and finding.get("line") == 10 for finding in persisted)


@pytest.mark.asyncio
async def test_distinct_omitted_same_line_findings_remain_separate_with_correct_models(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="Two independent issues create material risk and have limited test confidence.",
            comments=[],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "medium",
            "body": "Guard empty input before indexing.",
            "model": "correctness-model",
            "provider": "alpha",
            "domain": "quality",
            "source_id": "correctness-1",
        },
        {
            "path": "test.py",
            "line": 10,
            "severity": "high",
            "body": "Pass the argument without shell interpolation.",
            "model": "security-model",
            "provider": "beta",
            "domain": "security",
            "source_id": "security-1",
        },
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    assert len(posted["comments"]) == 2
    by_finding = {
        "empty input": next(comment["body"] for comment in posted["comments"] if "empty input" in comment["body"]),
        "shell interpolation": next(
            comment["body"] for comment in posted["comments"] if "shell interpolation" in comment["body"]
        ),
    }
    _assert_inline_footer(by_finding["empty input"], "correctness-model")
    assert "security-model" not in by_finding["empty input"]
    _assert_inline_footer(by_finding["shell interpolation"], "security-model")
    assert "correctness-model" not in by_finding["shell interpolation"]


@pytest.mark.asyncio
async def test_expanded_equivalent_sources_preserve_unique_trusted_details_and_all_models(output_harness):
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    common_body = "Validate the token before using it to authorize the request."
    expanded_body = (
        "Validate the token before using it to authorize the request, and reject expired credentials "
        "before caching the authorization decision."
    )
    ctx.llm = lambda name=None: SynthLLM(
        SynthResult(
            summary="The authorization finding creates material risk and has limited test confidence.",
            comments=[
                {
                    "path": "test.py",
                    "line": 10,
                    "severity": "high",
                    "body": common_body,
                    "models": ["model-a"],
                    "source_ids": ["source-a"],
                }
            ],
        )
    )
    findings = [
        {
            "path": "test.py",
            "line": 10,
            "severity": "high",
            "body": common_body,
            "model": "model-a",
            "provider": "alpha",
            "domain": "security",
            "source_id": "source-a",
        },
        {
            "path": "test.py",
            "line": 10,
            "severity": "high",
            "body": expanded_body,
            "model": "model-b",
            "provider": "beta",
            "domain": "security",
            "source_id": "source-b",
        },
    ]

    await synthesize_and_post(ctx, shell, _state(findings=findings))

    assert len(posted["comments"]) == 1
    body = posted["comments"][0]["body"]
    assert common_body in body
    assert "reject expired credentials before caching the authorization decision" in body
    _assert_inline_footer(body, "model-a", "model-b")


@pytest.mark.asyncio
async def test_partial_outage_restricts_mixed_prior_finding_to_failed_domains(output_harness, monkeypatch):
    from quality.agents.pr import memory as memory_module
    from quality.agents.pr.tasks.review import synthesize_and_post

    ctx, shell, posted = output_harness
    prior = {
        "head_sha": "old-sha",
        "summary": "Prior review.",
        "findings": [
            {
                "domains": ["quality", "security"],
                "domain": "quality",
                "path": "shared.py",
                "line": 30,
                "severity": "high",
                "title": "Shared validation affects both review domains.",
            }
        ],
    }
    monkeypatch.setattr(memory_module, "load_baseline", lambda _path: prior, raising=True)
    state = _state(
        notes=[
            {
                "domain": "security",
                "provider": "beta",
                "model": "security-model",
                "failed": True,
                "body": "_(reviewer branch security/beta failed: RuntimeError)_",
            }
        ],
        branch_executions=[
            {"domain": "quality", "provider": "alpha", "model": "quality-model", "succeeded": True},
            {"domain": "security", "provider": "beta", "model": "security-model", "succeeded": False},
        ],
        matrix=[("alpha", "quality-model"), ("beta", "security-model")],
    )

    await synthesize_and_post(ctx, shell, state)

    assert len(posted["baselines"]) == 1
    preserved = next(finding for finding in posted["baselines"][0]["findings"] if finding.get("path") == "shared.py")
    assert preserved["domains"] == ["security"]
    assert preserved["domain"] == "security"
