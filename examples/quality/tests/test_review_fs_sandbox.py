"""Adversarial tests for the reviewer's filesystem sandbox scope.

Now that ``fs_tools()`` are actually reachable by the model (#54/#55/#56), the
``allowed_paths`` those tools enforce is live security boundary for the first time.

``ReviewTask.run`` builds the Shell with ``allowed_paths=[runtime_dir]``
(review.py:1282-1287). ``runtime_dir`` is ``~/.quality`` — the whole app runtime
directory. The only thing the reviewer needs to read is the checkout, which lives
several levels down at ``runtime_dir/repos/<repo>.worktrees/...`` (review.py:244-266).

Everything else in ``runtime_dir`` is in scope too:

    ~/.quality/config.toml                  every provider's api_key
    ~/.quality/state.db                     task/event history for every task
    ~/.quality/state/review/**/pr-*.json    prior findings for every reviewed PR
    ~/.quality/ca-bundle.pem

The model's instructions come from the PR diff, which is attacker-controlled — that
is the whole premise the untrusted-model-output rules elsewhere in this module are
built on (see ``_resolve_event``, review.py:494). A diff that steers the reviewer
into ``read_file``/``grep_files`` on the runtime dir gets the operator's API keys
back as a tool result, in the model's own context, on the same turn loop that then
calls ``record_finding`` — whose body is posted to GitHub.

Redaction (``_redact.redact_secrets``) is explicitly documented as defense-in-depth,
not a boundary, and it does not cover every provider's key format.

The fix is scope, not redaction: pass the worktree (or at minimum
``runtime_dir / "repos"``) as ``allowed_paths``, not the whole runtime dir.

All imports are function-scoped to match the suite convention (see conftest.py).
"""

from __future__ import annotations

import pytest


class _StopAfterShell:
    """build_graph stand-in: lets ``run`` proceed past Shell construction and stop.

    Mirrors the helper in test_review_tool_loop.py — the graph itself is not under
    test here, only the Shell the task hands it.
    """

    def compile(self, checkpointer=None):
        return self

    async def aget_state(self, config):
        from types import SimpleNamespace

        return SimpleNamespace(values={})

    async def ainvoke(self, state, config=None):
        return {}


def _capture_task_shell_kwargs(monkeypatch, tmp_path) -> dict:
    """Run ``ReviewTask.run`` and return the kwargs it passed to ``Shell``.

    Derived from production rather than restated as a literal, so this stays honest
    if ``ReviewTask.run`` changes how it builds the Shell.
    """
    import asyncio

    from quality.agents.pr.tasks import review as review_module

    from switchplane.shell import Shell

    captured: dict = {}

    class RecordingShell(Shell):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(review_module, "Shell", RecordingShell, raising=True)
    monkeypatch.setattr(review_module, "build_graph", lambda ctx, shell: _StopAfterShell(), raising=True)

    from conftest import FakeAgentContext

    ctx = FakeAgentContext(
        config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "model-a"}}}},
        runtime_dir_path=tmp_path,
    )

    task = review_module.ReviewTask()
    task.pr = "https://github.com/org/repo/pull/1"
    task.local = True

    asyncio.run(task.run(ctx))

    assert captured, "ReviewTask.run must construct a Shell"
    return captured


class TestReviewerFilesystemScope:
    """The reviewer's fs_tools must not reach the operator's secrets."""

    def test_reviewer_cannot_read_runtime_config(self, monkeypatch, tmp_path):
        """``config.toml`` must be outside the reviewer's ``allowed_paths``.

        ``Shell.validate_path`` is the single chokepoint every fs tool goes through,
        so asserting on it covers ``read_file``, ``grep_files``, ``search_files`` and
        ``list_directory`` at once — and it does not depend on which tool a given
        model happens to reach for.

        Verified against the real classes: the allowed_paths value is read out of
        production's own ``Shell(...)`` call, and the check runs on a real ``Shell``.
        """
        from switchplane.shell import Shell

        captured = _capture_task_shell_kwargs(monkeypatch, tmp_path)

        allowed_paths = captured.get("allowed_paths")
        assert allowed_paths, "Shell must be constructed with an explicit allowed_paths"

        shell = Shell(
            allowed_paths=allowed_paths,
            allowed_commands=captured.get("allowed_commands", []),
            timeout=30.0,
        )

        config_path = tmp_path / "config.toml"
        config_path.write_text('[llm]\napi_key = "sk-ant-operator-secret"\n')

        with pytest.raises(PermissionError):
            shell.validate_path(str(config_path))

    def test_reviewer_cannot_read_other_prs_baselines(self, monkeypatch, tmp_path):
        """Prior findings for unrelated PRs must be outside the reviewer's scope.

        ``state/review/**/pr-*.json`` holds every finding this app has ever recorded,
        including embargoed security issues on other repositories. A reviewer scoped to
        the whole runtime dir can read all of them and quote them into a comment on an
        unrelated public PR.
        """
        from switchplane.shell import Shell

        captured = _capture_task_shell_kwargs(monkeypatch, tmp_path)

        shell = Shell(
            allowed_paths=captured["allowed_paths"],
            allowed_commands=captured.get("allowed_commands", []),
            timeout=30.0,
        )

        other = tmp_path / "state" / "review" / "github.com" / "other" / "repo"
        other.mkdir(parents=True, exist_ok=True)
        baseline = other / "pr-99.json"
        baseline.write_text('{"findings": [{"title": "unpatched RCE in auth"}]}')

        with pytest.raises(PermissionError):
            shell.validate_path(str(baseline))

    def test_reviewer_can_still_read_the_worktree(self, monkeypatch, tmp_path):
        """Whatever the scope becomes, the checkout itself must stay readable.

        Negative control. Without this, "fix" the scope to something empty and the two
        tests above pass while the reviewer can no longer read any code at all. The
        worktree layout is the one built by the setup node (review.py:244-266).
        """
        from switchplane.shell import Shell

        captured = _capture_task_shell_kwargs(monkeypatch, tmp_path)

        shell = Shell(
            allowed_paths=captured["allowed_paths"],
            allowed_commands=captured.get("allowed_commands", []),
            timeout=30.0,
        )

        worktree = tmp_path / "repos" / "github.com" / "org" / "repo.worktrees" / "pr-1-task"
        worktree.mkdir(parents=True, exist_ok=True)
        source = worktree / "auth.py"
        source.write_text("def check(): pass\n")

        # Must NOT raise — this is the reviewer's actual job.
        shell.validate_path(str(source))


class TestSecretRedactionCoverage:
    """Redaction is the last line before GitHub; provider key formats must be covered."""

    def test_google_api_key_is_redacted(self):
        """``AIzaSy...`` keys must be redacted.

        ``switchplane/llm.py`` routes Google models (``ChatGoogleGenerativeAI``), so a
        Google key is a first-class credential in ``config.toml``. But
        ``_SECRET_PATTERNS`` (quality/_redact.py:22-38) covers only ``sk-``, ``sk-ant-``,
        the ``gh*_`` family, ``github_pat_`` and ``AKIA`` — there is no Google prefix.

        An unlabelled ``AIzaSy...`` in a comment body therefore reaches GitHub verbatim.
        The labelled ``api_key = "AIzaSy..."`` form is caught by pattern 0, so the gap is
        specifically the bare-value case — which is exactly what a model produces when it
        paraphrases rather than quotes a config line.
        """
        from quality._redact import redact_secrets

        secret = "AIzaSy" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r"
        out = redact_secrets(f"The reviewer observed the value {secret} in the environment.")

        assert secret not in out, (
            f"Google API key survived redaction: {out!r}. _SECRET_PATTERNS has no "
            "AIzaSy prefix, so a bare Google key is posted to GitHub verbatim."
        )


def _capture_branch_fs_shell(monkeypatch, worktree_path: str):
    """Run ``review_branch`` and return the branch-scoped Shell it built for fs_tools.

    ``_capture_task_shell_kwargs`` above only proves the *task-level* Shell is scoped
    to ``repos/`` (#65 Part A). This helper drives the deeper property (#65 Part B):
    the Shell that actually backs the model's ``fs_tools`` is constructed *inside*
    ``review_branch`` and scoped to that branch's single ``worktree_path``.

    RecordingShell subclasses the real ``Shell`` so ``fs_tools()`` / ``validate_path``
    behave exactly as in production. Returns the captured Shell; assertions run on the
    real object, not a restated literal.
    """
    import asyncio
    from pathlib import Path
    from unittest.mock import AsyncMock, Mock

    from quality.agents.pr import memory as memory_module
    from quality.agents.pr import prompts as prompts_module
    from quality.agents.pr.tasks import review as review_module

    from switchplane.shell import Shell

    monkeypatch.setattr(memory_module, "load_baseline", lambda path: None)
    monkeypatch.setattr(prompts_module, "initial_prompt", lambda *a, **k: "prompt")

    captured: list[Shell] = []

    class RecordingShell(Shell):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(review_module, "Shell", RecordingShell, raising=True)

    from quality import ratelimit as ratelimit_module

    def fake_with_rate_limit_retry(runnable):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="Done", tool_calls=[]))
        return mock_llm

    monkeypatch.setattr(ratelimit_module, "with_rate_limit_retry", fake_with_rate_limit_retry)

    from conftest import FakeAgentContext, FakeShell

    ctx = FakeAgentContext(
        config={"llm": {"providers": {"alpha": {"model": "model-a"}}}},
        runtime_dir_path=Path("/fake/runtime"),
    )

    # ctx.llm must return something with a working bind_tools for the branch.
    def _fake_llm(name=None, *, model=None):
        m = Mock()
        m.bind_tools = Mock(return_value=m)
        return m

    monkeypatch.setattr(ctx, "llm", _fake_llm)

    outer_shell = FakeShell()
    state = {
        "domain": "quality",
        "provider": "alpha",
        "model": "model-a",
        "repo": "github.com/org/repo",
        "number": 1,
        "diff": "diff",
        "worktree_path": worktree_path,
    }

    asyncio.run(review_module.review_branch(ctx, outer_shell, state))

    assert captured, "review_branch must construct a worktree-scoped Shell for fs_tools"
    return captured[0]


class TestBranchWorktreeScope:
    """#65 Part B: a branch's fs_tools are scoped to its OWN worktree, not repos/.

    Part A (above) narrows the task-level Shell from the whole runtime dir to
    ``repos/``. That is not enough: with ``repos/`` as scope, a branch reviewing repo A
    could ``read_file``/``grep`` repo B's clone that happens to sit under the same
    ``repos/`` dir — cross-repo disclosure, still attacker-steerable via the diff.
    Part B moves the model's fs_tools onto a Shell scoped to the single worktree.
    """

    def test_branch_cannot_read_sibling_repo_checkout(self, monkeypatch, tmp_path):
        """A reviewer of repo A must not reach a sibling repo B's checkout under repos/.

        Production already prevents this (Part B scopes fs_tools to the single
        worktree), so this test PASSES and stands as a regression guard: a future
        refactor that re-widens the branch Shell back to ``repos/`` (or the runtime
        dir) fails here.
        """
        from switchplane.shell import Shell

        repos = tmp_path / "repos"
        wt_a = repos / "github.com" / "org" / "repo-a.worktrees" / "pr-1-quality"
        wt_a.mkdir(parents=True, exist_ok=True)

        fs_shell = _capture_branch_fs_shell(monkeypatch, str(wt_a))
        assert isinstance(fs_shell, Shell)

        # Sibling repo B's checkout, under the SAME repos/ dir the task-level shell allows.
        repo_b_secret = repos / "github.com" / "other" / "repo-b.worktrees" / "pr-9-quality" / "secret.py"
        repo_b_secret.parent.mkdir(parents=True, exist_ok=True)
        repo_b_secret.write_text("API_TOKEN = 'sk-cross-repo-leak'\n")

        with pytest.raises(PermissionError):
            fs_shell.validate_path(str(repo_b_secret))

    def test_branch_can_still_read_its_own_worktree(self, monkeypatch, tmp_path):
        """Negative control: the branch's own checkout must stay readable.

        Without this, scoping the branch Shell to something empty would pass the
        sibling-repo test while breaking the reviewer entirely.
        """
        wt_a = tmp_path / "repos" / "github.com" / "org" / "repo-a.worktrees" / "pr-1-quality"
        wt_a.mkdir(parents=True, exist_ok=True)
        own = wt_a / "auth.py"
        own.write_text("def check(): pass\n")

        fs_shell = _capture_branch_fs_shell(monkeypatch, str(wt_a))

        # Must NOT raise — reading the code under review is the reviewer's job.
        fs_shell.validate_path(str(own))

    def test_bare_filename_resolves_into_worktree(self, monkeypatch, tmp_path):
        """A bare ``auth.py`` from the model must resolve INTO the worktree.

        ``Shell.validate_path`` resolves a relative path against ``allowed_paths[0]``.
        Because the branch Shell is now worktree-scoped (not runtime-scoped), a bare
        filename the model emits resolves to ``<worktree>/auth.py`` — the file under
        review — rather than into the runtime dir. This pins the resolution anchor as
        a direct consequence of Part B's scope change.
        """
        wt_a = tmp_path / "repos" / "github.com" / "org" / "repo-a.worktrees" / "pr-1-quality"
        wt_a.mkdir(parents=True, exist_ok=True)
        (wt_a / "auth.py").write_text("def check(): pass\n")

        fs_shell = _capture_branch_fs_shell(monkeypatch, str(wt_a))

        resolved = fs_shell.validate_path("auth.py")
        assert resolved == (wt_a / "auth.py").resolve(), f"bare filename must resolve into the worktree, got {resolved}"
        # And explicitly NOT into the runtime dir.
        assert "repos" in resolved.parts, "bare filename must not escape to the runtime dir"
