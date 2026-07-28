"""Tests for quality.gh — GitHub/git plumbing without network or real repos.

Uses a fake Shell that records argv and returns canned stdout. The SSH retry
and parse_pr_url tests are SECURITY-CRITICAL: they pin the defenses against
SSRF/token-leak and path-traversal attacks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest


class FakeShell:
    """Test double for switchplane.Shell that records calls and returns canned output.

    Records every `run` and `run_ok` call's argv and kwargs. Returns stdout matched
    on argv prefixes, or raises RuntimeError if told to fail for that argv.
    """

    def __init__(self):
        self.calls: list[tuple[str, list[str], dict]] = []  # ("run" | "run_ok", argv, kwargs)
        self._responses: dict[tuple[str, ...], str] = {}
        self._failures: set[tuple[str, ...]] = set()

    def stub(self, argv_prefix: tuple[str, ...], stdout: str) -> None:
        """Stub a successful `run` call matching argv_prefix to return stdout."""
        self._responses[argv_prefix] = stdout

    def stub_failure(self, argv_prefix: tuple[str, ...]) -> None:
        """Stub a `run` call matching argv_prefix to raise RuntimeError."""
        self._failures.add(argv_prefix)

    async def run(self, cmd: list[str], **kwargs) -> str:
        self.calls.append(("run", cmd, kwargs))
        for prefix in self._failures:
            if tuple(cmd[: len(prefix)]) == prefix:
                raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        for prefix, out in self._responses.items():
            if tuple(cmd[: len(prefix)]) == prefix:
                return out
        return ""

    async def run_ok(self, cmd: list[str], **kwargs) -> bool:
        self.calls.append(("run_ok", cmd, kwargs))
        return all(tuple(cmd[: len(prefix)]) != prefix for prefix in self._failures)


@pytest.fixture
def fake_shell():
    return FakeShell()


@pytest.fixture
def no_sleep(monkeypatch):
    """Prevent real sleeping in tests — monkeypatch asyncio.sleep in the gh module."""
    from quality import gh

    sleep_spy = AsyncMock()
    monkeypatch.setattr(gh.asyncio, "sleep", sleep_spy)
    return sleep_spy


# -- parse_pr_url (SECURITY-CRITICAL) ----------------------------------------


def test_parse_pr_url_happy_path():
    """Valid PR URL is parsed into (host/org/repo, number)."""
    from quality.gh import parse_pr_url

    repo, num = parse_pr_url("https://github.com/myorg/myrepo/pull/42", ["github.com"])
    assert repo == "github.com/myorg/myrepo"
    assert num == 42


def test_parse_pr_url_strips_query_fragment_scheme():
    """Query, fragment, and scheme are stripped before parsing."""
    from quality.gh import parse_pr_url

    repo, num = parse_pr_url(
        "https://github.com/org/repo/pull/99?w=1#issuecomment-12345",
        ["github.com"],
    )
    assert repo == "github.com/org/repo"
    assert num == 99


def test_parse_pr_url_without_scheme():
    """Scheme is optional — github.com/org/repo/pull/1 is valid."""
    from quality.gh import parse_pr_url

    repo, num = parse_pr_url("github.com/org/repo/pull/1", ["github.com"])
    assert repo == "github.com/org/repo"
    assert num == 1


def test_parse_pr_url_host_allowlist_case_insensitive():
    """Host allowlist matching is case-insensitive."""
    from quality.gh import parse_pr_url

    repo, num = parse_pr_url("GitHub.COM/org/repo/pull/1", ["github.com"])
    assert repo == "GitHub.COM/org/repo"
    assert num == 1


def test_parse_pr_url_host_not_in_allowlist():
    """A host not in the allowlist is rejected with ValueError."""
    from quality.gh import parse_pr_url

    with pytest.raises(ValueError, match=r"not in the allowed hosts"):
        parse_pr_url("evil.example.com/org/repo/pull/1", ["github.com"])


def test_parse_pr_url_missing_pull_segment():
    """parts[3] must equal "pull" exactly."""
    from quality.gh import parse_pr_url

    with pytest.raises(ValueError, match="Invalid PR URL"):
        parse_pr_url("github.com/org/repo/issues/1", ["github.com"])


def test_parse_pr_url_rejects_dot_segment():
    """A segment of exactly "." is rejected (path traversal defense)."""
    from quality.gh import parse_pr_url

    with pytest.raises(ValueError, match="Invalid PR URL segment"):
        parse_pr_url("github.com/./repo/pull/1", ["github.com"])


def test_parse_pr_url_rejects_dotdot_segment():
    """A segment of exactly ".." is rejected (path traversal defense)."""
    from quality.gh import parse_pr_url

    with pytest.raises(ValueError, match="Invalid PR URL segment"):
        parse_pr_url("github.com/../repo/pull/1", ["github.com"])


def test_parse_pr_url_rejects_invalid_chars_in_segment():
    """Segments must match ^[A-Za-z0-9._-]+$ — special chars are rejected."""
    from quality.gh import parse_pr_url

    with pytest.raises(ValueError, match="Invalid PR URL segment"):
        parse_pr_url("github.com/org/repo$$/pull/1", ["github.com"])


def test_parse_pr_url_accepts_dots_dashes_underscores():
    """Dots, dashes, and underscores in segments are valid."""
    from quality.gh import parse_pr_url

    repo, num = parse_pr_url("github.com/my-org/repo.name_123/pull/1", ["github.com"])
    assert repo == "github.com/my-org/repo.name_123"
    assert num == 1


def test_parse_pr_url_non_integer_pr_number():
    """PR number must parse as int — non-integers raise ValueError."""
    from quality.gh import parse_pr_url

    with pytest.raises(ValueError):
        parse_pr_url("github.com/org/repo/pull/abc", ["github.com"])


def test_parse_pr_url_no_allowlist_accepts_any_host():
    """When allowed_hosts is None, any host matching the charset passes."""
    from quality.gh import parse_pr_url

    repo, num = parse_pr_url("git.example.com/org/repo/pull/1", allowed_hosts=None)
    assert repo == "git.example.com/org/repo"
    assert num == 1


def test_parse_pr_url_trailing_slash_stripped():
    """Trailing slashes are stripped before parsing."""
    from quality.gh import parse_pr_url

    repo, num = parse_pr_url("github.com/org/repo/pull/1/", ["github.com"])
    assert repo == "github.com/org/repo"
    assert num == 1


# -- commentable_lines (state machine from ava) ------------------------------


def test_commentable_lines_added_and_context():
    """Added (+) and context (no prefix) lines are commentable."""
    from quality.gh import commentable_lines

    diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,4 @@\n"
        " ctx1\n"
        "-removed\n"
        "+added2\n"
        "+added3\n"
        " ctx4\n"
    )
    result = commentable_lines(diff)
    # New-side lines: 1(ctx), 2(added), 3(added), 4(ctx)
    assert result["foo.py"] == {1, 2, 3, 4}


def test_commentable_lines_removed_not_commentable():
    """Removed (-) lines do not appear in the result."""
    from quality.gh import commentable_lines

    diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,1 @@\n"
        "-removed\n"
        " kept\n"
    )
    result = commentable_lines(diff)
    # Only line 1 (context) is commentable; removed line does not contribute
    assert result["foo.py"] == {1}


def test_commentable_lines_new_file():
    """A new file (--- /dev/null) has all + lines commentable."""
    from quality.gh import commentable_lines

    diff = (
        "diff --git a/new.py b/new.py\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+first\n"
        "+second\n"
    )
    result = commentable_lines(diff)
    assert result["new.py"] == {1, 2}


def test_commentable_lines_deleted_file():
    """A deleted file (+++ /dev/null) has no commentable lines and is absent."""
    from quality.gh import commentable_lines

    diff = (
        "diff --git a/gone.py b/gone.py\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-a\n"
        "-b\n"
    )
    result = commentable_lines(diff)
    assert result == {}


def test_commentable_lines_pure_rename():
    """A pure rename (no @@ hunk) yields no entry — pins the 422 avoidance."""
    from quality.gh import commentable_lines

    diff = (
        "diff --git a/old.py b/new.py\n"
        "similarity index 100%\n"
        "rename from old.py\n"
        "rename to new.py\n"
    )
    result = commentable_lines(diff)
    assert result == {}


def test_commentable_lines_mode_change():
    """A pure mode change (no content diff) yields no entry."""
    from quality.gh import commentable_lines

    diff = "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n"
    result = commentable_lines(diff)
    assert result == {}


def test_commentable_lines_double_reset():
    """Both "diff --git" and "index" reset path/new_no state."""
    from quality.gh import commentable_lines

    diff = (
        "diff --git a/a.py b/a.py\n"
        "index abc123..def456 100644\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,1 +1,2 @@\n"
        " a\n"
        "+b\n"
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,1 +1,1 @@\n"
        " x\n"
    )
    result = commentable_lines(diff)
    assert result["a.py"] == {1, 2}
    assert result["x.py"] == {1}


def test_commentable_lines_no_newline_marker():
    """The '\\ No newline at end of file' marker is ignored."""
    from quality.gh import commentable_lines

    diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "\\ No newline at end of file\n"
        "+new\n"
        "\\ No newline at end of file\n"
    )
    result = commentable_lines(diff)
    assert result["foo.py"] == {1}


def test_commentable_lines_strips_b_prefix():
    """The +++ b/path prefix is stripped."""
    from quality.gh import commentable_lines

    diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,1 +1,1 @@\n"
        " x\n"
    )
    result = commentable_lines(diff)
    assert "foo.py" in result


def test_commentable_lines_empty_diff():
    """An empty diff returns an empty dict."""
    from quality.gh import commentable_lines

    assert commentable_lines("") == {}


# -- SSH auth retry (SECURITY-CRITICAL: macOS SSH key eviction) -------------


async def test_ssh_retry_succeeds_first_attempt(fake_shell, no_sleep):
    """When the command succeeds, returns immediately without retry."""
    from quality.gh import run_git

    fake_shell.stub(("git", "fetch"), "ok")
    out = await run_git(fake_shell, ["git", "fetch"])
    assert out == "ok"
    assert len(fake_shell.calls) == 1
    no_sleep.assert_not_awaited()


async def test_ssh_retry_non_ssh_error_raises_immediately(fake_shell, no_sleep):
    """Non-SSH errors raise on first attempt without retry."""
    from quality.gh import run_git

    fake_shell.stub_failure(("git", "pull"))
    with pytest.raises(RuntimeError):
        await run_git(fake_shell, ["git", "pull"])
    # Only one call — no retry
    assert len(fake_shell.calls) == 1
    no_sleep.assert_not_awaited()


async def test_ssh_retry_recovers_after_transient_failure(fake_shell, no_sleep):
    """An SSH error triggers retry with delays (5, 15, 30)."""
    from quality.gh import run_git

    call_count = {"n": 0}

    async def failing_once(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Permission denied (publickey)")
        return "ok"

    fake_shell.run = failing_once

    out = await run_git(fake_shell, ["git", "fetch"])
    assert out == "ok"
    assert call_count["n"] == 2
    # First retry uses delay[0] = 5
    assert no_sleep.call_count == 1
    assert no_sleep.call_args.args == (5,)


async def test_ssh_retry_exhausts_and_raises_friendly_error(fake_shell, no_sleep):
    """After 3 SSH retries, raises RuntimeError naming ssh-add fix."""
    from quality.gh import run_git

    call_count = {"n": 0}

    async def always_ssh_fail(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError("Permission denied (publickey)")

    fake_shell.run = always_ssh_fail

    with pytest.raises(RuntimeError, match=r"ssh-add --apple-use-keychain"):
        await run_git(fake_shell, ["git", "fetch"])

    # 1 first attempt + 3 retries (delays = [5, 15, 30])
    assert call_count["n"] == 4
    # 3 sleeps between 4 attempts
    assert no_sleep.call_count == 3


async def test_ssh_retry_delays_are_5_15_30(fake_shell, no_sleep):
    """Pin the exact retry delays: (5, 15, 30)."""
    from quality.gh import run_git

    call_count = {"n": 0}

    async def fail_three_times(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 3:
            raise RuntimeError("Permission denied (publickey)")
        return "ok"

    fake_shell.run = fail_three_times

    await run_git(fake_shell, ["git", "fetch"])
    # 3 sleeps
    assert no_sleep.call_count == 3
    calls = [call.args[0] for call in no_sleep.call_args_list]
    assert calls == [5, 15, 30]


async def test_is_ssh_auth_error_matches_publickey():
    """_is_ssh_auth_error returns True only for 'Permission denied (publickey)'."""
    from quality.gh import _is_ssh_auth_error

    assert _is_ssh_auth_error(RuntimeError("Permission denied (publickey)"))
    assert _is_ssh_auth_error(RuntimeError("foo Permission denied (publickey) bar"))
    assert not _is_ssh_auth_error(RuntimeError("merge conflict"))
    assert not _is_ssh_auth_error(RuntimeError("Permission denied"))


# -- create_pr_worktree (argv order pins prune-before-add) ------------------


async def test_create_pr_worktree_prunes_before_add(fake_shell):
    """Pin argv order: git worktree prune MUST run before worktree add.

    A crashed git can leave the path registered while the directory is gone.
    Then exists() is False, cleanup is skipped, yet worktree add still fails.
    The unconditional prune is what fixes that — and only argv-order assertions
    pin the fix.
    """
    from quality.gh import create_pr_worktree

    fake_shell.stub(("git", "fetch"), "")
    fake_shell.stub(("git", "rev-parse"), "abc123")

    await create_pr_worktree(fake_shell, Path("/repo"), 42, "task.123")

    # Find the prune and add calls
    prune_idx = None
    add_idx = None
    for i, (method, cmd, _kwargs) in enumerate(fake_shell.calls):
        if method == "run_ok" and cmd[:3] == ["git", "worktree", "prune"]:
            prune_idx = i
        if method == "run" and cmd[:3] == ["git", "worktree", "add"]:
            add_idx = i

    assert prune_idx is not None, "git worktree prune was never called"
    assert add_idx is not None, "git worktree add was never called"
    assert prune_idx < add_idx, "prune must come before add"


async def test_create_pr_worktree_fetches_pr_head(fake_shell):
    """Fetches refs/pull/N/head and returns the head SHA."""
    from quality.gh import create_pr_worktree

    fake_shell.stub(("git", "fetch"), "")
    fake_shell.stub(("git", "rev-parse"), "deadbeef")

    _worktree_path, sha = await create_pr_worktree(
        fake_shell, Path("/repo"), 99, "task.456"
    )

    assert sha == "deadbeef"
    # Check that fetch was called for the PR ref
    fetch_calls = [c for m, c, _kw in fake_shell.calls if m == "run" and c[0] == "git" and c[1] == "fetch"]
    assert any("refs/pull/99/head" in " ".join(c) for c in fetch_calls)


async def test_create_pr_worktree_detaches_at_head_sha(fake_shell):
    """The worktree is created with --detach at the resolved SHA."""
    from quality.gh import create_pr_worktree

    fake_shell.stub(("git", "fetch"), "")
    fake_shell.stub(("git", "rev-parse"), "abc123")

    worktree_path, _sha = await create_pr_worktree(fake_shell, Path("/repo"), 1, "task.1")

    # Find the worktree add call
    add_calls = [c for m, c, _kw in fake_shell.calls if m == "run" and c[:3] == ["git", "worktree", "add"]]
    assert len(add_calls) == 1
    cmd = add_calls[0]
    assert "--detach" in cmd
    assert "abc123" in cmd
    # Assert task_id is in the returned path
    assert "task.1" in str(worktree_path)


async def test_create_pr_worktree_task_id_isolates_worktrees(fake_shell):
    """Different task_ids yield different worktree paths.

    This pins the requirement that task_id appears in the path, so leaked
    worktrees are never reused and show up as disk-fill instead.
    """
    from quality.gh import create_pr_worktree

    fake_shell.stub(("git", "fetch"), "")
    fake_shell.stub(("git", "rev-parse"), "abc123")

    path1, _ = await create_pr_worktree(fake_shell, Path("/repo"), 1, "task.100")
    path2, _ = await create_pr_worktree(fake_shell, Path("/repo"), 1, "task.200")

    assert path1 != path2
    assert "task.100" in str(path1)
    assert "task.200" in str(path2)


# -- clone_or_update_repo ----------------------------------------------------


async def test_clone_or_update_clones_when_missing(tmp_path, fake_shell):
    """When repo_path does not exist, clone via gh repo clone."""
    from quality.gh import clone_or_update_repo

    fake_shell.stub(("gh", "repo", "clone"), "")

    path = await clone_or_update_repo(fake_shell, "github.com/o/r", tmp_path)

    assert path == tmp_path / "github.com" / "o" / "r"
    clone_calls = [c for m, c, _kw in fake_shell.calls if m == "run" and c[:3] == ["gh", "repo", "clone"]]
    assert len(clone_calls) == 1
    assert "o/r" in clone_calls[0]


async def test_clone_or_update_fetches_when_present(tmp_path, fake_shell):
    """When repo_path exists, fetch --all --prune + checkout + pull."""
    from quality.gh import clone_or_update_repo

    repo_dir = tmp_path / "github.com" / "o" / "r"
    repo_dir.mkdir(parents=True)

    fake_shell.stub(("git", "fetch"), "")
    fake_shell.stub(("git", "symbolic-ref"), "refs/remotes/origin/main")
    fake_shell.stub(("git", "checkout"), "")
    fake_shell.stub(("git", "pull"), "")

    await clone_or_update_repo(fake_shell, "github.com/o/r", tmp_path)

    fetch_calls = [c for m, c, _kw in fake_shell.calls if m == "run" and c[:2] == ["git", "fetch"]]
    assert len(fetch_calls) == 1
    assert "--all" in fetch_calls[0]
    assert "--prune" in fetch_calls[0]


async def test_get_default_branch_symbolic_ref(fake_shell, tmp_path):
    """get_default_branch uses symbolic-ref and strips refs/remotes/origin/ prefix."""
    from quality.gh import get_default_branch

    fake_shell.stub(("git", "symbolic-ref"), "refs/remotes/origin/develop")

    branch = await get_default_branch(fake_shell, tmp_path)
    assert branch == "develop"


async def test_get_default_branch_fallback_to_main(fake_shell, tmp_path):
    """When symbolic-ref fails, falls back to "main"."""
    from quality.gh import get_default_branch

    async def run_ok_false(*args, **kwargs):
        return False

    fake_shell.run_ok = run_ok_false

    branch = await get_default_branch(fake_shell, tmp_path)
    assert branch == "main"


# -- remove_worktree ---------------------------------------------------------


async def test_remove_worktree_prunes_removes_and_rmtrees(fake_shell, tmp_path):
    """remove_worktree runs prune, remove --force, then shutil.rmtree backstop."""
    from quality.gh import remove_worktree

    repo = tmp_path / "repo"
    worktree = tmp_path / "repo.worktrees" / "task"
    worktree.mkdir(parents=True)

    await remove_worktree(fake_shell, repo, worktree)

    prune_calls = [c for m, c, _kw in fake_shell.calls if m == "run_ok" and c[:3] == ["git", "worktree", "prune"]]
    remove_calls = [c for m, c, _kw in fake_shell.calls if m == "run_ok" and c[:3] == ["git", "worktree", "remove"]]
    assert len(prune_calls) == 1
    assert len(remove_calls) == 1
    assert "--force" in remove_calls[0]
    # Assert rmtree backstop: directory must be gone
    assert not worktree.exists()


# -- GitHub API wrappers (host via _gh_env, -R org/repo) --------------------


async def test_get_pr_diff_uses_gh_env_and_dash_r(fake_shell, tmp_path):
    """get_pr_diff passes -R org/repo and sets GH_HOST from the validated host.

    SECURITY-CRITICAL: The host must come from parse_pr_url validation, never
    re-derived from git remote get-url (which would launder an unvalidated value).
    """
    from quality.gh import get_pr_diff

    fake_shell.stub(("gh", "pr", "diff"), "diff content")

    diff = await get_pr_diff(fake_shell, "github.com/org/repo", 7)

    assert diff == "diff content"
    # Find the gh pr diff call
    diff_calls = [(c, kw) for m, c, kw in fake_shell.calls if m == "run" and c[:3] == ["gh", "pr", "diff"]]
    assert len(diff_calls) == 1
    cmd, kwargs = diff_calls[0]
    # Assert -R org/repo is passed
    assert "-R" in cmd
    r_idx = cmd.index("-R")
    assert cmd[r_idx + 1] == "org/repo"
    # Assert GH_HOST is set in env to the validated host
    assert "env" in kwargs
    assert kwargs["env"]["GH_HOST"] == "github.com"
    # Assert the PR number is in argv
    assert "7" in cmd


async def test_get_pr_diff_host_from_repo_arg(fake_shell):
    """GH_HOST must derive from the repo arg, not hardcoded github.com.

    SECURITY-CRITICAL: An implementation that hardcodes github.com would pass
    the previous test but fail this one with a non-github.com host.
    """
    from quality.gh import get_pr_diff

    fake_shell.stub(("gh", "pr", "diff"), "diff")

    await get_pr_diff(fake_shell, "git.internal.example.com/org/repo", 1)

    diff_calls = [(c, kw) for m, c, kw in fake_shell.calls if m == "run" and c[:3] == ["gh", "pr", "diff"]]
    assert len(diff_calls) == 1
    _cmd, kwargs = diff_calls[0]
    assert kwargs["env"]["GH_HOST"] == "git.internal.example.com"


async def test_get_pr_head_sha_strips_whitespace(fake_shell, tmp_path):
    """get_pr_head_sha strips leading/trailing whitespace and sets GH_HOST."""
    from quality.gh import get_pr_head_sha

    fake_shell.stub(("gh", "pr", "view"), "  deadbeef\n")

    sha = await get_pr_head_sha(fake_shell, "github.com/org/repo", 7)
    assert sha == "deadbeef"
    # Assert GH_HOST and -R are passed
    view_calls = [(c, kw) for m, c, kw in fake_shell.calls if m == "run" and c[:3] == ["gh", "pr", "view"]]
    assert len(view_calls) == 1
    _cmd, kwargs = view_calls[0]
    assert kwargs["env"]["GH_HOST"] == "github.com"


async def test_get_pr_author_returns_login(fake_shell, tmp_path):
    """get_pr_author returns the author's login and sets GH_HOST."""
    from quality.gh import get_pr_author

    fake_shell.stub(("gh", "pr", "view"), "alice")

    author = await get_pr_author(fake_shell, "github.com/org/repo", 7)
    assert author == "alice"
    # Assert GH_HOST is passed
    view_calls = [(c, kw) for m, c, kw in fake_shell.calls if m == "run" and c[:3] == ["gh", "pr", "view"]]
    _cmd, kwargs = view_calls[0]
    assert kwargs["env"]["GH_HOST"] == "github.com"


async def test_get_authenticated_user_returns_login(fake_shell, tmp_path):
    """get_authenticated_user returns the authenticated gh user's login and sets GH_HOST."""
    from quality.gh import get_authenticated_user

    fake_shell.stub(("gh", "api", "user"), "bob")

    user = await get_authenticated_user(fake_shell, "github.com/org/repo")
    assert user == "bob"
    # Assert GH_HOST is passed
    api_calls = [(c, kw) for m, c, kw in fake_shell.calls if m == "run" and c[:3] == ["gh", "api", "user"]]
    _cmd, kwargs = api_calls[0]
    assert kwargs["env"]["GH_HOST"] == "github.com"


async def test_list_review_comments_returns_list(fake_shell, tmp_path):
    """list_review_comments returns a list of comment dicts and sets GH_HOST."""
    import json

    from quality.gh import list_review_comments

    fake_shell.stub(("gh", "api"), json.dumps([{"id": 1, "body": "test"}]))

    comments = await list_review_comments(fake_shell, "github.com/org/repo", 7)
    assert comments == [{"id": 1, "body": "test"}]
    # Assert GH_HOST is passed
    api_calls = [(c, kw) for m, c, kw in fake_shell.calls if m == "run" and c[0] == "gh" and c[1] == "api"]
    _cmd, kwargs = api_calls[0]
    assert kwargs["env"]["GH_HOST"] == "github.com"


async def test_submit_pr_review_posts_review(fake_shell, tmp_path):
    """submit_pr_review posts a review with event, body, and GH_HOST."""
    from quality.gh import submit_pr_review

    fake_shell.stub(("gh", "api"), "")

    await submit_pr_review(fake_shell, "github.com/org/repo", 7, "APPROVE", "lgtm")

    api_calls = [(c, kw) for m, c, kw in fake_shell.calls if m == "run" and c[0] == "gh" and c[1] == "api"]
    assert len(api_calls) == 1
    cmd, kwargs = api_calls[0]
    # Verify event and body were passed
    assert "event=APPROVE" in " ".join(cmd)
    assert "body=lgtm" in " ".join(cmd)
    # Assert GH_HOST is passed
    assert kwargs["env"]["GH_HOST"] == "github.com"


async def test_create_pr_review_comment_posts_comment(fake_shell, tmp_path):
    """create_pr_review_comment posts a comment with path, line, commit_id, and GH_HOST."""
    from quality.gh import create_pr_review_comment

    fake_shell.stub(("gh", "api"), "")

    await create_pr_review_comment(
        fake_shell, "github.com/org/repo", 7, "comment", "file.py", 10, "abc123"
    )

    api_calls = [(c, kw) for m, c, kw in fake_shell.calls if m == "run" and c[0] == "gh" and c[1] == "api"]
    assert len(api_calls) == 1
    cmd, kwargs = api_calls[0]
    # Verify path, line, commit_id were passed
    assert "path=file.py" in " ".join(cmd)
    assert "line=10" in " ".join(cmd)
    assert "commit_id=abc123" in " ".join(cmd)
    # Assert GH_HOST is passed
    assert kwargs["env"]["GH_HOST"] == "github.com"


# -- check_dependencies ------------------------------------------------------


async def test_check_dependencies_raises_when_both_missing(monkeypatch):
    """check_dependencies raises RuntimeError naming both git and gh when both absent."""
    from quality import gh

    monkeypatch.setattr(gh.shutil, "which", lambda cmd: None)

    with pytest.raises(RuntimeError) as exc_info:
        await gh.check_dependencies()
    # Assert both "git" and "gh" appear in the message
    msg = str(exc_info.value)
    assert "git" in msg
    assert "gh" in msg


async def test_check_dependencies_raises_when_only_gh_missing(monkeypatch):
    """check_dependencies raises RuntimeError naming gh when only gh is absent."""
    from quality import gh

    def fake_which(cmd):
        return "/usr/bin/git" if cmd == "git" else None

    monkeypatch.setattr(gh.shutil, "which", fake_which)

    with pytest.raises(RuntimeError) as exc_info:
        await gh.check_dependencies()
    msg = str(exc_info.value)
    assert "gh" in msg


async def test_check_dependencies_passes_when_present(monkeypatch):
    """check_dependencies returns None when both git and gh are on PATH."""
    from quality import gh

    monkeypatch.setattr(gh.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    await gh.check_dependencies()  # Should not raise
