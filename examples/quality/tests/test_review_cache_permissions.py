"""Adversarial test for repo-cache directory permissions.

The artifact and baseline paths were hardened with ``_mkdir_private`` (0o700 on every
component) because ``mkdir(mode=)`` only protects the leaf. The **repo cache** never
got the same treatment:

    cache_root = runtime_dir / "repos"
    cache_root.mkdir(parents=True, exist_ok=True)     # review.py:242 — no mode=
    ...
    repo_path.parent.mkdir(parents=True, exist_ok=True)  # gh.py:356 — no mode=

Both fall back to the process umask, so on a default umask of 022 the whole cache tree
lands 0o755. That tree holds a full clone of every reviewed repository — including
private ones — plus a per-PR worktree checkout, world-readable under the user's home.

This is the same defect class as the already-fixed artifact-permission bug, one
directory over. The suite misses it because ``test_review_local_mode`` asserts modes
only on ``runtime_dir/reviews``, and ``test_memory`` only on ``runtime_dir/state``.

All imports are function-scoped to match the suite convention (see conftest.py).
"""

from __future__ import annotations

import pytest


class TestRepoCachePermissions:
    """Cloned repo content must not be world-readable."""

    @pytest.mark.asyncio
    async def test_setup_creates_private_repo_cache(self, monkeypatch, tmp_path):
        """The repos/ cache the setup node creates must be 0o700, not umask-default.

        Driven through the real ``setup`` node so the assertion covers the directory
        production actually creates, at the mode production actually leaves it in.
        ``clone_or_update_repo`` is stubbed to mkdir the clone path the way gh.py:356
        does (``parents=True, exist_ok=True``, no ``mode=``), so the intermediate
        host/org components are created the same way the real call creates them.
        """
        import stat

        from quality.agents.pr.tasks.review import ReviewState, setup

        from quality import gh as gh_module

        created: dict[str, object] = {}

        async def fake_clone_or_update_repo(shell, repo, cache_root):
            # Mirrors gh.py:353-356: mkdir the clone's parent with no explicit mode.
            repo_path = cache_root / repo
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            repo_path.mkdir(parents=True, exist_ok=True)
            created["repo_path"] = repo_path
            return repo_path

        async def fake_create_pr_worktree(shell, repo_path, pr_number, task_id):
            worktree = tmp_path / "worktrees" / str(pr_number)
            worktree.mkdir(parents=True, exist_ok=True)
            return worktree, "stub-head-sha"

        async def fake_get_pr_diff(shell, repo, pr_number):
            return "diff"

        async def fake_get_pr_author(shell, repo, pr_number):
            return "pr-author"

        async def fake_get_authenticated_user(shell, repo):
            return "authed-user"

        monkeypatch.setattr(gh_module, "clone_or_update_repo", fake_clone_or_update_repo, raising=True)
        monkeypatch.setattr(gh_module, "create_pr_worktree", fake_create_pr_worktree, raising=True)
        monkeypatch.setattr(gh_module, "get_pr_diff", fake_get_pr_diff, raising=True)
        monkeypatch.setattr(gh_module, "get_pr_author", fake_get_pr_author, raising=True)
        monkeypatch.setattr(gh_module, "get_authenticated_user", fake_get_authenticated_user, raising=True)

        from conftest import FakeAgentContext, FakeShell

        ctx = FakeAgentContext(
            config={"llm": {"providers": {"alpha": {"api_key": "k", "model": "model-a"}}}},
            runtime_dir_path=tmp_path,
        )

        state = ReviewState(repo="github.com/org/private-repo", number=42, local=True)

        result = await setup(ctx, FakeShell(), state)

        assert not result.get("error"), f"setup must succeed: {result.get('error')}"

        cache_root = tmp_path / "repos"
        assert cache_root.is_dir(), "setup must create the repos/ cache"

        offenders = []
        for d in (cache_root, *sorted(p for p in cache_root.rglob("*") if p.is_dir())):
            mode = stat.S_IMODE(d.stat().st_mode)
            if mode & 0o077:
                offenders.append((str(d.relative_to(tmp_path)), oct(mode)))

        assert not offenders, (
            "Every directory in the repo cache must be private (0o700): it holds full "
            "clones of private repositories and the per-PR worktree checkout. "
            f"Group/other-readable: {offenders}. "
            "review.py:242 and gh.py:356 both mkdir without mode=, so they inherit the "
            "umask — the same defect already fixed for reviews/ and state/ via "
            "_mkdir_private."
        )
