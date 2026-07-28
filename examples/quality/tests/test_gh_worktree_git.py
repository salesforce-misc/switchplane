"""Integration-ish tests for quality.gh worktree operations using real git.

These tests use git init in tmp_path and real git commands to verify worktree
operations that can't be validated via argv-recording alone. Skips if git is
not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from switchplane import Shell

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not on PATH"
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a real git repo with a commit and a PR-like ref.

    Returns (repo_path, pr_sha) where pr_sha is the head of refs/pull/1/head.
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # Initialize repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create a file and commit
    (repo_path / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "file.txt"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create a second commit to stand in for a PR head
    (repo_path / "file.txt").write_text("pr content")
    subprocess.run(["git", "add", "file.txt"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "PR commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Get the PR head SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    pr_sha = result.stdout.strip()

    # Create refs/pull/1/head pointing to this commit
    subprocess.run(
        ["git", "update-ref", "refs/pull/1/head", pr_sha],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Add origin remote pointing to self so fetch operations work
    subprocess.run(
        ["git", "remote", "add", "origin", str(repo_path)],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    return repo_path, pr_sha


async def test_create_pr_worktree_detaches_at_pr_head(git_repo, tmp_path):
    """create_pr_worktree creates a detached worktree at the PR head SHA."""
    from quality.gh import create_pr_worktree

    repo_path, pr_sha = git_repo
    shell = Shell(allowed_paths=[tmp_path], allowed_commands=["git"], timeout=10.0)

    worktree_path, returned_sha = await create_pr_worktree(
        shell, repo_path, 1, "task.123"
    )

    # Verify the worktree was created
    assert worktree_path.exists()
    assert worktree_path.is_dir()
    assert returned_sha == pr_sha

    # Verify the worktree is detached at the PR head
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == pr_sha

    # Verify detached state
    result = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=worktree_path,
        capture_output=True,
    )
    assert result.returncode != 0  # Should fail because HEAD is detached


async def test_create_pr_worktree_cleans_stale_registration(git_repo, tmp_path):
    """Prune clears stale worktree registrations so add can succeed.

    Simulates a crashed git by manually removing a worktree directory while
    leaving it registered. Then verifies that create_pr_worktree can recover.
    """
    from quality.gh import create_pr_worktree

    repo_path, pr_sha = git_repo
    shell = Shell(allowed_paths=[tmp_path], allowed_commands=["git"], timeout=10.0)

    # Create a worktree first
    worktree_path, _ = await create_pr_worktree(shell, repo_path, 1, "task.123")
    assert worktree_path.exists()

    # Simulate a crash: remove the directory but leave it registered
    shutil.rmtree(worktree_path)
    assert not worktree_path.exists()

    # Calling create_pr_worktree again should succeed (prune clears the stale entry)
    worktree_path2, sha2 = await create_pr_worktree(shell, repo_path, 1, "task.456")
    assert worktree_path2.exists()
    assert sha2 == pr_sha


async def test_remove_worktree_cleans_up(git_repo, tmp_path):
    """remove_worktree removes both the registration and the directory."""
    from quality.gh import create_pr_worktree, remove_worktree

    repo_path, _ = git_repo
    shell = Shell(allowed_paths=[tmp_path], allowed_commands=["git"], timeout=10.0)

    worktree_path, _ = await create_pr_worktree(shell, repo_path, 1, "task.123")
    assert worktree_path.exists()

    await remove_worktree(shell, repo_path, worktree_path)

    # Directory should be gone
    assert not worktree_path.exists()

    # Should not appear in worktree list
    result = subprocess.run(
        ["git", "worktree", "list"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(worktree_path) not in result.stdout


async def test_worktree_isolation(git_repo, tmp_path):
    """Changes in a worktree do not affect the main repo."""
    from quality.gh import create_pr_worktree

    repo_path, _pr_sha = git_repo
    shell = Shell(allowed_paths=[tmp_path], allowed_commands=["git"], timeout=10.0)

    worktree_path, _ = await create_pr_worktree(shell, repo_path, 1, "task.123")

    # Modify a file in the worktree
    (worktree_path / "file.txt").write_text("modified in worktree")

    # Main repo should still have the original content
    main_content = (repo_path / "file.txt").read_text()
    assert main_content == "pr content"

    # Worktree should have the modified content
    worktree_content = (worktree_path / "file.txt").read_text()
    assert worktree_content == "modified in worktree"
