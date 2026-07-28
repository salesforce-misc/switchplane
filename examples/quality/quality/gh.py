"""GitHub and git plumbing for PR review operations.

This module provides security-hardened wrappers for GitHub CLI and git operations.
It implements defenses against SSRF/token-leak (via parse_pr_url host validation)
and path-traversal attacks (via segment validation), along with macOS SSH key
eviction recovery.

Design notes:
- GH_HOST must always derive from the validated repo argument, never re-derived
  from git remote get-url (which would launder an unvalidated value).
- SSH retry is macOS-specific: the SSH agent can evict keys, causing transient
  "Permission denied (publickey)" errors. Retries with delays (5, 15, 30) seconds.
- Worktree operations run prune unconditionally before add to clear stale
  registrations left by crashed git processes.
"""

from __future__ import annotations

import asyncio  # noqa: F401 — imported for test monkeypatching
import re
import shutil
from pathlib import Path

from quality._concurrency import retry_async


def parse_pr_url(url: str, allowed_hosts: list[str] | None) -> tuple[str, int]:
    """Parse and validate a GitHub PR URL, defending against SSRF and path traversal.

    SECURITY-CRITICAL: The returned host becomes GH_HOST for gh CLI calls made with
    the user's ambient credentials (SSRF/token-leak risk), and the segments become
    filesystem path components (traversal risk).

    Validation rules:
    - Strip scheme, query, fragment, and trailing slashes
    - Require parts[3] == "pull" exactly
    - Validate every segment against ^[A-Za-z0-9._-]+$
    - Explicitly reject segments that are "." or ".."
    - Host allowlist matching is case-insensitive, but original casing is returned
    - allowed_hosts=None means any charset-valid host is accepted

    Args:
        url: PR URL (e.g. "https://github.com/org/repo/pull/42")
        allowed_hosts: Case-insensitive host allowlist, or None to allow any valid host

    Returns:
        (repo_with_host, pr_number) e.g. ("github.com/org/repo", 42)

    Raises:
        ValueError: If URL is malformed, host not in allowlist, or segments fail validation
    """
    # Strip scheme
    without_scheme = re.sub(r"^https?://", "", url)

    # Strip query and fragment
    without_query = without_scheme.split("?")[0].split("#")[0]

    # Strip trailing slashes
    normalized = without_query.rstrip("/")

    # Split into segments
    parts = normalized.split("/")

    if len(parts) < 5:
        raise ValueError("Invalid PR URL")

    host, org, repo, pull_keyword, pr_num_str = parts[0], parts[1], parts[2], parts[3], parts[4]

    # Require "pull" exactly at parts[3]
    if pull_keyword != "pull":
        raise ValueError("Invalid PR URL")

    # Validate host against allowlist (case-insensitive match)
    if allowed_hosts is not None and not any(host.lower() == allowed.lower() for allowed in allowed_hosts):
        raise ValueError(f"Host {host} not in the allowed hosts")

    # Validate all segments (host, org, repo, pr_num_str) against charset and reject dot/dotdot
    segment_pattern = re.compile(r"^[A-Za-z0-9._-]+$")
    for segment in [host, org, repo, pr_num_str]:
        if segment in (".", ".."):
            raise ValueError("Invalid PR URL segment")
        if not segment_pattern.match(segment):
            raise ValueError("Invalid PR URL segment")

    # Parse PR number
    try:
        pr_number = int(pr_num_str)
    except ValueError as exc:
        raise ValueError(f"Invalid PR number: {pr_num_str}") from exc

    return f"{host}/{org}/{repo}", pr_number


def commentable_lines(diff: str) -> dict[str, set[int]]:
    """Parse a unified diff and return commentable line numbers per file.

    This is a diff state machine that tracks new-side line numbers. Only added (+)
    and context (space-prefix) lines are commentable — removed (-) lines don't
    advance the new-side counter. GitHub returns 422 for comments on non-RIGHT-side
    lines, which is why this exists.

    Special cases that yield NO entry (to avoid 422):
    - Pure renames (no @@ hunk)
    - Mode changes (no content diff)
    - Deleted files (+++ /dev/null)

    Both "diff --git" and "index" lines reset per-file state.

    Args:
        diff: Unified diff text from git or gh pr diff

    Returns:
        Dict mapping file paths to sets of commentable line numbers
    """
    result: dict[str, set[int]] = {}
    current_path: str | None = None
    new_line_no = 0
    in_deleted_file = False

    for line in diff.splitlines():
        # Reset state on diff --git or index
        if line.startswith("diff --git") or line.startswith("index "):
            current_path = None
            new_line_no = 0
            in_deleted_file = False
            continue

        # Capture path from +++ b/...
        if line.startswith("+++ "):
            path = line[4:]  # Strip "+++ "
            if path == "/dev/null":
                in_deleted_file = True
                current_path = None
            else:
                # Strip "b/" prefix if present
                if path.startswith("b/"):
                    path = path[2:]
                current_path = path
                in_deleted_file = False
            continue

        # Skip if we're in a deleted file or haven't seen +++ yet
        if in_deleted_file or current_path is None:
            continue

        # Hunk header resets new-side line counter
        if line.startswith("@@"):
            # Parse @@ -old_start,old_count +new_start,new_count @@
            match = re.match(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@", line)
            if match:
                new_line_no = int(match.group(1))
            continue

        # Ignore "\ No newline at end of file" marker
        if line.startswith("\\"):
            continue

        # Process diff content lines
        if line.startswith("+"):
            # Added line
            if current_path not in result:
                result[current_path] = set()
            result[current_path].add(new_line_no)
            new_line_no += 1
        elif line.startswith("-"):
            # Removed line — does not advance new-side counter
            pass
        elif line.startswith(" "):
            # Context line
            if current_path not in result:
                result[current_path] = set()
            result[current_path].add(new_line_no)
            new_line_no += 1

    return result


def _is_ssh_auth_error(exc: Exception) -> bool:
    """Check if an exception is a transient SSH key eviction error.

    macOS can evict SSH keys from the agent, causing transient auth failures.
    We only retry on "Permission denied (publickey)" — other errors raise immediately.

    Args:
        exc: Exception to inspect

    Returns:
        True if the error is "Permission denied (publickey)"
    """
    return isinstance(exc, RuntimeError) and "Permission denied (publickey)" in str(exc)


async def run_git(shell, cmd: list[str], cwd: Path | None = None) -> str:
    """Run a git command with SSH retry on macOS key eviction.

    macOS evicts SSH keys from the agent, causing transient "Permission denied (publickey)"
    errors. This wrapper retries with delays (5, 15, 30) seconds on SSH errors only.
    Non-SSH errors raise immediately with no retry.

    On exhaustion (1 initial + 3 retries = 4 total attempts), raises a friendly
    RuntimeError naming the ssh-add fix.

    **CRITICAL:** Always pass an explicit `cwd` for git commands operating on a specific
    repository. Without it, git resolves paths against the process's ambient working
    directory, which can cause destructive operations (worktree add/remove, checkout)
    to silently target the developer's own repository instead of the intended worktree.

    Args:
        shell: switchplane.Shell instance
        cmd: Command argv
        cwd: Working directory (pass explicitly for all repo-scoped operations)

    Returns:
        Command stdout

    Raises:
        RuntimeError: On non-SSH errors (immediately) or SSH retry exhaustion (friendly msg)
    """

    async def attempt():
        return await shell.run(cmd, cwd=cwd)

    try:
        return await retry_async(
            attempt,
            should_retry=_is_ssh_auth_error,
            delays=[5, 15, 30],
        )
    except RuntimeError as exc:
        if _is_ssh_auth_error(exc):
            raise RuntimeError(
                "SSH authentication failed after retries. "
                "Run: ssh-add --apple-use-keychain"
            ) from exc
        raise


def _gh_env(host: str) -> dict[str, str]:
    """Build environment dict for gh CLI calls with validated GH_HOST.

    SECURITY-CRITICAL: The host must come from parse_pr_url validation,
    never re-derived from git remote get-url.

    Args:
        host: Validated host from parse_pr_url

    Returns:
        Environment dict with GH_HOST set
    """
    return {"GH_HOST": host}


async def create_pr_worktree(
    shell, repo_path: Path, pr_number: int, task_id: str
) -> tuple[Path, str]:
    """Create a detached git worktree at the PR head SHA.

    CRITICAL: Runs `git worktree prune` BEFORE `git worktree add` unconditionally.
    A crashed git can leave a path registered while the directory is gone; then
    exists() is False so cleanup is skipped, yet add still fails. The unconditional
    prune fixes that.

    The worktree path embeds the task_id so a leaked worktree is never silently reused.

    Args:
        shell: switchplane.Shell instance
        repo_path: Path to the main git repository
        pr_number: PR number to fetch
        task_id: Task ID for worktree isolation

    Returns:
        (worktree_path, head_sha) tuple
    """
    # Fetch the PR ref
    await run_git(
        shell,
        ["git", "fetch", "origin", f"refs/pull/{pr_number}/head"],
        cwd=repo_path,
    )

    # Resolve the PR head SHA
    sha_output = await run_git(
        shell,
        ["git", "rev-parse", "FETCH_HEAD"],
        cwd=repo_path,
    )
    head_sha = sha_output.strip()

    # Build worktree path with task_id for isolation
    worktree_path = repo_path.parent / f"{repo_path.name}.worktrees" / f"pr-{pr_number}-{task_id}"

    # Prune stale worktree registrations BEFORE attempting add
    await shell.run_ok(["git", "worktree", "prune"], cwd=repo_path)

    # Create detached worktree at the PR head SHA
    await run_git(
        shell,
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(worktree_path),
            head_sha,
        ],
        cwd=repo_path,
    )

    return worktree_path, head_sha


async def remove_worktree(shell, repo_path: Path, worktree_path: Path) -> None:
    """Remove a git worktree: prune, remove --force, then shutil.rmtree backstop.

    Args:
        shell: switchplane.Shell instance
        repo_path: Path to the main git repository
        worktree_path: Path to the worktree to remove
    """
    # Prune first
    await shell.run_ok(["git", "worktree", "prune"], cwd=repo_path)

    # Remove the worktree (--force allows removal even if dirty)
    await shell.run_ok(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        cwd=repo_path,
    )

    # Backstop: ensure directory is actually gone
    if worktree_path.exists():
        shutil.rmtree(worktree_path)


async def clone_or_update_repo(shell, repo: str, cache_root: Path) -> Path:
    """Clone a repository if missing, or fetch + pull if present.

    Args:
        shell: switchplane.Shell instance
        repo: Repository identifier (e.g. "github.com/org/repo")
        cache_root: Root directory for cached repos

    Returns:
        Path to the local repository
    """
    # Build repo path: cache_root / host / org / repo
    repo_path = cache_root / repo

    if not repo_path.exists():
        # Clone via gh repo clone
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        host, rest = repo.split("/", 1)  # Split "github.com/org/repo" -> ("github.com", "org/repo")
        await shell.run(
            ["gh", "repo", "clone", rest, str(repo_path)],
            env=_gh_env(host),
        )
    else:
        # Fetch all + prune, checkout default branch, pull
        await run_git(shell, ["git", "fetch", "--all", "--prune"], cwd=repo_path)

        default_branch = await get_default_branch(shell, repo_path)
        await run_git(shell, ["git", "checkout", default_branch], cwd=repo_path)
        await run_git(shell, ["git", "pull"], cwd=repo_path)

    return repo_path


async def get_default_branch(shell, repo_path: Path) -> str:
    """Get the default branch name, falling back to "main" if symbolic-ref fails.

    Args:
        shell: switchplane.Shell instance
        repo_path: Path to the git repository

    Returns:
        Default branch name (e.g. "main", "develop")
    """
    success = await shell.run_ok(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=repo_path,
    )

    if success:
        output = await shell.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=repo_path,
        )
        # Strip "refs/remotes/origin/" prefix
        full_ref = output.strip()
        if full_ref.startswith("refs/remotes/origin/"):
            return full_ref[len("refs/remotes/origin/") :]
        return full_ref

    # Fallback to main
    return "main"


async def get_pr_diff(shell, repo: str, pr_number: int) -> str:
    """Get the unified diff for a PR via gh pr diff.

    Args:
        shell: switchplane.Shell instance
        repo: Repository identifier (e.g. "github.com/org/repo")
        pr_number: PR number

    Returns:
        Unified diff text
    """
    host, rest = repo.split("/", 1)
    output = await shell.run(
        ["gh", "pr", "diff", str(pr_number), "-R", rest],
        env=_gh_env(host),
    )
    return output


async def get_pr_head_sha(shell, repo: str, pr_number: int) -> str:
    """Get the head SHA of a PR via gh pr view.

    Args:
        shell: switchplane.Shell instance
        repo: Repository identifier (e.g. "github.com/org/repo")
        pr_number: PR number

    Returns:
        Head SHA (stripped of whitespace)
    """
    host, rest = repo.split("/", 1)
    output = await shell.run(
        ["gh", "pr", "view", str(pr_number), "-R", rest, "--json", "headRefOid", "-q", ".headRefOid"],
        env=_gh_env(host),
    )
    return output.strip()


async def get_pr_author(shell, repo: str, pr_number: int) -> str:
    """Get the author login of a PR via gh pr view.

    Args:
        shell: switchplane.Shell instance
        repo: Repository identifier (e.g. "github.com/org/repo")
        pr_number: PR number

    Returns:
        Author login
    """
    host, rest = repo.split("/", 1)
    output = await shell.run(
        ["gh", "pr", "view", str(pr_number), "-R", rest, "--json", "author", "-q", ".author.login"],
        env=_gh_env(host),
    )
    return output.strip()


async def get_authenticated_user(shell, repo: str) -> str:
    """Get the authenticated gh user's login via gh api user.

    Args:
        shell: switchplane.Shell instance
        repo: Repository identifier (e.g. "github.com/org/repo")

    Returns:
        Authenticated user login
    """
    host, _rest = repo.split("/", 1)
    output = await shell.run(
        ["gh", "api", "user", "-q", ".login"],
        env=_gh_env(host),
    )
    return output.strip()


async def list_review_comments(shell, repo: str, pr_number: int) -> list[dict]:
    """List all review comments on a PR via gh api.

    Args:
        shell: switchplane.Shell instance
        repo: Repository identifier (e.g. "github.com/org/repo")
        pr_number: PR number

    Returns:
        List of comment dicts
    """
    import json

    host, rest = repo.split("/", 1)
    output = await shell.run(
        ["gh", "api", f"repos/{rest}/pulls/{pr_number}/comments"],
        env=_gh_env(host),
    )
    return json.loads(output)


async def submit_pr_review(
    shell, repo: str, pr_number: int, event: str, body: str
) -> None:
    """Submit a PR review via gh api.

    Args:
        shell: switchplane.Shell instance
        repo: Repository identifier (e.g. "github.com/org/repo")
        pr_number: PR number
        event: Review event (e.g. "APPROVE", "REQUEST_CHANGES", "COMMENT")
        body: Review body text
    """
    host, rest = repo.split("/", 1)
    await shell.run(
        [
            "gh",
            "api",
            f"repos/{rest}/pulls/{pr_number}/reviews",
            "-X",
            "POST",
            "-f",
            f"event={event}",
            "-f",
            f"body={body}",
        ],
        env=_gh_env(host),
    )


async def create_pr_review_comment(
    shell,
    repo: str,
    pr_number: int,
    body: str,
    path: str,
    line: int,
    commit_id: str,
) -> None:
    """Create a single PR review comment via gh api.

    Args:
        shell: switchplane.Shell instance
        repo: Repository identifier (e.g. "github.com/org/repo")
        pr_number: PR number
        body: Comment body text
        path: File path
        line: Line number
        commit_id: Commit SHA
    """
    host, rest = repo.split("/", 1)
    await shell.run(
        [
            "gh",
            "api",
            f"repos/{rest}/pulls/{pr_number}/comments",
            "-X",
            "POST",
            "-f",
            f"body={body}",
            "-f",
            f"path={path}",
            "-f",
            f"commit_id={commit_id}",
            "-F",
            f"line={line}",
        ],
        env=_gh_env(host),
    )


async def check_dependencies() -> None:
    """Check that git and gh are on PATH.

    Raises:
        RuntimeError: If git and/or gh are missing
    """
    missing = []
    if shutil.which("git") is None:
        missing.append("git")
    if shutil.which("gh") is None:
        missing.append("gh")

    if missing:
        raise RuntimeError(f"Missing required dependencies: {', '.join(missing)}")
