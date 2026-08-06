"""Per-PR review baseline persistence.

Storing prior findings (not full message history) lets a subsequent review of the
same PR run as a follow-up: branches are told what was raised before and focus on
what changed.

Local (dry-run) and non-local (GitHub-posting) reviews keep SEPARATE baseline files.
A cross-mode share would let a GitHub-posted baseline degrade a later ``--local``
artifact to a thin follow-up (or vice versa), even though the two modes never surface
findings the other saw — namespacing on the ``local`` flag keeps each mode idempotent
with respect to itself and never leaks state across the boundary.

Layout:
    <root>/state/review/<host>/<org>/<repo>/pr-<n>.json         (non-local)
    <root>/state/review/<host>/<org>/<repo>/pr-<n>.local.json   (local)

JSON shape:
    {
        "head_sha": str,
        "summary": str,
        "updated_at": str (ISO8601 UTC),
        "findings": [
            {
                "path": str|null,
                "line": int|null,
                "severity": str,
                "title": str,
                "body": str,
                "models": [str],
                "domain": str
            }
        ]
    }
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import structlog

from quality._paths import mkdir_private

logger = structlog.get_logger()


def baseline_path(root: Path, repo: str, number: int, *, local: bool = False) -> Path:
    """Build the baseline file path for a PR.

    The repo string is "host/org/repo" (e.g. "github.com/myorg/myrepo") and its
    segments are used as the directory tree. Local and non-local baselines live
    in the same directory with different filenames (.local.json vs .json) so they
    can coexist and be listed side-by-side.

    **Path safety:** The resolved path is verified to be relative to ``root`` after
    symlink resolution to prevent traversal attacks via malicious repo strings or
    symlinks.

    Args:
        root: Runtime root directory (typically the app's runtime dir).
        repo: Full repository path (host/org/repo).
        number: PR number.
        local: If True, uses .local.json suffix for local/dry-run baselines.

    Returns:
        The baseline file path.

    Raises:
        ValueError: If the resolved path is not relative to root (path traversal attempt).
    """
    suffix = ".local.json" if local else ".json"
    path = root / "state" / "review" / repo / f"pr-{number}{suffix}"

    # Defense-in-depth: verify the resolved path is under root after symlink resolution
    try:
        resolved_root = root.resolve()
        resolved_path = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve path: {exc}") from exc

    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"resolved path {resolved_path} not relative to root {resolved_root}")

    return path


def load_baseline(path: Path) -> dict[str, Any] | None:
    """Load a baseline file, returning None on any failure.

    Graceful degradation: a missing, corrupt, or unreadable baseline doesn't block
    a review — it just forces a fresh one. This is always safe.

    Args:
        path: Path to the baseline file.

    Returns:
        The parsed baseline dict, or None if the file is missing, unreadable,
        contains invalid JSON, or is not a dict at the top level.
    """
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            logger.warning("baseline_not_dict", path=str(path))
            return None
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("baseline_load_failed", path=str(path), error=str(exc))
        return None


def save_baseline(
    root: Path,
    *,
    repo: str,
    number: int,
    head_sha: str,
    summary: str,
    findings: list[dict],
    updated_at: str | None = None,
    local: bool = False,
) -> Path:
    """Persist review findings, replacing any prior file atomically.

    Creates parent directories with mode 0o700 (user-only) and writes the file with
    mode 0o600 (user read/write only) for privacy — findings may contain security
    issues and must not be world-readable.

    Atomic write: uses ``tempfile.mkstemp`` in the same directory + ``os.replace``
    to prevent partial reads. The temp file is cleaned up on failure.

    Args:
        root: Runtime root directory.
        repo: Full repository path (host/org/repo).
        number: PR number.
        head_sha: Git SHA of the PR head at review time.
        summary: Human-readable review summary.
        findings: List of finding dicts.
        updated_at: Optional ISO8601 UTC timestamp.
        local: If True, writes to the .local.json file.

    Returns:
        The path where the baseline was written.

    Raises:
        OSError: If the write fails (e.g. disk full, permission denied).
    """
    path = baseline_path(root, repo, number, local=local)
    mkdir_private(path.parent, root)

    payload = {
        "head_sha": head_sha,
        "summary": summary,
        "findings": findings,
        "updated_at": updated_at or "",
    }

    # Atomic write: create temp file in the same directory, write, chmod, then replace
    fd = None
    temp_path = None
    try:
        fd, temp_path_str = tempfile.mkstemp(dir=path.parent, prefix=".baseline-", suffix=".tmp")
        temp_path = Path(temp_path_str)

        # Write JSON to the temp file
        os.write(fd, json.dumps(payload, indent=2).encode("utf-8"))
        os.close(fd)
        fd = None

        # Set restrictive permissions before moving into place
        os.chmod(temp_path, 0o600)

        # Atomic replace
        os.replace(temp_path, path)
        temp_path = None  # Successfully moved, don't clean up

        return path

    finally:
        if fd is not None:
            os.close(fd)
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
