"""Tests for quality/agents/pr/memory.py — per-PR review baseline persistence.

The path-safety test is critical: it verifies that a malicious repo string
cannot escape the memory root via path traversal.
"""

import json
import stat

import pytest


class TestBaselinePath:
    """Tests for baseline_path — path resolution and safety."""

    def test_non_local_path_structure(self, tmp_path):
        """Non-local baseline path follows <root>/state/review/<host>/<org>/<repo>/pr-<n>.json."""
        from quality.agents.pr.memory import baseline_path

        path = baseline_path(tmp_path, "github.com/myorg/myrepo", 42, local=False)
        expected = tmp_path / "state" / "review" / "github.com" / "myorg" / "myrepo" / "pr-42.json"
        assert path == expected

    def test_local_path_has_local_suffix(self, tmp_path):
        """Local baseline path uses .local.json suffix."""
        from quality.agents.pr.memory import baseline_path

        path = baseline_path(tmp_path, "github.com/myorg/myrepo", 42, local=True)
        expected = tmp_path / "state" / "review" / "github.com" / "myorg" / "myrepo" / "pr-42.local.json"
        assert path == expected

    def test_local_and_non_local_paths_differ(self, tmp_path):
        """Local and non-local baselines use different filenames in the same directory.

        This pins the split that prevents a GitHub-posted baseline from silently
        degrading a later --local artifact into a thin follow-up.
        """
        from quality.agents.pr.memory import baseline_path

        non_local = baseline_path(tmp_path, "github.com/org/repo", 99, local=False)
        local = baseline_path(tmp_path, "github.com/org/repo", 99, local=True)

        assert non_local.parent == local.parent
        assert non_local != local
        assert non_local.name == "pr-99.json"
        assert local.name == "pr-99.local.json"

    def test_path_traversal_rejected(self, tmp_path):
        """A repo string with path-traversal attempts must raise ValueError.

        Defense-in-depth: even though parse_pr_url validates the repo string,
        baseline_path must independently verify the resolved path is under root.
        """
        from quality.agents.pr.memory import baseline_path

        with pytest.raises(ValueError, match=r"resolved path .* not relative to root"):
            baseline_path(tmp_path, "../../../etc/passwd", 1, local=False)

    def test_path_traversal_via_symlink_rejected(self, tmp_path):
        """A symlink-based traversal attempt is rejected.

        This pins the is_relative_to check that prevents escaping via symlinks.
        """
        from quality.agents.pr.memory import baseline_path

        # Create a symlink pointing outside tmp_path
        other_dir = tmp_path.parent / "other"
        other_dir.mkdir(exist_ok=True)
        link = tmp_path / "state" / "review" / "escape"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(other_dir)

        with pytest.raises(ValueError, match=r"resolved path .* not relative to root"):
            # This would resolve to other_dir/org/repo/pr-1.json
            baseline_path(tmp_path, "escape/org/repo", 1, local=False)


class TestLoadBaseline:
    """Tests for load_baseline — reading persisted findings."""

    def test_missing_file_returns_none(self, tmp_path):
        """When the baseline file doesn't exist, returns None."""
        from quality.agents.pr.memory import load_baseline

        result = load_baseline(tmp_path / "nonexistent.json")
        assert result is None

    def test_valid_json_returned(self, tmp_path):
        """A valid baseline file is parsed and returned as a dict."""
        from quality.agents.pr.memory import load_baseline

        path = tmp_path / "baseline.json"
        data = {"head_sha": "abc123", "summary": "test", "findings": []}
        path.write_text(json.dumps(data))

        result = load_baseline(path)
        assert result == data

    def test_corrupt_json_returns_none(self, tmp_path):
        """A file with invalid JSON returns None rather than raising.

        This pins the graceful-degradation behavior: a corrupt baseline doesn't
        block a review, it just forces a fresh one.
        """
        from quality.agents.pr.memory import load_baseline

        path = tmp_path / "corrupt.json"
        path.write_text("{broken json")

        result = load_baseline(path)
        assert result is None

    def test_non_dict_top_level_returns_none(self, tmp_path):
        """Valid JSON that isn't a dict at the top level returns None.

        This pins the type guard: baseline_path expects a dict with specific
        keys, so a JSON array or scalar is treated as corrupt.
        """
        from quality.agents.pr.memory import load_baseline

        path = tmp_path / "array.json"
        path.write_text(json.dumps([1, 2, 3]))

        result = load_baseline(path)
        assert result is None

    def test_unreadable_file_returns_none(self, tmp_path):
        """An existing but unreadable file returns None."""
        from quality.agents.pr.memory import load_baseline

        path = tmp_path / "unreadable.json"
        path.write_text("{}")
        path.chmod(0o000)

        try:
            result = load_baseline(path)
            assert result is None
        finally:
            path.chmod(0o600)  # Restore for cleanup


class TestSaveBaseline:
    """Tests for save_baseline — persisting findings with secure permissions."""

    def test_round_trip(self, tmp_path):
        """Data saved by save_baseline can be loaded back identically."""
        from quality.agents.pr.memory import load_baseline, save_baseline

        findings = [{"path": "file.py", "line": 10, "severity": "medium", "body": "issue"}]
        path = save_baseline(
            tmp_path, repo="github.com/org/repo", number=5, head_sha="abc", summary="summary", findings=findings
        )

        loaded = load_baseline(path)
        assert loaded["head_sha"] == "abc"
        assert loaded["summary"] == "summary"
        assert loaded["findings"] == findings

    def test_parent_created_with_0o700(self, tmp_path):
        """Parent directories are created with mode 0o700 (user-only).

        This pins the privacy constraint: findings are user-local and must not
        be world-readable.
        """
        from quality.agents.pr.memory import save_baseline

        path = save_baseline(
            tmp_path,
            repo="github.com/org/repo",
            number=7,
            head_sha="xyz",
            summary="test",
            findings=[],
        )

        parent = path.parent
        assert parent.exists()
        assert stat.S_IMODE(parent.stat().st_mode) == 0o700

    def test_file_mode_0o600(self, tmp_path):
        """The baseline file is written with mode 0o600 (user read/write only).

        This pins the privacy constraint: findings contain potential security
        issues and must not be group- or world-readable.
        """
        from quality.agents.pr.memory import save_baseline

        path = save_baseline(
            tmp_path,
            repo="github.com/org/repo",
            number=8,
            head_sha="def",
            summary="test",
            findings=[],
        )

        assert path.exists()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_replaces_existing_file(self, tmp_path):
        """save_baseline replaces an existing baseline file atomically.

        This pins the atomic-write behavior that prevents partial reads.
        """
        from quality.agents.pr.memory import load_baseline, save_baseline

        # Write initial baseline
        path1 = save_baseline(tmp_path, repo="github.com/org/repo", number=9, head_sha="v1", summary="old", findings=[])

        # Overwrite with new data
        path2 = save_baseline(tmp_path, repo="github.com/org/repo", number=9, head_sha="v2", summary="new", findings=[])

        assert path1 == path2
        loaded = load_baseline(path2)
        assert loaded["head_sha"] == "v2"
        assert loaded["summary"] == "new"

    def test_local_vs_non_local_separate_files(self, tmp_path):
        """Local and non-local baselines are stored in separate files.

        This is the critical property: a GitHub-posted review and a --local
        artifact must not share state.
        """
        from quality.agents.pr.memory import save_baseline

        non_local_path = save_baseline(
            tmp_path,
            repo="github.com/org/repo",
            number=10,
            head_sha="abc",
            summary="non-local",
            findings=[],
            local=False,
        )

        local_path = save_baseline(
            tmp_path,
            repo="github.com/org/repo",
            number=10,
            head_sha="xyz",
            summary="local",
            findings=[],
            local=True,
        )

        assert non_local_path != local_path
        assert non_local_path.parent == local_path.parent
        assert non_local_path.read_text() != local_path.read_text()

    def test_cleanup_on_write_failure(self, tmp_path, monkeypatch):
        """The temp file is cleaned up if the write fails mid-operation.

        This pins the try/finally cleanup that prevents temp-file leaks.
        """
        # Monkeypatch os.replace to raise after the temp file is created
        import os

        from quality.agents.pr.memory import save_baseline

        call_count = {"n": 0}

        def failing_replace(src, dst):
            call_count["n"] += 1
            raise OSError("Simulated failure")

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(OSError, match="Simulated failure"):
            save_baseline(
                tmp_path,
                repo="github.com/org/repo",
                number=11,
                head_sha="fail",
                summary="fail",
                findings=[],
            )

        # The temp file should be cleaned up — count files in the target directory
        target_dir = tmp_path / "state" / "review" / "github.com" / "org" / "repo"
        if target_dir.exists():
            # Should have no files (the final pr-11.json was never written)
            assert list(target_dir.glob("*")) == []

    def test_updated_at_field_persisted(self, tmp_path):
        """The optional updated_at field is persisted when provided."""
        from quality.agents.pr.memory import load_baseline, save_baseline

        path = save_baseline(
            tmp_path,
            repo="github.com/org/repo",
            number=12,
            head_sha="abc",
            summary="test",
            findings=[],
            updated_at="2025-01-01T00:00:00Z",
        )

        loaded = load_baseline(path)
        assert loaded["updated_at"] == "2025-01-01T00:00:00Z"
