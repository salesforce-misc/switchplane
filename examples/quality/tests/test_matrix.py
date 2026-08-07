"""Tests for _resolve_matrix provider pool fan-out.

These tests pin the behavior of resolving (provider, model) pairs from ctx.providers,
including fallback, skip, and error cases. Each test uses provider names and
models that cannot be guessed from the code — the resolution logic must be
correct for tests to pass.
"""

from unittest.mock import Mock

import pytest

from switchplane.config import DEFAULT_PROVIDER, resolve_provider


class TestResolveMatrix:
    """Test suite for _resolve_matrix provider pool fan-out logic."""

    def test_two_entry_pool_yields_both_providers(self, monkeypatch):
        """A two-entry pool returns both (provider, model) pairs.

        Detects: missing entries, wrong ordering, duplicates.
        """
        from quality.agents.pr.tasks.review import _resolve_matrix

        ctx = Mock()
        ctx.providers = ["alpha", "beta"]
        ctx.config = {
            "llm": {
                "providers": {
                    "alpha": {"api_key": "key-a", "model": "model-a-1"},
                    "beta": {"api_key": "key-b", "model": "model-b-2"},
                }
            }
        }

        result = _resolve_matrix(ctx)

        # Both providers should appear as (name, model) tuples
        expected = [
            ("alpha", "model-a-1"),
            ("beta", "model-b-2"),
        ]
        assert len(result) == 2, f"Expected 2 provider entries, got {len(result)}"
        assert result == expected, f"Provider list mismatch: {result} != {expected}"

    def test_entry_without_api_key_is_skipped_with_progress(self, monkeypatch):
        """An entry lacking api_key is skipped AND ctx.progress is called.

        Detects: silent drops (no progress call), or inclusion of unusable entries.
        """
        from quality.agents.pr.tasks.review import _resolve_matrix

        ctx = Mock()
        ctx.providers = ["alpha", "gamma"]
        ctx.config = {
            "llm": {
                "providers": {
                    "alpha": {"api_key": "key-a", "model": "model-a-1"},
                    "gamma": {"model": "model-g-3"},  # No api_key
                }
            }
        }

        result = _resolve_matrix(ctx)

        # Only alpha should appear; gamma skipped
        expected = [("alpha", "model-a-1")]
        assert result == expected, f"Expected only alpha entry, got {result}"

        # Assert progress was called noting the skip
        ctx.progress.assert_called()
        progress_call_text = " ".join(str(call) for call in ctx.progress.call_args_list)
        assert "gamma" in progress_call_text.lower(), "Progress must mention the skipped provider 'gamma'"

    def test_empty_pool_with_llm_api_key_falls_back_to_default(self, monkeypatch):
        """Empty pool with [llm] api_key falls back to one default entry.

        Detects: missing fallback, or multiple entries from a single default.
        """
        from quality.agents.pr.tasks.review import _resolve_matrix

        ctx = Mock()
        ctx.providers = []  # Empty pool
        ctx.config = {
            "llm": {
                "api_key": "default-key",
                "model": "model-default",
            }
        }

        result = _resolve_matrix(ctx)

        # Exactly one default provider entry
        expected = [(DEFAULT_PROVIDER, "model-default")]
        assert len(result) == 1, f"Expected 1 default entry, got {len(result)}"
        assert result == expected, f"Default fallback mismatch: {result} != {expected}"

    def test_empty_pool_without_api_key_yields_empty(self, monkeypatch):
        """Empty pool with no api_key yields an empty list.

        Detects: crash on missing credentials, or incorrectly creating branches.
        """
        from quality.agents.pr.tasks.review import _resolve_matrix

        ctx = Mock()
        ctx.providers = []
        ctx.config = {
            "llm": {
                "model": "model-default",
                # No api_key
            }
        }

        result = _resolve_matrix(ctx)

        assert result == [], f"Expected empty list for unconfigured fallback, got {result}"

    def test_default_alias_excluded_when_named_entries_exist(self, monkeypatch):
        """A 'default' alias alongside named entries is excluded to avoid duplication.

        Detects: double-review with the same model when a user configures a default alias.
        """
        from quality.agents.pr.tasks.review import _resolve_matrix

        ctx = Mock()
        ctx.providers = ["alpha", "default"]
        ctx.config = {
            "llm": {
                "api_key": "base-key",
                "model": "model-base",
                "providers": {
                    "alpha": {"api_key": "key-a", "model": "model-a-1"},
                    "default": {"api_key": "default-alias-key", "model": "model-default-alias"},
                },
            }
        }

        result = _resolve_matrix(ctx)

        # Only alpha should appear; 'default' filtered out
        expected = [("alpha", "model-a-1")]
        assert result == expected, f"Expected only alpha entry (default excluded), got {result}"

    def test_unknown_provider_name_surfaces_clear_message(self, monkeypatch):
        """Unknown provider name raises with a message listing configured providers.

        Detects: bare traceback, or unclear error without listing available names.
        """
        from quality.agents.pr.tasks.review import _resolve_matrix

        ctx = Mock()
        ctx.providers = ["unknown"]
        ctx.config = {
            "llm": {
                "providers": {
                    "alpha": {"api_key": "key-a", "model": "model-a-1"},
                    "beta": {"api_key": "key-b", "model": "model-b-2"},
                }
            }
        }

        # _resolve_matrix should call ctx.fail with a message listing 'alpha' and 'beta'
        ctx.fail.side_effect = RuntimeError("simulated fail")

        with pytest.raises(RuntimeError, match="simulated fail"):
            _resolve_matrix(ctx)

        # Assert ctx.fail was called with a message containing the configured provider names
        ctx.fail.assert_called()
        fail_message = str(ctx.fail.call_args[0][0])
        assert "alpha" in fail_message, f"Fail message must list 'alpha', got: {fail_message}"
        assert "beta" in fail_message, f"Fail message must list 'beta', got: {fail_message}"
        assert "unknown" in fail_message, f"Fail message must mention 'unknown', got: {fail_message}"


class TestProviderResolutionIntegration:
    """Integration tests verifying ctx.config shape matches resolve_provider expectations."""

    def test_resolve_provider_with_named_entry(self):
        """resolve_provider returns the correct ProviderConfig for a named entry."""
        config = {
            "llm": {
                "api_key": "base-key",
                "model": "base-model",
                "providers": {
                    "alpha": {"api_key": "key-a", "model": "model-a-1"},
                },
            }
        }

        provider = resolve_provider(config, "alpha")

        assert provider.api_key == "key-a"
        assert provider.model == "model-a-1"

    def test_resolve_provider_default_fallback(self):
        """resolve_provider with name=None or 'default' falls back to [llm] block."""
        config = {
            "llm": {
                "api_key": "base-key",
                "model": "base-model",
            }
        }

        provider_none = resolve_provider(config, None)
        provider_default = resolve_provider(config, DEFAULT_PROVIDER)

        assert provider_none.api_key == "base-key"
        assert provider_none.model == "base-model"
        assert provider_default.api_key == "base-key"
        assert provider_default.model == "base-model"

    def test_resolve_provider_raises_on_unknown_name(self):
        """resolve_provider raises KeyError with configured names listed."""
        config = {
            "llm": {
                "providers": {
                    "alpha": {"api_key": "key-a", "model": "model-a-1"},
                    "beta": {"api_key": "key-b", "model": "model-b-2"},
                }
            }
        }

        with pytest.raises(KeyError) as exc_info:
            resolve_provider(config, "unknown")

        error_msg = str(exc_info.value)
        assert "alpha" in error_msg, f"KeyError must list 'alpha', got: {error_msg}"
        assert "beta" in error_msg, f"KeyError must list 'beta', got: {error_msg}"
