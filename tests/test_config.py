import json
from typing import ClassVar

import pytest
from structlog.testing import capture_logs

from switchplane.config import (
    DEFAULT_MODEL,
    AppConfig,
    LLMConfig,
    ProviderConfig,
    TuiConfig,
    _deep_merge,
    load_config,
    resolve_provider,
)


class TestDeepMerge:
    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"llm": {"provider": "anthropic", "model": "claude"}, "x": 1}
        _deep_merge(base, {"llm": {"model": "gpt-4", "api_key": "sk-xxx"}})
        assert base == {
            "llm": {"provider": "anthropic", "model": "gpt-4", "api_key": "sk-xxx"},
            "x": 1,
        }

    def test_override_non_dict_with_dict(self):
        base = {"a": "string"}
        _deep_merge(base, {"a": {"nested": True}})
        assert base == {"a": {"nested": True}}

    def test_empty_override(self):
        base = {"a": 1}
        _deep_merge(base, {})
        assert base == {"a": 1}


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "anthropic"
        assert cfg.api_key is None
        assert cfg.base_url is None
        assert cfg.model == "claude-sonnet-4-20250514"

    def test_custom(self):
        cfg = LLMConfig(provider="openai", model="gpt-4", api_key="sk-123")
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-123"


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.tui, TuiConfig)

    def test_no_agents_section(self):
        """Per-agent overrides were removed; the section is not modelled."""
        assert not hasattr(AppConfig(), "agents")

    def test_stale_agents_section_ignored(self):
        """A leftover [agents.*] block parses without error but is dropped."""
        cfg = AppConfig.model_validate({"agents": {"worker": {"timeout": 30}}})
        assert "agents" not in cfg.model_dump()


class TestTuiConfig:
    """`TuiConfig` knobs cap per-frame TUI render cost.
    Defaults are intentionally conservative — see config.py."""

    def test_defaults(self):
        cfg = TuiConfig()
        # 2_000 (was 10_000) — render cost grows linearly with this.
        assert cfg.max_buffer_lines == 2_000
        # 0.5s (was 0.2s hardcoded in tui.py) — 2.5× slower spinner
        # tick cuts baseline render rate proportionally without
        # crossing the threshold where the spinner reads as "stuck"
        # rather than "alive".
        assert cfg.spinner_interval == 0.5

    def test_overrides(self):
        cfg = TuiConfig(max_buffer_lines=500, spinner_interval=2.0)
        assert cfg.max_buffer_lines == 500
        assert cfg.spinner_interval == 2.0

    def test_loaded_via_app_config(self):
        """The TuiConfig is reachable as `AppConfig().tui`, which is
        how the cli.py TUI launch path reads it."""
        cfg = AppConfig(tui={"max_buffer_lines": 1234, "spinner_interval": 0.5})
        assert cfg.tui.max_buffer_lines == 1234
        assert cfg.tui.spinner_interval == 0.5


class TestLoadConfig:
    def test_no_files(self):
        cfg = load_config(None, None)
        assert isinstance(cfg, AppConfig)
        assert cfg.llm.provider == "anthropic"

    def test_nonexistent_paths(self, tmp_path):
        cfg = load_config(tmp_path / "nope.toml", tmp_path / "also_nope.toml")
        assert isinstance(cfg, AppConfig)

    def test_app_defaults_only(self, tmp_path):
        default_cfg = tmp_path / "defaults.toml"
        default_cfg.write_text('[llm]\nprovider = "openai"\nmodel = "gpt-4"\n')
        cfg = load_config(None, default_cfg)
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4"

    def test_user_config_only(self, tmp_path):
        user_cfg = tmp_path / "config.toml"
        user_cfg.write_text('[llm]\napi_key = "sk-user"\n')
        cfg = load_config(user_cfg, None)
        assert cfg.llm.api_key == "sk-user"

    def test_merge_user_overrides_app_defaults(self, tmp_path):
        default_cfg = tmp_path / "defaults.toml"
        default_cfg.write_text('[llm]\nprovider = "anthropic"\nmodel = "claude"\n')

        user_cfg = tmp_path / "config.toml"
        user_cfg.write_text('[llm]\napi_key = "sk-user"\nmodel = "gpt-4"\n')

        cfg = load_config(user_cfg, default_cfg)
        assert cfg.llm.provider == "anthropic"  # from defaults
        assert cfg.llm.model == "gpt-4"  # user override
        assert cfg.llm.api_key == "sk-user"  # user addition


    def test_stale_agents_section_warns(self, tmp_path):
        """Silence is the wrong failure mode for a removed feature, so
        load_config warns when it sees a section that no longer applies."""
        user_cfg = tmp_path / "config.toml"
        user_cfg.write_text('[agents.bot]\nsystem_prompt = "hi"\n')

        with capture_logs() as logs:
            load_config(user_cfg, None)

        warned = [e for e in logs if e["event"] == "config_section_ignored"]
        assert len(warned) == 1
        assert warned[0]["section"] == "agents"
        assert str(user_cfg) == warned[0]["path"]

    def test_app_defaults_agents_section_warns(self, tmp_path):
        """Both config layers are checked, not just the user's."""
        default_cfg = tmp_path / "defaults.toml"
        default_cfg.write_text('[agents.bot]\nsystem_prompt = "hi"\n')

        with capture_logs() as logs:
            load_config(None, default_cfg)

        assert [e for e in logs if e["event"] == "config_section_ignored"]

    def test_no_warning_without_agents_section(self, tmp_path):
        user_cfg = tmp_path / "config.toml"
        user_cfg.write_text('[llm]\napi_key = "k"\n')

        with capture_logs() as logs:
            load_config(user_cfg, None)

        assert not [e for e in logs if e["event"] == "config_section_ignored"]


class TestProviderPool:
    def test_empty_by_default(self):
        assert AppConfig().llm.providers == {}

    def test_multiple_independent_entries(self):
        cfg = AppConfig.model_validate(
            {
                "llm": {
                    "model": "claude-sonnet-4-20250514",
                    "api_key": "ant",
                    "providers": {
                        "cheap": {"model": "gemini-2.5-flash", "api_key": "gai"},
                        "gw": {"model": "gpt-4o", "api_key": "gw-tok", "base_url": "https://gw/v1"},
                    },
                }
            }
        )
        assert cfg.llm.providers["cheap"].api_key == "gai"
        assert cfg.llm.providers["cheap"].base_url is None
        assert cfg.llm.providers["gw"].base_url == "https://gw/v1"
        # Pool does not disturb the sibling [llm] fields.
        assert cfg.llm.api_key == "ant"
        assert cfg.llm.model == "claude-sonnet-4-20250514"

    def test_same_vendor_two_endpoints(self):
        """The case a vendor-keyed design could not express: one vendor,
        direct and via gateway, with different credentials."""
        cfg = AppConfig.model_validate(
            {
                "llm": {
                    "providers": {
                        "direct": {"model": "claude-opus-4-6-v1", "api_key": "sk-ant"},
                        "gateway": {
                            "model": "claude-opus-4-6-v1",
                            "api_key": "gw-tok",
                            "base_url": "https://gw/v1",
                        },
                    }
                }
            }
        )
        assert cfg.llm.providers["direct"].base_url is None
        assert cfg.llm.providers["gateway"].base_url == "https://gw/v1"
        assert cfg.llm.providers["direct"].api_key != cfg.llm.providers["gateway"].api_key

    def test_entry_model_defaults(self):
        cfg = AppConfig.model_validate({"llm": {"providers": {"x": {"api_key": "k"}}}})
        assert cfg.llm.providers["x"].model == DEFAULT_MODEL

    def test_provider_field_reserved_and_unset(self):
        assert ProviderConfig().provider is None

    def test_user_entry_deep_merges_onto_app_default(self, tmp_path):
        """Per-field merge within a pool entry, not whole-entry replacement."""
        default_cfg = tmp_path / "defaults.toml"
        default_cfg.write_text('[llm.providers.cheap]\nmodel = "gemini-2.5-flash"\nbase_url = "https://app/v1"\n')

        user_cfg = tmp_path / "config.toml"
        user_cfg.write_text('[llm.providers.cheap]\napi_key = "sk-user"\n')

        cfg = load_config(user_cfg, default_cfg)
        entry = cfg.llm.providers["cheap"]
        assert entry.model == "gemini-2.5-flash"  # from app defaults
        assert entry.base_url == "https://app/v1"  # from app defaults
        assert entry.api_key == "sk-user"  # user addition


class TestResolveProvider:
    POOL: ClassVar[dict] = {
        "llm": {
            "model": "claude-sonnet-4-20250514",
            "api_key": "ant-key",
            "base_url": "https://default/v1",
            "providers": {
                "cheap": {"model": "gemini-2.5-flash", "api_key": "gai-key"},
                "bare": {"model": "gpt-4o"},
            },
        }
    }

    def test_none_resolves_to_llm_block(self):
        p = resolve_provider(self.POOL)
        assert (p.model, p.api_key, p.base_url) == (
            "claude-sonnet-4-20250514",
            "ant-key",
            "https://default/v1",
        )

    def test_default_name_resolves_to_llm_block(self):
        assert resolve_provider(self.POOL, "default").api_key == "ant-key"

    def test_explicit_default_entry_wins(self):
        cfg = {"llm": {"api_key": "block", "providers": {"default": {"api_key": "entry", "model": "gpt-4o"}}}}
        assert resolve_provider(cfg, "default").api_key == "entry"
        assert resolve_provider(cfg).api_key == "entry"

    def test_named_entry(self):
        p = resolve_provider(self.POOL, "cheap")
        assert (p.model, p.api_key) == ("gemini-2.5-flash", "gai-key")

    def test_no_inherit_from_llm_block(self):
        """Pins the no-inherit decision: an entry declaring only `model`
        resolves with api_key/base_url None, NOT the [llm] block's values.
        Without this a well-meaning 'convenience' merge would pass."""
        p = resolve_provider(self.POOL, "bare")
        assert p.model == "gpt-4o"
        assert p.api_key is None
        assert p.base_url is None

    def test_providers_key_not_carried_into_default(self):
        """The resolved default exposes only provider fields, never the pool.

        Note for future editors: this test cannot detect removal of the
        `k != "providers"` strip in `resolve_provider`. `ProviderConfig` inherits
        Pydantic's `extra="ignore"`, so passing the raw `[llm]` dict through
        produces an identical model — the strip is defensive, not load-bearing,
        and no assertion here can prove otherwise. It becomes load-bearing only if
        `ProviderConfig` ever sets `extra="forbid"` or gains a `providers` field;
        this test then pins the shape that must survive that change.
        """
        dumped = resolve_provider(self.POOL).model_dump()
        assert "providers" not in dumped
        assert set(dumped) == {"provider", "api_key", "base_url", "model"}

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError) as exc:
            resolve_provider(self.POOL, "nope")
        msg = str(exc.value)
        assert "nope" in msg
        assert "bare" in msg and "cheap" in msg  # lists configured names

    def test_unknown_name_with_empty_pool_raises(self):
        """No silent fallback to the default — that would send the wrong
        credential to the wrong vendor, the bug the pool exists to prevent."""
        with pytest.raises(KeyError):
            resolve_provider({"llm": {"api_key": "k"}}, "nope")

    def test_missing_llm_section(self):
        p = resolve_provider({})
        assert p.model == DEFAULT_MODEL
        assert p.api_key is None


class TestLLMBlockCompat:
    """The [llm] block is a working production config. These pin it."""

    def test_existing_llm_block_unchanged_by_pool(self):
        cfg = AppConfig.model_validate({"llm": {"model": "claude-sonnet-4-20250514", "api_key": "k"}})
        d = cfg.model_dump()

        llm = d["llm"]
        assert llm.pop("providers") == {}  # the only addition
        assert json.dumps(llm, sort_keys=True) == (
            '{"api_key": "k", "base_url": null, "model": "claude-sonnet-4-20250514", "provider": "anthropic"}'
        )
        # No new top-level sections, and `agents` is gone.
        assert sorted(d) == ["llm", "logging", "tui"]

    def test_legacy_reader_access_pattern(self):
        """How every pre-existing consumer reads config (chat.py, review.py)."""
        delivered = AppConfig.model_validate({"llm": {"model": "gpt-4o", "api_key": "k"}}).model_dump()
        llm_config = delivered.get("llm", {})
        assert llm_config.get("model") == "gpt-4o"
        assert llm_config.get("api_key") == "k"
        assert llm_config.get("base_url") is None
