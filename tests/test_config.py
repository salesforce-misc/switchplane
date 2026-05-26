from switchplane.config import (
    AppConfig,
    LLMConfig,
    TuiConfig,
    _deep_merge,
    get_agent_config,
    load_config,
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
        assert cfg.agents == {}

    def test_with_agents(self):
        cfg = AppConfig(agents={"worker": {"timeout": 30}})
        assert cfg.agents["worker"]["timeout"] == 30


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


class TestGetAgentConfig:
    def test_existing_agent(self):
        cfg = AppConfig(agents={"worker": {"timeout": 60}})
        assert get_agent_config(cfg, "worker") == {"timeout": 60}

    def test_missing_agent(self):
        cfg = AppConfig()
        assert get_agent_config(cfg, "nonexistent") == {}
