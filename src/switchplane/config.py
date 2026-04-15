"""Configuration loading for Switchplane.

Two-layer cascading config: app-bundled defaults deep-merged with
user-level overrides from ~/.{app_name}/config.toml.
"""

import tomllib
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from switchplane._util import deep_merge

# Backward-compatible alias
_deep_merge = deep_merge


DEFAULT_MODEL = "claude-sonnet-4-20250514"


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    api_key: str | None = None
    base_url: str | None = None
    model: str = DEFAULT_MODEL


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "debug"  # log level: debug, info, warning, error


class AppConfig(BaseModel):
    """Top-level configuration."""

    llm: LLMConfig = LLMConfig()
    logging: LoggingConfig = LoggingConfig()
    agents: dict[str, dict[str, Any]] = {}


_C = TypeVar("_C", bound=AppConfig)


def load_config(
    config_path: Path | None = None,
    default_config_path: Path | None = None,
    config_class: type[_C] = AppConfig,  # type: ignore[assignment]
) -> _C:
    """Load config from a TOML file.

    Args:
        config_path: Explicit path to config file
        default_config_path: Optional path to default config file bundled with the app
        config_class: Pydantic model class to validate the merged config into.
            Defaults to AppConfig; pass a subclass to support app-specific sections.
    """
    # Load app defaults if they exist
    app_defaults = {}
    if default_config_path and default_config_path.exists():
        with open(default_config_path, "rb") as f:
            app_defaults = tomllib.load(f)

    # Load user config if it exists
    user_config = {}
    if config_path and config_path.exists():
        with open(config_path, "rb") as f:
            user_config = tomllib.load(f)

    # Merge: user config overrides app defaults
    if app_defaults and user_config:
        merged = app_defaults.copy()
        deep_merge(merged, user_config)
        return config_class.model_validate(merged)
    elif user_config:
        return config_class.model_validate(user_config)
    elif app_defaults:
        return config_class.model_validate(app_defaults)
    else:
        return config_class()


def get_agent_config(config: AppConfig, agent_name: str) -> dict[str, Any]:
    """Get the config section for a specific agent."""
    return config.agents.get(agent_name, {})
