"""Configuration loading for Switchplane.

Two-layer cascading config: app-bundled defaults deep-merged with
user-level overrides from ~/.{app_name}/config.toml.
"""

import tomllib
from pathlib import Path
from typing import Any

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


class TuiConfig(BaseModel):
    """TUI tuning knobs.

    Defaults are conservative — they trade scrollback depth and
    spinner liveness for bounded per-frame render cost. The
    TUI's main thread renders the **entire** scrollback buffer on
    every redraw (prompt_toolkit's `FormattedTextControl.create_content`
    is per-frame O(buffer_size), not O(visible-area)), so a buffer
    much larger than these defaults can pin the daemon's CPU on
    long-running tasks even when the user isn't actively scrolling.
    """

    max_buffer_lines: int = 2_000
    """Maximum lines retained per tab before oldest are trimmed.

    Was 10_000; that produced sustained 99% CPU spins on the daemon
    main thread for long-running tasks (LLM tool loops with hundreds
    of events). The render cost grows linearly with this; halving it
    halves baseline render cost while still giving the operator a
    deep-enough scrollback for routine debugging.
    """

    spinner_interval: float = 0.5
    """How often the active-task spinner redraws, in seconds.

    Was 0.2 (5 fps), raised here to 0.5 (2 fps). The original 5 fps
    pinned a redraw-every-200ms cadence on every active-task tab
    regardless of whether content changed; combined with a large
    `max_buffer_lines` it was the load-bearing contributor to the
    daemon-CPU pin.

    2 fps is a 2.5× cost reduction without sacrificing liveness —
    fast enough that operators read it as "alive" rather than "stuck"
    (1 fps was tested and felt like the latter). The smaller buffer
    cap and `_REFRESH_DEBOUNCE_SECONDS` are doing most of the
    heavy lifting on render cost; the spinner change here is the
    polish on top.
    """


class AppConfig(BaseModel):
    """Top-level configuration."""

    llm: LLMConfig = LLMConfig()
    logging: LoggingConfig = LoggingConfig()
    tui: TuiConfig = TuiConfig()
    agents: dict[str, dict[str, Any]] = {}


def load_config[C: AppConfig](
    config_path: Path | None = None,
    default_config_path: Path | None = None,
    config_class: type[C] = AppConfig,  # type: ignore[assignment]
) -> C:
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
