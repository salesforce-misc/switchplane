"""Configuration loading for Switchplane.

Two-layer cascading config: app-bundled defaults deep-merged with
user-level overrides from ~/.{app_name}/config.toml.
"""

import tomllib
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

from switchplane._util import deep_merge

# Backward-compatible alias
_deep_merge = deep_merge

logger = structlog.get_logger(__name__)


DEFAULT_MODEL = "claude-sonnet-4-20250514"

DEFAULT_PROVIDER = "default"
"""Name that resolves to the `[llm]` block itself rather than a pool entry."""


class ProviderConfig(BaseModel):
    """One entry in the `[llm.providers]` pool.

    Field-compatible with the provider fields on `LLMConfig`, so the `[llm]`
    block and a pool entry are structurally interchangeable — `resolve_provider`
    treats them as one type.

    Unset fields are **not** inherited from `[llm]`: an entry declaring only a
    `model` resolves with `api_key=None`, exactly as a bare `[llm]` block does.
    Each entry is meant to be readable in isolation, and inheriting a credential
    across vendors is the failure mode the pool exists to prevent. To reuse the
    `[llm]` credential with a different model, pass `model=` to `ctx.llm()`
    instead of defining an entry.
    """

    provider: str | None = None
    """Reserved for explicit provider routing. Unread today — `build_llm` routes
    on model-name prefix. Present so adding it later is not a schema change."""

    api_key: str | None = None
    base_url: str | None = None
    model: str = DEFAULT_MODEL


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    api_key: str | None = None
    base_url: str | None = None
    model: str = DEFAULT_MODEL
    providers: dict[str, ProviderConfig] = {}
    """Named alternates, e.g. `[llm.providers.cheap]`. The fields above act as
    the default provider; see `resolve_provider`."""


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


def _warn_removed_sections(raw: dict[str, Any], path: Path) -> None:
    """Warn about config sections that are no longer honored.

    `[agents.<name>]` per-agent overrides were removed — the pool
    (`[llm.providers]`) supersedes their only use, varying models per agent.
    Unknown sections are ignored by the model, so without this the values would
    silently stop taking effect.
    """
    if "agents" in raw:
        logger.warning(
            "config_section_ignored",
            section="agents",
            path=str(path),
            detail="Per-agent overrides were removed; use [llm.providers.<name>] and ctx.llm(<name>).",
        )


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
        _warn_removed_sections(app_defaults, default_config_path)

    # Load user config if it exists
    user_config = {}
    if config_path and config_path.exists():
        with open(config_path, "rb") as f:
            user_config = tomllib.load(f)
        _warn_removed_sections(user_config, config_path)

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


def resolve_provider(config: dict[str, Any], name: str | None = None) -> ProviderConfig:
    """Resolve an LLM provider from a delivered config dict.

    Args:
        config: The config dict as delivered to an agent (``ctx.config``).
        name: Pool entry name from ``[llm.providers.<name>]``. ``None`` or
            ``"default"`` resolves to the ``[llm]`` block's own fields, unless a
            ``[llm.providers.default]`` entry is explicitly defined.

    Unset fields are never filled in from the ``[llm]`` block — the returned
    config is either the default's fields or a named entry's fields, never a
    blend of both. See ``ProviderConfig``.

    Raises:
        KeyError: If *name* is not a configured pool entry.
    """
    llm = config.get("llm") or {}
    pool = llm.get("providers") or {}

    if name is None or name == DEFAULT_PROVIDER:
        # An explicit [llm.providers.default] wins, letting a user redirect the
        # default without editing [llm] itself.
        if DEFAULT_PROVIDER in pool:
            return ProviderConfig.model_validate(pool[DEFAULT_PROVIDER])
        return ProviderConfig.model_validate({k: v for k, v in llm.items() if k != "providers"})

    if name not in pool:
        configured = ", ".join(sorted(pool)) or "none"
        raise KeyError(f"Unknown LLM provider {name!r}. Configured providers: {configured}")

    return ProviderConfig.model_validate(pool[name])
