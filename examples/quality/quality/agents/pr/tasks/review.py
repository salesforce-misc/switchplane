"""Pull request review task with multi-provider fan-out."""

from __future__ import annotations

from switchplane.agent_runtime import AgentContext
from switchplane.config import DEFAULT_PROVIDER, resolve_provider

DOMAINS = ("quality", "security")
"""Review domains for the fan-out cross-product."""


def _resolve_matrix(ctx: AgentContext) -> list[tuple[str, str]]:
    """Resolve (provider, model) pairs from ctx.providers.

    Returns a list of (provider_name, model) tuples for the fan-out matrix.
    Each tuple represents one LLM configuration that will independently review
    the pull request.

    Behavior:
    - Named pool entries with api_key are included in the order ctx.providers returns them
    - Entries missing api_key are skipped with a ctx.progress notification
    - DEFAULT_PROVIDER is filtered out when other named entries exist (prevents
      double-review when a user creates a [llm.providers.default] alias)
    - Empty pool with [llm] api_key falls back to one (DEFAULT_PROVIDER, model) entry
      (ensures the example works with a stock config.toml)
    - Empty pool without api_key returns [] (clean no-op, no branches)
    - Unknown provider names call ctx.fail with a clear message listing configured names

    Args:
        ctx: Agent context providing providers list and config.

    Returns:
        List of (provider_name, model) tuples. Empty if no usable providers.
    """
    matrix: list[tuple[str, str]] = []

    # Filter out DEFAULT_PROVIDER if other named entries exist to avoid duplication
    providers = ctx.providers
    if len(providers) > 1 and DEFAULT_PROVIDER in providers:
        providers = [p for p in providers if p != DEFAULT_PROVIDER]

    for name in providers:
        try:
            provider = resolve_provider(ctx.config, name)
        except KeyError as exc:
            # resolve_provider already lists configured names in the message
            ctx.fail(str(exc))
            raise  # Ensure control does not continue past ctx.fail

        if not provider.api_key:
            ctx.progress(f"Skipping provider '{name}' (no api_key configured)")
            continue

        matrix.append((name, provider.model))

    # Empty pool fallback: if no named entries and [llm] has api_key, use default
    if not matrix and not ctx.providers:
        provider = resolve_provider(ctx.config, None)
        if provider.api_key:
            matrix.append((DEFAULT_PROVIDER, provider.model))

    return matrix
