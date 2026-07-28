"""Pull request review task with multi-provider fan-out."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.types import Send
from pydantic import BaseModel

from switchplane.agent_runtime import AgentContext
from switchplane.config import DEFAULT_PROVIDER, resolve_provider
from switchplane.shell import Shell

DOMAINS = ("quality", "security")
"""Review domains for the fan-out cross-product."""


class ReviewState(BaseModel):
    """LangGraph state for the PR review fan-out.

    Per-branch fields (cur_domain, cur_provider, cur_model, cur_prior) are read-only
    inside branches and NEVER returned — attribution rides inside each finding/note dict.
    Returning non-reducer fields from concurrent branches raises InvalidUpdateError.

    The reducer fields (findings, notes) are the only fields that branches return,
    and their Annotated[list, operator.add] reducer allows concurrent writes to merge.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Input fields (set once, never updated)
    repo: str = ""
    number: int = 0
    diff: str = ""
    worktree_path: str = ""
    matrix: list[tuple[str, str]] = []  # list of (provider, model) tuples
    error: str | None = None
    is_followup: bool = False
    ctx: Any = None  # AgentContext, stashed so the conditional-edge router can resolve the matrix

    # Per-branch fields — read-only inside branches, set via Send payload
    cur_domain: str = ""
    cur_provider: str = ""
    cur_model: str = ""
    cur_prior: str = ""  # Formatted prior findings for this domain

    # Reducer fields — concurrent branches merge via operator.add
    findings: Annotated[list[dict[str, Any]], operator.add] = []
    notes: Annotated[list[dict[str, Any]], operator.add] = []


def _state_accessor(state: dict | ReviewState, key: str, default: Any = None) -> Any:
    """Access a field from state that may be a dict or ReviewState.

    LangGraph hands nodes a ``ReviewState``, but ``Send`` payloads and the
    branch entry points arrive as plain dicts, so both forms must work.
    """
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def route_to_branches(state: dict | ReviewState) -> str | list[Send]:
    """Conditional-edge router for the review fan-out.

    Fan-out lives here rather than in a node because only a conditional edge
    may return ``Send`` objects. Takes a single argument, as LangGraph requires.

    Guard order is load-bearing: the error and empty-diff checks short-circuit
    before the matrix is resolved, so a setup failure never pays for provider
    resolution and an empty diff skips the fan-out entirely.

    Returns:
        - END if state.error is set (setup failed)
        - "synthesize_and_post" if diff is empty
        - "synthesize_and_post" if matrix is empty (no usable providers)
        - list[Send] dispatching to "review_branch" for each (domain x provider) pair
    """
    from quality.agents.pr.tasks import review as review_module

    error = _state_accessor(state, "error")
    if error:
        return END

    diff = _state_accessor(state, "diff", "")
    if not diff:
        return "synthesize_and_post"

    matrix = _state_accessor(state, "matrix", [])

    # Called through the module so monkeypatch can reach it.
    if not matrix:
        ctx = _state_accessor(state, "ctx")
        matrix = review_module._resolve_matrix(ctx)

    # If still no matrix after resolving, route to synthesis
    if not matrix:
        return "synthesize_and_post"

    # Build the cross-product: domains x (provider, model) pairs
    sends = []
    for domain in DOMAINS:
        for provider_name, model in matrix:
            # Create a copy of the state with per-branch fields set
            if isinstance(state, dict):
                branch_state = ReviewState(**state)
            else:
                branch_state = state.model_copy()

            branch_state.cur_domain = domain
            branch_state.cur_provider = provider_name
            branch_state.cur_model = model
            # cur_prior is set by a prior node or defaults to ""

            sends.append(Send("review_branch", branch_state))

    return sends


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


async def review_branch(ctx: AgentContext, shell: Shell, state: dict | ReviewState) -> dict[str, list]:
    """Execute one review branch for a (domain, provider) pair.

    This is the concurrent branch node invoked by the fan-out Send dispatcher.
    It loads the baseline, selects the appropriate prompt (initial vs follow-up),
    constructs an LLM with the branch's provider, binds tools (fs_tools + recording),
    and runs the tool loop.

    Branch failure is isolated: exceptions preserve partial findings and append a
    failed=True note. Only asyncio.CancelledError propagates (for task cancellation).

    Returns ONLY reducer fields (findings, notes) — attribution rides inside each dict.
    Returning non-reducer fields raises InvalidUpdateError under concurrency.
    """
    import asyncio

    from quality import ratelimit as ratelimit_module
    from quality.agents.pr import memory as memory_module
    from quality.agents.pr import prompts as prompts_module

    # Check for cancellation before starting work
    await ctx.check_cancelled()

    # Extract per-branch state (tolerate dict or ReviewState)
    domain = _state_accessor(state, "domain") or _state_accessor(state, "cur_domain")
    provider_name = _state_accessor(state, "provider") or _state_accessor(state, "cur_provider")
    model = _state_accessor(state, "model") or _state_accessor(state, "cur_model")
    repo = _state_accessor(state, "repo")
    number = _state_accessor(state, "number")
    diff = _state_accessor(state, "diff")
    worktree_path = _state_accessor(state, "worktree_path")
    is_followup = _state_accessor(state, "is_followup", False)
    cur_prior = _state_accessor(state, "cur_prior", "")

    # Per-branch closure for recording findings and notes
    # These MUST be created fresh per branch to avoid cross-contamination
    findings_list: list[dict[str, Any]] = []
    notes_list: list[dict[str, Any]] = []

    @tool
    async def record_finding(path: str, line: int, severity: str, body: str) -> str:
        """Record a code issue tied to a specific file and line.

        Args:
            path: File path relative to repo root.
            line: Line number in the file.
            severity: One of: info, low, medium, high, critical.
            body: Markdown explanation of the issue.

        Returns:
            Success message.
        """
        # Coerce invalid severity to "medium" (LLM is caller, don't lose a real finding over formatting)
        sev = severity.strip().lower()
        if sev not in {"info", "low", "medium", "high", "critical"}:
            sev = "medium"

        finding = {
            "domain": domain,
            "provider": provider_name,
            "model": model,
            "path": path,
            "line": line,
            "severity": sev,
            "body": body,
        }
        findings_list.append(finding)
        return f"Recorded {sev} finding at {path}:{line}"

    @tool
    async def record_note(body: str) -> str:
        """Record a general observation not tied to a specific line.

        Args:
            body: Markdown text of the note.

        Returns:
            Success message.
        """
        note = {
            "domain": domain,
            "provider": provider_name,
            "model": model,
            "body": body,
        }
        notes_list.append(note)
        return "Recorded note"

    try:
        # Load baseline to determine if we have prior findings for this domain
        from pathlib import Path

        runtime_dir = ctx.runtime_dir()
        if not isinstance(runtime_dir, Path):
            runtime_dir = Path(runtime_dir)
        baseline = memory_module.load_baseline(memory_module.baseline_path(runtime_dir, repo, number))

        # Compute is_followup and cur_prior if not already set in state
        # is_followup is True if a baseline exists (regardless of domain-specific findings)
        # cur_prior is the formatted prior findings for THIS domain (may be empty even if baseline exists)
        if not is_followup:
            is_followup = baseline is not None

        if not cur_prior and baseline:
            cur_prior = prompts_module._format_prior(baseline, domain)

        # Select prompt variant: followup requires BOTH is_followup AND cur_prior
        # This ensures follow-up runs with no prior findings for THIS domain get initial_prompt
        if is_followup and cur_prior:
            prompt_text = prompts_module.followup_prompt(domain, repo, number, worktree_path, diff, cur_prior)
        else:
            prompt_text = prompts_module.initial_prompt(domain, repo, number, worktree_path, diff)

        # Build LLM for this branch's provider
        llm = ctx.llm(provider_name)

        # Bind tools: fs_tools + the two recording tools
        fs_tools = shell.fs_tools()
        recording_tools = [record_finding, record_note]
        all_tools = fs_tools + recording_tools

        # Wrap with rate-limit retry AFTER bind_tools (retry sits on outermost ainvoke)
        llm = ratelimit_module.with_rate_limit_retry(llm.bind_tools(all_tools))

        # Run the tool loop
        messages = [HumanMessage(content=prompt_text)]
        await llm.ainvoke(messages)

    except asyncio.CancelledError:
        # CancelledError must propagate (task cancellation)
        raise

    except Exception as exc:
        # Branch failure is isolated: preserve partial findings and add a failed note
        # Type name only (no message) — exception text can carry tokens/credentials
        exc_type_name = type(exc).__name__
        failed_note = {
            "domain": domain,
            "provider": provider_name,
            "model": model,
            "failed": True,
            "body": f"_(reviewer branch {domain}/{provider_name} failed: {exc_type_name})_",
        }
        notes_list.append(failed_note)
        ctx.progress(f"Branch {domain}/{provider_name} failed: {exc_type_name}")

    # Return ONLY reducer fields — no cur_domain/domain/model at top level
    # Attribution rides inside each finding/note dict
    return {"findings": findings_list, "notes": notes_list}
