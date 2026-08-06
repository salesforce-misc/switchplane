"""Pull request review task with multi-provider fan-out."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from quality._paths import mkdir_private
from switchplane import Field, Task
from switchplane.agent_runtime import AgentContext
from switchplane.config import DEFAULT_PROVIDER, resolve_provider
from switchplane.shell import Shell

DOMAINS = ("quality", "security")
"""Review domains for the fan-out cross-product."""

_SHELL_ALLOWED_COMMANDS = ["git", "gh", "ls", "find", "grep"]
"""Shell allowlist for ReviewTask: git/gh for repo operations, ls/find/grep for fs_tools."""


class ReviewState(BaseModel):
    """LangGraph state for the PR review fan-out.

    Per-branch fields (cur_domain, cur_provider, cur_model, cur_prior) are read-only
    inside branches and NEVER returned — attribution rides inside each finding/note dict.
    Returning non-reducer fields from concurrent branches raises InvalidUpdateError.

    The reducer fields (findings, notes) are the only fields that branches return,
    and their Annotated[list, operator.add] reducer allows concurrent writes to merge.

    ctx and shell are NOT in state — they're passed via closure in build_graph() to
    avoid msgpack serialization errors during checkpointing.
    """

    # Input fields (set once, never updated)
    repo: str = ""
    number: int = 0
    diff: str = ""
    worktree_path: str = ""
    head_sha: str = ""
    authed_user: str = ""
    is_self_review: bool = False
    matrix: list[tuple[str, str]] = []  # list of (provider, model) tuples
    local: bool = False  # If True, skip GitHub writes and emit artifact instead
    error: str | None = None
    is_followup: bool = False

    # Per-branch fields — read-only inside branches, set via Send payload
    cur_domain: str = ""
    cur_provider: str = ""
    cur_model: str = ""
    cur_prior: str = ""  # Formatted prior findings for this domain

    # Reducer fields — concurrent branches merge via operator.add
    findings: Annotated[list[dict[str, Any]], operator.add] = []
    notes: Annotated[list[dict[str, Any]], operator.add] = []

    # Output fields — set by synthesize_and_post, no reducer (runs once after join)
    local_artifact_path: str = ""
    posted_comments: int = 0
    failed_comments: int = 0
    findings_written: int = 0  # Count of findings written to artifact (local) or posted (GitHub)


def _repo_paths(runtime_dir: Path, repo: str) -> tuple[Path, Path]:
    """Clone dir and lock file for a repo. Single source of truth for both
    setup and cleanup — they MUST agree or the lock is not a lock.

    Args:
        runtime_dir: Runtime directory root
        repo: Full repository path (host/org/repo)

    Returns:
        (clone_path, lock_path) tuple
    """
    cache_root = runtime_dir / "repos"
    clone_path = cache_root / repo
    # String-append suffix — never with_suffix on names with dots (repo starts with hostname)
    lock_path = cache_root / f"{repo.replace('/', '-')}.lock"
    return clone_path, lock_path


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

    The matrix is resolved in the setup node (which has ctx) and passed via state,
    so this router never needs ctx.

    Guard order is load-bearing: the error and empty-diff checks short-circuit
    before checking the matrix, so a setup failure never pays for branch dispatch
    and an empty diff skips the fan-out entirely.

    Returns:
        - END if state.error is set (setup failed)
        - "synthesize_and_post" if diff is empty
        - "synthesize_and_post" if matrix is empty (no usable providers)
        - list[Send] dispatching to "review_branch" for each (domain x provider) pair
    """
    error = _state_accessor(state, "error")
    if error:
        return END

    diff = _state_accessor(state, "diff", "")
    if not diff:
        return "synthesize_and_post"

    matrix = _state_accessor(state, "matrix", [])
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


async def setup(ctx: AgentContext, shell: Shell, state: ReviewState) -> dict:
    """Setup node: clone repo, create worktree, fetch metadata, resolve matrix.

    Runs under gh.file_lock to serialize concurrent worktree operations on the same repo.
    The lock path is BESIDE the clone (not inside .git) to avoid conflicts when the
    clone doesn't exist yet.

    Returns dict with: diff, worktree_path, head_sha, authed_user, is_self_review, matrix.
    On error, returns {"error": "..."} which short-circuits the fan-out via route_to_branches.
    """
    from quality import _concurrency
    from quality import gh as gh_module

    repo = state.repo
    number = state.number

    try:
        # Resolve provider matrix (needs ctx, so must happen in a node not the router)
        matrix = _resolve_matrix(ctx)

        # Get runtime dir and derive clone/lock paths
        runtime_dir = ctx.runtime_dir
        clone_path, lock_path = _repo_paths(runtime_dir, repo)
        cache_root = runtime_dir / "repos"
        # Ensure cache_root is private (0o700) — all clones and worktrees under it
        mkdir_private(cache_root, runtime_dir)
        # Pre-create clone parent with 0o700 so gh.py:356's mkdir is a no-op
        mkdir_private(clone_path.parent, runtime_dir)

        # Clone/update and create worktree under lock
        async with _concurrency.file_lock(lock_path):
            # Clone or update the repo
            clone_path = await gh_module.clone_or_update_repo(shell, repo, cache_root)
            # Ensure clone dir is private (gh command uses default umask)
            clone_path.chmod(0o700)

            # Pre-create worktree parent as private (git worktree add uses default umask)
            worktrees_root = clone_path.parent / f"{clone_path.name}.worktrees"
            mkdir_private(worktrees_root, runtime_dir)

            # Create PR worktree
            worktree_path, head_sha = await gh_module.create_pr_worktree(shell, clone_path, number, ctx.task_id)
            # Ensure worktree is private (git worktree add uses default umask)
            worktree_path.chmod(0o700)

        # Fetch metadata (outside lock - just API calls)
        diff = await gh_module.get_pr_diff(shell, repo, number)
        pr_author = await gh_module.get_pr_author(shell, repo, number)
        authed_user = await gh_module.get_authenticated_user(shell, repo)

        # Determine self-review (both must be non-empty)
        is_self_review = bool(pr_author and authed_user and pr_author == authed_user)

        return {
            "diff": diff,
            "worktree_path": str(worktree_path),
            "head_sha": head_sha,
            "authed_user": authed_user,
            "is_self_review": is_self_review,
            "matrix": matrix,
        }

    except Exception as exc:
        # Let KeyError from unknown provider propagate (ctx.fail already called)
        if isinstance(exc, KeyError):
            raise
        # Other errors get wrapped
        # Exception text from Shell.run can echo credentialed remote URLs from
        # git/gh stderr, so redact before propagating to ctx.progress and task state
        from quality import _redact as redact_module

        error_msg = redact_module.redact_secrets(str(exc))
        ctx.progress(f"Setup failed: {error_msg}")
        return {"error": f"Setup failed: {error_msg}"}


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
    local = _state_accessor(state, "local", False)

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
        runtime_dir = ctx.runtime_dir
        baseline = memory_module.load_baseline(memory_module.baseline_path(runtime_dir, repo, number, local=local))

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

        # Guard: worktree_path must be present for fs_tools scope
        if not worktree_path:
            note = {
                "domain": domain,
                "provider": provider_name,
                "model": model,
                "body": "Branch failed: worktree_path missing from state",
                "failed": True,
            }
            notes_list.append(note)
            return {"findings": findings_list, "notes": notes_list}

        # Build LLM for this branch's provider
        llm = ctx.llm(provider_name)

        # Build a worktree-scoped Shell for fs_tools to prevent reading other repos
        # or runtime config. Bare filenames from the model resolve into the worktree.
        fs_shell = Shell(
            allowed_paths=[Path(worktree_path)],
            allowed_commands=_SHELL_ALLOWED_COMMANDS,
            timeout=300.0,
            ctx=ctx,
        )

        # fs_tools returns Tool wrappers; unwrap for bind_tools (which requires BaseTool),
        # but keep wrappers in tool_map for run_tool_loop's progress rendering
        fs_tools = fs_shell.fs_tools()
        recording_tools = [record_finding, record_note]
        tool_map = {t.name: t for t in (fs_tools + recording_tools)}

        # bind_tools requires unwrapped BaseTool instances
        bind_tools_list = [t.tool if hasattr(t, "tool") else t for t in (fs_tools + recording_tools)]

        # Wrap with rate-limit retry AFTER bind_tools (retry sits on outermost ainvoke)
        llm = ratelimit_module.with_rate_limit_retry(llm.bind_tools(bind_tools_list))

        # Run the tool loop until the model produces a final answer
        from switchplane.llm import run_tool_loop

        messages = [HumanMessage(content=prompt_text)]
        await run_tool_loop(
            llm,
            messages,
            tool_map,
            ctx,
            model,
            label=f"{domain}/{provider_name}",
            max_turns=100,
            progress_every=10,
        )

    except asyncio.CancelledError:
        # CancelledError must propagate (task cancellation)
        raise

    except Exception as exc:
        # Branch failure is isolated: preserve partial findings and add a failed note
        # Exception text can carry credentials from PR content or API responses,
        # so redact before logging. A redacted traceback is more diagnostic than
        # type-only at the same disclosure risk.
        import traceback

        from quality import _redact as redact_module

        exc_msg = redact_module.redact_secrets(traceback.format_exc())
        failed_note = {
            "domain": domain,
            "provider": provider_name,
            "model": model,
            "failed": True,
            "body": f"_(reviewer branch {domain}/{provider_name} failed: {type(exc).__name__})_",
        }
        notes_list.append(failed_note)
        ctx.progress(f"Branch {domain}/{provider_name} failed:\n{exc_msg}")

    # Return ONLY reducer fields — no cur_domain/domain/model at top level
    # Attribution rides inside each finding/note dict
    return {"findings": findings_list, "notes": notes_list}


# -- Event resolution security constants -------------------------------------
# These three functions implement the untrusted-model-output invariant:
# the synthesis model's verdict is derived from the PR diff (attacker-controlled),
# so it cannot be trusted verbatim. See memory: llm-review-gate-untrusted-model-output.

_VALID_EVENTS = {"APPROVE", "REQUEST_CHANGES", "COMMENT"}
_EVENT_STRICTNESS = {"APPROVE": 0, "COMMENT": 1, "REQUEST_CHANGES": 2}
_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_BLOCKING_SEVERITIES = {"high", "critical"}


def _synth_event(comments: list[dict]) -> str:
    """Fallback event selection from comment severities.

    Args:
        comments: List of comment dicts (each with optional "severity" key)

    Returns:
        "APPROVE" if no comments, "REQUEST_CHANGES" if any blocking severity,
        otherwise "COMMENT"
    """
    if not comments:
        return "APPROVE"
    # Normalize severity defensively (synthesis output should already be normalized,
    # but this makes the function robust to direct test calls)
    if any(str(c.get("severity", "")).strip().lower() in _BLOCKING_SEVERITIES for c in comments):
        return "REQUEST_CHANGES"
    return "COMMENT"


def _resolve_event(model_event: object, comments: list[dict]) -> str:
    """Pick the review event, treating model output as untrusted.

    The synthesis model's event is derived from findings whose text comes from
    attacker-controlled PR content, so it can't be forwarded to GitHub verbatim:
    - An unrecognized value (typo, wrong case, "REQUEST CHANGES" with a space)
      is discarded in favor of the severity-derived event
    - We never let the model relax the verdict below what the severities justify:
      take max(model_event, synth_event) so an injected diff can't turn the
      reviewer into a rubber stamp
    - APPROVE is never returned: a prompt-injected diff could suppress every
      record_finding call and earn a clean run, so APPROVE → COMMENT

    Args:
        model_event: Event string from synthesis model (may be invalid)
        comments: List of comment dicts for severity-based fallback

    Returns:
        One of: "REQUEST_CHANGES", "COMMENT" (never "APPROVE")
    """
    severity_event = _synth_event(comments)
    event = model_event if isinstance(model_event, str) else ""
    event = event.strip().upper().replace(" ", "_")
    if event not in _VALID_EVENTS:
        return severity_event
    # The model may escalate, but can never relax the gate below what the
    # retained severities justify
    resolved = max(event, severity_event, key=_EVENT_STRICTNESS.get)
    # APPROVE is never submitted (security invariant)
    if resolved == "APPROVE":
        return "COMMENT"
    return resolved


def _effective_event(event: str, is_self_review: bool) -> str:
    """Clamp the review event to what we are permitted to submit.

    Two independent clamps, both landing on COMMENT:
    1. APPROVE is never submitted. A prompt-injected diff could suppress every
       record_finding call and earn a clean run. If approvals counted toward
       branch protection that would be a merge-gate bypass.
    2. On a self-authored PR GitHub forbids APPROVE and REQUEST_CHANGES with
       HTTP 422, so REQUEST_CHANGES is also downgraded to COMMENT there.

    Args:
        event: Resolved event from _resolve_event
        is_self_review: Whether the PR author is the authenticated user

    Returns:
        One of: "COMMENT", "REQUEST_CHANGES"
    """
    if event == "APPROVE":
        return "COMMENT"
    if is_self_review and event == "REQUEST_CHANGES":
        return "COMMENT"
    return event


# -- Synthesis and posting ----------------------------------------------------


def _comments_from_findings(findings: list[dict]) -> list[dict]:
    """Build comments directly from raw findings, merging by (path, line).

    Used when synthesis returns no structured comments: the raw findings are
    posted instead of being lost. Findings on the same path+line are merged —
    the highest severity wins, every contributing model is listed, and distinct
    bodies are concatenated.

    Args:
        findings: List of finding dicts from review branches

    Returns:
        List of merged comment dicts
    """
    merged: dict[tuple, dict] = {}
    for f in findings:
        path = f.get("path") or ""
        line = f.get("line")
        key = (path, line)
        # Normalize severity defensively (findings from record_finding are already normalized)
        sev = str(f.get("severity", "medium")).strip().lower()
        body = f.get("body", "")
        model = f.get("model", "")
        if key not in merged:
            merged[key] = {
                "path": path,
                "line": line,
                "severity": sev,
                "body": body,
                "models": [model] if model else [],
            }
            continue
        existing = merged[key]
        if _SEVERITY_ORDER.get(sev, 2) > _SEVERITY_ORDER.get(existing["severity"], 2):
            existing["severity"] = sev
        if body and body not in existing["body"]:
            existing["body"] = f"{existing['body']}\n\n{body}" if existing["body"] else body
        if model and model not in existing["models"]:
            existing["models"].append(model)
    return list(merged.values())


def _model_attrib(models: list[str]) -> str:
    """Render a [m1 | m2] attribution suffix from a model id list.

    De-duplicates while preserving order. Empty when there are no models.

    Args:
        models: List of model ids

    Returns:
        Attribution string like " [model-1 | model-2]" or "" if empty
    """
    uniq: list[str] = []
    for m in models:
        if m and m not in uniq:
            uniq.append(m)
    if not uniq:
        return ""
    return f" [{' | '.join(uniq)}]"


def _coerce_line(line: object) -> int | None:
    """Best-effort coercion of an LLM-supplied line to an int, else None.

    Synthesis output is model-generated, so line may be missing or a non-numeric
    string ("N/A", "multiple"). Returning None lets the caller fold the comment
    into the summary rather than crash the task.

    Args:
        line: Value from synthesis model (may be None, int, or str)

    Returns:
        Integer line number or None
    """
    if line is None:
        return None
    try:
        return int(line)
    except (TypeError, ValueError):
        return None


def _is_commentable(commentable: dict, path: str, line: int) -> bool:
    """Whether GitHub will accept a line comment at (path, line).

    Args:
        commentable: Dict from commentable_lines (path -> set of line numbers)
        path: File path
        line: Line number

    Returns:
        True if the line is in the diff and commentable
    """
    return line in commentable.get(path, set())


def _render_unpostable(comments: list[dict]) -> str:
    """Render comments that can't be posted inline as a markdown section.

    GitHub rejects review comments on lines outside the diff (HTTP 422), so
    those findings are folded into the review summary body instead of dropped.

    Args:
        comments: List of comment dicts that couldn't be posted

    Returns:
        Markdown section text (empty string if no comments)
    """
    if not comments:
        return ""
    lines = ["", "---", "", "### Additional findings (not on changed lines)", ""]
    for c in comments:
        raw_path = c.get("path")
        path = raw_path or "(general)"
        loc = path
        line = _coerce_line(c.get("line"))
        if raw_path and line is not None:
            loc += f":{line}"
        sev = c.get("severity", "medium")
        attrib = _model_attrib(c.get("models", [])).strip()
        suffix = f" {attrib}" if attrib else ""
        lines.append(f"- **{loc}** [{sev}] — {c.get('body', '')}{suffix}")
    return "\n".join(lines)


def _render_notes(notes: list[dict]) -> str:
    """Render reviewer notes (including failures) as a markdown section.

    Partial outage must be disclosed deterministically — if security ran and quality
    failed, the review summary must surface that quality failed. The synthesis model
    cannot be trusted to preserve notes verbatim, so we render them directly.

    Args:
        notes: List of note dicts from branches (may include failed=True)

    Returns:
        Markdown section text (empty string if no notes)
    """
    if not notes:
        return ""
    lines = ["", "---", "", "### Reviewer notes", ""]
    for n in notes:
        domain = n.get("domain", "")
        provider = n.get("provider", "")
        body = n.get("body", "")
        failed = n.get("failed", False)
        prefix = f"**{domain}/{provider}**:"
        if failed:
            prefix = f"⚠️ {prefix}"
        lines.append(f"- {prefix} {body}")
    return "\n".join(lines)


async def _existing_comment_lines(
    shell: Shell, repo: str, pr_number: int, task_name: str, authed_user: str
) -> set[tuple]:
    """(path, line) pairs already carrying a prior comment from this task.

    Used to skip re-posting still-valid findings on a follow-up run. Requires BOTH:
    - the task-specific marker "**quality/{task_name}**"
    - the comment to be authored by authed_user

    The marker is publicly visible, so the PR author could pre-seed a comment
    containing it to suppress a finding on exactly the line carrying their bug.
    Gating on author identity closes that suppression oracle. When authed_user
    is unknown we skip dedup entirely rather than trust the spoofable marker alone.

    Args:
        repo_path: Path to the repository (unused in production, kept for test compat)
        pr_number: PR number
        task_name: Task name for marker (e.g. "review")
        authed_user: Authenticated user login

    Returns:
        Set of (path, line) tuples with existing comments
    """
    from quality import gh as gh_module

    if not authed_user:
        return set()

    marker = f"**quality/{task_name}**"
    try:
        comments = await gh_module.list_review_comments(shell, repo, pr_number)
    except Exception:
        # Best-effort: any failure returns empty so a transient API error
        # never blocks the review (at worst we re-post)
        return set()

    seen: set[tuple] = set()
    for c in comments:
        body = c.get("body") or ""
        path = c.get("path")
        line = c.get("line")
        user_obj = c.get("user")

        # GitHub API returns user as a dict with {"login": ..., "id": ..., ...}.
        # Extract login string for comparison. Tolerate flat string for test stubs.
        login = user_obj.get("login") if isinstance(user_obj, dict) else user_obj

        if path and line is not None and marker in body and login and login == authed_user:
            seen.add((path, line))
    return seen


def _persist_baseline(
    ctx, memory_module, comments, findings, *, runtime_dir, repo, number, head_sha, summary, local
) -> None:
    """Build domain attribution and persist baseline for follow-up runs.

    Implements the three-tier attribution fallback: exact (path, line), path-level, all-domains.
    """
    # Build exact-line attribution map and path-level fallback
    domains_by_loc: dict[tuple, set[str]] = {}
    domains_by_path: dict[str, set[str]] = {}
    for f in findings:
        path = f.get("path", "")
        line = _coerce_line(f.get("line"))
        domain = f.get("domain", "")
        domains_by_loc.setdefault((path, line), set()).add(domain)
        domains_by_path.setdefault(path, set()).add(domain)

    persisted = []
    for c in comments:
        path = c.get("path", "")
        line = _coerce_line(c.get("line"))
        # Try exact (path, line) match first, fall back to path-level attribution
        domains_at_loc = domains_by_loc.get((path, line), set())
        if not domains_at_loc and path:
            # Synthesis consolidated findings across lines — use path-level attribution
            domains_at_loc = domains_by_path.get(path, set())

        domains_sorted = sorted(domains_at_loc)
        if not domains_sorted and findings:
            # Last-resort fallback: union of all domains that ran
            all_domains = {f.get("domain", "") for f in findings if f.get("domain")}
            domains_sorted = sorted(all_domains)
            ctx.progress(f"WARNING: Empty domain attribution for {path}:{line}, using fallback: {domains_sorted}")

        persisted.append(
            {
                "domains": domains_sorted,
                "domain": domains_sorted[0] if domains_sorted else "",
                "path": path,
                "line": line,
                "severity": c.get("severity", "medium"),
                "title": c.get("body", "")[:80],
            }
        )
    memory_module.save_baseline(
        runtime_dir,
        repo=repo,
        number=number,
        head_sha=head_sha,
        summary=summary,
        findings=persisted,
        local=local,
    )


async def synthesize_and_post(ctx: AgentContext, shell: Shell, state: dict | ReviewState) -> dict:
    """Fan-in: consolidate findings, post line comments + a single review.

    This node is the graph's only writer to GitHub and is not fully idempotent.
    If the task crashes after posting but before the node returns, a checkpoint
    resume re-enters here and re-runs synthesis. Inline line comments are
    de-duplicated against already-posted comments, so those aren't doubled — but
    the top-level review would be submitted a second time. That is a benign
    duplicate (an extra COMMENT review), not corruption, so we accept it rather
    than thread a "review already submitted" flag through checkpointed state.

    Args:
        ctx: Agent context
        shell: Shell instance for gh/git operations
        state: Review state (dict or ReviewState)

    Returns:
        Dict with updated state fields (error, posted_comments, failed_comments)
    """
    from quality import _redact as redact_module
    from quality import gh as gh_module
    from quality.agents.pr import memory as memory_module
    from quality.agents.pr import prompts as prompts_module

    error = _state_accessor(state, "error")
    if error:
        return {}

    repo = _state_accessor(state, "repo")
    number = _state_accessor(state, "number")
    diff = _state_accessor(state, "diff", "")
    head_sha = _state_accessor(state, "head_sha", "")
    findings = _state_accessor(state, "findings", [])
    notes = _state_accessor(state, "notes", [])
    is_self_review = _state_accessor(state, "is_self_review", False)
    authed_user = _state_accessor(state, "authed_user", "")
    local = _state_accessor(state, "local", False)
    matrix = _state_accessor(state, "matrix", [])

    # Total reviewer outage: every branch raised, so we have failure notes but
    # zero findings. That is NOT a clean PR — posting "review complete" would
    # mislead and persist an empty baseline. Surface it as an error.
    if notes and all(n.get("failed") for n in notes) and not findings:
        failed = ", ".join(f"{n['domain']}/{n['provider']}" for n in notes if n.get("failed"))
        ctx.progress("All reviewer branches failed — review could not be completed")
        return {"error": f"All reviewer branches failed ({failed}); no review performed."}

    if not findings and not notes:
        # Empty provider matrix: no reviewers ran because no provider has an api_key.
        # This is NOT a clean PR — posting "no issues found" or persisting an empty
        # baseline would misrepresent the review state. Surface as configuration error.
        if not matrix:
            ctx.progress("No reviewers ran — no provider has an api_key configured")
            return {
                "error": "No LLM provider has an api_key configured. Add an api_key to "
                "[llm] or at least one [llm.providers.<name>] entry in ~/.quality/config.toml"
            }

        # Clean PR: reviewers ran and found nothing
        if local:
            ctx.progress("No findings (local mode)")
            # Write empty artifact
            runtime_dir = ctx.runtime_dir
            artifact_dir = runtime_dir / "reviews" / repo
            mkdir_private(artifact_dir, runtime_dir)
            artifact_path = artifact_dir / f"pr-{number}.md"
            artifact_path.write_text(f"# PR #{number}: {repo}\n\n✅ No quality or security issues found.\n")
            artifact_path.chmod(0o600)

            # Persist an empty baseline (same as GitHub branch below)
            memory_module.save_baseline(
                runtime_dir,
                repo=repo,
                number=number,
                head_sha=head_sha,
                summary="No quality or security issues found.",
                findings=[],
                local=local,
            )
            return {"local_artifact_path": str(artifact_path), "findings_written": 0}
        else:
            ctx.progress("No findings — submitting approval")
            event = _effective_event("APPROVE", is_self_review)
            body = "No quality or security issues found."
            try:
                await gh_module.submit_pr_review(shell, repo, number, event, body)
            except Exception as exc:
                exc_msg = redact_module.redact_secrets(str(exc))
                ctx.progress(f"Failed to submit review: {exc_msg}")

            # Persist an empty baseline
            runtime_dir = ctx.runtime_dir
            memory_module.save_baseline(
                runtime_dir,
                repo=repo,
                number=number,
                head_sha=head_sha,
                summary="No quality or security issues found.",
                findings=[],
                local=local,
            )
            return {}

    # Synthesize with the default provider
    ctx.progress(f"Synthesizing {len(findings)} finding(s)...")
    llm = ctx.llm()

    notes_text = "\n".join(f"- ({n['domain']}/{n.get('provider', '?')}) {n['body']}" for n in notes) or "(none)"
    import json

    synthesis_prompt = prompts_module.SYNTHESIS_PROMPT
    user_prompt = f"""\
The reviewers produced the following raw findings (JSON array of dicts):

```json
{json.dumps(findings, indent=2)}
```

And the following notes:

{notes_text}

Consolidate these into a single review: merge findings that describe the same issue, \
keep findings that are real and actionable, drop noise. Emit the result as:
- `summary`: 2-4 sentence overview (no enumeration of findings)
- `event`: APPROVE | REQUEST_CHANGES | COMMENT
- `comments`: array of {{path, line, severity, body, models}}
"""

    # Create a Pydantic model for structured output
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field as PydField

    class SynthComment(PydanticBaseModel):
        path: str = PydField("", description="File path relative to repo root")
        line: int | None = PydField(None, description="Line number in the diff")
        severity: str = PydField("medium", description="info, low, medium, high, or critical")
        body: str = PydField("", description="Markdown comment explaining the issue")
        models: list[str] = PydField(default_factory=list, description="Models that reported this")

    class SynthResult(PydanticBaseModel):
        summary: str = PydField("", description="Markdown review-body summary")
        event: str = PydField("COMMENT", description="APPROVE, REQUEST_CHANGES, or COMMENT")
        comments: list[SynthComment] = PydField(default_factory=list)

    from quality import ratelimit as ratelimit_module

    structured = ratelimit_module.with_rate_limit_retry(llm.with_structured_output(SynthResult))

    messages = [
        {"role": "system", "content": synthesis_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Retry synthesis a few times on malformed tool calls
    import asyncio
    import random

    for attempt in range(1, 4):
        try:
            result: SynthResult = await structured.ainvoke(messages)
            synthesized = result.model_dump()
            break
        except Exception as exc:
            exc_msg = redact_module.redact_secrets(str(exc))
            ctx.progress(f"Synthesis attempt {attempt} failed: {exc_msg}")
            if attempt < 3:
                await asyncio.sleep(2.0**attempt + random.uniform(0, 1))
    else:
        # All attempts failed
        ctx.progress("Synthesis failed after retries — using raw findings")
        synthesized = {"summary": "", "event": "COMMENT", "comments": []}

    comments = synthesized.get("comments", [])
    summary = synthesized.get("summary", "")
    event = synthesized.get("event")

    # Redact secrets from synthesis output immediately (before any use)
    # All downstream consumers (artifact, review body, baseline) are then safe by construction
    summary = redact_module.redact_secrets(summary)
    for c in comments:
        if "body" in c:
            c["body"] = redact_module.redact_secrets(c["body"])
        # Normalize severity to lowercase and strip whitespace (model returns untrusted case)
        if "severity" in c:
            c["severity"] = str(c["severity"]).strip().lower()

    # Defensive fallback: synthesis sometimes returns no structured comments
    if not comments and findings:
        ctx.progress("Synthesis returned no comments — deriving from raw findings")
        comments = _comments_from_findings(findings)
        # Redact and normalize fallback-generated comments too
        for c in comments:
            if "body" in c:
                c["body"] = redact_module.redact_secrets(c["body"])
            if "severity" in c:
                c["severity"] = str(c["severity"]).strip().lower()

    # Resolve event (untrusted model output)
    event = _effective_event(_resolve_event(event, comments), is_self_review)

    # Parse commentable lines from the diff (needed for both local and GitHub modes)
    commentable = gh_module.commentable_lines(diff)

    # In local mode, write artifact instead of posting to GitHub
    if local:
        runtime_dir = ctx.runtime_dir
        artifact_dir = runtime_dir / "reviews" / repo
        mkdir_private(artifact_dir, runtime_dir)
        artifact_path = artifact_dir / f"pr-{number}.md"

        # Build markdown artifact
        lines = [f"# PR #{number}: {repo}\n"]
        lines.append(f"**Event**: {event}\n")
        if summary:
            lines.append(f"## Summary\n\n{summary}\n")
        if comments:
            lines.append(f"## Findings ({len(comments)})\n")
            for c in comments:
                sev = c.get("severity", "medium")
                path = c.get("path", "")
                line = c.get("line")
                body = c.get("body", "")
                models = c.get("models", [])
                attrib = _model_attrib(models) if models else ""
                lines.append(f"\n### {sev.upper()}: {path}:{line}\n\n{body}\n")
                if attrib:
                    lines.append(f"\n{attrib}\n")

        # Render reviewer notes (including failures) — partial outage must be disclosed
        notes_section = _render_notes(notes)
        if notes_section:
            lines.append(notes_section)

        artifact_path.write_text("".join(lines))
        artifact_path.chmod(0o600)
        ctx.progress(f"Wrote artifact: {artifact_path}")

        # Persist baseline even in local mode
        _persist_baseline(
            ctx,
            memory_module,
            comments,
            findings,
            runtime_dir=runtime_dir,
            repo=repo,
            number=number,
            head_sha=head_sha,
            summary=summary,
            local=local,
        )

        return {"local_artifact_path": str(artifact_path), "findings_written": len(comments)}

    # GitHub mode: load existing comments to skip duplicates
    already = await _existing_comment_lines(shell, repo, number, "review", authed_user)

    # Post each consolidated line comment
    posted = 0
    failed = 0
    skipped = 0
    unpostable: list[dict] = []

    # Sort comments by severity (descending), then path, then line for determinism
    def comment_sort_key(c: dict):
        sev = str(c.get("severity", "medium")).strip().lower()
        path = c.get("path", "")
        line = _coerce_line(c.get("line")) or 0
        return (-_SEVERITY_ORDER.get(sev, 2), path, line)

    sorted_comments = sorted(comments, key=comment_sort_key)

    for c in sorted_comments:
        path = c.get("path")
        line = _coerce_line(c.get("line"))
        if not path or line is None or not _is_commentable(commentable, path, line):
            unpostable.append(c)
            continue
        if (path, line) in already:
            skipped += 1
            continue

        body = c.get("body", "")
        attrib = _model_attrib(c.get("models", []))
        if attrib:
            body = f"{body}\n\n---\n{attrib}"

        # Redact secrets from the body before posting
        body = redact_module.redact_secrets(body)
        body += "\n\n**quality/review**"

        try:
            await gh_module.create_pr_review_comment(shell, repo, number, body, path, line, head_sha)
            posted += 1
        except Exception as exc:
            exc_msg = redact_module.redact_secrets(str(exc))
            ctx.progress(f"Failed to post comment on {path}:{line}: {exc_msg}")
            failed += 1
            unpostable.append(c)

    if skipped:
        ctx.progress(f"Skipped {skipped} existing comment(s)")

    # Build review body
    review_body = summary or "Review complete."
    review_body += _render_unpostable(unpostable)
    review_body += _render_notes(notes)
    # Redact secrets from the review body before posting
    review_body = redact_module.redact_secrets(review_body)

    # Submit the consolidated review
    try:
        await gh_module.submit_pr_review(shell, repo, number, event, review_body)
        ctx.progress(
            f"Submitted {event} review with {posted} inline comment(s)"
            + (f", {len(unpostable)} folded into summary" if unpostable else "")
        )
    except Exception as exc:
        exc_msg = redact_module.redact_secrets(str(exc))
        ctx.progress(f"Failed to submit review: {exc_msg}")
        return {"error": f"Failed to submit {event} review: {exc_msg}"}

    # Persist the synthesized comments as the new baseline
    runtime_dir = ctx.runtime_dir
    _persist_baseline(
        ctx,
        memory_module,
        sorted_comments,
        findings,
        runtime_dir=runtime_dir,
        repo=repo,
        number=number,
        head_sha=head_sha,
        summary=summary,
        local=local,
    )

    return {"posted_comments": posted, "failed_comments": failed, "findings_written": posted}


def build_graph(ctx: AgentContext, shell: Shell) -> StateGraph:
    """Build the LangGraph StateGraph for PR review.

    Creates 1-arg node adapters that close over ctx and shell. The underlying functions
    remain 3-arg (ctx, shell, state), and ctx/shell are NOT checkpointed (they're passed
    via closure to avoid msgpack serialization errors).

    Args:
        ctx: Agent context (not checkpointed - passed via closure)
        shell: Shell instance (not checkpointed - passed via closure)

    Returns:
        Uncompiled StateGraph (caller compiles with checkpointer)
    """

    # Create 1-arg adapters that close over ctx and shell
    def make_node(fn):
        async def adapter(state):
            return await fn(ctx, shell, state)

        return adapter

    graph = StateGraph(ReviewState)
    graph.add_node("setup", make_node(setup))
    graph.add_node("review_branch", make_node(review_branch))
    graph.add_node("synthesize_and_post", make_node(synthesize_and_post))

    # Topology: __start__ → setup → [route_to_branches] → review_branch* | synthesize_and_post | END
    #                                                       ↓
    #                                                  synthesize_and_post → END
    graph.set_entry_point("setup")
    graph.add_conditional_edges(
        "setup",
        route_to_branches,
        ["review_branch", "synthesize_and_post", END],
    )
    # Fan-in: all review branches flow to synthesis
    graph.add_edge("review_branch", "synthesize_and_post")
    # Synthesis is terminal
    graph.add_edge("synthesize_and_post", END)

    return graph


# -- Task entry point ---------------------------------------------------------


class ReviewTask(Task):
    """Review a pull request with multi-provider fan-out."""

    name = "review"
    description = "Multi-model code review on a pull request (fan-out/fan-in)"

    pr: str = Field(description="PR URL to review")
    local: bool = Field(
        default=False,
        description="If true, skip GitHub writes and emit a Markdown artifact instead",
    )

    def startup_info(self) -> dict:
        return {"pr": self.pr, "local": self.local}

    async def run(self, ctx: AgentContext) -> None:
        """Execute the PR review task lifecycle.

        The worktree cleanup runs in a finally block to guarantee execution on
        success, exception, AND asyncio.CancelledError. Cleanup failures are caught
        and logged via ctx.progress so they never mask the original error.

        Checkpoint/local-flag mismatch guard: a task checkpointed with --local and
        resumed without it (or vice versa) fails fast with a clear message before
        executing the graph.
        """
        from quality import gh as gh_module

        # Stash ctx for _cleanup_worktree access
        # (ctx must not go in checkpointed state - msgpack cannot serialize it)
        self._ctx = ctx

        # Fail fast if no API key configured anywhere (top-level or pool)
        llm_config = ctx.config.get("llm", {})
        pool = llm_config.get("providers") or {}
        has_key = bool(llm_config.get("api_key")) or any(p.get("api_key") for p in pool.values() if isinstance(p, dict))
        if not has_key:
            ctx.fail(
                "No LLM API key configured. Set [llm].api_key, or an api_key on at "
                "least one [llm.providers.<name>] entry, in ~/.quality/config.toml"
            )
            return

        # Parse and validate the PR URL against the allowed_hosts config
        allowed_hosts = ctx.config.get("review", {}).get("allowed_hosts")
        if not allowed_hosts:
            from quality import config as config_module

            allowed_hosts = list(config_module.DEFAULT_ALLOWED_HOSTS)

        try:
            repo, number = gh_module.parse_pr_url(self.pr, allowed_hosts)
        except ValueError as exc:
            # Defense-in-depth: redact any credentials that might appear in the error.
            # The source fix in gh.parse_pr_url rejects userinfo before building any
            # error message, so this redaction should be a no-op in practice.
            from quality import _redact as redact_module

            ctx.fail(redact_module.redact_secrets(str(exc)))
            return

        # Construct Shell with security-relevant allowlists
        # git/gh for repo operations, ls/find/grep for fs_tools (LLM filesystem access)
        # All commands restricted to runtime_dir via allowed_paths
        runtime_dir = ctx.runtime_dir

        shell = Shell(
            allowed_paths=[runtime_dir / "repos"],
            allowed_commands=_SHELL_ALLOWED_COMMANDS,
            timeout=300.0,  # 5 minutes for potentially slow git operations
            ctx=ctx,  # Pass ctx so shell output is observable
        )

        # Stash shell for _cleanup_worktree access
        self._shell = shell

        # Build the graph with ctx and shell closed over
        graph = build_graph(ctx, shell).compile(checkpointer=ctx.checkpointer)
        config = {"configurable": {"thread_id": ctx.task_id}}

        result: dict = {}
        try:
            # Check for existing checkpoint
            existing = await graph.aget_state(config)
            if existing.values:
                # Checkpoint exists — guard against local flag mismatch
                persisted_local = bool(existing.values.get("local", False))
                if persisted_local != self.local:
                    ctx.fail(
                        f"Checkpointed review has local={persisted_local} but this run "
                        f"requested local={self.local}. Refusing to resume — the persisted "
                        "run's GitHub-write choice can't be flipped mid-run. "
                        "Start a fresh task to change the mode."
                    )
                    return

                ctx.progress("Resuming from checkpoint...")
                # Pass None to ainvoke when resuming from checkpoint
                result = await graph.ainvoke(None, config)
            else:
                # Fresh run — create initial state with minimal fields
                # The setup node populates diff, worktree_path, head_sha, authed_user,
                # is_self_review, and matrix
                initial_state = ReviewState(
                    repo=repo,
                    number=number,
                    local=self.local,
                )

                result = await graph.ainvoke(initial_state, config)

        finally:
            # Unconditional cleanup: remove the per-PR worktree even if ainvoke raised
            # (gateway outage, cancellation, etc.). result is empty on failure, so recover
            # worktree_path/target_repo from the persisted graph state instead.
            # Wrap in try/except so cleanup failure never masks the original error.
            try:
                await self._cleanup_worktree(graph, config, result)
            except Exception:
                # Cleanup failed — logged inside _cleanup_worktree via ctx.progress.
                # Don't re-raise: the original error (if any) must propagate.
                pass

        # Check for errors
        if result.get("error"):
            ctx.fail(result["error"])
            return

        # Complete with results
        completion = {
            "pr": self.pr,
            "repo": repo,
            "number": number,
            "raw_findings": len(result.get("findings", [])),
            "failed_comments": result.get("failed_comments", 0),
            "self_review": result.get("is_self_review", False),
        }

        # Conditional fields based on local mode
        if self.local:
            completion["local_artifact_path"] = result.get("local_artifact_path", "")
            completion["findings_written"] = result.get("findings_written", 0)
        else:
            completion["posted_comments"] = result.get("posted_comments", 0)

        ctx.complete(completion)

    async def _cleanup_worktree(self, graph, config: dict, result: dict) -> None:
        """Remove the per-PR worktree, best-effort — never fails the task.

        Prefers the worktree path from the completed graph result; if ainvoke
        raised before returning, result is empty, so fall back to the checkpointed
        graph state.

        Cleanup runs under the repo lock (same as creation) to avoid racing
        concurrent reviews of the same repo. A git worktree prune from one review
        can race another's git worktree add if cleanup doesn't hold the lock.

        Cleanup failure is caught and logged via ctx.progress so it never masks
        the original error that triggered unwinding.

        Args:
            graph: Compiled StateGraph
            config: LangGraph config dict with thread_id
            result: Result dict from ainvoke (may be empty on failure)
        """
        from quality import _concurrency
        from quality import _redact as redact_module
        from quality import gh as gh_module

        # Access ctx via self._ctx (stashed in run())
        ctx = self._ctx
        worktree = result.get("worktree_path")
        repo = result.get("repo", "")

        if not worktree:
            try:
                snapshot = await graph.aget_state(config)
                worktree = snapshot.values.get("worktree_path")
                repo = snapshot.values.get("repo", repo)
            except Exception:
                # Can't retrieve worktree path, nothing to clean
                return

        if not worktree or not repo:
            return

        try:
            runtime_dir = ctx.runtime_dir
            shell = self._shell

            clone_path, lock_path = _repo_paths(runtime_dir, repo)

            # Hold the repo lock across cleanup (same as creation)
            async with _concurrency.file_lock(lock_path):
                await gh_module.remove_worktree(shell, clone_path, Path(worktree))
        except Exception as exc:
            # Cleanup failure must not mask the original error
            exc_msg = redact_module.redact_secrets(str(exc))
            ctx.progress(f"WORKTREE CLEANUP FAILED (disk may leak): {exc_msg}")
