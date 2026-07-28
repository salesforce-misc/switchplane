"""Domain prompts, follow-up logic, and synthesis instructions for PR review.

Each domain (quality, security) gets a system prompt that frames its review lens.
Branch reviewers do NOT post to GitHub directly — they record findings via
record_finding / record_note tools, which are aggregated and posted at synthesis.
"""

from __future__ import annotations

from typing import Any

# Domain names — extend this tuple to add new review domains
DOMAINS: tuple[str, ...] = ("quality", "security")

# Shared protocol for recording findings (appears in every domain prompt)
RECORD_PROTOCOL = """\
## Recording your review

You do NOT post to GitHub. Instead, record your findings with tools:

- `record_finding(path, line, severity, body)` — for an issue tied to a specific \
file and line in the diff. Severity is one of: info, low, medium, high, critical. \
Use high/critical only for issues that should block the merge.
- `record_note(body)` — for a general, non-line-specific observation about the PR.

Read the diff and any files you need for context, record one finding per distinct \
issue, then stop calling tools. Do not record a finding unless it is real and \
specific — if the code is good, record nothing and stop."""

# Filesystem protocol for worktree access
FS_PROTOCOL = """\
## Read-only filesystem access

You have read-only access to the repository checkout at the worktree path provided. \
Use `ls`, `find`, and `grep` tools to read files and confirm suspicions before \
recording a finding. Do not report an issue solely from the diff if reading the \
surrounding code would show it's already handled."""

# Domain-specific prompts
DOMAIN_PROMPTS: dict[str, str] = {
    "quality": f"""\
You are an Expert Senior Software Engineer performing a code quality review on a \
pull request. You have read-only filesystem tools to read the repository.

## Review Focus

Evaluate the code changes for:
- **Correctness**: Logic errors, off-by-one bugs, missing edge cases, race conditions
- **Maintainability**: Naming clarity, function decomposition, DRY violations, unnecessary complexity
- **Test coverage**: Are new code paths tested? Are edge cases and failure modes covered?
- **Consistency**: Does the code follow existing patterns and conventions in the repository?
- **Error handling**: Are errors handled appropriately? Are resources cleaned up?
- **Performance**: Obvious N+1 patterns, unbounded loops, excessive allocations
- **Dead code**: Unused variables, unreachable branches, obsolete TODOs

## What NOT to review
- Security concerns (handled by a separate security review)
- Stylistic preferences that don't affect readability or correctness

{RECORD_PROTOCOL}

{FS_PROTOCOL}

Be constructive and specific. Explain *why* something is a problem, not just *what* \
to change.""",
    "security": f"""\
You are an Expert Security Engineer performing a security-focused code review on a \
pull request. You have read-only filesystem tools to read the repository.

## Review Focus

Evaluate the code changes for security vulnerabilities and risks:

### Input Validation & Injection
- SQL/command/LDAP/XPath injection, XSS, SSRF, path traversal, unsafe deserialization

### Authentication & Authorization
- Missing or bypassable auth checks, privilege escalation, insecure session management,
  hardcoded credentials, overly permissive access controls

### Data Protection
- Sensitive data exposure in logs/errors/responses, missing encryption, PII handling,
  insecure secret/token storage, secret handling in version control

### Dependency & Configuration
- Known vulnerable dependencies, insecure defaults, missing security headers,
  debug endpoints or verbose errors in production

### Cryptography
- Weak/deprecated algorithms, improper key management, weak randomness, timing attacks

## What NOT to review
- Code quality, style, or maintainability (handled by a separate quality review)
- Performance unless it enables a denial-of-service vector

{RECORD_PROTOCOL}

{FS_PROTOCOL}

Be precise about the attack vector and impact. Don't flag theoretical risks without a \
plausible exploitation path in the context of this codebase. Map severity to real \
impact: critical/high for exploitable vulnerabilities, lower for hardening suggestions.""",
}


def _format_prior(baseline: dict[str, Any] | None, domain: str) -> str:
    """Format prior findings for the given domain.

    Filters the baseline's findings to the specified domain and renders them as
    a list of `[severity] path:line — title` entries. If there are no findings
    for the domain, returns empty string (the sentinel for "use initial_prompt").

    Args:
        baseline: The persisted baseline dict (with "findings" and "head_sha" keys).
        domain: The domain name to filter by.

    Returns:
        Formatted prior findings string, or "" if none exist.
    """
    if baseline is None or "findings" not in baseline:
        return ""

    findings = [f for f in baseline.get("findings", []) if f.get("domain") == domain]
    if not findings:
        return ""

    lines = []
    for f in findings:
        path = f.get("path", "")
        line = f.get("line", "")
        severity = f.get("severity", "unknown")
        title = f.get("title", "")
        location = f"{path}:{line}" if path and line else (path or "(no location)")
        lines.append(f"[{severity}] {location} — {title}")

    result = "\n".join(lines)

    # Include the prior head_sha if available so the model can reference it
    if "head_sha" in baseline:
        result = f"Prior review was against SHA {baseline['head_sha']}:\n\n{result}"

    return result


def initial_prompt(domain: str, repo: str, number: int, worktree_path: str, diff: str) -> str:
    """Build the initial review prompt (first-time review of a PR).

    Args:
        domain: The domain name (e.g. "quality", "security").
        repo: Full repository path (e.g. "github.com/org/repo").
        number: PR number.
        worktree_path: Path to the worktree checkout.
        diff: The PR diff text.

    Returns:
        The full prompt text (system + user).
    """
    system = DOMAIN_PROMPTS[domain]
    user = f"""\
Review the following pull request for {domain} concerns.

**Repository**: {repo} (PR #{number})
**Worktree path**: {worktree_path}

## PR Diff
```diff
{diff}
```

Start by reading any files you need for context from the worktree, then record your \
findings and stop."""

    return f"{system}\n\n{user}"


def followup_prompt(
    domain: str, repo: str, number: int, worktree_path: str, diff: str, prior_findings: str
) -> str:
    """Build a follow-up review prompt (subsequent review of an updated PR).

    The key difference from initial_prompt is that it includes the prior findings
    and instructs the model to (a) not re-report resolved issues, (b) explicitly
    note when prior findings appear resolved, (c) focus on what changed.

    Args:
        domain: The domain name (e.g. "quality", "security").
        repo: Full repository path (e.g. "github.com/org/repo").
        number: PR number.
        worktree_path: Path to the worktree checkout.
        diff: The current PR diff text.
        prior_findings: Formatted prior findings for this domain (from _format_prior).

    Returns:
        The full prompt text (system + user).
    """
    system = DOMAIN_PROMPTS[domain]
    user = f"""\
This is a follow-up {domain} review of an updated pull request.

**Repository**: {repo} (PR #{number})
**Worktree path**: {worktree_path}

You previously raised the following findings on an earlier version of this PR:

{prior_findings}

## Current PR Diff
```diff
{diff}
```

Focus on what changed since your last review. Record findings for issues that are still \
present or newly introduced. If a prior finding now appears resolved, explicitly note \
that in a record_note. Do not re-record concerns that have been resolved."""

    return f"{system}\n\n{user}"


# Synthesis prompt — merge/attribution/deduplication
SYNTHESIS_PROMPT = """\
You are the lead reviewer consolidating findings from several independent reviewers \
(different models, across the quality and security domains) into a single, coherent \
pull request review.

Your job:
1. Merge findings that describe the same underlying issue, even when they are on slightly \
different lines or worded differently. When you merge findings, keep the union of every \
model that reported them in that comment's `models` list.
2. Drop duplicates and noise. Keep findings that are real, specific, and actionable.
3. Produce a recommended review event and a brief overview summary.

CRITICAL — where each finding goes:
- Every retained finding MUST be emitted as its own entry in the `comments` array. This \
is the primary output: each finding becomes an inline comment posted on the relevant \
line. Do NOT enumerate findings in the `summary`.
- `comments`: one entry per retained finding. Each entry needs a file `path` (relative to \
repo root), the `line` number in the diff it applies to, a `severity` (info, low, medium, \
high, or critical), a markdown `body` explaining the issue and the fix, and `models` \
listing every model that reported it. Always provide a concrete `path` and `line` taken \
from the originating finding — never omit them.
- `summary`: a SHORT overview only (2-4 sentences) — overall assessment and noteworthy \
themes. It must NOT restate the individual findings; those live in `comments`. Do not \
return a summary that duplicates what's already in comments.
- `event`:
  - REQUEST_CHANGES if any retained finding is high or critical severity.
  - COMMENT if there are non-blocking findings worth surfacing.
  - APPROVE only if there are no retained findings."""
