# Quality — PR review with multi-model fan-out

**Quality** is a Switchplane application that performs automated code review on pull requests. It fans out the review across multiple LLM providers (e.g. Anthropic Claude, OpenAI GPT) and multiple review domains (code quality and security), synthesizes the findings into a single review, and either posts it to GitHub or saves it as a local Markdown artifact.

This is a real-world example of LangGraph's concurrent fan-out/fan-in topology, LLM provider pooling, and LLM-driven tool use with sandboxed filesystem access. It demonstrates how to structure a multi-step agent workflow that coordinates multiple LLMs while maintaining deterministic control over what gets posted and where.

## What it does

1. **Multi-provider fan-out**: The review graph fans out to (providers × domains), invoking each LLM independently. If you configure two providers (Claude Opus, GPT-5.5) and two domains (quality, security), you get four concurrent review branches.
2. **Domain-specific prompts**: Each domain (quality vs. security) gets its own review lens. Quality reviewers focus on correctness, maintainability, test coverage, and error handling. Security reviewers focus on auth, injection, data exposure, and crypto.
3. **LLM tool use**: Each branch gets read-only filesystem tools (ls, find, grep) and two recording tools (record_finding, record_note) to structure its output. The LLM reads files from the PR's worktree and records findings as structured data, not freeform text.
4. **Synthesis and deduplication**: After all branches complete, a synthesis node (driven by another LLM call) merges duplicate findings, drops noise, and resolves the final review event (APPROVE, COMMENT, or REQUEST_CHANGES) by cross-checking the model's verdict against the actual finding severities.
5. **Follow-up review with memory**: The baseline from the prior review is persisted and reloaded on subsequent runs. Each branch is prompted with its domain's prior findings and instructed not to re-report resolved issues.
6. **GitHub posting or local artifact**: By default, the synthesized review is posted to GitHub via inline comments and a labeled review submission. With `--local`, the review is saved to a Markdown file under `~/.quality/reviews/` instead — useful for iterating on prompts or reviewing PRs you don't have push access to.

## Installation

```bash
# From the switchplane repo root
uv pip install -e . -e examples/quality

# Or from the quality directory
cd examples/quality
uv venv .venv && source .venv/bin/activate
uv pip install -e ../.. -e .
```

## Configuration

Quality ships with app defaults at `examples/quality/quality/config.toml` that define the provider pool and default model. You provide your personal config at `~/.quality/config.toml` with API keys:

```toml
# ~/.quality/config.toml
[llm]
api_key = "sk-ant-..."

# The review fans out across every configured pool entry × every domain.
# Each entry needs its own api_key. Entries without one are skipped with
# a progress note rather than failing the review.
[llm.providers.opus]
api_key = "sk-ant-..."

[llm.providers.gpt]
api_key = "sk-..."
```

The shipped defaults configure two providers:
- `opus` → `claude-opus-4-8`
- `gpt` → `gpt-5.5`

If both have API keys, the review fans out to 4 branches (2 providers × 2 domains). If you set only `[llm].api_key` and no pool entries, the review runs with a single provider on the default model.

## Usage

### Review a pull request and post to GitHub

```bash
quality run pr review --pr https://github.com/org/repo/pull/123
```

This clones the repo (or updates an existing clone), creates a worktree for the PR's head commit, fans out the review across (providers × domains), synthesizes the findings, and posts the review to GitHub. Events stream inline. Ctrl+C detaches without killing the task.

### Review locally (no GitHub writes)

```bash
quality run pr review --pr https://github.com/org/repo/pull/123 --local
```

The `--local` flag skips all GitHub writes (no inline comments, no review submission) and saves the synthesized review to a Markdown artifact at:

```
~/.quality/reviews/<repo>/pr-<number>.md
```

**Note:** The artifact filename does not include the head SHA, so a follow-up review on the same PR overwrites the previous artifact. Each run replaces the prior local artifact for that PR.

This is useful for:
- Iterating on review prompts without spamming the PR
- Reviewing PRs on repos you don't have push access to
- Generating a review you want to manually edit before posting

The artifact directory mode is `0o700` and the file mode is `0o600` (secrets are redacted, but defense-in-depth applies).

### Follow-up review

Run the same command again after the PR is updated. The task loads the prior baseline (keyed by repo + PR number), passes prior findings to each domain's LLM branch, and prompts it to focus on what changed. Resolved findings are not re-reported.

The baseline is persisted at:
```
~/.quality/state/review/<repo>/pr-<number>.json        (GitHub mode)
~/.quality/state/review/<repo>/pr-<number>.local.json  (--local mode)
```

Local and non-local baselines are kept separate: a GitHub-posted baseline does not degrade a subsequent `--local` run to a follow-up, and vice versa. Each mode is idempotent with respect to itself.

Each domain's findings are tracked with a `domains` list for attribution. When synthesis consolidates findings across lines, the three-tier attribution strategy (exact location → path-level → last-resort fallback) ensures follow-up dedup works correctly.

### Detached execution

```bash
quality run pr review --pr https://github.com/org/repo/pull/123 -d
```

The `-d` flag (or `--detach`) submits the task and returns immediately. The review runs in the background. Check on it with:

```bash
quality task list
quality task follow <task_id>
```

### Full-screen TUI

Invoke `quality` with no subcommand to open the full-screen TUI dashboard, which shows all running tasks, streams events in real time, and lets you submit new reviews with `:run pr review --pr <url>`.

## Architecture

The review graph has this topology:

```
setup → route_to_branches (fan-out) → [review_branch × (providers × domains)] → synthesize_and_post → END
```

**setup**: Clones the repo (or updates an existing clone under `~/.quality/repos/`), creates a PR worktree, fetches the diff and PR metadata, resolves the provider matrix from config, and returns. Setup runs under a per-repo file lock to serialize concurrent worktree operations.

**route_to_branches**: A conditional edge that inspects the setup result. If `error` is set (e.g. repo clone failed), it short-circuits to `synthesize_and_post` with empty findings. Otherwise, it generates one Send per (provider, domain) pair and dispatches them to `review_branch`.

**review_branch**: Each branch loads the baseline (if it exists), selects the appropriate prompt (initial vs follow-up), builds an LLM with the branch's provider, binds filesystem + recording tools, and runs the tool loop. Branch failures are isolated: exceptions are caught, partial findings are preserved, and a failed note is appended. Only asyncio.CancelledError propagates (for task cancellation). Returns only reducer-compatible fields (findings, notes) — attribution rides inside each dict.

**synthesize_and_post**: Takes the aggregated findings and notes (concatenated across all branches), uses an LLM with structured output to merge duplicates and consolidate the review, redacts secrets from the synthesis output, resolves the review event by cross-checking the model's verdict against actual severities, then either posts to GitHub or writes a local artifact. Also saves a new baseline for follow-up review. Returns the output fields expected by the task (posted_comments, failed_comments, local_artifact_path, findings_written).

## Design notes

### Security

**Model output is untrusted.** The synthesis model's verdict (APPROVE/COMMENT/REQUEST_CHANGES) is derived from the PR diff, which is attacker-controlled. The event-resolution logic validates the model's choice by scanning the actual comment severities and escalates to a stricter event if needed. This prevents an attacker from suppressing a critical finding by carefully crafting the diff to manipulate the model into returning APPROVE.

**Secrets are redacted at source.** Immediately after synthesis (before any use), the summary and all comment bodies are passed through `quality._redact.redact_secrets()`, which pattern-matches common credential formats (API keys, tokens, private keys) and replaces them with a redacted placeholder. All downstream consumers (artifact, review body, baseline) are then safe by construction.

**Artifact directories are private.** The local-artifact path is created with `0o700` on every directory component (not just the leaf), and the file itself is written with `0o600`. This prevents other users on the same machine from reading reviews that might contain repo-specific context.

### Deduplication and attribution

Each finding is tagged with `domain`, `provider`, and `model` when recorded. Synthesis merges findings that describe the same issue (even across slightly different lines or wording) and keeps the union of all models that reported it in the `models` list. The persisted baseline also stores a `domains` list (sorted) for each finding, supporting both the new list format and the old scalar `domain` for backward compat.

On follow-up review, each branch matches prior findings by domain using a helper that checks both `domains` (new) and `domain` (old). When synthesis consolidates findings across multiple lines, a three-tier attribution fallback ensures the domains list is populated:

1. Exact (path, line) match against raw findings
2. Path-level match (all domains that flagged the file)
3. Last-resort fallback to all domains present in raw findings (logs a warning)

This prevents the follow-up dedup from breaking when the synthesis model shifts a consolidated comment to a different line than the individual branches reported.

### Checkpoint and resume

The review graph is compiled with `checkpointer=ctx.checkpointer`, so LangGraph saves state after each node. If the review fails mid-execution (e.g. rate limit, transient API error), you can retry from the last checkpoint:

```bash
quality task retry <task_id>
```

The task guards against local-flag mismatch: a review checkpointed with `--local` and resumed without it (or vice versa) fails fast with a clear message, because the GitHub-write choice can't be flipped mid-run.

### Worktree lifecycle

Each PR gets its own worktree. For a repo cloned at `~/.quality/repos/<repo>`, the worktree is created as a sibling directory:

```
~/.quality/repos/<repo-name>.worktrees/pr-<number>-<task-id>/
```

The worktree is created in the setup node and removed in a finally block inside `ReviewTask.run()`, even if the graph raises or is cancelled. Cleanup runs under the same per-repo lock as creation to avoid racing concurrent reviews of the same repo (a `git worktree prune` from one review can race another's `git worktree add` if cleanup doesn't hold the lock).

Cleanup failures are caught and logged via `ctx.progress` so they never mask the original error that triggered unwinding.

## CLI Reference

```bash
# Review and post to GitHub
quality run pr review --pr <url>

# Review locally (no GitHub writes)
quality run pr review --pr <url> --local

# Detached
quality run pr review --pr <url> -d

# List all tasks
quality task list

# Follow a running task
quality task follow <task_id>

# Retry from checkpoint
quality task retry <task_id>

# Show runtime status
quality runtime status

# Full-screen TUI (bare invocation)
quality
```

## Testing

```bash
cd examples/quality
pytest
```

The test suite includes:
- Unit tests for graph nodes with mocked LLM/shell boundaries
- End-to-end graph execution tests that compile and invoke the real graph with a real SqliteCheckpointSaver
- Offline test harness with stub seams for gh/shell operations

The e2e tests found 11 production defects that the unit suite missed, because they exercised the real LangGraph compilation, state filtering, and checkpoint serialization paths. See `CLAUDE.md` memory: `execute-the-real-path` for the full postmortem.

## Related documentation

- [Switchplane README](../../README.md) — Runtime control plane architecture, CLI reference, LLM integration, MCP servers
- [Provider pool docs](../../README.md#provider-pool) — How to configure multiple LLM providers for fan-out workflows like this one
- [LangGraph docs](https://langchain-ai.github.io/langgraph/) — StateGraph, Send, conditional edges, checkpointing
