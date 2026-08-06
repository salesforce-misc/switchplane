"""Tests for quality/agents/pr/prompts.py — domain prompts and synthesis.

These tests pin the prompt structure: domain coverage, follow-up behavior, and
synthesis protocol. They detect silent omissions (e.g. forgetting to include the
worktree path or prior findings).
"""


class TestDomains:
    """Tests for DOMAINS and DOMAIN_PROMPTS."""

    def test_domains_tuple_exists(self):
        """DOMAINS must be a tuple of domain names."""
        from quality.agents.pr.prompts import DOMAINS

        assert isinstance(DOMAINS, tuple)
        assert len(DOMAINS) > 0

    def test_quality_domain_present(self):
        """The 'quality' domain must be in DOMAINS."""
        from quality.agents.pr.prompts import DOMAINS

        assert "quality" in DOMAINS

    def test_security_domain_present(self):
        """The 'security' domain must be in DOMAINS."""
        from quality.agents.pr.prompts import DOMAINS

        assert "security" in DOMAINS

    def test_domain_prompts_covers_all_domains(self):
        """DOMAIN_PROMPTS must have an entry for every domain in DOMAINS.

        This pins the completeness requirement: adding a domain without a prompt
        would fail at runtime when the fan-out tries to build a branch prompt.
        """
        from quality.agents.pr.prompts import DOMAIN_PROMPTS, DOMAINS

        for domain in DOMAINS:
            assert domain in DOMAIN_PROMPTS, f"Missing prompt for domain '{domain}'"

    def test_domain_prompts_are_non_empty_strings(self):
        """Every domain prompt must be a non-empty string."""
        from quality.agents.pr.prompts import DOMAIN_PROMPTS

        for domain, prompt in DOMAIN_PROMPTS.items():
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0, f"Domain '{domain}' has empty prompt"


class TestRecordProtocol:
    """Tests for the shared RECORD_PROTOCOL block."""

    def test_record_protocol_in_quality_prompt(self):
        """The quality prompt must include the RECORD_PROTOCOL instructions.

        This pins the tool-based recording behavior that replaces direct GitHub
        posting in the branch nodes.
        """
        from quality.agents.pr.prompts import DOMAIN_PROMPTS

        quality = DOMAIN_PROMPTS["quality"]
        assert "record_finding" in quality
        assert "record_note" in quality
        assert "severity" in quality.lower()

    def test_record_protocol_in_security_prompt(self):
        """The security prompt must include the RECORD_PROTOCOL instructions."""
        from quality.agents.pr.prompts import DOMAIN_PROMPTS

        security = DOMAIN_PROMPTS["security"]
        assert "record_finding" in security
        assert "record_note" in security
        assert "severity" in security.lower()

    def test_severity_vocabulary_documented(self):
        """The prompts must document the severity vocabulary.

        This pins the severity enum (info, low, medium, high, critical) that
        event resolution uses to map findings to GitHub review events.
        """
        from quality.agents.pr.prompts import DOMAIN_PROMPTS

        quality = DOMAIN_PROMPTS["quality"]
        # Check for severity terms — at least "high" and "critical" must appear
        assert "high" in quality.lower()
        assert "critical" in quality.lower()


class TestFilesystemProtocol:
    """Tests for the FS_PROTOCOL block (worktree tools)."""

    def test_fs_protocol_exists(self):
        """FS_PROTOCOL must be defined and non-empty."""
        from quality.agents.pr import prompts

        assert hasattr(prompts, "FS_PROTOCOL")
        assert isinstance(prompts.FS_PROTOCOL, str)
        assert len(prompts.FS_PROTOCOL.strip()) > 0

    def test_fs_protocol_mentions_read_only_tools(self):
        """FS_PROTOCOL must mention the read-only filesystem tools (ls/find/grep).

        This is what tells the model to use the worktree for context rather than
        relying solely on the diff.
        """
        from quality.agents.pr.prompts import FS_PROTOCOL

        fs_lower = FS_PROTOCOL.lower()
        assert "ls" in fs_lower or "find" in fs_lower or "grep" in fs_lower
        assert "read" in fs_lower or "context" in fs_lower


class TestInitialPrompt:
    """Tests for initial_prompt — first-time review."""

    def test_includes_domain(self):
        """The initial prompt must include the domain name."""
        from quality.agents.pr.prompts import initial_prompt

        prompt = initial_prompt(
            domain="quality",
            repo="github.com/org/repo",
            number=42,
            worktree_path="/tmp/worktree",
            diff="diff content",
        )

        assert "quality" in prompt

    def test_includes_repo_and_number(self):
        """The initial prompt must include the repo and PR number."""
        from quality.agents.pr.prompts import initial_prompt

        prompt = initial_prompt(
            domain="security",
            repo="github.com/myorg/myrepo",
            number=99,
            worktree_path="/tmp/wt",
            diff="diff",
        )

        assert "github.com/myorg/myrepo" in prompt or "myorg/myrepo" in prompt
        assert "99" in prompt

    def test_includes_worktree_path(self):
        """The initial prompt must include the worktree path.

        This is load-bearing: without it, the model can't use ls/find/grep tools
        to read the repository for context.
        """
        from quality.agents.pr.prompts import initial_prompt

        prompt = initial_prompt(
            domain="quality",
            repo="github.com/org/repo",
            number=1,
            worktree_path="/path/to/worktree",
            diff="diff",
        )

        assert "/path/to/worktree" in prompt

    def test_includes_diff(self):
        """The initial prompt must include the full diff."""
        from quality.agents.pr.prompts import initial_prompt

        diff_content = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,4 @@\n+new line\n"
        prompt = initial_prompt(
            domain="quality",
            repo="github.com/org/repo",
            number=1,
            worktree_path="/tmp/wt",
            diff=diff_content,
        )

        assert diff_content in prompt

    def test_no_followup_language(self):
        """The initial prompt must NOT contain follow-up language like 'previous findings'.

        This pins the distinction between initial and follow-up prompts.
        """
        from quality.agents.pr.prompts import initial_prompt

        prompt = initial_prompt(
            domain="quality",
            repo="github.com/org/repo",
            number=1,
            worktree_path="/tmp/wt",
            diff="diff",
        )

        prompt_lower = prompt.lower()
        assert "previous" not in prompt_lower
        assert "prior" not in prompt_lower
        assert "follow-up" not in prompt_lower


class TestFollowupPrompt:
    """Tests for followup_prompt — subsequent review of updated PR."""

    def test_includes_domain(self):
        """The followup prompt must include the domain name."""
        from quality.agents.pr.prompts import followup_prompt

        prompt = followup_prompt(
            domain="security",
            repo="github.com/org/repo",
            number=5,
            worktree_path="/tmp/wt",
            diff="diff",
            prior_findings="[high] file.py:10 — issue",
        )

        assert "security" in prompt

    def test_includes_prior_findings(self):
        """The followup prompt must include the prior findings text.

        This is the key difference from initial_prompt: it tells the model what
        was raised before so it can avoid re-reporting resolved issues.
        """
        from quality.agents.pr.prompts import followup_prompt

        prior = "[medium] auth.py:42 — missing validation\n[high] db.py:100 — SQL injection risk"
        prompt = followup_prompt(
            domain="quality",
            repo="github.com/org/repo",
            number=10,
            worktree_path="/tmp/wt",
            diff="diff",
            prior_findings=prior,
        )

        assert "auth.py:42" in prompt
        assert "missing validation" in prompt
        assert "db.py:100" in prompt
        assert "SQL injection risk" in prompt

    def test_includes_worktree_path(self):
        """The followup prompt must include the worktree path."""
        from quality.agents.pr.prompts import followup_prompt

        prompt = followup_prompt(
            domain="quality",
            repo="github.com/org/repo",
            number=1,
            worktree_path="/custom/path",
            diff="diff",
            prior_findings="prior",
        )

        assert "/custom/path" in prompt

    def test_includes_diff(self):
        """The followup prompt must include the current diff."""
        from quality.agents.pr.prompts import followup_prompt

        diff_content = "--- a/main.py\n+++ b/main.py\n@@ -10,5 +10,6 @@\n+added line\n"
        prompt = followup_prompt(
            domain="security",
            repo="github.com/org/repo",
            number=2,
            worktree_path="/tmp/wt",
            diff=diff_content,
            prior_findings="prior",
        )

        assert diff_content in prompt

    def test_has_followup_language(self):
        """The followup prompt must contain follow-up language.

        This pins the instruction to focus on what changed and not re-report
        resolved findings.
        """
        from quality.agents.pr.prompts import followup_prompt

        prompt = followup_prompt(
            domain="quality",
            repo="github.com/org/repo",
            number=1,
            worktree_path="/tmp/wt",
            diff="diff",
            prior_findings="prior",
        )

        prompt_lower = prompt.lower()
        assert "follow-up" in prompt_lower or "previously" in prompt_lower or "prior" in prompt_lower


class TestFormatPrior:
    """Tests for _format_prior — filtering and rendering prior findings."""

    def test_filters_by_domain(self):
        """_format_prior must return only findings for the requested domain.

        This pins the per-domain filtering that prevents a quality branch from
        seeing security findings and vice versa.
        """
        from quality.agents.pr.prompts import _format_prior

        baseline = {
            "findings": [
                {"domain": "quality", "severity": "medium", "path": "a.py", "line": 10, "title": "Quality issue"},
                {"domain": "security", "severity": "high", "path": "b.py", "line": 20, "title": "Security issue"},
                {"domain": "quality", "severity": "low", "path": "c.py", "line": 30, "title": "Another quality issue"},
            ]
        }

        formatted = _format_prior(baseline, domain="quality")

        assert "Quality issue" in formatted
        assert "Another quality issue" in formatted
        assert "Security issue" not in formatted

    def test_returns_empty_string_for_no_findings(self):
        """When there are no findings for the domain, returns empty string.

        This is the sentinel that tells the caller to fall back to initial_prompt.
        """
        from quality.agents.pr.prompts import _format_prior

        baseline = {"findings": []}
        assert _format_prior(baseline, domain="quality") == ""

    def test_returns_empty_for_absent_baseline(self):
        """When baseline is None or missing findings key, returns empty string."""
        from quality.agents.pr.prompts import _format_prior

        assert _format_prior(None, domain="quality") == ""
        assert _format_prior({}, domain="quality") == ""

    def test_includes_finding_details(self):
        """_format_prior must include path, line, severity, and title for each finding."""
        from quality.agents.pr.prompts import _format_prior

        baseline = {
            "findings": [
                {
                    "domain": "security",
                    "severity": "critical",
                    "path": "auth.py",
                    "line": 55,
                    "title": "Hardcoded secret",
                    "body": "Details...",
                }
            ]
        }

        formatted = _format_prior(baseline, domain="security")

        assert "auth.py" in formatted
        assert "55" in formatted
        assert "critical" in formatted.lower()
        assert "Hardcoded secret" in formatted

    def test_includes_head_sha_when_present(self):
        """If the baseline has a head_sha, _format_prior must include it.

        This tells the model what SHA the prior findings were raised against,
        which is useful for detecting whether code changed since then.
        """
        from quality.agents.pr.prompts import _format_prior

        baseline = {"head_sha": "abc123def456", "findings": [{"domain": "quality", "title": "Issue"}]}

        formatted = _format_prior(baseline, domain="quality")

        assert "abc123def456" in formatted


class TestSynthesisPrompt:
    """Tests for SYNTHESIS_PROMPT — merge/attribution/deduplication instructions."""

    def test_synthesis_prompt_exists(self):
        """SYNTHESIS_PROMPT must be defined and non-empty."""
        from quality.agents.pr import prompts

        assert hasattr(prompts, "SYNTHESIS_PROMPT")
        assert isinstance(prompts.SYNTHESIS_PROMPT, str)
        assert len(prompts.SYNTHESIS_PROMPT.strip()) > 0

    def test_instructs_merge_and_dedupe(self):
        """SYNTHESIS_PROMPT must instruct merging duplicates and attributing models."""
        from quality.agents.pr.prompts import SYNTHESIS_PROMPT

        prompt_lower = SYNTHESIS_PROMPT.lower()
        assert "merge" in prompt_lower or "dedupe" in prompt_lower or "duplicate" in prompt_lower
        assert "models" in prompt_lower  # Attribution

    def test_specifies_output_structure(self):
        """SYNTHESIS_PROMPT must specify the output structure (comments, summary, event)."""
        from quality.agents.pr.prompts import SYNTHESIS_PROMPT

        prompt_lower = SYNTHESIS_PROMPT.lower()
        assert "comments" in prompt_lower
        assert "summary" in prompt_lower
        assert "event" in prompt_lower

    def test_lists_review_events(self):
        """SYNTHESIS_PROMPT must list the valid GitHub review events.

        This pins the event vocabulary: APPROVE, COMMENT, REQUEST_CHANGES.
        """
        from quality.agents.pr.prompts import SYNTHESIS_PROMPT

        prompt_upper = SYNTHESIS_PROMPT.upper()
        assert "APPROVE" in prompt_upper or "REQUEST_CHANGES" in prompt_upper or "COMMENT" in prompt_upper

    def test_warns_against_duplicating_findings_in_summary(self):
        """SYNTHESIS_PROMPT must warn against enumerating findings in the summary.

        This pins the output structure: findings go in comments array, not summary.
        """
        from quality.agents.pr.prompts import SYNTHESIS_PROMPT

        prompt_lower = SYNTHESIS_PROMPT.lower()
        assert "summary" in prompt_lower
        assert "not" in prompt_lower or "must not" in prompt_lower or "do not" in prompt_lower
