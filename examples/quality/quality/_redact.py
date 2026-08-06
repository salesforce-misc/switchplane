"""Secret redaction for review comments and error messages.

Security-critical: PR diffs contain user code, which can include tokens, API keys,
and credentialed URLs. When those snippets are quoted in review findings posted to
GitHub (potentially public), they create credential exposure. This module provides
a best-effort redaction pass that catches common patterns.

The pattern set is conservative: it's better to over-redact (occasional false
positive) than under-redact (credential leak). All patterns are case-insensitive.

**IMPORTANT:** Redaction is defense-in-depth, not a substitute for secure secret
management. Code under review should never contain inline secrets in the first place.
"""

from __future__ import annotations

import re

# Each pattern captures exactly one leading group (the prefix/label to preserve)
# and consumes the secret, allowing uniform substitution.
# Patterns normalize whitespace: "api_key = <v>" becomes "api_key=<REDACTED>".
_SECRET_PATTERNS = (
    # Labelled: api_key: <v>, api-key=<v>, API_KEY=<v>, token=<v>
    # Requires underscore or hyphen separator to avoid matching camelCase identifiers.
    # Captures just the label (no delimiter), then consumes delimiter + whitespace + secret.
    re.compile(r"((?:api[_-]key|token))\s*[:=]\s*\S+", re.IGNORECASE),
    # Authorization: Bearer <v> / Authorization: token <v>
    # Captures prefix without trailing space
    re.compile(r"(Authorization\s*:\s*(?:Bearer|token))\s+\S+", re.IGNORECASE),
    # Bare values by known prefix — no label required. Catches tokens appearing
    # standalone in diffs, error messages, or env var values.
    re.compile(r"\b(sk-ant-|sk-|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|AIzaSy)[A-Za-z0-9_\-]+"),
    # AWS access keys (bare prefix match)
    re.compile(r"\b(AKIA)[0-9A-Z]{16}"),
    # Password in URL userinfo: https://user:<secret>@host
    # The lookahead requires the @, so ordinary URLs and ssh://git@host are left alone.
    re.compile(r"(://[^/\s:@]+:)[^/\s@]+(?=@)"),
)


def redact_secrets(text: str) -> str:
    """Redact common secret patterns from text.

    Replaces the secret value itself with ``<REDACTED>``, preserving the key name
    and structural context so error messages and findings remain actionable.

    Patterns cover:
    - Labelled forms: api_key=..., token:..., Authorization: Bearer ...
    - Bare credential prefixes: sk-ant-..., ghp_..., AKIA...
    - Credentialed URLs: https://user:pass@host/path

    Whitespace is normalized: "api_key = <v>" becomes "api_key=<REDACTED>".

    Args:
        text: Input text (error message, comment body, log entry)

    Returns:
        Text with secret values replaced by ``<REDACTED>``
    """
    out = text
    # Labelled forms need special handling to normalize whitespace
    # Pattern 0: api_key/token forms
    out = _SECRET_PATTERNS[0].sub(lambda m: m.group(1).rstrip() + "=<REDACTED>", out)
    # Pattern 1: Authorization forms
    out = _SECRET_PATTERNS[1].sub(lambda m: m.group(1).rstrip() + " <REDACTED>", out)
    # Remaining patterns (bare prefixes, URLs) use uniform substitution
    for pattern in _SECRET_PATTERNS[2:]:
        out = pattern.sub(r"\1<REDACTED>", out)
    return out
