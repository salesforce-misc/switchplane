"""Secret redaction for review comments and error messages.

Security-critical: PR diffs contain user code, which can include tokens, API keys,
and credentialed URLs. When those snippets are quoted in review findings posted to
GitHub (potentially public), they create credential exposure. This module provides
a best-effort redaction pass that catches common patterns.

The pattern set is conservative, but labeled values are parsed by shape so redaction
preserves JSON/YAML/Python delimiters and avoids obvious references and status values.

**IMPORTANT:** Redaction is defense-in-depth, not a substitute for secure secret
management. Code under review should never contain inline secrets in the first place.
"""

from __future__ import annotations

import re

_REDACTED = "<REDACTED>"
_LABEL = (
    r"api[_-]?key|api-token|token|access[_-](?:token|secret)|client[_-]secret|"
    r"refresh[_-]token|private[_-]key|secret[_-]key|secret|password|passwd|pwd|session"
)
_LABEL_PREFIX = (
    rf"(?<![.\w])(?P<label_quote>['\"]?)(?P<label>(?:{_LABEL}))(?P=label_quote)"
    r"(?P<separator>\s*[:=]\s*)"
)

# Process PEM blocks before scalar values so a YAML block marker does not get
# redacted while leaving the private key body behind.
_LABELED_PRIVATE_KEY = re.compile(
    r"^(?P<indent>[ \t]*)(?P<label_quote>['\"]?)(?P<label>private[_-]key)(?P=label_quote)"
    r"\s*[:=]\s*(?:[|>][+-]?)?[ \t]*\r?\n"
    r"[ \t]*-----BEGIN (?P<kind>[A-Z0-9 ]*PRIVATE KEY)-----\r?\n"
    r".*?^[ \t]*-----END (?P=kind)-----[ \t]*(?P<newline>\r?\n|$)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?P<kind>[A-Z0-9 ]*PRIVATE KEY)-----.*?-----END (?P=kind)-----",
    re.IGNORECASE | re.DOTALL,
)

_DOUBLE_QUOTED_VALUE = re.compile(
    _LABEL_PREFIX + r'"(?P<value>(?:\\.|[^"\\])*)"',
    re.IGNORECASE,
)
_SINGLE_QUOTED_VALUE = re.compile(
    _LABEL_PREFIX + r"'(?P<value>(?:\\.|[^'\\])*)'",
    re.IGNORECASE,
)
# Stop before structural punctuation. In particular, never consume the closing
# bracket of the deterministic ``quality/review: [...]`` attribution line.
_UNQUOTED_VALUE = re.compile(
    _LABEL_PREFIX + r"(?!['\"])(?P<value>os\.environ\[['\"][^\]\r\n]+['\"]\]|\$\{[^}\r\n]+\}|[^\s,}\])]+)",
    re.IGNORECASE,
)

_AUTHORIZATION = re.compile(
    r"(Authorization\s*:\s*(?:Bearer|token))\s+([^\s,}\]]+)",
    re.IGNORECASE,
)
_BARE_PATTERNS = (
    re.compile(r"\b(sk-ant-|sk-proj-|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|AIzaSy)[A-Za-z0-9_\-]+"),
    # Generic sk-* is ambiguous with ordinary identifiers, so require a long,
    # credential-shaped suffix while preserving the prefix for useful context.
    re.compile(r"\b(sk-)[A-Za-z0-9_\-]{24,}"),
    re.compile(r"\b(AKIA)[0-9A-Z]{16}"),
    # Password in URL userinfo. The lookahead preserves the @ and URL structure.
    re.compile(r"(://[^/\s:@]+:)[^/\s@]+(?=@)"),
)

_DOTTED_REFERENCE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_ENVIRONMENT_REFERENCE = re.compile(r"(?:os\.environ\[['\"][^\]\r\n]+['\"]\]|\$\{[^}\r\n]+\})")
_STATUS_VALUES = {
    "active",
    "closed",
    "disabled",
    "enabled",
    "expired",
    "inactive",
    "invalid",
    "open",
    "pending",
    "valid",
}


def _label_prefix(match: re.Match[str], *, normalize: bool) -> str:
    """Render the matched label and delimiter without changing quoted syntax."""
    quote = match.group("label_quote")
    label = match.group("label")
    if quote or not normalize:
        return f"{quote}{label}{quote}{match.group('separator')}"
    return f"{label}="


def _redact_quoted(match: re.Match[str], quote: str) -> str:
    """Redact a complete quoted value while retaining both quote delimiters."""
    return f"{_label_prefix(match, normalize=False)}{quote}{_REDACTED}{quote}"


def _redact_unquoted(match: re.Match[str]) -> str:
    """Redact a scalar credential unless it is an obvious reference or status."""
    value = match.group("value")
    label = match.group("label").lower().replace("_", "-")
    if _ENVIRONMENT_REFERENCE.fullmatch(value):
        return match.group(0)
    dotted = _DOTTED_REFERENCE.fullmatch(value)
    if dotted:
        terminal = value.rsplit(".", 1)[-1]
        normalized_label = re.sub(r"[-_]", "", label).lower()
        normalized_terminal = re.sub(r"[-_]", "", terminal).lower()
        if normalized_terminal == normalized_label:
            return match.group(0)
    if label == "session" and value.lower() in _STATUS_VALUES:
        return match.group(0)
    return f"{_label_prefix(match, normalize=True)}{_REDACTED}"


def redact_secrets(text: str) -> str:
    """Redact common secret forms while preserving surrounding syntax.

    Covers labeled scalar and quoted values, PEM private-key blocks, authorization
    headers, known token prefixes, AWS access keys, and credentialed URLs. Obvious
    dotted references and ordinary session statuses remain unchanged.
    """
    out = _LABELED_PRIVATE_KEY.sub(
        lambda match: (
            f"{match.group('indent')}{match.group('label_quote')}{match.group('label')}"
            f"{match.group('label_quote')}={_REDACTED}{match.group('newline')}"
        ),
        text,
    )
    out = _PRIVATE_KEY_BLOCK.sub(_REDACTED, out)
    out = _DOUBLE_QUOTED_VALUE.sub(lambda match: _redact_quoted(match, '"'), out)
    out = _SINGLE_QUOTED_VALUE.sub(lambda match: _redact_quoted(match, "'"), out)
    out = _UNQUOTED_VALUE.sub(_redact_unquoted, out)
    out = _AUTHORIZATION.sub(lambda match: f"{match.group(1)} {_REDACTED}", out)
    for pattern in _BARE_PATTERNS:
        out = pattern.sub(rf"\1{_REDACTED}", out)
    return out
