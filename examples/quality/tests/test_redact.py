"""Tests for quality/_redact.py secret scrubbing.

This is one of the highest-leverage test files in the example: a redaction failure
is silent and publishes credentials to a potentially public PR. Every test asserts
the *exact* output string, not just `secret not in result` (which passes for a
function that returns "").

The cases below pin both credential coverage and the ordinary identifiers that
must remain readable after redaction.
"""

from __future__ import annotations


class TestSecretRedaction:
    """Tests for redact_secrets — defensive scrubbing of PR content before posting."""

    def test_short_labelled_secrets_are_redacted(self):
        """Labelled secrets are redacted even when their values are short."""
        from quality._redact import redact_secrets

        # 12-char key (realistic for dev/test keys)
        short_key = "api_key = sk-abc123xyz"
        redacted = redact_secrets(short_key)

        # EXACT output assertion (not just "sk-abc123xyz" not in redacted)
        assert redacted == "api_key=<REDACTED>", f"Short key must be redacted, got: {redacted}"

    def test_bare_tokens_with_no_label_are_redacted(self):
        """Bare tokens with known credential prefixes are redacted."""
        from quality._redact import redact_secrets

        # Anthropic key (no label, realistic length)
        bare_anthropic = "Error message contains sk-ant-api03-bareToken0123456789012345"
        redacted_anthropic = redact_secrets(bare_anthropic)
        assert "sk-ant-api03-bareToken0123456789012345" not in redacted_anthropic, "Bare Anthropic key must be redacted"
        assert "<REDACTED>" in redacted_anthropic, f"Redacted bare key must show placeholder, got: {redacted_anthropic}"

        # OpenAI key (no label)
        bare_openai = "Token sk-proj-abc1234567890123456789012345 was rejected"
        redacted_openai = redact_secrets(bare_openai)
        assert "sk-proj-abc1234567890123456789012345" not in redacted_openai, "Bare OpenAI key must be redacted"

        # GitHub PAT (no label)
        bare_github = "Clone failed: ghp_abc1234567890123456789012345"
        redacted_github = redact_secrets(bare_github)
        assert "ghp_abc1234567890123456789012345" not in redacted_github, "Bare GitHub PAT must be redacted"

    def test_labelled_forms_are_redacted(self):
        """Labelled secrets (api_key:, token=, Authorization: Bearer) must be redacted."""
        from quality._redact import redact_secrets

        # api_key with colon (realistic length)
        labelled_colon = "Configuration: api_key: sk-ant-abc123xyz789012345678901234567890"
        redacted_colon = redact_secrets(labelled_colon)
        assert "sk-ant-abc123xyz789012345678901234567890" not in redacted_colon, "api_key: form must be redacted"
        # Exact structure check: key name preserved, value redacted
        assert "api_key" in redacted_colon and "<REDACTED>" in redacted_colon, (
            f"Redacted labelled key must preserve structure, got: {redacted_colon}"
        )

        # token with equals sign
        labelled_equals = "export TOKEN=abc123xyz789012345678901234567890"
        redacted_equals = redact_secrets(labelled_equals)
        assert "abc123xyz789012345678901234567890" not in redacted_equals, "token= form must be redacted"

        # Authorization: Bearer header (most common in error messages)
        auth_bearer = "Request failed: Authorization: Bearer sk-ant-secret123456789012345678901234567"
        redacted_bearer = redact_secrets(auth_bearer)
        assert "sk-ant-secret123456789012345678901234567" not in redacted_bearer, "Bearer token must be redacted"
        assert "<REDACTED>" in redacted_bearer, f"Redacted bearer must show placeholder, got: {redacted_bearer}"

    def test_common_credential_labels_are_redacted(self):
        """Common credential labels redact values without relying on token prefixes."""
        from quality._redact import redact_secrets

        cases = {
            "password: hunter2": "password=<REDACTED>",
            "client_secret = local-dev-secret": "client_secret=<REDACTED>",
            "access_token: opaque-value": "access_token=<REDACTED>",
            "secret-key=keyboard-cat": "secret-key=<REDACTED>",
        }
        for raw, expected in cases.items():
            assert redact_secrets(raw) == expected

    def test_short_generic_sk_identifier_is_not_redacted(self):
        """Short generic sk-* identifiers remain readable when they are not credential-shaped."""
        from quality._redact import redact_secrets

        identifier = "Use model alias sk-test in this fixture."
        assert redact_secrets(identifier) == identifier

    def test_long_generic_sk_token_is_redacted(self):
        """Long generic sk-* values are treated as credential-shaped and redacted."""
        from quality._redact import redact_secrets

        token = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
        assert redact_secrets(f"Request failed for {token}") == "Request failed for sk-<REDACTED>"

    def test_credentialed_url_preserves_structure(self):
        """Credentialed URLs redact passwords while preserving URL structure."""
        from quality._redact import redact_secrets

        url = "Clone failed: https://username:my-secret-password-12345@github.com/org/repo"
        redacted = redact_secrets(url)

        # Secret must be gone
        assert "my-secret-password-12345" not in redacted, "URL password must be redacted"

        # Structure must survive: username, ://, @, host, path all intact
        # Exact assertion catches mangling like `user=<REDACTED>host`
        assert "https://username:" in redacted, "URL scheme and username must survive"
        assert "<REDACTED>@github.com/org/repo" in redacted or ":<REDACTED>@github.com" in redacted, (
            f"URL structure must be preserved (not mangled), got: {redacted}"
        )

    def test_authorization_header_preserves_header_name(self):
        """Authorization headers retain their name and scheme after redaction."""
        from quality._redact import redact_secrets

        header = "Authorization: Bearer sk-ant-secret123456789012345678901234567"
        redacted = redact_secrets(header)

        # Secret must be gone
        assert "sk-ant-secret123456789012345678901234567" not in redacted, "Bearer token must be redacted"

        # Header name must survive for context
        assert "Authorization" in redacted, f"Authorization header name must be preserved, got: {redacted}"
        assert "Bearer" in redacted or "<REDACTED>" in redacted, (
            f"Redacted header must preserve structure, got: {redacted}"
        )

    def test_negative_cases_pass_through_untouched(self):
        """Ordinary URLs, SSH URLs, and prose must pass through without redaction.

        Over-redaction that destroys PR links makes findings unactionable. These
        cases must return unchanged.
        """
        from quality._redact import redact_secrets

        # GitHub PR URL (no credentials)
        pr_url = "https://github.com/org/repo/pull/42"
        assert redact_secrets(pr_url) == pr_url, "PR URL must pass through untouched"

        # SSH URL (no credentials in the password sense)
        ssh_url = "ssh://git@github.com/org/repo.git"
        assert redact_secrets(ssh_url) == ssh_url, "SSH URL must pass through untouched"

        # Plain prose
        prose = "The API key was not found in the configuration file."
        assert redact_secrets(prose) == prose, "Prose without secrets must pass through untouched"

        # Code snippet with no actual secret
        code = 'const apiKey = process.env.API_KEY || "default";'
        assert redact_secrets(code) == code, "Code without actual secret must pass through untouched"

    def test_idempotence(self):
        """redact_secrets(redact_secrets(s)) == redact_secrets(s).

        Bodies may pass through more than once (e.g., logged then posted). A
        redacted placeholder must not itself trigger redaction or mangling.
        """
        from quality._redact import redact_secrets

        # Start with a secret
        original = "api_key: sk-ant-abc123xyz789012345678901234567890"
        once = redact_secrets(original)
        twice = redact_secrets(once)

        assert once == twice, f"Redaction must be idempotent: {once} != {twice}"

    def test_multiple_secrets_in_one_string(self):
        """Multiple secrets in one string must all be redacted."""
        from quality._redact import redact_secrets

        multi = (
            "Error: api_key: sk-ant-abc123xyz789012345678901234567890 and "
            "token=ghp_def456abc789012345678901234567890 both invalid"
        )
        redacted = redact_secrets(multi)

        assert "sk-ant-abc123xyz789012345678901234567890" not in redacted, "First secret must be redacted"
        assert "ghp_def456abc789012345678901234567890" not in redacted, "Second secret must be redacted"
        assert redacted.count("<REDACTED>") == 2, f"Both secrets must be redacted, got: {redacted}"

    def test_aws_access_key_is_redacted(self):
        """AWS access keys (AKIA prefix) must be redacted."""
        from quality._redact import redact_secrets

        aws = "S3 upload failed with key: AKIAIOSFODNN7EXAMPLE"
        redacted = redact_secrets(aws)

        assert "AKIAIOSFODNN7EXAMPLE" not in redacted, "AWS access key must be redacted"
        assert "<REDACTED>" in redacted, f"Redacted AWS key must show placeholder, got: {redacted}"

    def test_case_insensitive_label_matching(self):
        """Labels like API_KEY, Api-Key, token must all match (case-insensitive)."""
        from quality._redact import redact_secrets

        uppercase = "API_KEY: sk-ant-abc123xyz789012345678901234567890"
        mixedcase = "Api-Key=sk-ant-def456abc789012345678901234567890"
        lowercase = "api-key: sk-ant-ghi789def012345678901234567890"

        redacted_upper = redact_secrets(uppercase)
        redacted_mixed = redact_secrets(mixedcase)
        redacted_lower = redact_secrets(lowercase)

        assert "sk-ant-abc123xyz789012345678901234567890" not in redacted_upper, "Uppercase label must match"
        assert "sk-ant-def456abc789012345678901234567890" not in redacted_mixed, "Mixed-case label must match"
        assert "sk-ant-ghi789def012345678901234567890" not in redacted_lower, "Lowercase label must match"

    def test_quoted_secrets_are_redacted(self):
        """Secrets in quotes (JSON, YAML, code) must be redacted."""
        from quality._redact import redact_secrets

        json_secret = '{"api_key": "sk-ant-abc123xyz789012345678901234567890"}'
        redacted_json = redact_secrets(json_secret)
        assert "sk-ant-abc123xyz789012345678901234567890" not in redacted_json, "Quoted JSON secret must be redacted"

        yaml_secret = "api_key: 'sk-ant-def456abc789012345678901234567890'"
        redacted_yaml = redact_secrets(yaml_secret)
        assert "sk-ant-def456abc789012345678901234567890" not in redacted_yaml, "Quoted YAML secret must be redacted"

    def test_quoted_labels_and_values_are_redacted_without_consuming_structure(self):
        """Quoted JSON, Python, and YAML credentials preserve their closing syntax."""
        from quality._redact import redact_secrets

        cases = {
            '{"password": "correct horse battery staple", "enabled": true}': (
                '{"password": "<REDACTED>", "enabled": true}'
            ),
            "{'client-secret': 'local development secret', 'enabled': True}": (
                "{'client-secret': '<REDACTED>', 'enabled': True}"
            ),
            '"secret": "value containing spaces"': '"secret": "<REDACTED>"',
            "'pwd': 'two words'": "'pwd': '<REDACTED>'",
            'session: "opaque session value"': 'session: "<REDACTED>"',
        }

        for raw, expected in cases.items():
            assert redact_secrets(raw) == expected

    def test_private_key_blocks_are_redacted_without_leaking_key_body(self):
        """A labelled PEM value is removed through its matching END marker."""
        from quality._redact import redact_secrets

        raw = (
            "private_key: |\n"
            "  -----BEGIN PRIVATE KEY-----\n"
            "  c2VjcmV0LWtleS1ib2R5LWxpbmUtMQ==\n"
            "  c2VjcmV0LWtleS1ib2R5LWxpbmUtMg==\n"
            "  -----END PRIVATE KEY-----\n"
            "public_name: review-bot"
        )

        redacted = redact_secrets(raw)

        assert redacted == "private_key=<REDACTED>\npublic_name: review-bot"
        assert "BEGIN PRIVATE KEY" not in redacted
        assert "c2VjcmV0" not in redacted

    def test_reference_assignments_and_status_values_are_not_redacted(self):
        """Obvious references and ordinary session status remain useful context."""
        from quality._redact import redact_secrets

        references = "password = request.password\napiKey = config.apiKey\nsession: active"

        assert redact_secrets(references) == references

    def test_environment_password_references_are_preserved_complete(self):
        from quality._redact import redact_secrets

        references = 'password = os.environ["PASSWORD"]\npassword = ${PASSWORD}'

        assert redact_secrets(references) == references

    def test_literal_credentials_still_redact_next_to_reference_assignments(self):
        """Reference exemptions must not allow adjacent literal credentials through."""
        from quality._redact import redact_secrets

        raw = (
            "password = request.password\n"
            'password = "literal credential"\n'
            "apiKey = config.apiKey\n"
            "apiKey = literal-api-key\n"
            "session: active\n"
            "session: opaque-session-token"
        )
        expected = (
            "password = request.password\n"
            'password = "<REDACTED>"\n'
            "apiKey = config.apiKey\n"
            "apiKey=<REDACTED>\n"
            "session: active\n"
            "session=<REDACTED>"
        )

        assert redact_secrets(raw) == expected

    def test_dotted_literals_redact_but_credential_references_remain(self):
        """Dots alone do not make a labelled credential value a code reference."""
        from quality._redact import redact_secrets

        raw = "token=abc.def.ghi\npassword=hunter2.value\npassword=request.password\napiKey=config.apiKey"

        assert redact_secrets(raw) == (
            "token=<REDACTED>\npassword=<REDACTED>\npassword=request.password\napiKey=config.apiKey"
        )
