"""OAuth 2.0 support for MCP HTTP transport.

Two authentication modes are supported:

**MCP-spec OAuth** — When ``OAuthConfig`` has no explicit endpoints, the
MCP SDK's ``OAuthClientProvider`` (an ``httpx.Auth`` subclass) handles
metadata discovery, PKCE, token exchange, refresh, and retry-on-401.

**Direct OIDC** — When ``OAuthConfig`` has ``auth_url`` and ``token_url``
set, a lightweight ``DirectOIDCAuth`` flow talks to the identity provider
directly.  This covers external IdPs like Keycloak/QuantumK that are not
discoverable via MCP server metadata.

Both modes share the same ``FileTokenStorage``, ``OAuthCallbackServer``,
and CLI integration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import structlog

from switchplane.app import McpServerConfig, OAuthConfig

logger = structlog.get_logger(__name__)


def _resolve_ssl_verify(runtime_dir: Path | None) -> str | bool:
    """Return the CA bundle path if present, else default verification."""
    if runtime_dir:
        ca_bundle = runtime_dir / "ca-bundle.pem"
        if ca_bundle.exists():
            return str(ca_bundle)
    return True


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------


class FileTokenStorage:
    """Persist OAuth tokens and client info as JSON files.

    Layout::

        {storage_dir}/
            tokens.json        — OAuthToken
            client_info.json   — OAuthClientInformationFull
    """

    def __init__(self, storage_dir: Path, oauth_config: OAuthConfig) -> None:
        self._dir = storage_dir
        self._oauth_config = oauth_config
        self._dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Ensure permissions are correct even if the directory already existed
        self._dir.chmod(0o700)

    @property
    def _tokens_path(self) -> Path:
        return self._dir / "tokens.json"

    @property
    def _client_info_path(self) -> Path:
        return self._dir / "client_info.json"

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken

        if not self._tokens_path.exists():
            return None
        data = json.loads(self._tokens_path.read_text())
        return OAuthToken.model_validate(data)

    async def set_tokens(self, tokens) -> None:
        import os
        import stat

        self._tokens_path.write_text(tokens.model_dump_json(indent=2))
        os.chmod(self._tokens_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull

        if self._client_info_path.exists():
            data = json.loads(self._client_info_path.read_text())
            return OAuthClientInformationFull.model_validate(data)

        # Pre-seed with the configured client_id so OAuthClientProvider
        # skips dynamic client registration (needed for servers like Slack
        # that hand out a known client_id).
        redirect_uri = f"http://localhost:{self._oauth_config.callback_port}/callback"
        client_info = OAuthClientInformationFull(
            client_id=self._oauth_config.client_id,
            client_secret=self._oauth_config.client_secret,
            redirect_uris=[redirect_uri],
        )
        await self.set_client_info(client_info)
        return client_info

    async def set_client_info(self, client_info) -> None:
        import os
        import stat

        self._client_info_path.write_text(client_info.model_dump_json(indent=2))
        os.chmod(self._client_info_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600

    # -- Direct OIDC helpers (not part of MCP SDK protocol) --

    def get_token_timestamp(self) -> float | None:
        """Return the mtime of the tokens file, or None if absent."""
        if self._tokens_path.exists():
            return self._tokens_path.stat().st_mtime
        return None


# ---------------------------------------------------------------------------
# Callback server
# ---------------------------------------------------------------------------

_SUCCESS_HTML = """\
HTTP/1.1 200 OK\r
Content-Type: text/html\r
Connection: close\r
\r
<!DOCTYPE html>
<html><body>
<h3>Authorization successful</h3>
<p>You can close this tab and return to your terminal.</p>
</body></html>"""


class OAuthCallbackServer:
    """Ephemeral async TCP server that receives one OAuth redirect callback."""

    def __init__(self, port: int) -> None:
        self._port = port
        self._result: asyncio.Future[tuple[str, str | None]] | None = None
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._result = loop.create_future()
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", self._port)
        logger.debug("oauth_callback_server_started", port=self._port)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            parts = request_line.decode().split()
            if len(parts) >= 2:
                path = parts[1]
                qs = parse_qs(urlparse(path).query)
                code = qs.get("code", [None])[0]
                state = qs.get("state", [None])[0]

                writer.write(_SUCCESS_HTML.encode())
                await writer.drain()

                if code and self._result and not self._result.done():
                    self._result.set_result((code, state))
        except Exception:
            logger.exception("oauth_callback_handler_error")
        finally:
            writer.close()
            await writer.wait_closed()

    async def wait_for_callback(self) -> tuple[str, str | None]:
        if self._result is None:
            raise RuntimeError("Callback server not started")
        return await self._result

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.debug("oauth_callback_server_stopped")


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def _generate_pkce() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for S256 PKCE."""
    import base64

    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# ---------------------------------------------------------------------------
# Direct OIDC auth (for external IdPs like QuantumK)
# ---------------------------------------------------------------------------


class DirectOIDCAuth(httpx.Auth):
    """httpx.Auth backend for OAuth2 providers with known endpoints.

    Loads stored tokens from ``FileTokenStorage``, injects them as Bearer
    headers, and transparently refreshes on 401.  Does *not* perform
    interactive authorization — if no valid token exists and refresh fails,
    the request fails with a clear error message.
    """

    requires_response_body = True

    def __init__(self, oauth: OAuthConfig, storage: FileTokenStorage, ssl_verify: str | bool = True) -> None:
        self._oauth = oauth
        self._storage = storage
        self._ssl_verify = ssl_verify
        self._tokens = None
        self._token_acquired_at: float | None = None
        self._initialized = False

    async def _load_tokens(self) -> None:
        self._tokens = await self._storage.get_tokens()
        self._token_acquired_at = self._storage.get_token_timestamp()
        self._initialized = True

    def _is_token_expired(self) -> bool:
        if not self._tokens or not self._tokens.expires_in or not self._token_acquired_at:
            return True
        elapsed = time.time() - self._token_acquired_at
        return elapsed >= (self._tokens.expires_in - 30)  # 30s buffer

    async def _refresh(self, client: httpx.AsyncClient) -> bool:
        """Attempt a token refresh.  Returns True on success."""
        if not self._tokens or not self._tokens.refresh_token:
            return False

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._tokens.refresh_token,
            "client_id": self._oauth.client_id,
        }
        resp = await client.post(
            self._oauth.token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            logger.warning("oauth_refresh_failed", status=resp.status_code, body=resp.text[:200])
            return False

        from mcp.shared.auth import OAuthToken

        self._tokens = OAuthToken.model_validate_json(resp.content)
        self._token_acquired_at = time.time()
        await self._storage.set_tokens(self._tokens)
        logger.info("oauth_token_refreshed")
        return True

    async def async_auth_flow(self, request: httpx.Request):
        if not self._initialized:
            await self._load_tokens()

        if not self._tokens:
            raise RuntimeError("No stored OAuth tokens — run 'auth login <server>' to authenticate")

        # Proactively refresh before sending if expired
        if self._is_token_expired() and self._tokens.refresh_token:
            async with httpx.AsyncClient(verify=self._ssl_verify, follow_redirects=True) as refresh_client:
                await self._refresh(refresh_client)

        request.headers["Authorization"] = f"Bearer {self._tokens.access_token}"
        response = yield request

        if response.status_code == 401:
            # Token was rejected — attempt refresh
            async with httpx.AsyncClient(verify=self._ssl_verify, follow_redirects=True) as refresh_client:
                if await self._refresh(refresh_client):
                    request.headers["Authorization"] = f"Bearer {self._tokens.access_token}"
                    yield request
                else:
                    raise RuntimeError(
                        "OAuth token expired and refresh failed — run 'auth login <server>' to re-authenticate"
                    )


async def run_direct_oidc_login(
    oauth: OAuthConfig,
    storage: FileTokenStorage,
    runtime_dir: Path | None = None,
) -> None:
    """Run an interactive OIDC authorization-code + PKCE flow.

    Opens a browser for user consent, exchanges the code for tokens,
    and persists the result via *storage*.
    """
    redirect_uri = f"http://localhost:{oauth.callback_port}/callback"
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    params = {
        "response_type": "code",
        "client_id": oauth.client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if oauth.scopes:
        params["scope"] = oauth.scopes
    params.update(oauth.extra_authorize_params)

    auth_url = f"{oauth.auth_url}?{urlencode(params)}"

    callback_server = OAuthCallbackServer(oauth.callback_port)
    await callback_server.start()

    try:
        logger.info("oauth_opening_browser", url=auth_url)
        webbrowser.open(auth_url)

        code, returned_state = await asyncio.wait_for(
            callback_server.wait_for_callback(),
            timeout=300.0,
        )
    finally:
        await callback_server.stop()

    if returned_state is None or not secrets.compare_digest(returned_state, state):
        raise RuntimeError(f"OAuth state mismatch: expected {state!r}, got {returned_state!r}")
    if not code:
        raise RuntimeError("No authorization code received")

    # Exchange code for tokens
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": oauth.client_id,
        "code_verifier": verifier,
    }

    ssl_verify = _resolve_ssl_verify(runtime_dir)
    async with httpx.AsyncClient(verify=ssl_verify, follow_redirects=True) as client:
        resp = await client.post(
            oauth.token_url,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Token exchange failed ({resp.status_code}): {resp.text[:500]}")

    from mcp.shared.auth import OAuthToken

    tokens = OAuthToken.model_validate_json(resp.content)
    await storage.set_tokens(tokens)
    logger.info("oauth_tokens_stored")


# ---------------------------------------------------------------------------
# Client builder (unified entry point)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rate-limit (HTTP 429) retry transport
# ---------------------------------------------------------------------------

_RETRY_BACKOFF_BASE_SECONDS = 2.0
_RETRY_BACKOFF_MAX_SECONDS = 60.0


def _retry_after_seconds(response: httpx.Response | None, attempt: int) -> float:
    """Seconds to wait before the next retry.

    Combines the server's ``Retry-After`` header (delta-seconds form) with a
    capped exponential backoff keyed on the 0-based attempt number, taking the
    *larger* of the two. This honors a server that asks for a longer cooldown,
    while ensuring a small constant header (e.g. Slack's ``Retry-After: 1`` on
    every 429) can't defeat escalation — without it, three retries would wait
    1s each (~3s total) and never outlast a real throttle window. Both inputs
    are clamped to a sane maximum.

    ``response`` is ``None`` for transport faults (timeouts, connection errors)
    where no HTTP response was received; those fall back to pure exponential
    backoff.
    """
    backoff = _RETRY_BACKOFF_BASE_SECONDS * (2**attempt)
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        try:
            backoff = max(backoff, float(retry_after))
        except ValueError:
            pass  # HTTP-date form is unusual for 429; fall through to backoff
    return min(max(0.0, backoff), _RETRY_BACKOFF_MAX_SECONDS)


class RetryTransport(httpx.AsyncBaseTransport):
    """Wraps an httpx transport to retry transient failures transparently.

    Retries two classes of failure, both up to ``max_retries`` times with
    capped exponential backoff:

    - **HTTP 429** (rate-limited) responses, honoring ``Retry-After``.
    - **Transient transport faults** raised by the wrapped transport —
      ``ReadTimeout``/``ConnectTimeout`` (a hung request that would otherwise
      ride the full client timeout into a fatal session teardown),
      ``ConnectError``, and ``RemoteProtocolError``.

    This sits *below* the MCP SDK's streamable-HTTP layer. The SDK runs each
    request inside an anyio task group and calls ``response.raise_for_status()``
    there; a 429 — or an exception escaping the request — surfaces only at
    session teardown as a ``CancelledError`` and tears down the whole session.
    Absorbing both here — before the SDK ever sees them — means transient
    failures are retried instead of fatal, for both the ``initialize``
    handshake and ordinary tool calls.

    Only 429 responses that will be retried are drained; any other response
    (including streaming/SSE success bodies) is returned untouched so the SDK
    can consume it normally. A transport fault raised *after* response headers
    arrive (e.g. mid-SSE-body) escapes this layer — the request has already
    returned — but the header-receipt hang that motivates this is caught.
    """

    # Transport faults worth retrying: a transient timeout/connection drop on a
    # replayable JSON-RPC request. Deliberately excludes things like
    # ``httpx.ProtocolError`` on a malformed request, which won't fix on retry.
    _RETRYABLE_EXC = (
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
    )

    def __init__(self, wrapped: httpx.AsyncBaseTransport, max_retries: int, server_name: str):
        self._wrapped = wrapped
        self._max_retries = max_retries
        self._server_name = server_name

    async def __aenter__(self) -> RetryTransport:
        await self._wrapped.__aenter__()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._wrapped.__aexit__(*exc_info)

    async def aclose(self) -> None:
        await self._wrapped.aclose()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Retrying re-sends the same `request`; this assumes a buffered (i.e.
        # replayable) request body, which httpx guarantees for the small JSON-RPC
        # payloads the MCP SDK sends. A streaming/consume-once upload body would
        # not be replayable — not a concern for MCP, but the invariant is here.
        #
        # Semantic caveat: a 429 is safe to retry because the server rejected the
        # request *before* processing it. A transport fault is not in the same
        # category — a `ReadTimeout` or mid-flight `RemoteProtocolError` can occur
        # after the server has already begun (or finished) executing the call and
        # only the response was lost. MCP `tools/call` requests are not guaranteed
        # idempotent, so retrying a transport fault can double-execute a
        # state-mutating tool. We accept this: the alternative (a fatal session
        # teardown on every transient blip) is worse for the long-running tasks
        # this serves, and tool idempotency is the server's contract to uphold.
        attempt = 0
        while True:
            try:
                response = await self._wrapped.handle_async_request(request)
            except self._RETRYABLE_EXC as exc:
                if attempt >= self._max_retries:
                    raise
                delay = _retry_after_seconds(None, attempt)
                logger.warning(
                    "mcp_transport_retrying",
                    server=self._server_name,
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    error=type(exc).__name__,
                    delay_seconds=round(delay, 1),
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue
            if response.status_code != 429 or attempt >= self._max_retries:
                return response
            # We're going to retry: fully drain and close this response so the
            # connection is released before we sleep and re-send.
            await response.aread()
            await response.aclose()
            delay = _retry_after_seconds(response, attempt)
            logger.warning(
                "mcp_rate_limited_retrying",
                server=self._server_name,
                attempt=attempt + 1,
                max_retries=self._max_retries,
                delay_seconds=round(delay, 1),
            )
            await asyncio.sleep(delay)
            attempt += 1


def _build_transport(config: McpServerConfig, ssl_verify: str | bool) -> httpx.AsyncBaseTransport:
    """Build the httpx transport for an MCP OAuth client.

    Returns an ``AsyncHTTPTransport`` with TLS verification configured, wrapped
    in ``RetryTransport`` when 429 retries are enabled. Constructed explicitly
    and passed via the client's public ``transport=`` parameter rather than
    mutating ``client._transport``/``_mounts`` after the fact — that kept the
    retry wiring on httpx's private internals, which could silently break (or
    drop TLS verification) on an httpx upgrade. ``verify`` lives on the
    transport because that is where httpx applies it once ``transport=`` is set.
    """
    transport: httpx.AsyncBaseTransport = httpx.AsyncHTTPTransport(verify=ssl_verify)
    if config.max_retries > 0:
        transport = RetryTransport(transport, config.max_retries, config.name)
    return transport


async def build_oauth_http_client(
    config: McpServerConfig,
    runtime_dir: Path,
    *,
    interactive: bool = False,
) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with OAuth2 authentication.

    Selects the appropriate auth backend based on ``OAuthConfig``:

    - **Direct OIDC** (``auth_url``/``token_url`` set): Uses
      ``DirectOIDCAuth`` which handles Bearer injection and refresh.
      Interactive login is handled separately via ``run_direct_oidc_login``.

    - **MCP-spec OAuth** (no explicit endpoints): Uses the MCP SDK's
      ``OAuthClientProvider`` which discovers endpoints from the MCP
      server itself.

    Args:
        config: MCP server config (must have ``oauth`` set).
        runtime_dir: Application runtime directory (e.g. ``~/.ava``).
        interactive: When True (and MCP-spec OAuth), provide browser
            redirect and callback handlers for the full auth flow.

    Returns:
        An authenticated ``httpx.AsyncClient``.
    """
    if config.oauth is None:
        raise ValueError(f"MCP server '{config.name}' has no OAuth config")

    oauth = config.oauth
    storage_dir = runtime_dir / "oauth" / config.oauth_storage_key
    storage = FileTokenStorage(storage_dir, oauth)

    ssl_verify = _resolve_ssl_verify(runtime_dir)

    if oauth.is_direct:
        auth = DirectOIDCAuth(oauth, storage, ssl_verify=ssl_verify)
        return httpx.AsyncClient(
            auth=auth,
            transport=_build_transport(config, ssl_verify),
            timeout=httpx.Timeout(config.timeout),
            follow_redirects=True,
        )

    # MCP-spec OAuth (Slack, etc.)
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    redirect_uri = f"http://localhost:{oauth.callback_port}/callback"
    client_metadata = OAuthClientMetadata(
        client_name=f"switchplane:{config.name}",
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        scope=oauth.scopes,
    )

    redirect_handler = None
    callback_handler = None
    _callback_server: OAuthCallbackServer | None = None

    if interactive:
        _callback_server = OAuthCallbackServer(oauth.callback_port)
        await _callback_server.start()

        async def _redirect(url: str) -> None:
            # OAuthClientProvider builds the authorize URL internally;
            # append OAuthConfig.extra_authorize_params (e.g. Slack's
            # `team` to pin an enterprise) before opening the browser.
            if oauth.extra_authorize_params:
                parsed = urlparse(url)
                query = parse_qs(parsed.query, keep_blank_values=True)
                for k, v in oauth.extra_authorize_params.items():
                    query[k] = [v]
                new_query = urlencode(query, doseq=True)
                url = parsed._replace(query=new_query).geturl()
            logger.info("oauth_opening_browser", url=url)
            webbrowser.open(url)

        async def _callback() -> tuple[str, str | None]:
            assert _callback_server is not None
            try:
                return await asyncio.wait_for(_callback_server.wait_for_callback(), timeout=300.0)
            finally:
                await _callback_server.stop()

        redirect_handler = _redirect
        callback_handler = _callback

    auth = OAuthClientProvider(
        server_url=config.url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )

    return httpx.AsyncClient(
        auth=auth,
        transport=_build_transport(config, ssl_verify),
        timeout=httpx.Timeout(config.timeout),
        follow_redirects=True,
    )
