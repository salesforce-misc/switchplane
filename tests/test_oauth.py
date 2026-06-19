"""Tests for switchplane.oauth.

Focused on the RetryTransport / 429-handling logic. The interactive OAuth
flows (browser redirect, callback server) are exercised via integration tests
elsewhere; here we cover the rate-limit retry that wraps the MCP HTTP client.
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from switchplane.app import McpServerConfig, OAuthConfig
from switchplane.oauth import (
    RetryTransport,
    _build_transport,
    _retry_after_seconds,
    build_oauth_http_client,
)


class _CountingTransport(httpx.AsyncBaseTransport):
    """Returns queued responses in order, counting how many requests it saw.

    Records, at the moment each new request arrives, whether the previously
    returned response had been closed — so tests can assert the retry loop
    drains/releases a 429 before re-sending.
    """

    def __init__(self, statuses, headers=None):
        self._statuses = list(statuses)
        self._headers = headers or {}
        self.request_count = 0
        self.closed = False
        self._last_response: httpx.Response | None = None
        self.prev_closed_at_next_request: list[bool] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._last_response is not None:
            self.prev_closed_at_next_request.append(self._last_response.is_closed)
        self.request_count += 1
        status = self._statuses.pop(0)
        hdrs = self._headers if status == 429 else {}
        resp = httpx.Response(status, headers=hdrs, request=request, content=b"body")
        self._last_response = resp
        return resp

    async def aclose(self) -> None:
        self.closed = True


# -- _retry_after_seconds ---------------------------------------------------


def _resp(status=429, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(status, headers=headers, request=httpx.Request("POST", "http://x"))


def test_retry_after_header_used_when_larger_than_backoff():
    # Header (7s) exceeds the attempt-0 exponential floor (2s) -> honor header.
    assert _retry_after_seconds(_resp(retry_after="7"), attempt=0) == 7.0


def test_retry_after_header_clamped_to_max():
    assert _retry_after_seconds(_resp(retry_after="9999"), attempt=0) == 60.0


def test_small_constant_header_cannot_defeat_exponential_backoff():
    # Slack sends Retry-After: 1 on every 429; escalation must still win so we
    # actually outlast the throttle window instead of retrying every ~1s.
    assert _retry_after_seconds(_resp(retry_after="1"), attempt=0) == 2.0
    assert _retry_after_seconds(_resp(retry_after="1"), attempt=1) == 4.0
    assert _retry_after_seconds(_resp(retry_after="1"), attempt=2) == 8.0


def test_retry_after_falls_back_to_exponential_backoff():
    assert _retry_after_seconds(_resp(), attempt=0) == 2.0
    assert _retry_after_seconds(_resp(), attempt=1) == 4.0
    assert _retry_after_seconds(_resp(), attempt=2) == 8.0


def test_retry_after_backoff_capped():
    # 2 * 2**6 = 128 -> clamped to 60
    assert _retry_after_seconds(_resp(), attempt=6) == 60.0


def test_retry_after_non_numeric_header_falls_back():
    # HTTP-date form isn't parsed; falls back to backoff.
    assert _retry_after_seconds(_resp(retry_after="Wed, 21 Oct 2025 07:28:00 GMT"), attempt=0) == 2.0


# -- RetryTransport ---------------------------------------------------------


async def test_retry_transport_retries_429_then_returns_success():
    inner = _CountingTransport([429, 429, 200])
    rt = RetryTransport(inner, max_retries=3, server_name="slack")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()) as sleep:
        resp = await rt.handle_async_request(httpx.Request("POST", "https://mcp.slack.com/mcp"))
    assert resp.status_code == 200
    assert inner.request_count == 3
    assert sleep.await_count == 2  # two 429s -> two sleeps


async def test_retry_transport_gives_up_after_max_retries():
    inner = _CountingTransport([429, 429, 429, 429, 429])
    rt = RetryTransport(inner, max_retries=3, server_name="slack")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()):
        resp = await rt.handle_async_request(httpx.Request("POST", "https://mcp.slack.com/mcp"))
    # Returns the final 429 rather than raising; caller (SDK) decides.
    assert resp.status_code == 429
    assert inner.request_count == 4  # initial + 3 retries


async def test_retry_transport_passes_through_non_429():
    inner = _CountingTransport([503])
    rt = RetryTransport(inner, max_retries=3, server_name="slack")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()) as sleep:
        resp = await rt.handle_async_request(httpx.Request("POST", "https://mcp.slack.com/mcp"))
    assert resp.status_code == 503
    assert inner.request_count == 1
    sleep.assert_not_awaited()


async def test_retry_transport_honors_retry_after_header():
    inner = _CountingTransport([429, 200], headers={"Retry-After": "3"})
    rt = RetryTransport(inner, max_retries=3, server_name="slack")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()) as sleep:
        await rt.handle_async_request(httpx.Request("POST", "https://mcp.slack.com/mcp"))
    sleep.assert_awaited_once_with(3.0)


async def test_retry_transport_drains_429_before_resending():
    inner = _CountingTransport([429, 429, 200])
    rt = RetryTransport(inner, max_retries=3, server_name="slack")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()):
        await rt.handle_async_request(httpx.Request("POST", "https://mcp.slack.com/mcp"))
    # Each retry observed the prior (429) response already closed.
    assert inner.prev_closed_at_next_request == [True, True]


async def test_retry_transport_aclose_delegates():
    inner = _CountingTransport([200])
    rt = RetryTransport(inner, max_retries=3, server_name="slack")
    await rt.aclose()
    assert inner.closed is True


# -- RetryTransport: transport faults ---------------------------------------


class _FaultTransport(httpx.AsyncBaseTransport):
    """Replays a queued sequence of outcomes.

    Each item is either an ``Exception`` instance (raised) or an int status
    (returned as a response). Mirrors a flaky transport that times out before
    eventually answering, and records the request bodies it saw so a test can
    assert the request was replayed unchanged across retries.
    """

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.request_count = 0
        self.seen_bodies: list[bytes] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.request_count += 1
        self.seen_bodies.append(request.content)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, request=request, content=b"ok")


def _req():
    return httpx.Request("POST", "https://mcp.codesearch/mcp", content=b'{"jsonrpc":"2.0"}')


async def test_retry_transport_retries_read_timeout_then_succeeds():
    inner = _FaultTransport([httpx.ReadTimeout("hung"), httpx.ReadTimeout("hung"), 200])
    rt = RetryTransport(inner, max_retries=3, server_name="codesearch")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()) as sleep:
        resp = await rt.handle_async_request(_req())
    assert resp.status_code == 200
    assert inner.request_count == 3
    assert sleep.await_count == 2


async def test_retry_transport_replays_body_unchanged_across_retries():
    inner = _FaultTransport([httpx.ReadTimeout("hung"), 200])
    rt = RetryTransport(inner, max_retries=3, server_name="codesearch")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()):
        await rt.handle_async_request(_req())
    assert inner.seen_bodies == [b'{"jsonrpc":"2.0"}', b'{"jsonrpc":"2.0"}']


async def test_retry_transport_reraises_timeout_after_max_retries_without_id():
    # `_req()`'s body carries no JSON-RPC `id`, so there is no foreground waiter
    # to route a synthesized error response to — the transport must re-raise.
    inner = _FaultTransport([httpx.ReadTimeout("hung")] * 5)
    rt = RetryTransport(inner, max_retries=2, server_name="codesearch")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()), pytest.raises(httpx.ReadTimeout):
        await rt.handle_async_request(_req())
    assert inner.request_count == 3  # initial + 2 retries


def _req_with_id(request_id=7):
    # An id-bearing JSON-RPC request, as the MCP SDK sends for `tools/call`.
    body = f'{{"jsonrpc":"2.0","id":{request_id},"method":"tools/call"}}'.encode()
    return httpx.Request("POST", "https://mcp.codesearch/mcp", content=body)


async def test_retry_transport_synthesizes_in_band_error_on_exhaustion_with_id():
    # An id-bearing request whose retries are exhausted must NOT re-raise (that
    # would crash the SDK's session task group); it returns a JSON-RPC error
    # response echoing the request id so the SDK surfaces a catchable McpError.
    inner = _FaultTransport([httpx.ReadTimeout("hung")] * 5)
    rt = RetryTransport(inner, max_retries=2, server_name="codesearch")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()):
        resp = await rt.handle_async_request(_req_with_id(7))
    assert inner.request_count == 3  # initial + 2 retries
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "application/json"
    payload = json.loads(resp.content)
    assert payload["id"] == 7
    assert payload["error"]["code"] == int(httpx.codes.REQUEST_TIMEOUT)
    assert "ReadTimeout" in payload["error"]["message"]


async def test_retry_transport_connect_error_exhaustion_also_synthesizes():
    inner = _FaultTransport([httpx.ConnectError("refused")] * 5)
    rt = RetryTransport(inner, max_retries=1, server_name="codesearch")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()):
        resp = await rt.handle_async_request(_req_with_id(3))
    payload = json.loads(resp.content)
    assert payload["id"] == 3
    assert "ConnectError" in payload["error"]["message"]


async def test_retry_transport_retries_connect_error():
    inner = _FaultTransport([httpx.ConnectError("refused"), 200])
    rt = RetryTransport(inner, max_retries=3, server_name="codesearch")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()):
        resp = await rt.handle_async_request(_req())
    assert resp.status_code == 200
    assert inner.request_count == 2


async def test_retry_transport_does_not_retry_non_transient_exception():
    # A bad-request protocol error won't fix on retry; it must propagate at once.
    inner = _FaultTransport([httpx.UnsupportedProtocol("bad")])
    rt = RetryTransport(inner, max_retries=3, server_name="codesearch")
    with (
        patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()) as sleep,
        pytest.raises(httpx.UnsupportedProtocol),
    ):
        await rt.handle_async_request(_req())
    assert inner.request_count == 1
    sleep.assert_not_awaited()


async def test_retry_transport_zero_retries_raises_on_first_timeout():
    inner = _FaultTransport([httpx.ReadTimeout("hung"), 200])
    rt = RetryTransport(inner, max_retries=0, server_name="codesearch")
    with patch("switchplane.oauth.asyncio.sleep", new=AsyncMock()), pytest.raises(httpx.ReadTimeout):
        await rt.handle_async_request(_req())
    assert inner.request_count == 1


def test_retry_after_seconds_handles_none_response():
    # Transport faults pass response=None -> pure exponential backoff.
    assert _retry_after_seconds(None, attempt=0) == 2.0
    assert _retry_after_seconds(None, attempt=1) == 4.0


# -- _build_transport -------------------------------------------------------


def test_build_transport_wraps_in_retry_when_enabled():
    config = McpServerConfig(name="slack", url="https://mcp.slack.com/mcp", max_retries=3)
    transport = _build_transport(config, ssl_verify=True)
    assert isinstance(transport, RetryTransport)
    # The wrapped transport is a real httpx transport, not the client's guts.
    assert isinstance(transport._wrapped, httpx.AsyncHTTPTransport)


def test_build_transport_plain_when_disabled():
    config = McpServerConfig(name="slack", url="https://mcp.slack.com/mcp", max_retries=0)
    transport = _build_transport(config, ssl_verify=True)
    assert not isinstance(transport, RetryTransport)
    assert isinstance(transport, httpx.AsyncHTTPTransport)


# -- build_oauth_http_client integration ------------------------------------


async def test_build_oauth_http_client_direct_oidc_wraps_retry(tmp_path):
    config = McpServerConfig(
        name="dxmcp",
        url="https://example.invalid/mcp",
        oauth=OAuthConfig(
            client_id="cid",
            auth_url="https://idp.invalid/auth",
            token_url="https://idp.invalid/token",
        ),
        max_retries=3,
    )
    client = await build_oauth_http_client(config, tmp_path)
    try:
        assert isinstance(client._transport, RetryTransport)
    finally:
        await client.aclose()


async def test_build_oauth_http_client_mcp_spec_oauth_wraps_retry(tmp_path):
    # The Slack path: MCP-spec OAuth (no explicit auth_url/token_url), which
    # uses the SDK's OAuthClientProvider. This is the actual motivating case.
    pytest.importorskip("mcp")  # MCP-spec branch imports the optional mcp package
    config = McpServerConfig(
        name="slack",
        url="https://mcp.slack.com/mcp",
        oauth=OAuthConfig(client_id="cid", callback_port=3118),
        max_retries=3,
    )
    client = await build_oauth_http_client(config, tmp_path)
    try:
        assert isinstance(client._transport, RetryTransport)
    finally:
        await client.aclose()


async def test_build_oauth_http_client_respects_max_retries_zero(tmp_path):
    pytest.importorskip("mcp")  # MCP-spec branch imports the optional mcp package
    config = McpServerConfig(
        name="slack",
        url="https://mcp.slack.com/mcp",
        oauth=OAuthConfig(client_id="cid", callback_port=3118),
        max_retries=0,
    )
    client = await build_oauth_http_client(config, tmp_path)
    try:
        assert not isinstance(client._transport, RetryTransport)
    finally:
        await client.aclose()
