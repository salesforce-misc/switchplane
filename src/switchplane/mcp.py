"""MCP client lifecycle management and LangChain tool integration."""

import datetime
import importlib
import inspect
import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import structlog

from switchplane.app import McpServerConfig

logger = structlog.get_logger()


def _import_transport_factory(dotted_path: str):
    """Import and validate an HTTP transport factory from a dotted path.

    The factory must be a callable that accepts a single positional argument
    (McpServerConfig) and returns an httpx.AsyncClient.
    """
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(
            f"Invalid transport factory path '{dotted_path}': must be a dotted path (e.g. 'myapp.transports.build_client')"
        )
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(f"Cannot import transport factory module '{module_path}': {e}") from e

    factory = getattr(module, attr_name, None)
    if factory is None:
        raise ImportError(f"Module '{module_path}' has no attribute '{attr_name}'")
    if not callable(factory):
        raise TypeError(f"Transport factory '{dotted_path}' is not callable")

    sig = inspect.signature(factory)
    # Must accept at least one positional argument (the config).
    positional_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    positional = [p for p in sig.parameters.values() if p.kind in positional_kinds]
    required_positional = [p for p in positional if p.default is inspect.Parameter.empty]
    if len(required_positional) != 1:
        raise TypeError(
            f"Transport factory '{dotted_path}' must accept exactly one required "
            f"positional argument (McpServerConfig), got {len(required_positional)}"
        )

    import httpx

    ret = sig.return_annotation
    if ret is not inspect.Signature.empty and ret is not httpx.AsyncClient:
        raise TypeError(f"Transport factory '{dotted_path}' return annotation must be httpx.AsyncClient, got {ret}")

    return factory


class McpSession:
    """Manages a single MCP client session."""

    def __init__(self, config: McpServerConfig, runtime_dir: Path | None = None):
        self.config = config
        self.session = None
        self._runtime_dir = runtime_dir
        self._stack: AsyncExitStack | None = None

    async def start(self, stack: AsyncExitStack) -> None:
        """Start the MCP session within the given async exit stack."""
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.streamable_http import streamable_http_client

        if self.config.command:
            params = StdioServerParameters(
                command=self.config.command[0],
                args=self.config.command[1:],
                env=self.config.env or None,
            )
            read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
        else:
            import httpx

            http_client = None
            if self.config.oauth:
                from switchplane.oauth import build_oauth_http_client

                if self._runtime_dir is None:
                    raise RuntimeError(f"MCP server '{self.config.name}' uses OAuth but runtime_dir was not provided")
                http_client = await build_oauth_http_client(self.config, self._runtime_dir)
                logger.debug("mcp_oauth_transport", server=self.config.name)
            elif self.config.http_transport:
                factory = _import_transport_factory(self.config.http_transport)
                http_client = factory(self.config)
                logger.debug("mcp_custom_transport", server=self.config.name, factory=self.config.http_transport)

            if http_client is not None:
                http_client = await stack.enter_async_context(http_client)

            # Pre-flight connectivity check.  The MCP streamable HTTP
            # transport uses anyio task groups internally which swallow
            # the real error and surface only a CancelledError.  A
            # quick GET probe surfaces actionable DNS/SSL/auth errors
            # before we enter the MCP protocol layer.  Timeouts are
            # expected (MCP endpoints are POST-oriented) and treated
            # as "reachable".  For OAuth servers a bare (unauthenticated)
            # client is used — 401 is fine, it proves connectivity.
            if self.config.oauth:
                from switchplane.oauth import _resolve_ssl_verify

                ssl_verify = _resolve_ssl_verify(self._runtime_dir)
                preflight_client = httpx.AsyncClient(
                    verify=ssl_verify,
                    timeout=httpx.Timeout(5.0),
                    follow_redirects=True,
                )
            else:
                preflight_client = http_client or httpx.AsyncClient(
                    timeout=httpx.Timeout(5.0),
                    follow_redirects=True,
                )

            try:
                async with preflight_client.stream("GET", self.config.url) as resp:
                    logger.info("mcp_preflight_ok", server=self.config.name, status=resp.status_code)
            except httpx.TimeoutException:
                logger.info("mcp_preflight_timeout", server=self.config.name)
            except Exception as e:
                raise ConnectionError(f"Cannot reach MCP server '{self.config.name}' at {self.config.url}: {e}") from e
            finally:
                if self.config.oauth or http_client is None:
                    await preflight_client.aclose()

            logger.info("mcp_http_client_connecting", server=self.config.name)
            read_stream, write_stream, _get_session_id = await stack.enter_async_context(
                streamable_http_client(self.config.url, http_client=http_client)
            )
            logger.info("mcp_http_client_ready", server=self.config.name)

        self.session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        logger.info("mcp_session_initializing", server=self.config.name)
        await self.session.initialize()
        logger.info("mcp_session_ready", server=self.config.name, transport=self.config.transport)

    async def list_tools(self) -> list:
        """List available tools from this MCP server."""
        if not self.session:
            return []
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None):
        """Call a tool on this MCP server."""
        if not self.session:
            raise RuntimeError(f"MCP session '{self.config.name}' not initialized")
        # The MCP SDK applies its own ``read_timeout_seconds`` *around* the
        # whole request, independently of the httpx client timeout. When retries
        # are enabled (``max_retries > 0``), the RetryTransport — using the
        # per-attempt ``config.timeout`` — is the sole timeout authority: a fixed
        # SDK ceiling would cancel a live retry sequence mid-flight. So we leave
        # it unbounded and let the transport govern. With retries disabled, the
        # SDK ceiling mirrors the client timeout (unchanged legacy behavior).
        if self.config.max_retries > 0:
            read_timeout = None
        else:
            read_timeout = datetime.timedelta(seconds=self.config.timeout) if self.config.timeout is not None else None
        return await self.session.call_tool(name, arguments, read_timeout_seconds=read_timeout)


class McpManager:
    """Manages multiple MCP sessions for an agent."""

    def __init__(self, configs: list[McpServerConfig], runtime_dir: Path | None = None):
        self._configs = configs
        self._runtime_dir = runtime_dir
        self._sessions: dict[str, McpSession] = {}
        self._stack: AsyncExitStack | None = None

    def __getitem__(self, name: str) -> McpSession:
        """Get an MCP session by server name."""
        if name not in self._sessions:
            raise KeyError(f"MCP server '{name}' not found")
        return self._sessions[name]

    def get(self, name: str, default=None) -> McpSession | None:
        """Get an MCP session by server name, returning *default* if not found."""
        return self._sessions.get(name, default)

    async def start(self) -> list[tuple[str, str]]:
        """Start all MCP sessions.

        Returns a list of ``(server_name, error_message)`` tuples for sessions
        that failed to start. The name is reported structurally rather than only
        embedded in the message so callers can attribute a failure to its config
        (e.g. to honor a per-server ``optional`` flag) without re-parsing text.
        """
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        errors: list[tuple[str, str]] = []
        for config in self._configs:
            session = McpSession(config, runtime_dir=self._runtime_dir)
            try:
                await session.start(self._stack)
                self._sessions[config.name] = session
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                msg = f"Failed to start MCP server '{config.name}': {e}"
                logger.error("mcp_server_start_failed", server=config.name, error=str(e))
                errors.append((config.name, msg))
        return errors

    async def stop(self) -> None:
        """Stop all MCP sessions."""
        if self._stack:
            await self._stack.aclose()
            self._sessions.clear()

    async def langchain_tools(self) -> list:
        """Convert all MCP tools to LangChain StructuredTool instances."""

        tools = []
        for server_name, session in self._sessions.items():
            mcp_tools = await session.list_tools()
            for mcp_tool in mcp_tools:
                tool = _mcp_tool_to_langchain(session, mcp_tool, server_name)
                tools.append(tool)
        return tools


def _mcp_tool_to_langchain(session: McpSession, mcp_tool, server_name: str):
    """Convert a single MCP tool definition to a LangChain StructuredTool."""
    from langchain_core.tools import StructuredTool

    tool_name = mcp_tool.name
    description = mcp_tool.description or f"MCP tool '{tool_name}' from {server_name}"
    input_schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}

    async def _invoke(tool_name=tool_name, **kwargs) -> str:
        result = await session.call_tool(tool_name, kwargs if kwargs else None)
        # MCP returns CallToolResult with content list
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            else:
                parts.append(json.dumps(content.model_dump()))
        return "\n".join(parts)

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=tool_name,
        description=description,
        args_schema=_json_schema_to_pydantic(tool_name, input_schema),
    )


def _json_schema_to_pydantic(name: str, schema: dict) -> type:
    """Build a Pydantic model from a JSON schema (for LangChain args_schema)."""
    from pydantic import create_model
    from pydantic.fields import FieldInfo

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields = {}
    for prop_name, prop_schema in properties.items():
        py_type = _resolve_python_type(prop_schema)
        default = ... if prop_name in required else None
        field_info = FieldInfo(default=default, description=prop_schema.get("description"))
        if prop_name not in required:
            py_type = py_type | None
        fields[prop_name] = (py_type, field_info)

    return create_model(f"_{name}_Args", **fields)


def _resolve_python_type(prop_schema: dict) -> type:
    """Resolve a JSON Schema property to a Python type.

    Handles ``anyOf``/``oneOf`` unions, list-valued ``type`` fields,
    and typed arrays (``items``).
    """
    if "anyOf" in prop_schema or "oneOf" in prop_schema:
        variants = prop_schema.get("anyOf") or prop_schema.get("oneOf", [])
        types = [type(None) if v.get("type") == "null" else _resolve_python_type(v) for v in variants]
        types = list(dict.fromkeys(types))  # dedupe, preserve order
        if len(types) == 1:
            return types[0]
        result = types[0]
        for t in types[1:]:
            result = result | t
        return result

    json_type = prop_schema.get("type", "string")

    if isinstance(json_type, list):
        types = [type(None) if t == "null" else _json_type_to_python(t) for t in json_type]
        types = list(dict.fromkeys(types))
        if len(types) == 1:
            return types[0]
        result = types[0]
        for t in types[1:]:
            result = result | t
        return result

    base = _json_type_to_python(json_type)
    if base is list and "items" in prop_schema:
        item_type = _resolve_python_type(prop_schema["items"])
        return list[item_type]
    return base


def _json_type_to_python(json_type: str) -> type:
    """Map JSON schema types to Python types."""
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)
