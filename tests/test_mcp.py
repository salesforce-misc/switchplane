import os
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from switchplane.app import McpServerConfig
from switchplane.mcp import (
    McpManager,
    McpSession,
    _json_schema_to_pydantic,
    _json_type_to_python,
    _mcp_tool_to_langchain,
    _resolve_python_type,
)

itest = pytest.mark.skipif(os.environ.get("ITEST") != "1", reason="ITEST=1 not set")


class TestJsonTypeToPython:
    def test_string(self):
        assert _json_type_to_python("string") is str

    def test_integer(self):
        assert _json_type_to_python("integer") is int

    def test_number(self):
        assert _json_type_to_python("number") is float

    def test_boolean(self):
        assert _json_type_to_python("boolean") is bool

    def test_array(self):
        assert _json_type_to_python("array") is list

    def test_object(self):
        assert _json_type_to_python("object") is dict

    def test_unknown_defaults_to_str(self):
        assert _json_type_to_python("unknown") is str


class TestResolvePythonType:
    def test_simple_string(self):
        assert _resolve_python_type({"type": "string"}) is str

    def test_simple_array(self):
        assert _resolve_python_type({"type": "array"}) is list

    def test_typed_array(self):
        assert _resolve_python_type({"type": "array", "items": {"type": "string"}}) == list[str]

    def test_nested_typed_array(self):
        schema = {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}
        assert _resolve_python_type(schema) == list[list[int]]

    def test_anyof_array_or_null(self):
        schema = {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ]
        }
        assert _resolve_python_type(schema) == list[str] | None

    def test_oneof_string_or_int(self):
        schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        assert _resolve_python_type(schema) == str | int

    def test_list_valued_type(self):
        schema = {"type": ["string", "null"]}
        assert _resolve_python_type(schema) == str | None

    def test_list_valued_type_single(self):
        assert _resolve_python_type({"type": ["integer"]}) is int

    def test_anyof_single_variant(self):
        schema = {"anyOf": [{"type": "string"}]}
        assert _resolve_python_type(schema) is str

    def test_defaults_to_str(self):
        assert _resolve_python_type({}) is str


class TestJsonSchemaToPydantic:
    def test_simple_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "age": {"type": "integer", "description": "User age"},
            },
            "required": ["name"],
        }
        model = _json_schema_to_pydantic("test_tool", schema)
        assert "name" in model.model_fields
        assert "age" in model.model_fields

        instance = model(name="Alice")
        assert instance.name == "Alice"
        assert instance.age is None

    def test_all_required(self):
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x", "y"],
        }
        model = _json_schema_to_pydantic("coords", schema)

        with pytest.raises(ValidationError):
            model()  # missing required fields

        instance = model(x=1, y=2)
        assert instance.x == 1

    def test_no_properties(self):
        schema = {"type": "object"}
        model = _json_schema_to_pydantic("empty", schema)
        instance = model()
        assert instance is not None

    def test_optional_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
        }
        model = _json_schema_to_pydantic("search", schema)
        instance = model()
        assert instance.query is None

    def test_anyof_array_or_null(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "columns": {
                    "default": None,
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                },
            },
            "required": ["query"],
        }
        model = _json_schema_to_pydantic("query_splunk", schema)

        instance = model(query="index=foo", columns=["sms"])
        assert instance.columns == ["sms"]

        instance = model(query="index=foo", columns=None)
        assert instance.columns is None

        instance = model(query="index=foo")
        assert instance.columns is None

        with pytest.raises(ValidationError):
            model(query="index=foo", columns=123)

    def test_list_valued_type(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"]},
            },
        }
        model = _json_schema_to_pydantic("flexible", schema)
        assert model(value="hello").value == "hello"
        assert model(value=None).value is None

    def test_typed_array_items(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tags"],
        }
        model = _json_schema_to_pydantic("tagged", schema)
        instance = model(tags=["a", "b"])
        assert instance.tags == ["a", "b"]

    def test_multiple_types(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "flag": {"type": "boolean"},
                "items": {"type": "array"},
                "meta": {"type": "object"},
            },
            "required": ["name", "count", "ratio", "flag", "items", "meta"],
        }
        model = _json_schema_to_pydantic("multi", schema)
        instance = model(
            name="test",
            count=5,
            ratio=0.5,
            flag=True,
            items=[1, 2],
            meta={"k": "v"},
        )
        assert instance.name == "test"
        assert instance.flag is True


class TestMcpSession:
    def test_init(self):
        cfg = McpServerConfig(name="test", command=["echo"])
        session = McpSession(cfg)
        assert session.config.name == "test"
        assert session.session is None

    @pytest.mark.asyncio
    async def test_list_tools_not_initialized(self):
        cfg = McpServerConfig(name="test", command=["echo"])
        session = McpSession(cfg)
        tools = await session.list_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_call_tool_not_initialized(self):
        cfg = McpServerConfig(name="test", command=["echo"])
        session = McpSession(cfg)
        with pytest.raises(RuntimeError, match="not initialized"):
            await session.call_tool("test_tool")


class TestMcpManager:
    def test_getitem_missing(self):
        manager = McpManager([])
        with pytest.raises(KeyError, match="not found"):
            manager["nonexistent"]

    def test_getitem_existing(self):
        manager = McpManager([])
        cfg = McpServerConfig(name="test", command=["echo"])
        session = McpSession(cfg)
        manager._sessions["test"] = session
        assert manager["test"] is session

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        manager = McpManager([])
        await manager.stop()  # should not raise

    def test_init_stores_configs(self):
        configs = [
            McpServerConfig(name="a", command=["echo"]),
            McpServerConfig(name="b", url="http://x"),
        ]
        manager = McpManager(configs)
        assert len(manager._configs) == 2
        assert manager._sessions == {}

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        manager = McpManager([])
        await manager.start()
        assert manager._stack is not None
        await manager.stop()
        assert manager._sessions == {}

    @pytest.mark.asyncio
    async def test_start_returns_name_keyed_errors(self, monkeypatch):
        """A failed session is reported as (name, message), not just text, so
        callers can attribute the failure to its config without parsing."""
        cfg = McpServerConfig(name="boomsrv", command=["echo"])
        manager = McpManager([cfg])

        async def _boom(self, stack):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(McpSession, "start", _boom)

        errors = await manager.start()
        assert len(errors) == 1
        name, msg = errors[0]
        assert name == "boomsrv"
        assert "boomsrv" in msg and "kaboom" in msg
        assert manager._sessions == {}  # the failed session is not registered
        await manager.stop()

    @pytest.mark.asyncio
    async def test_langchain_tools_with_sessions(self):
        cfg = McpServerConfig(name="test", command=["echo"])
        manager = McpManager([cfg])

        mock_session = MagicMock(spec=McpSession)
        mock_tool = MagicMock()
        mock_tool.name = "my_tool"
        mock_tool.description = "Does stuff"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"x": {"type": "integer", "description": "The x value"}},
            "required": ["x"],
        }
        mock_session.list_tools = AsyncMock(return_value=[mock_tool])
        manager._sessions["test"] = mock_session

        tools = await manager.langchain_tools()
        assert len(tools) == 1
        assert tools[0].name == "my_tool"

    @pytest.mark.asyncio
    async def test_langchain_tools_empty(self):
        manager = McpManager([])
        tools = await manager.langchain_tools()
        assert tools == []


class TestMcpToolToLangchain:
    def test_converts_tool(self):
        session = MagicMock(spec=McpSession)
        mcp_tool = MagicMock()
        mcp_tool.name = "search"
        mcp_tool.description = "Search for things"
        mcp_tool.inputSchema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }

        tool = _mcp_tool_to_langchain(session, mcp_tool, "server1")
        assert tool.name == "search"
        assert "Search" in tool.description

    def test_no_description(self):
        session = MagicMock(spec=McpSession)
        mcp_tool = MagicMock()
        mcp_tool.name = "tool1"
        mcp_tool.description = None
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        tool = _mcp_tool_to_langchain(session, mcp_tool, "myserver")
        assert "myserver" in tool.description

    def test_no_input_schema(self):
        session = MagicMock(spec=McpSession)
        mcp_tool = MagicMock()
        mcp_tool.name = "simple"
        mcp_tool.description = "Simple tool"
        mcp_tool.inputSchema = None

        tool = _mcp_tool_to_langchain(session, mcp_tool, "srv")
        assert tool.name == "simple"


class TestPreflightIntegration:
    """Integration tests against a live MCP endpoint. Only run with ITEST=1."""

    @itest
    @pytest.mark.asyncio
    async def test_preflight_succeeds(self):
        config = McpServerConfig(name="deepwiki", url="https://mcp.deepwiki.com/mcp")
        session = McpSession(config)
        stack = AsyncExitStack()
        async with stack:
            await session.start(stack)
            assert session.session is not None
            tools = await session.list_tools()
            assert len(tools) > 0

    @itest
    @pytest.mark.asyncio
    async def test_preflight_unreachable_raises(self):
        config = McpServerConfig(name="bad", url="https://unreachable.invalid/mcp")
        session = McpSession(config)
        stack = AsyncExitStack()
        async with stack:
            with pytest.raises(ConnectionError):
                await session.start(stack)
