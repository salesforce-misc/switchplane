from pathlib import Path

import pytest
from pydantic import ValidationError

from switchplane.agent import AgentSpec
from switchplane.app import Application, McpServerConfig


class TestMcpServerConfig:
    def test_stdio_transport(self):
        cfg = McpServerConfig(name="fs", command=["npx", "mcp-server-fs"])
        assert cfg.transport == "stdio"
        assert cfg.url is None
        assert cfg.env == {}

    def test_http_transport(self):
        cfg = McpServerConfig(name="remote", url="http://localhost:8080")
        assert cfg.transport == "http"
        assert cfg.command is None

    def test_both_raises(self):
        with pytest.raises(ValidationError, match="not both"):
            McpServerConfig(name="bad", command=["x"], url="http://x")

    def test_neither_raises(self):
        with pytest.raises(ValidationError, match="Provide either"):
            McpServerConfig(name="bad")

    def test_with_env(self):
        cfg = McpServerConfig(
            name="fs",
            command=["node", "server.js"],
            env={"API_KEY": "secret"},
        )
        assert cfg.env == {"API_KEY": "secret"}

    def test_strip_whitespace(self):
        cfg = McpServerConfig(name="  server  ", url="http://x")
        assert cfg.name == "server"


class TestApplication:
    def test_defaults(self):
        app = Application(name="myapp")
        assert app.name == "myapp"
        assert app.runtime_dir == Path.home() / ".myapp"
        assert app.default_config_path is None
        assert app.agents == {}
        assert app.mcp_servers == {}
        assert app._discovery_roots == []

    def test_custom_runtime_dir(self, tmp_path):
        app = Application(name="myapp", runtime_dir=tmp_path / "custom")
        assert app.runtime_dir == (tmp_path / "custom").expanduser()

    def test_with_default_config(self, tmp_path):
        cfg_path = tmp_path / "defaults.toml"
        app = Application(name="myapp", default_config=cfg_path)
        assert app.default_config_path == cfg_path

    def test_discover_agents(self):
        app = Application(name="myapp")
        app.discover_agents("myapp.agents")
        app.discover_agents("myapp.extra_agents")
        assert app._discovery_roots == ["myapp.agents", "myapp.extra_agents"]

    def test_register_agent(self):
        app = Application(name="myapp")
        spec = AgentSpec(agent_name="worker")
        app.register_agent(spec)
        assert "worker" in app.agents
        assert app.agents["worker"] is spec

    def test_register_mcp_server(self):
        app = Application(name="myapp")
        cfg = McpServerConfig(name="fs", command=["node", "fs-server.js"])
        app.register_mcp_server(cfg)
        assert "fs" in app.mcp_servers
        assert app.mcp_servers["fs"] is cfg

    def test_register_overwrites(self):
        app = Application(name="myapp")
        spec1 = AgentSpec(agent_name="worker", module_path="v1")
        spec2 = AgentSpec(agent_name="worker", module_path="v2")
        app.register_agent(spec1)
        app.register_agent(spec2)
        assert app.agents["worker"].module_path == "v2"

    def test_invalid_name_path_traversal(self):
        with pytest.raises(ValueError, match="Invalid application name"):
            Application(name="../../etc")

    def test_invalid_name_empty(self):
        with pytest.raises(ValueError, match="Invalid application name"):
            Application(name="")

    def test_invalid_name_starts_with_digit(self):
        with pytest.raises(ValueError, match="Invalid application name"):
            Application(name="123app")

    def test_valid_name_with_hyphens_underscores(self):
        app = Application(name="my-app_v2")
        assert app.name == "my-app_v2"
