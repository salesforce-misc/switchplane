"""Application and MCP server configuration."""

import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from switchplane.agent import AgentSpec
    from switchplane.config import AppConfig

_VALID_APP_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


class OAuthConfig(BaseModel):
    """OAuth 2.0 configuration for an MCP server.

    Two modes are supported:

    **MCP-spec OAuth** (default): Leave ``auth_url`` and ``token_url``
    unset.  The MCP SDK's ``OAuthClientProvider`` discovers endpoints
    from the MCP server's protected-resource metadata automatically.

    **Direct OIDC**: Set ``auth_url`` and ``token_url`` to point at an
    external identity provider (e.g. Keycloak/QuantumK).  Switchplane
    runs the PKCE authorization-code flow against those endpoints
    directly, bypassing MCP metadata discovery.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    client_id: str
    client_secret: str | None = None
    callback_port: int = 3118
    scopes: str | None = None
    auth_url: str | None = None
    token_url: str | None = None
    extra_authorize_params: dict[str, str] = {}

    @property
    def is_direct(self) -> bool:
        """True when explicit OIDC endpoints are configured."""
        return self.auth_url is not None and self.token_url is not None

    def model_post_init(self, __context) -> None:
        if bool(self.auth_url) != bool(self.token_url):
            raise ValueError("Provide both 'auth_url' and 'token_url', or neither")


class McpServerConfig(BaseModel):
    """Configuration for an MCP server.

    Provide `command` for stdio transport (Switchplane manages the process)
    or `url` for HTTP transport (Switchplane connects to an existing server).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    name: str
    command: list[str] | None = None
    url: str | None = None
    env: dict[str, str] = {}
    http_transport: str | None = None
    oauth: OAuthConfig | None = None
    oauth_group: str | None = None
    timeout: float | None = 30.0

    @property
    def oauth_storage_key(self) -> str:
        """Key for OAuth token storage. Servers sharing an oauth_group share tokens."""
        return self.oauth_group or self.name

    @property
    def transport(self) -> str:
        return "stdio" if self.command else "http"

    def model_post_init(self, __context) -> None:
        if self.command and self.url:
            raise ValueError("Provide either 'command' (stdio) or 'url' (http), not both")
        if not self.command and not self.url:
            raise ValueError("Provide either 'command' (stdio) or 'url' (http)")
        if self.oauth and self.http_transport:
            raise ValueError("Provide either 'oauth' or 'http_transport', not both")
        if self.oauth and not self.url:
            raise ValueError("OAuth requires HTTP transport ('url' must be set)")


class Application:
    """Application container for agents and MCP servers."""

    def __init__(
        self,
        name: str,
        runtime_dir: str | Path | None = None,
        default_config: Path | None = None,
        config_class: "type[AppConfig] | None" = None,
    ) -> None:
        """Initialize application.

        Args:
            name: Application name
            runtime_dir: Optional runtime directory path (defaults to ~/.{name})
            default_config: Optional path to default config file bundled with the app
            config_class: Pydantic model class used to parse the merged config.
                Defaults to AppConfig; pass a subclass to support app-specific
                config sections that are otherwise dropped by the base model.
        """
        from switchplane.config import AppConfig as _AppConfig

        if not _VALID_APP_NAME.match(name):
            raise ValueError(
                f"Invalid application name {name!r}. "
                "Names must start with a letter and contain only letters, digits, hyphens, or underscores."
            )
        self.name = name
        self.runtime_dir = Path(runtime_dir).expanduser() if runtime_dir else Path.home() / f".{name}"
        self.default_config_path = default_config
        self.config_class: type[AppConfig] = config_class or _AppConfig
        self.agents: dict[str, AgentSpec] = {}
        self.mcp_servers: dict[str, McpServerConfig] = {}
        self._discovery_roots: list[str] = []

    def discover_agents(self, root: str) -> None:
        """Store root directory for later agent discovery.

        Args:
            root: Root directory path to discover agents from
        """
        self._discovery_roots.append(root)

    def register_agent(self, spec: "AgentSpec") -> None:
        """Register an agent specification.

        Args:
            spec: Agent specification to register
        """
        self.agents[spec.agent_name] = spec

    def register_mcp_server(self, config: McpServerConfig) -> None:
        """Register an MCP server configuration.

        Args:
            config: MCP server configuration to register
        """
        self.mcp_servers[config.name] = config

    def run(self) -> None:
        """Discover agents and start the CLI."""
        from switchplane.cli import build_cli
        from switchplane.discovery import discover_agents_for_app

        discover_agents_for_app(self)
        cli = build_cli(self)
        cli()
