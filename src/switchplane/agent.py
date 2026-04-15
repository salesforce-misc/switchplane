"""Agent specifications and records."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentStatus(StrEnum):
    """Lifecycle status of an agent subprocess."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"


class AgentRecord(BaseModel):
    """Record of an agent instance."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    agent_id: str
    agent_name: str
    pid: int | None = None
    status: AgentStatus = AgentStatus.IDLE
    capabilities_json: str = "{}"
    started_at: datetime | None = None
    last_heartbeat: datetime | None = None


class AgentSpec(BaseModel):
    """Specification for an agent."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    agent_name: str
    module_path: str = ""  # Set by discovery — dotted Python path to agent module
    mcp_servers: list[str] = []  # Allowed MCP server names
    tasks: dict[str, Any] = {}  # task_name -> Task class
