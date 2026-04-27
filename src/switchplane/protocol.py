"""IPC message types for Switchplane.

Two distinct protocols, both using 4-byte big-endian length-prefixed JSON:
1. CLI ↔ Control Plane: Unix domain socket at ~/.{app}/runtime.sock
2. Agent ↔ Control Plane: Unix socketpair passed via --ipc-fd
"""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# CLI ↔ Control Plane Protocol


class CliRequest(BaseModel):
    """Request envelope from CLI to Control Plane."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class CliResponse(BaseModel):
    """Response envelope from Control Plane to CLI."""

    id: str
    ok: bool
    result: Any = None
    error: str | None = None


class StreamEvent(BaseModel):
    """Streaming task output from Control Plane to CLI."""

    task_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: datetime
    event_id: int = 0
    task_status: str = ""


# Agent ↔ Control Plane Protocol


class AgentEvent(BaseModel):
    """Event from Agent to Control Plane over the IPC socketpair."""

    type: Literal[
        "task.started",
        "task.progress",
        "task.completed",
        "task.failed",
        "task.cancelled",
        "task.interrupted",
        "task.resumed",
        "checkpoint.save",
        "llm.usage",
        "log",
        "task.command_result",
    ]
    task_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentCommand(BaseModel):
    """Command from Control Plane to Agent over the IPC socketpair."""

    type: Literal["execute_task", "cancel", "shutdown", "user_command"]
    task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
