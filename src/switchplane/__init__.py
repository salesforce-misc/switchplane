"""Switchplane - Local runtime harness for agent-based task execution."""

__version__ = "0.1.0"

from pydantic import Field

from switchplane import fmt
from switchplane.agent import AgentSpec
from switchplane.agent_runtime import AgentContext
from switchplane.app import Application, McpServerConfig
from switchplane.shell import Shell
from switchplane.task import Task, TaskStatus, command

__all__ = [
    "AgentContext",
    "AgentSpec",
    "Application",
    "Field",
    "McpServerConfig",
    "Shell",
    "Task",
    "TaskStatus",
    "command",
    "fmt",
]
