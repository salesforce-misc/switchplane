"""Switchplane - Local runtime harness for agent-based task execution."""

__version__ = "0.1.0"

from pydantic import Field

from switchplane import fmt
from switchplane.app import Application
from switchplane.shell import Shell
from switchplane.task import Task, command

__all__ = ["Application", "Field", "Shell", "Task", "command", "fmt"]
