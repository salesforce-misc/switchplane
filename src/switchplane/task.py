"""Task definitions and execution framework."""

import asyncio
import inspect
from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from switchplane.agent_runtime import AgentContext


class TaskStatus(StrEnum):
    """Status of a task execution."""

    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRecord(BaseModel):
    """Record of a task execution."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    task_id: str
    agent_name: str
    task_name: str
    status: TaskStatus = TaskStatus.PENDING
    input_json: str = "{}"  # Serialized Pydantic model JSON
    result_json: str | None = None
    error_json: str | None = None
    created_at: datetime
    updated_at: datetime
    workflow_identity_json: str | None = None
    checkpoint_metadata_json: str | None = None
    parent_task_id: str | None = None


def command(fn):
    """Mark a method as a task command handler.

    Methods decorated with ``@command`` become dispatchable actions on a
    running task. The control plane (or CLI ``task <id> <action>`` command)
    sends a ``user_command`` message whose ``action`` field matches the
    method name. Parameters are passed as ``--key value`` pairs from the
    CLI and automatically coerced to the types declared in the handler's
    signature via a generated Pydantic model.

    The handler receives the current ``AgentContext`` as its first argument
    (after ``self``) plus any declared keyword parameters. It may be sync
    or async. If it returns a dict, that dict is emitted as a
    ``task.command_result`` event; otherwise an empty result is emitted.

    Example::

        class WeatherWatch(Task):
            name = "watch"
            mode = "long_running"

            @command
            def set_location(self, ctx: AgentContext, lat: float, lon: float):
                self.lat = lat
                self.lon = lon
                return {"status": "location updated"}
    """
    fn._is_command = True
    return fn


class Task(ABC):
    """Base class for task implementations.

    Subclass this to define a unit of work that runs inside an agent
    subprocess. Each task must set ``name`` and ``description`` as class
    attributes and implement the async ``run`` method.

    Class attributes:
        name: Short identifier used in CLI commands (e.g. ``"hello"``).
        description: Human-readable summary shown in ``agent list`` output.
        mode: ``"ephemeral"`` (default) for one-shot tasks, or
            ``"long_running"`` for tasks that loop until cancelled.

    Parameters are declared as class-level annotations with ``Field()``
    defaults. They are validated by a generated Pydantic model before
    ``run()`` is called and set as instance attributes::

        class Greet(Task):
            name = "hello"
            description = "Say hello"
            recipient: str = Field(description="Who to greet")

            async def run(self, ctx: AgentContext) -> None:
                ctx.complete({"message": f"Hello, {self.recipient}!"})

    The ``run`` method receives an ``AgentContext`` for IPC with the
    control plane. Signal completion with ``ctx.complete(result)``, or
    failure with ``ctx.fail(error)``. If ``run`` returns a non-None
    value without calling ``complete``/``fail``, auto-completion is
    triggered with that value.

    For long-running tasks, decorate methods with ``@command`` to accept
    runtime commands from the CLI or TUI while the task is executing.
    Use ``process_commands`` or ``dispatch_command`` to handle them
    inside your run loop.
    """

    name: str = ""
    description: str = ""
    mode: Literal["ephemeral", "long_running"] = "ephemeral"
    mcp_servers: ClassVar[list[str]] = []
    _ctx: "AgentContext | None" = None

    @classmethod
    def parameters_model(cls) -> type[BaseModel] | None:
        """Build a Pydantic model from Field-annotated attributes.

        Walks the MRO so that subclasses inherit parameter fields from
        parent Task classes (e.g. ReviewerTask → QualityTask).
        """
        fields = {}
        base_attrs = set(Task.__annotations__)
        # Walk MRO in reverse so subclass annotations override parents
        for klass in reversed(cls.__mro__):
            for attr_name, annotation in getattr(klass, "__annotations__", {}).items():
                if attr_name in base_attrs:
                    continue
                default = getattr(cls, attr_name, ...)
                if isinstance(default, FieldInfo):
                    fields[attr_name] = (annotation, default)
        if not fields:
            return None
        return create_model(f"_{cls.__name__}Params", **fields)

    async def _dispatch_command(self, ctx: "AgentContext", cmd: dict) -> None:
        """Dispatch a single command dict to the matching @command handler."""
        action = cmd.get("action")
        if action == "__input__":
            return  # consumed by ctx.wait_for_input(), not dispatched
        raw_params = cmd.get("params", {})
        for attr_name in dir(type(self)):
            attr = getattr(type(self), attr_name, None)
            if callable(attr) and getattr(attr, "_is_command", False) and attr_name == action:
                handler = getattr(self, attr_name)
                params = _coerce_command_params(attr, raw_params)
                result = handler(ctx, **params)
                if asyncio.iscoroutine(result):
                    result = await result
                ctx.command_result(action, result if isinstance(result, dict) else {})
                return
        ctx.command_result(action or "unknown", {"error": f"Unknown command: {action}"})

    async def process_commands(self, ctx: "AgentContext") -> None:
        """Dispatch all pending commands to @command-decorated methods (non-blocking)."""
        while cmd := ctx.poll_command():
            await self._dispatch_command(ctx, cmd)

    async def dispatch_command(self, ctx: "AgentContext") -> None:
        """Block until one command arrives, then dispatch to the matching @command handler."""
        cmd = await ctx.receive_command()
        await self._dispatch_command(ctx, cmd)

    def open(self, path: str, mode: str = "rb"):
        if self._ctx is None:
            raise RuntimeError("Task.open() requires an active AgentContext")
        data_dir = (Path(self._ctx.runtime_dir) / "data" / self._ctx.task_id).resolve()
        full_path = (data_dir / path).resolve()
        if not full_path.is_relative_to(data_dir):
            raise PermissionError(f"Path '{path}' escapes the task data directory")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path.open(mode)

    @abstractmethod
    async def run(self, ctx: "AgentContext") -> None: ...


def _coerce_command_params(handler, raw_params: dict) -> dict:
    """Coerce string params to the types declared in the handler's signature."""
    if not hasattr(handler, "_command_model"):
        handler._command_model = _build_command_model(handler)
    model = handler._command_model
    if model is None:
        return raw_params
    validated = model.model_validate(raw_params)
    return {k: getattr(validated, k) for k in validated.model_fields}


def _build_command_model(handler) -> type[BaseModel] | None:
    """Build a Pydantic model from a command handler's type hints."""
    hints = get_type_hints(handler)
    sig = inspect.signature(handler)
    fields = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "ctx"):
            continue
        annotation = hints.get(param_name, str)
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[param_name] = (annotation, default)
    if not fields:
        return None
    return create_model(f"_{handler.__name__}_cmd", **fields)
