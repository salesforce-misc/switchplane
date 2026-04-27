"""Agent-side runtime harness for Switchplane.

This module runs inside agent subprocesses and provides the execution harness
that agent code calls into. Communication with the control plane is
bidirectional over a Unix socketpair passed via --ipc-fd, using
4-byte length-prefixed JSON framing.
"""

import argparse
import asyncio
import importlib
import logging as _logging
import os
import socket
import struct
import traceback
from pathlib import Path
from typing import Any

import structlog

from switchplane._util import MAX_MESSAGE_SIZE
from switchplane.protocol import AgentCommand, AgentEvent
from switchplane.usage import LLMUsageRecord, estimate_cost_usd

_logger = structlog.get_logger()


class _IPCLogHandler(_logging.Handler):
    """Forwards stdlib log records to the control plane as 'log' AgentEvents.

    Uses StreamMessageFormatter so the format can be swapped without touching
    this handler.  Logger name comes from record.name rather than the formatter
    since Formatter.format() returns a single string.
    """

    def __init__(self, ctx: "AgentContext"):
        super().__init__()
        self._ctx = ctx
        self.addFilter(self._no_recursion)
        from switchplane.logging import StreamMessageFormatter

        self.setFormatter(StreamMessageFormatter())

    @staticmethod
    def _no_recursion(record: _logging.LogRecord) -> bool:
        return not record.name.startswith("switchplane.agent_runtime")

    def emit(self, record: _logging.LogRecord) -> None:
        try:
            self._ctx.emit(
                "log",
                {
                    "message": self.format(record),
                    "level": record.levelname.lower(),
                    "logger": record.name,
                },
            )
        except Exception:
            pass  # never let logging failures crash the agent


async def _read_message(reader: asyncio.StreamReader) -> bytes:
    """Read a length-prefixed message from the IPC socket."""
    length_bytes = await reader.readexactly(4)
    length = struct.unpack(">I", length_bytes)[0]
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message size {length} exceeds limit of {MAX_MESSAGE_SIZE}")
    return await reader.readexactly(length)


def _write_message_sync(sock: socket.socket, data: bytes) -> None:
    """Write a length-prefixed message synchronously.

    Temporarily sets the socket to blocking mode if needed, because asyncio
    puts sockets into non-blocking mode and sock.sendall() on a non-blocking
    socket can send partial data then raise BlockingIOError for large messages,
    corrupting the IPC framing.
    """
    message = struct.pack(">I", len(data)) + data
    was_blocking = sock.getblocking()
    if not was_blocking:
        sock.setblocking(True)
    try:
        sock.sendall(message)
    finally:
        if not was_blocking:
            sock.setblocking(False)


class AgentContext:
    """Context injected into agent task execution. Provides IPC helpers.

    An ``AgentContext`` is created by the agent runtime and passed to
    ``Task.run()``. It is the primary interface for task code to
    communicate with the control plane.

    Key attributes:
        task_id: Unique identifier for this task execution.
        task_name: The registered name of the task.
        config: Dict of merged app + user configuration (from TOML
            config cascade). Access agent-specific settings, API keys,
            model names, etc. via this dict.

    Logging:
        Standard library ``logging`` calls are automatically forwarded
        to the control plane as ``log`` events via an IPC log handler
        installed at subprocess startup. Use ``logging.getLogger()`` as
        normal — there is no need for a special logging method.

    Lifecycle methods:
        Use ``complete(result)`` to signal success, ``fail(error)`` to
        signal failure, and ``progress(message)`` for intermediate
        status updates. For low-level custom events, use ``emit()``.
    """

    def __init__(
        self,
        task_id: str,
        task_name: str,
        ipc_sock: socket.socket,
        config: dict[str, Any],
        db_path: str | None = None,
    ):
        self.task_id = task_id
        self.task_name = task_name
        self._sock = ipc_sock
        self.config = config
        self._cancelled = asyncio.Event()
        self._completed = False
        self._command_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._mcp: Any = None  # McpManager, set during startup if MCP servers configured
        self._db_path = db_path
        self._checkpointer: Any = None
        self._db_conn: Any = None
        self._task: Any = None

    @property
    def runtime_dir(self) -> Path:
        if self._db_path is None:
            raise RuntimeError("runtime_dir not available")
        return Path(self._db_path).parent

    @property
    def mcp(self):
        """Access MCP sessions. Returns McpManager or None if no MCP servers configured."""
        return self._mcp

    async def mcp_tools(self) -> dict[str, Any]:
        """Get all MCP tools as LangChain StructuredTool instances."""
        if self._mcp is None:
            return {}
        return {t.name: t for t in await self._mcp.langchain_tools()}

    @property
    def checkpointer(self):
        """LangGraph checkpoint saver for resumable workflows. Returns None if db_path not set."""
        return self._checkpointer

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Send an event to the control plane over the IPC socket.

        This is the low-level event primitive. Prefer the higher-level
        helpers for standard lifecycle events:

        - ``progress(message)`` — emits ``task.progress``
        - ``complete(result)`` — emits ``task.completed``
        - ``fail(error)`` — emits ``task.failed``

        Use ``emit`` directly for custom event types (e.g.
        ``"task.metric"``, ``"task.warning"``). The ``event_type`` string
        is stored as-is in the events table and forwarded to CLI/TUI
        subscribers. The ``payload`` dict is serialized as JSON; keep
        values JSON-serializable.

        Args:
            event_type: Dot-namespaced event type string.
            payload: Arbitrary JSON-serializable dict. Defaults to ``{}``.
        """
        event = AgentEvent(
            type=event_type,
            task_id=self.task_id,
            payload=payload or {},
        )
        _write_message_sync(self._sock, event.model_dump_json().encode())

    def progress(self, message: str, detail: str | list[str] | None = None, **extra) -> None:
        payload: dict[str, Any] = {"message": message, **extra}
        if detail is not None:
            payload["detail"] = detail if isinstance(detail, list) else detail.split("\n")
        self.emit("task.progress", payload)

    def complete(self, result: Any) -> None:
        self._completed = True
        self.emit("task.completed", {"result": result})

    def fail(self, error: str, traceback_str: str | None = None) -> None:
        self._completed = True
        payload = {"error": error}
        if traceback_str:
            payload["traceback"] = traceback_str
        self.emit("task.failed", payload)

    @property
    def is_cancelled(self) -> bool:
        """Whether a cancellation has been requested for this task.

        Check this in long-running loops to exit gracefully. For an
        async-friendly alternative that raises ``CancelledError``, use
        ``check_cancelled()``. For cancellation-aware sleeping, use
        ``sleep()`` which returns ``False`` when cancelled.
        """
        return self._cancelled.is_set()

    async def check_cancelled(self) -> None:
        """Raise asyncio.CancelledError if a cancel command has been received."""
        if self._cancelled.is_set():
            raise asyncio.CancelledError("Task cancelled by control plane")

    async def receive_command(self) -> dict[str, Any]:
        """Block until a command arrives from the queue."""
        return await self._command_queue.get()

    def poll_command(self) -> dict[str, Any] | None:
        """Non-blocking check for commands, returns None if empty."""
        try:
            return self._command_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def command_result(self, action: str, result: dict[str, Any]) -> None:
        """Emit a ``task.command_result`` event back to the control plane.

        Called automatically by ``Task._dispatch_command`` after a
        ``@command``-decorated handler returns. You generally do not
        need to call this directly unless you are implementing custom
        command dispatch logic outside the ``@command`` decorator
        framework.

        Args:
            action: The command action name that was executed.
            result: Dict payload to include in the result event. An
                ``{"error": ...}`` key signals failure to the CLI/TUI.
        """
        self.emit("task.command_result", {"action": action, "result": result})

    def record_llm_usage(
        self,
        *,
        model: str,
        node_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        estimated_raw_prompt_tokens: int | None = None,
        estimated_tokens_saved: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMUsageRecord:
        """Emit a structured ``llm.usage`` event for one model call."""

        total = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        cost = estimated_cost_usd
        if cost is None:
            cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)

        record = LLMUsageRecord(
            task_id=self.task_id,
            model=model,
            node_name=node_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            estimated_cost_usd=cost,
            estimated_raw_prompt_tokens=estimated_raw_prompt_tokens,
            estimated_tokens_saved=estimated_tokens_saved,
            metadata=metadata or {},
        )
        self.emit("llm.usage", record.model_dump(mode="json"))
        return record

    async def wait_for_input(self, prompt: str | None = None) -> str:
        """Block until the user sends freeform text input.

        Emits a task.interrupted event (with optional prompt), waits for
        user input via the command queue, then emits task.resumed and returns
        the text. Non-input commands are dispatched normally while waiting.

        Requires a checkpointer to be configured.
        """
        if self._checkpointer is None:
            raise RuntimeError(
                "wait_for_input requires a checkpointer — compile your graph with checkpointer=ctx.checkpointer"
            )

        payload: dict[str, Any] = {"prompt": prompt} if prompt is not None else {}
        self.emit("task.interrupted", payload)

        while True:
            cmd = await self._command_queue.get()
            if cmd["action"] == "__input__":
                self.emit("task.resumed", {})
                return cmd["params"]["text"]
            # Dispatch non-input commands to the task
            if self._task is not None:
                await self._task._dispatch_command(self, cmd)

    async def sleep(self, seconds: float) -> bool:
        """Cancellation-aware sleep.

        Returns ``True`` if the full duration elapsed, ``False`` if the
        task was cancelled during the wait.
        """
        try:
            await asyncio.wait_for(self._cancelled.wait(), timeout=seconds)
            return False
        except TimeoutError:
            return True

    async def poll_until(
        self,
        callback,
        interval: int | float = 60,
        task: Any = None,
    ):
        """Repeatedly call *callback* every *interval* seconds until it
        returns a non-``None`` value or the task is cancelled.

        *callback* may be sync or async.
        If *task* is provided, pending commands are dispatched between polls.
        Returns the callback result, or ``None`` if cancelled.
        """
        while not self.is_cancelled:
            try:
                await asyncio.wait_for(self._cancelled.wait(), timeout=interval)
                return None
            except TimeoutError:
                pass

            if task is not None:
                await task.process_commands(self)

            result = callback()
            if asyncio.iscoroutine(result):
                result = await result
            if result is not None:
                return result
        return None


async def _listen_for_commands(
    reader: asyncio.StreamReader,
    ctx: AgentContext,
    task_handle: asyncio.Task,
) -> None:
    """Listen for incoming commands from the control plane (cancel, shutdown)."""
    try:
        while True:
            data = await _read_message(reader)
            try:
                command = AgentCommand.model_validate_json(data)
            except Exception:
                _logger.warning("malformed_command", data=data[:200])
                continue

            match command.type:
                case "cancel":
                    ctx._cancelled.set()
                    task_handle.cancel()
                    return
                case "shutdown":
                    ctx._cancelled.set()
                    task_handle.cancel()
                    return
                case "user_command":
                    # Put the command payload onto the command queue
                    await ctx._command_queue.put(command.payload)
    except (asyncio.IncompleteReadError, ConnectionError, OSError):
        pass  # Socket closed — control plane is gone, task will finish or be orphaned


async def _start_checkpointer(ctx: AgentContext) -> None:
    """Initialize the checkpoint saver if db_path is available."""
    if not ctx._db_path:
        return
    try:
        import aiosqlite

        from switchplane.checkpoint import SqliteCheckpointSaver

        ctx._db_conn = await aiosqlite.connect(ctx._db_path)
        ctx._db_conn.row_factory = aiosqlite.Row
        await ctx._db_conn.execute("PRAGMA journal_mode=WAL")
        saver = SqliteCheckpointSaver(ctx._db_conn)
        await saver.setup()
        ctx._checkpointer = saver
    except Exception as e:
        _logger.warning("checkpointer_init_failed", error=str(e))


async def _stop_checkpointer(ctx: AgentContext) -> None:
    """Close the checkpoint database connection."""
    if ctx._db_conn is not None:
        try:
            await ctx._db_conn.close()
        except Exception:
            pass


async def _start_mcp(ctx: AgentContext, mcp_configs: list[dict[str, Any]]) -> None:
    """Start MCP sessions if configured.

    Raises ``RuntimeError`` if all configured servers fail to start so
    the caller can abort the task before execution begins.
    """
    if not mcp_configs:
        return

    try:
        from switchplane.app import McpServerConfig
        from switchplane.mcp import McpManager
    except ImportError:
        raise RuntimeError(
            "MCP support requires the 'mcp' package. Install with: pip install switchplane[mcp]"
        ) from None

    configs = [McpServerConfig.model_validate(c) for c in mcp_configs]
    runtime_dir = Path(ctx._db_path).parent if ctx._db_path else None
    manager = McpManager(configs, runtime_dir=runtime_dir)
    errors = await manager.start()
    for err in errors:
        _logger.error("mcp_server_start_failed", error=err)

    if len(errors) == len(configs):
        await manager.stop()
        raise RuntimeError(f"All MCP servers failed to start: {'; '.join(errors)}")

    ctx._mcp = manager


async def _stop_mcp(ctx: AgentContext) -> None:
    """Stop MCP sessions if running."""
    if ctx._mcp is not None:
        try:
            await ctx._mcp.stop()
        except Exception:
            pass


async def _run_task(ctx: AgentContext, task_module_path: str, raw_params: dict) -> None:
    """Import and execute the user task."""
    try:
        module = importlib.import_module(task_module_path)

        from switchplane.task import Task

        task_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Task) and obj is not Task:
                task_class = obj
                break

        if task_class is None:
            raise RuntimeError(f"No Task subclass found in {task_module_path}")

        task_instance = task_class()
        task_instance._ctx = ctx
        ctx._task = task_instance

        # Validate and set parameter fields
        params_model = task_class.parameters_model()
        if params_model is not None:
            validated = params_model.model_validate(raw_params)
            for field_name in params_model.model_fields:
                setattr(task_instance, field_name, getattr(validated, field_name))

        result = await task_instance.run(ctx)
        # Only auto-complete if the task returned a value AND hasn't already
        # emitted a terminal event (complete/fail) itself.
        if result is not None and not ctx._completed:
            ctx.complete(result)
    except Exception:
        # Re-raise to be caught by agent_main which will include traceback
        raise


async def agent_main(ipc_fd: int, entry_point: str) -> None:
    """Main entry point for agent subprocess.

    Opens the IPC socket from the passed fd, reads the initial execute_task
    command, runs the task, and listens for cancel/shutdown concurrently.
    """
    # Reconstruct socket from the inherited fd
    sock = socket.fromfd(ipc_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    os.close(ipc_fd)  # fromfd duped the fd

    # Wrap for async reading (commands from CP)
    try:
        reader, _writer = await asyncio.open_connection(sock=sock)
    except Exception:
        sock.close()
        raise

    # Read the initial command
    try:
        data = await _read_message(reader)
    except (asyncio.IncompleteReadError, ConnectionError):
        sock.close()
        return

    command = AgentCommand.model_validate_json(data)
    if command.type != "execute_task":
        sock.close()
        return

    task_id = command.task_id
    task_name = command.payload.get("task_name", "")
    params = command.payload.get("params", {})
    task_module_path = command.payload.get("task_module", "")
    config = command.payload.get("config", {})
    mcp_configs = command.payload.get("mcp_servers", [])
    db_path = command.payload.get("db_path")

    ctx = AgentContext(
        task_id=task_id,
        task_name=task_name,
        ipc_sock=sock,
        config=config,
        db_path=db_path,
    )

    # Configure log level and install IPC handler so structlog output
    # from agent code is forwarded as "log" events to the control plane.
    log_level = command.payload.get("log_level", "debug")
    _logging.getLogger().setLevel(getattr(_logging, log_level.upper(), _logging.DEBUG))
    _ipc_handler = _IPCLogHandler(ctx)
    _logging.getLogger().addHandler(_ipc_handler)

    # Start checkpointer and MCP sessions before task execution
    await _start_checkpointer(ctx)
    try:
        await _start_mcp(ctx, mcp_configs)
    except RuntimeError as e:
        ctx.fail(str(e))
        await _stop_checkpointer(ctx)
        _writer.close()
        sock.close()
        return

    ctx.emit("task.started", {})

    # Run the task and the command listener concurrently
    task_handle = asyncio.create_task(_run_task(ctx, task_module_path, params))
    listener_handle = asyncio.create_task(_listen_for_commands(reader, ctx, task_handle))

    try:
        await task_handle
    except asyncio.CancelledError:
        ctx.emit("task.cancelled", {})
    except BaseException as e:
        ctx.fail(f"{type(e).__name__}: {e}", traceback.format_exc())
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        _logging.getLogger().removeHandler(_ipc_handler)
        await _stop_mcp(ctx)
        await _stop_checkpointer(ctx)
        listener_handle.cancel()
        try:
            await listener_handle
        except asyncio.CancelledError:
            pass
        _writer.close()
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switchplane agent runtime")
    parser.add_argument("--entry-point", required=True)
    parser.add_argument("--ipc-fd", required=True, type=int)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    from pathlib import Path as _Path

    from switchplane import logging

    logging.configure(log_file=_Path(args.log_file) if args.log_file else None)

    asyncio.run(agent_main(args.ipc_fd, args.entry_point))
