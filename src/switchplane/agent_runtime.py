"""Agent-side runtime harness for Switchplane.

This module runs inside agent subprocesses and provides the execution harness
that agent code calls into. Communication with the control plane is
bidirectional over a Unix socketpair passed via --ipc-fd, using
4-byte length-prefixed JSON framing.

Remote debugging
----------------
Set ``SWITCHPLANE_DEBUG_AGENT`` in the environment that launches the daemon to
have each agent subprocess host a debugpy listener on 127.0.0.1 and block
until a debugger attaches. Values: ``1``/``true`` -> port 5678, any integer ->
that port (``auto`` for an ephemeral port; the bound port is logged to the task
log). Requires ``pip install switchplane[debug]``.

Example VS Code ``launch.json`` entry::

    {"name": "Attach: switchplane agent", "type": "debugpy", "request": "attach",
     "connect": {"host": "127.0.0.1", "port": 5678}, "justMyCode": false}
"""

import argparse
import asyncio
import importlib
import json
import logging as _logging
import os
import socket
import struct
import traceback
from pathlib import Path
from typing import Any

import structlog

from switchplane._util import MAX_MESSAGE_SIZE
from switchplane.protocol import AgentCommand, AgentEvent, AgentRequest, AgentResponse

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
            payload = {
                "message": self.format(record),
                "level": record.levelname.lower(),
                "logger": record.name,
            }
            # Carry the formatted traceback when the caller logged with
            # `exc_info=...`. Without this, exception logs cross the IPC
            # boundary as one-line messages and the file:line that
            # actually identifies the failure is dropped — the per-task
            # JSON log file keeps it (different formatter) but the
            # events DB and TUI never see it.
            if record.exc_info:
                payload["traceback"] = "".join(traceback.format_exception(*record.exc_info))
            self._ctx.emit("log", payload)
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


# Paired module-level state shared between `_maybe_attach_debugger`
# (which binds the listener and is called pre-`AgentContext`) and
# `_wait_for_debugger` (which blocks on a client and is called once
# `ctx.progress(...)` is available). Both are populated atomically
# at the end of a successful `_maybe_attach_debugger` call; either
# both are non-None or `_debug_bound` is None and `_wait_for_debugger`
# is a no-op. Holding the imported `debugpy` module here too means
# `_wait_for_debugger` doesn't re-import — when `_debug_bound` is
# set, `debugpy` is guaranteed importable in this process.
#
# This pairing is intentional for the one-shot subprocess startup
# flow (the only caller). Don't invoke `_wait_for_debugger`
# standalone, and don't call `_maybe_attach_debugger` twice — the
# second invocation would silently overwrite the first listener's
# port without unbinding it.
_debug_bound: tuple[str, int] | None = None
_debugpy: Any = None


def _maybe_attach_debugger() -> None:
    """Optionally start a debugpy listener (non-blocking).

    Controlled by the ``SWITCHPLANE_DEBUG_AGENT`` env var. ``1``/``true`` means
    the debugpy default port (5678); ``auto`` requests an ephemeral free port;
    any other integer is used as-is. Binds 127.0.0.1 only. No-op when the env
    var is unset or empty. Logs a warning and returns instead of crashing if
    debugpy is not installed.

    The blocking ``wait_for_client()`` call is deferred to
    ``_wait_for_debugger`` so an ``AgentContext`` is available for user-facing
    progress events. The two functions are tied together via the
    ``_debug_bound`` / ``_debugpy`` module-state pair documented above
    — call them in order, exactly once each.

    Note: debugpy permits arbitrary code execution by any client that can reach
    the listening port; binding 127.0.0.1 keeps this loopback-only.
    """
    global _debug_bound, _debugpy
    raw = os.environ.get("SWITCHPLANE_DEBUG_AGENT", "").strip()
    if not raw:
        return
    if raw.lower() in ("1", "true"):
        port = 5678
    elif raw.lower() == "auto":
        port = 0
    else:
        try:
            port = int(raw)
        except ValueError:
            _logger.warning("debug_attach_invalid_value", value=raw)
            return
        if port == 0:
            _logger.warning(
                "debug_attach_invalid_value",
                value=raw,
                message="use SWITCHPLANE_DEBUG_AGENT=auto for an ephemeral port",
            )
            return
    try:
        import debugpy
    except ImportError:
        _logger.warning(
            "debug_attach_debugpy_missing",
            message="SWITCHPLANE_DEBUG_AGENT is set but debugpy is not installed; install switchplane[debug]",
        )
        return
    try:
        bound_host, bound_port = debugpy.listen(("127.0.0.1", port))
    except OSError as e:
        _logger.warning("debug_attach_listen_failed", port=port, error=str(e))
        return
    _logger.info("debug_attach_listening", host=bound_host, port=bound_port)
    _debug_bound = (bound_host, bound_port)
    _debugpy = debugpy


def _wait_for_debugger(ctx: "AgentContext") -> None:
    """Block until a debugpy client attaches, emitting user-facing progress.

    No-op if ``_maybe_attach_debugger`` did not successfully start a listener.
    Reuses the `debugpy` module reference cached by `_maybe_attach_debugger`
    — see the `_debug_bound` / `_debugpy` pairing comment above.
    """
    if _debug_bound is None:
        return
    host, port = _debug_bound
    ctx.progress(f"execution paused: debugpy listening on {host}:{port}, waiting for client to attach")
    _debugpy.wait_for_client()
    _logger.info("debug_attach_client_connected", port=port)


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
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._notification_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    @property
    def runtime_dir(self) -> Path:
        if self._db_path is None:
            raise RuntimeError("runtime_dir not available")
        return Path(self._db_path).parent

    @property
    def mcp(self):
        """Access MCP sessions. Returns McpManager or empty dict if no MCP servers configured."""
        if self._mcp is None:
            return {}
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

    def stream_token(self, text: str) -> None:
        """Emit a streaming text chunk (ephemeral, not persisted)."""
        self.emit("stream.chunk", {"text": text})

    def stream_flush(self, text: str) -> None:
        """End a streaming sequence. The text is the final complete output that replaces accumulated chunks."""
        self.emit("stream.flush", {"text": text})

    def tool_invoke(self, name: str, summary: str = "") -> None:
        """Emit a tool invocation event."""
        payload: dict[str, Any] = {"name": name}
        if summary:
            payload["summary"] = summary
        self.emit("tool.invoke", payload)

    def tool_result(self, name: str, summary: str = "") -> None:
        """Emit a tool result event."""
        payload: dict[str, Any] = {"name": name}
        if summary:
            payload["summary"] = summary
        self.emit("tool.result", payload)

    def file_edit(self, path: str, diff: str) -> None:
        """Emit a file edit event with a unified diff."""
        self.emit("file.edit", {"path": path, "diff": diff})

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

    async def _send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a request to the control plane and wait for the response.

        Raises RuntimeError if the control plane returns an error.
        """
        request = AgentRequest(method=method, params=params or {})
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_requests[request.request_id] = future
        try:
            _write_message_sync(self._sock, request.model_dump_json().encode())
            response = await future
        finally:
            self._pending_requests.pop(request.request_id, None)
        if not response.ok:
            raise RuntimeError(f"Control plane error: {response.error}")
        return response.result

    async def submit_task(
        self,
        agent_name: str,
        task_name: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Submit a new task for execution, returns the task_id.

        The submitted task is linked to this task as its parent.
        """
        result = await self._send_request(
            "submit_task",
            {
                "agent_name": agent_name,
                "task_name": task_name,
                "input": params or {},
                "parent_task_id": self.task_id,
            },
        )
        return result["task_id"]

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Get the current state of a task.

        Returns a dict with 'task' (task record) and 'events' keys.
        """
        return await self._send_request("get_task", {"task_id": task_id})

    async def wait_for_task(self, task_id: str, poll_interval: float = 10) -> dict[str, Any]:
        """Poll until a task reaches a terminal state, then return its record.

        Terminal states: completed, failed, cancelled.
        Returns the task record dict.
        """
        terminal = {"completed", "failed", "cancelled"}
        while True:
            result = await self._send_request("get_task", {"task_id": task_id})
            task_record = result["task"]
            if task_record["status"] in terminal:
                return task_record
            if not await self.sleep(poll_interval):
                raise asyncio.CancelledError("Parent task cancelled while waiting for child")

    async def wait_for_tasks(self, task_ids: list[str], poll_interval: float = 10) -> list[dict[str, Any]]:
        """Poll until all tasks reach terminal states, then return their records.

        Returns task records in the same order as the input task_ids.
        """
        results: dict[str, dict[str, Any]] = {}
        terminal = {"completed", "failed", "cancelled"}
        remaining = set(task_ids)
        while remaining:
            for tid in list(remaining):
                result = await self._send_request("get_task", {"task_id": tid})
                task_record = result["task"]
                if task_record["status"] in terminal:
                    results[tid] = task_record
                    remaining.discard(tid)
            if remaining and not await self.sleep(poll_interval):
                raise asyncio.CancelledError("Parent task cancelled while waiting for children")
        return [results[tid] for tid in task_ids]

    async def notify_task(self, task_id: str, payload: dict[str, Any] | None = None) -> None:
        """Send a notification to another running task.

        The notification is delivered to the target task's notification
        queue.  If the target is blocked in ``wait_for_notification()``,
        it wakes up immediately.

        Args:
            task_id: The task to notify.
            payload: Arbitrary JSON-serializable dict delivered as the notification body.
        """
        await self._send_request(
            "notify_task",
            {
                "task_id": task_id,
                "payload": payload or {},
            },
        )

    async def wait_for_notification(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Block until a notification arrives or the task is cancelled.

        Returns the notification payload dict, or ``None`` if cancelled
        or timed out.
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(
                    self._wait_notification_or_cancel(),
                    timeout=timeout,
                )
            return await self._wait_notification_or_cancel()
        except (TimeoutError, asyncio.CancelledError):
            return None

    async def _wait_notification_or_cancel(self) -> dict[str, Any] | None:
        """Wait for either a notification or cancellation."""
        cancel_task = asyncio.create_task(self._cancelled.wait())
        notify_task = asyncio.create_task(self._notification_queue.get())
        done, pending = await asyncio.wait(
            {cancel_task, notify_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        if notify_task in done:
            return notify_task.result()
        return None

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

            # Check if this is a response to a pending request
            raw = json.loads(data)
            if raw.get("kind") == "response":
                response = AgentResponse.model_validate(raw)
                future = ctx._pending_requests.get(response.request_id)
                if future and not future.done():
                    future.set_result(response)
                continue

            try:
                command = AgentCommand.model_validate(raw)
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
                case "notify":
                    await ctx._notification_queue.put(command.payload)
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
        await ctx._db_conn.execute("PRAGMA busy_timeout=5000")
        saver = SqliteCheckpointSaver(ctx._db_conn, task_id=ctx.task_id)
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

    if errors:
        await manager.stop()
        failed_names = []
        for err in errors:
            for c in configs:
                if c.name in err:
                    failed_names.append(c.name)
                    break
        names = ", ".join(failed_names) if failed_names else "unknown"
        raise RuntimeError(
            f"MCP server(s) failed to start ({names}): {'; '.join(errors)}. "
            f"Check authentication with 'auth login' for the failed server(s)."
        )

    ctx._mcp = manager


async def _stop_mcp(ctx: AgentContext) -> None:
    """Stop MCP sessions if running."""
    if ctx._mcp is not None:
        try:
            await ctx._mcp.stop()
        except Exception:
            pass


def _import_task_class(task_module_path: str) -> type:
    """Import the module at *task_module_path* and return the Task subclass.

    Only considers classes defined in the module itself (not imported base
    classes), matching the discovery logic in ``discovery.py``.
    """
    module = importlib.import_module(task_module_path)

    from switchplane.task import Task

    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Task)
            and obj is not Task
            and getattr(obj, "__module__", None) == module.__name__
        ):
            return obj

    raise RuntimeError(f"No Task subclass found in {task_module_path}")


def _instantiate_task(ctx: AgentContext, task_class: type, raw_params: dict):
    """Construct a Task subclass and bind validated parameter fields.

    Split from `_run_task` so callers (specifically `agent_main`) can
    reach into the populated instance for `startup_info()` *before*
    `task.started` is emitted. Without this, the emit would have to
    fire either pre-binding (no access to params) or post-`run()`
    (defeats the lifecycle signal).
    """
    task_instance = task_class()
    task_instance._ctx = ctx
    ctx._task = task_instance

    params_model = task_class.parameters_model()
    if params_model is not None:
        validated = params_model.model_validate(raw_params)
        for field_name in params_model.model_fields:
            setattr(task_instance, field_name, getattr(validated, field_name))
    return task_instance


async def _run_task(ctx: AgentContext, task_instance) -> None:
    """Execute a previously-instantiated Task."""
    try:
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

    # Import the task class early so we can inspect mcp_servers before
    # starting any MCP connections.
    try:
        task_class = _import_task_class(task_module_path)
    except Exception as e:
        ctx.fail(f"{type(e).__name__}: {e}", traceback.format_exc())
        _writer.close()
        sock.close()
        return

    # Only start MCP servers the task actually needs
    if task_class.mcp_servers:
        available = {c["name"]: c for c in mcp_configs}
        mcp_configs = [available[n] for n in task_class.mcp_servers if n in available]
        missing = [n for n in task_class.mcp_servers if n not in available]
        for name in missing:
            _logger.warning("task_mcp_server_not_available", server=name, task=task_class.name)
    else:
        mcp_configs = []

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

    # Instantiate and bind parameters BEFORE emitting `task.started`,
    # so `startup_info()` can surface task-specific metadata (resolved
    # model, input identifiers, etc.) on the lifecycle event for
    # post-hoc inspection. A failure here surfaces as `task.failed` —
    # nothing else has run yet.
    try:
        task_instance = _instantiate_task(ctx, task_class, params)
    except Exception as e:
        ctx.fail(f"{type(e).__name__}: {e}", traceback.format_exc())
        await _stop_mcp(ctx)
        await _stop_checkpointer(ctx)
        _writer.close()
        sock.close()
        return

    try:
        startup_info = task_instance.startup_info() or {}
    except Exception:
        # `startup_info` is best-effort metadata; never block startup.
        startup_info = {}
    ctx.emit("task.started", startup_info)

    _wait_for_debugger(ctx)

    # Run the task and the command listener concurrently
    task_handle = asyncio.create_task(_run_task(ctx, task_instance))
    listener_handle = asyncio.create_task(_listen_for_commands(reader, ctx, task_handle))

    try:
        await task_handle
    except asyncio.CancelledError:
        # `_cancelled` is set only by `_listen_for_commands` on receipt
        # of a `cancel`/`shutdown` from the control plane. A
        # CancelledError reaching here without that flag came from
        # *inside* the task body — a leaked `wait_for` cancel, a
        # cancelled child task awaited up the stack, an aborted
        # `gather`, or similar. Surfacing those as a bare
        # `task.cancelled` looks identical to an operator cancel and
        # erases every clue about what actually went wrong; treat them
        # as failures with the captured traceback.
        if ctx._cancelled.is_set():
            ctx.emit("task.cancelled", {})
        else:
            ctx.fail(
                "CancelledError raised internally (no cancel command received)",
                traceback.format_exc(),
            )
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

    _maybe_attach_debugger()

    asyncio.run(agent_main(args.ipc_fd, args.entry_point))
