"""Central control plane that ties everything together."""

import asyncio
import collections
import inspect
import json
import logging as _logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints
from uuid import uuid4

import structlog
from pydantic import ValidationError

from switchplane.app import Application
from switchplane.checkpoint import SqliteCheckpointSaver
from switchplane.config import AppConfig, load_config
from switchplane.daemon import IDLE_TIMEOUT, IdleTimer, RuntimePaths
from switchplane.logging import StreamMessageFormatter
from switchplane.persistence import Store
from switchplane.protocol import CliRequest, CliResponse, StreamEvent
from switchplane.subprocess_manager import SubprocessManager
from switchplane.task import TaskRecord, TaskStatus
from switchplane.transport import SocketServer, write_message

logger = structlog.get_logger()


class _SystemLogHandler(_logging.Handler):
    """Routes stdlib log records to system log subscribers and ring buffer.

    Uses StreamMessageFormatter so the format can be swapped without touching
    this handler.  The re-entrancy guard prevents the broadcast itself (which
    may trigger further logging) from recursing back into emit().
    """

    def __init__(self, cp: "ControlPlane"):
        super().__init__()
        self._cp = cp
        self._in_emit = False
        self.setFormatter(StreamMessageFormatter())

    def emit(self, record: _logging.LogRecord) -> None:
        if self._in_emit:
            return
        self._in_emit = True
        try:
            self._cp._broadcast_system_log(record, self.format(record))
        finally:
            self._in_emit = False


class ControlPlane:
    """Central orchestrator for the Switchplane runtime."""

    def __init__(self, paths: RuntimePaths, app: Application):
        self.paths = paths
        self.app = app
        self.store = Store(paths.db_path)
        self.checkpoint_saver: SqliteCheckpointSaver | None = None
        self.subprocess_mgr: SubprocessManager | None = None
        self.server: SocketServer | None = None
        self.config: AppConfig | None = None
        self._idle_timer: IdleTimer | None = None
        self._shutdown_event = asyncio.Event()
        self._config_watch_task: asyncio.Task | None = None
        # task_id -> set of asyncio.Queue[StreamEvent | None]
        # None in the queue is the terminal sentinel (stream.end)
        self._stream_subscribers: dict[str, set[asyncio.Queue]] = {}
        # System log streaming
        self._system_log_subscribers: set[asyncio.Queue] = set()
        self._system_log_buffer: collections.deque = collections.deque(maxlen=500)
        self._system_log_seq: int = 0

    async def start(self) -> None:
        """Initialize and start the control plane."""
        # Initialize persistence
        await self.store.initialize()

        # Recover orphaned tasks from previous runs
        orphaned_count = await self.store.recover_orphaned_tasks()
        if orphaned_count > 0:
            logger.warning("recovered_orphaned_tasks", count=orphaned_count)

        # Agent rows track live subprocesses; none survive a restart, so any
        # rows left by a previous run are stale.
        stale_agents = await self.store.clear_all_agents()
        if stale_agents > 0:
            logger.warning("cleared_stale_agents", count=stale_agents)

        # Set up checkpoint saver
        self.checkpoint_saver = SqliteCheckpointSaver(self.store.connection)
        await self.checkpoint_saver.setup()

        # Set up subprocess manager
        self.subprocess_mgr = SubprocessManager(
            self.store,
            app=self.app,
            event_callback=self._on_agent_event,
            request_handler=self.handle_request,
        )

        # Load config — uses the app's config_class so subclasses can define
        # app-specific sections without them being dropped by the base AppConfig model.
        self.config = load_config(self.paths.config_path, self.app.default_config_path, self.app.config_class)
        logger.info("config_loaded", path=str(self.paths.config_path))

        # Apply configured log level
        level = getattr(_logging, self.config.logging.level.upper(), _logging.DEBUG)
        _logging.getLogger().setLevel(level)

        # Install system log handler to broadcast CP logs to subscribers
        self._system_log_handler = _SystemLogHandler(self)
        _logging.getLogger().addHandler(self._system_log_handler)

        # Start socket server
        self.server = SocketServer(
            self.paths.sock_path,
            self.handle_request,
            stream_handler=self._stream_subscribe,
            system_stream_handler=self._subscribe_system,
        )
        await self.server.start()
        logger.info("control_plane_listening", sock=str(self.paths.sock_path))

        # Set up idle timer
        self._idle_timer = IdleTimer(IDLE_TIMEOUT, self._on_idle)
        self._idle_timer.reset()

        # Start config file watcher
        self._config_watch_task = asyncio.create_task(self._watch_config())

    async def run(self) -> None:
        """Run until shutdown."""
        await self.start()
        try:
            await self._shutdown_event.wait()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("shutting_down_control_plane")

        # Remove system log handler before tearing down
        if hasattr(self, "_system_log_handler"):
            _logging.getLogger().removeHandler(self._system_log_handler)

        if self._idle_timer:
            self._idle_timer.cancel()

        if self._config_watch_task:
            self._config_watch_task.cancel()
            try:
                await self._config_watch_task
            except asyncio.CancelledError:
                pass

        # Kill all agent processes
        if self.subprocess_mgr:
            await self.subprocess_mgr.kill_all()

        # Stop server
        if self.server:
            await self.server.stop()

        # Close persistence
        await self.store.close()

        logger.info("control_plane_shutdown_complete")

    def _on_idle(self) -> None:
        """Called when idle timeout expires."""
        # Only shut down if no active tasks or connections
        active = self.subprocess_mgr.active_count if self.subprocess_mgr else 0
        connections = self.server.connection_count if self.server else 0

        if active == 0 and connections == 0:
            logger.info("idle_timeout_shutdown")
            self._shutdown_event.set()
        else:
            logger.info("idle_timeout_reset", active_tasks=active, active_connections=connections)
            if self._idle_timer:
                self._idle_timer.reset()

    async def _cancel_children(self, parent_task_id: str) -> int:
        """Recursively cancel all non-terminal children of a task.

        Returns the number of tasks cancelled.
        """
        children = await self.store.get_child_tasks(parent_task_id)
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
        count = 0
        for child in children:
            if child.status in terminal:
                continue
            # Try to cancel the running subprocess
            await self.subprocess_mgr.cancel_task(child.task_id)
            # If the task has no running subprocess (e.g. stuck in PENDING),
            # update the status directly
            task = await self.store.get_task(child.task_id)
            if task and task.status not in terminal:
                await self.store.update_task(
                    child.task_id,
                    status=TaskStatus.CANCELLED,
                    error_json=json.dumps({"error": f"Cascade cancelled: parent {parent_task_id} terminated"}),
                )
                await self.store.add_event(
                    child.task_id,
                    "task.cancelled",
                    {"reason": f"Parent task {parent_task_id} terminated"},
                )
            count += 1
            # Recurse into this child's children
            count += await self._cancel_children(child.task_id)
        return count

    # Event types that mark a task as terminal
    _TERMINAL_EVENT_TYPES = frozenset({"task.completed", "task.failed", "task.cancelled"})

    async def _on_agent_event(self, event, event_id: int) -> None:
        """Push incoming agent events to all streaming subscribers for that task."""
        task_id = event.task_id

        # Cascade cancellation when a task fails or is cancelled
        if event.type in ("task.failed", "task.cancelled"):
            children_cancelled = await self._cancel_children(task_id)
            if children_cancelled:
                logger.info("cascade_cancelled_children", parent=task_id, count=children_cancelled, trigger=event.type)

        subscribers = self._stream_subscribers.get(task_id)
        if not subscribers:
            return

        task = await self.store.get_task(task_id)
        task_status = task.status.value if task else ""

        stream_event = StreamEvent(
            task_id=task_id,
            event_type=event.type,
            payload=event.payload,
            ts=event.ts,
            event_id=event_id,
            task_status=task_status,
        )

        for queue in list(subscribers):
            await queue.put(stream_event)

        if event.type in self._TERMINAL_EVENT_TYPES:
            for queue in list(subscribers):
                await queue.put(None)  # terminal sentinel

    async def _stream_subscribe(self, request: CliRequest, writer: asyncio.StreamWriter) -> None:
        """Handle a subscribe_task streaming connection.

        Replays stored events then pushes new ones as they arrive.  A ``None``
        in the internal queue signals the end of the stream (task finished).
        """
        task_id = request.params.get("task_id")
        after_event_id = int(request.params.get("after_event_id", 0))
        if not task_id:
            return

        queue: asyncio.Queue = asyncio.Queue()

        # Register before reading history to avoid a race between the DB read
        # and the first live event arriving from the agent.
        if task_id not in self._stream_subscribers:
            self._stream_subscribers[task_id] = set()
        self._stream_subscribers[task_id].add(queue)

        try:
            task = await self.store.get_task(task_id)
            past_events = await self.store.get_events_since(task_id, after_event_id)

            for ev_dict in past_events:
                stream_event = StreamEvent(
                    task_id=task_id,
                    event_type=ev_dict["event_type"],
                    payload=ev_dict.get("payload", {}),
                    ts=datetime.fromisoformat(ev_dict["timestamp"]),
                    event_id=ev_dict["event_id"],
                    task_status=task.status.value if task else "",
                )
                await write_message(writer, stream_event.model_dump_json().encode())

            # If the task is already terminal, close out the stream immediately.
            if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                end_event = StreamEvent(
                    task_id=task_id,
                    event_type="stream.end",
                    payload={},
                    ts=datetime.now(UTC),
                    task_status=task.status.value,
                )
                await write_message(writer, end_event.model_dump_json().encode())
                return

            # Forward live events from the queue until the terminal sentinel.
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    continue  # keepalive – just re-enter the wait

                if item is None:
                    # Task reached a terminal state; send stream.end.
                    refreshed = await self.store.get_task(task_id)
                    end_event = StreamEvent(
                        task_id=task_id,
                        event_type="stream.end",
                        payload={},
                        ts=datetime.now(UTC),
                        task_status=refreshed.status.value if refreshed else "",
                    )
                    try:
                        await write_message(writer, end_event.model_dump_json().encode())
                    except OSError:
                        pass
                    break

                try:
                    await write_message(writer, item.model_dump_json().encode())
                except OSError:
                    break  # client disconnected

        except OSError:
            pass  # client disconnected before or during history replay
        finally:
            subscribers = self._stream_subscribers.get(task_id)
            if subscribers:
                subscribers.discard(queue)
                if not subscribers:
                    del self._stream_subscribers[task_id]

    def _broadcast_system_event(self, event_type: str, payload: dict) -> None:
        """Push a structured event to all system log subscribers."""
        if not self._system_log_subscribers:
            return
        event = StreamEvent(
            task_id="_system",
            event_type=event_type,
            payload=payload,
            ts=datetime.now(UTC),
            event_id=0,
        )
        for queue in list(self._system_log_subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def _broadcast_system_log(self, record: _logging.LogRecord, message: str) -> None:
        """Push a log record to all system log subscribers and the ring buffer."""
        ts = datetime.fromtimestamp(record.created, UTC)
        logger_name = record.name
        level = record.levelname.lower()
        self._system_log_seq += 1
        self._system_log_buffer.append(
            {
                "message": message,
                "level": level,
                "logger": logger_name,
                "ts": ts.isoformat(),
                "seq": self._system_log_seq,
            }
        )

        if not self._system_log_subscribers:
            return

        event = StreamEvent(
            task_id="_system",
            event_type="system.log",
            payload={
                "message": message,
                "level": level,
                "logger": logger_name,
            },
            ts=ts,
            event_id=0,
        )
        for queue in list(self._system_log_subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def _subscribe_system(self, request: CliRequest, writer: asyncio.StreamWriter) -> None:
        """Handle a subscribe_system streaming connection.

        Pushes CP log events as they arrive until the client disconnects.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._system_log_subscribers.add(queue)

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    continue  # keepalive — re-enter the wait

                try:
                    await write_message(writer, item.model_dump_json().encode())
                except OSError:
                    break  # client disconnected
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            self._system_log_subscribers.discard(queue)

    async def _get_system_logs(self, request: CliRequest) -> CliResponse:
        """Return buffered CP log records after a given sequence number."""
        after_seq = int(request.params.get("after_seq", 0))
        records = [r for r in self._system_log_buffer if r["seq"] > after_seq]
        return CliResponse(
            id=request.id,
            ok=True,
            result={"logs": records, "latest_seq": self._system_log_seq},
        )

    async def handle_request(self, request: CliRequest) -> CliResponse:
        """Dispatch CLI requests to handlers."""
        # Reset idle timer
        if self._idle_timer:
            self._idle_timer.reset()

        if request.method == "get_events_since":
            logger.debug("handling_request", method=request.method)
        else:
            logger.info("handling_request", method=request.method)

        try:
            match request.method:
                case "submit_task":
                    return await self._submit_task(request)
                case "list_tasks":
                    return await self._list_tasks(request)
                case "get_task":
                    return await self._get_task(request)
                case "get_events_since":
                    return await self._get_events_since(request)
                case "cancel_task":
                    return await self._cancel_task(request)
                case "list_agents":
                    return await self._list_agents(request)
                case "status":
                    return await self._status(request)
                case "task_command":
                    return await self._task_command(request)
                case "notify_task":
                    return await self._notify_task(request)
                case "retry_from_checkpoint":
                    return await self._retry_from_checkpoint(request)
                case "clear_tasks":
                    return await self._clear_tasks(request)
                case "purge_tasks":
                    return await self._purge_tasks(request)
                case "get_system_logs":
                    return await self._get_system_logs(request)
                case "stop":
                    self._shutdown_event.set()
                    return CliResponse(id=request.id, ok=True, result="Shutting down")
                case _:
                    return CliResponse(id=request.id, ok=False, error=f"Unknown method: {request.method}")
        except Exception as e:
            logger.error("request_error", method=request.method, error=str(e), exc_info=True)
            return CliResponse(id=request.id, ok=False, error=str(e))

    def _reload_config(self) -> None:
        self.config = load_config(self.paths.config_path, self.app.default_config_path, self.app.config_class)
        level = getattr(_logging, self.config.logging.level.upper(), _logging.DEBUG)
        _logging.getLogger().setLevel(level)
        logger.info("config_reloaded", path=str(self.paths.config_path))

    def _config_mtime(self, path: Path | None) -> float:
        if path is None:
            return 0.0
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    async def _watch_config(self) -> None:
        user_mtime = self._config_mtime(self.paths.config_path)
        default_mtime = self._config_mtime(self.app.default_config_path)

        while True:
            await asyncio.sleep(2)
            try:
                new_user_mtime = self._config_mtime(self.paths.config_path)
                new_default_mtime = self._config_mtime(self.app.default_config_path)

                if new_user_mtime != user_mtime or new_default_mtime != default_mtime:
                    self._reload_config()
                    self._broadcast_system_event("config.reloaded", {"path": str(self.paths.config_path)})
                    user_mtime = new_user_mtime
                    default_mtime = new_default_mtime
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("config_watch_error")

    async def _submit_task(self, request: CliRequest) -> CliResponse:
        """Submit a new task for execution."""
        agent_name = request.params.get("agent_name")
        task_name = request.params.get("task_name")
        input_data = request.params.get("input", {})

        if not agent_name or not task_name:
            return CliResponse(id=request.id, ok=False, error="agent_name and task_name required")

        # Find the agent spec
        agent_spec = self.app.agents.get(agent_name)
        if not agent_spec:
            return CliResponse(id=request.id, ok=False, error=f"Agent '{agent_name}' not found")

        if task_name not in agent_spec.tasks:
            return CliResponse(id=request.id, ok=False, error=f"Task '{task_name}' not found in agent '{agent_name}'")

        # Validate task parameters if the task has a parameters model
        task_cls = agent_spec.tasks[task_name]
        params_model = task_cls.parameters_model()
        if params_model is not None:
            try:
                params_model.model_validate(input_data)
            except ValidationError as e:
                return CliResponse(id=request.id, ok=False, error=f"Invalid parameters: {e}")

        # Create task record
        parent_task_id = request.params.get("parent_task_id")
        now = datetime.now(UTC)
        task = TaskRecord(
            task_id=uuid4().hex[:12],
            agent_name=agent_name,
            task_name=task_name,
            status=TaskStatus.PENDING,
            input_json=json.dumps(input_data),
            created_at=now,
            updated_at=now,
            parent_task_id=parent_task_id,
        )
        await self.store.create_task(task)

        # Launch agent to execute task with config
        agent_id = await self.subprocess_mgr.launch_agent(agent_spec, task, self.config)

        self._broadcast_system_event(
            "task.created",
            {
                "task_id": task.task_id,
                "agent_name": task.agent_name,
                "task_name": task.task_name,
                "status": task.status.value,
                "parent_task_id": task.parent_task_id,
            },
        )

        return CliResponse(
            id=request.id,
            ok=True,
            result={"task_id": task.task_id, "agent_id": agent_id},
        )

    async def _retry_from_checkpoint(self, request: CliRequest) -> CliResponse:
        """Retry a failed or cancelled task from its last checkpoint."""
        task_id = request.params.get("task_id")
        if not task_id:
            return CliResponse(id=request.id, ok=False, error="task_id required")

        # Look up the original task
        original = await self.store.get_task(task_id)
        if not original:
            return CliResponse(id=request.id, ok=False, error=f"Task {task_id} not found")

        if original.status not in (TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.COMPLETED):
            return CliResponse(
                id=request.id,
                ok=False,
                error=f"Task {task_id} has status '{original.status}' — only failed, cancelled, or completed tasks can be retried",
            )

        # Find the agent spec
        agent_spec = self.app.agents.get(original.agent_name)
        if not agent_spec:
            return CliResponse(id=request.id, ok=False, error=f"Agent '{original.agent_name}' not found")

        # Reset task status and re-launch with the same task_id. Tasks that
        # checkpoint resume from their last state (the thread_id they chose is
        # recorded in checkpoint_threads).
        await self.store.update_task(task_id, status=TaskStatus.PENDING)
        agent_id = await self.subprocess_mgr.launch_agent(agent_spec, original, self.config)

        return CliResponse(
            id=request.id,
            ok=True,
            result={"task_id": task_id, "agent_id": agent_id},
        )

    async def _list_tasks(self, request: CliRequest) -> CliResponse:
        """List tasks with optional status filter.

        Cleared tasks are hidden unless explicitly requested via status=cleared.
        """
        status_str = request.params.get("status")
        status = TaskStatus(status_str) if status_str else None
        tasks = await self.store.list_tasks(status=status)
        if status is None:
            tasks = [t for t in tasks if t.status != TaskStatus.CLEARED]
        return CliResponse(
            id=request.id,
            ok=True,
            result=[t.model_dump(mode="json") for t in tasks],
        )

    async def _get_task(self, request: CliRequest) -> CliResponse:
        """Get a single task by ID."""
        task_id = request.params.get("task_id")
        if not task_id:
            return CliResponse(id=request.id, ok=False, error="task_id required")

        task = await self.store.get_task(task_id)
        if not task:
            return CliResponse(id=request.id, ok=False, error=f"Task {task_id} not found")

        # Also get events
        events = await self.store.get_events(task_id)

        return CliResponse(
            id=request.id,
            ok=True,
            result={"task": task.model_dump(mode="json"), "events": events},
        )

    async def _cancel_task(self, request: CliRequest) -> CliResponse:
        """Cancel a running task."""
        task_id = request.params.get("task_id")
        if not task_id:
            return CliResponse(id=request.id, ok=False, error="task_id required")

        cancelled = await self.subprocess_mgr.cancel_task(task_id)

        # Cascade cancellation to children
        children_cancelled = await self._cancel_children(task_id)
        if children_cancelled:
            logger.info("cascade_cancelled_children", parent=task_id, count=children_cancelled)

        return CliResponse(
            id=request.id, ok=True, result={"cancelled": cancelled, "children_cancelled": children_cancelled}
        )

    async def _list_agents(self, request: CliRequest) -> CliResponse:
        """List registered agents with task details."""
        agents = []
        for name, spec in self.app.agents.items():
            tasks = {}
            for task_name, task_cls in spec.tasks.items():
                task_info: dict = {
                    "description": getattr(task_cls, "description", ""),
                    "mode": getattr(task_cls, "mode", "ephemeral"),
                }
                params_model = task_cls.parameters_model()
                if params_model:
                    params = {}
                    for field_name, field_info in params_model.model_fields.items():
                        params[field_name] = {
                            "type": str(field_info.annotation) if field_info.annotation else "str",
                            "required": field_info.is_required(),
                        }
                        if not field_info.is_required() and field_info.default is not None:
                            params[field_name]["default"] = field_info.default
                        if field_info.description:
                            params[field_name]["description"] = field_info.description
                    task_info["parameters"] = params
                commands = {}
                for attr_name in dir(task_cls):
                    attr = getattr(task_cls, attr_name, None)
                    if callable(attr) and getattr(attr, "_is_command", False):
                        cmd_info: dict = {}
                        try:
                            sig = inspect.signature(attr)
                            cmd_params = {}
                            hints = get_type_hints(attr)
                            for pname, p in sig.parameters.items():
                                if pname in ("self", "ctx"):
                                    continue
                                ptype = hints.get(pname, str)
                                pentry: dict = {"type": getattr(ptype, "__name__", str(ptype))}
                                if p.default is not inspect.Parameter.empty:
                                    pentry["default"] = p.default
                                else:
                                    pentry["required"] = True
                                cmd_params[pname] = pentry
                            if cmd_params:
                                cmd_info["parameters"] = cmd_params
                        except (ValueError, TypeError):
                            pass
                        commands[attr_name] = cmd_info
                if commands:
                    task_info["commands"] = commands
                tasks[task_name] = task_info
            agents.append(
                {
                    "name": name,
                    "tasks": tasks,
                }
            )
        return CliResponse(id=request.id, ok=True, result=agents)

    async def _status(self, request: CliRequest) -> CliResponse:
        """Get runtime status."""
        active_agents = self.subprocess_mgr.active_count if self.subprocess_mgr else 0
        connections = self.server.connection_count if self.server else 0
        running_tasks = await self.store.list_tasks(status=TaskStatus.RUNNING)

        return CliResponse(
            id=request.id,
            ok=True,
            result={
                "active_agents": active_agents,
                "active_connections": connections,
                "running_tasks": len(running_tasks),
                "app": self.app.name,
            },
        )

    async def _get_events_since(self, request: CliRequest) -> CliResponse:
        """Get events for a task since a given event_id."""
        task_id = request.params.get("task_id")
        after_event_id = request.params.get("after_event_id", 0)
        if not task_id:
            return CliResponse(id=request.id, ok=False, error="task_id required")
        events = await self.store.get_events_since(task_id, after_event_id)
        task = await self.store.get_task(task_id)
        return CliResponse(
            id=request.id,
            ok=True,
            result={
                "events": events,
                "status": task.status.value if task else "unknown",
            },
        )

    async def _clear_tasks(self, request: CliRequest) -> CliResponse:
        """Soft-delete terminal tasks (mark CLEARED) so they drop out of view.

        Data is preserved until purge — only the status changes.
        """
        count = await self.store.clear_terminal_tasks()
        return CliResponse(id=request.id, ok=True, result={"cleared": count})

    async def _purge_tasks(self, request: CliRequest) -> CliResponse:
        active = self.subprocess_mgr.active_count if self.subprocess_mgr else 0
        if active > 0:
            return CliResponse(id=request.id, ok=False, error="Cannot purge while tasks are running")

        task_ids = await self.store.get_terminal_task_ids()
        await self.store.purge_terminal_tasks()

        for task_id in task_ids:
            shutil.rmtree(self.paths.data_dir / task_id, ignore_errors=True)
            (self.paths.runtime_dir / "logs" / "tasks" / f"{task_id}.log").unlink(missing_ok=True)

        return CliResponse(id=request.id, ok=True, result={"purged": len(task_ids)})

    async def _task_command(self, request: CliRequest) -> CliResponse:
        """Send a user command to a running task."""
        task_id = request.params.get("task_id")
        action = request.params.get("action")
        params = request.params.get("params", {})

        if not task_id or not action:
            return CliResponse(id=request.id, ok=False, error="task_id and action required")

        sent = await self.subprocess_mgr.send_user_command(task_id, action, params)
        if sent:
            return CliResponse(id=request.id, ok=True, result={"sent": True})
        else:
            return CliResponse(id=request.id, ok=False, error="Task not found or command send failed")

    async def _notify_task(self, request: CliRequest) -> CliResponse:
        """Send a notification to a running task."""
        task_id = request.params.get("task_id")
        payload = request.params.get("payload", {})

        if not task_id:
            return CliResponse(id=request.id, ok=False, error="task_id required")

        sent = await self.subprocess_mgr.send_notification(task_id, payload)
        if sent:
            return CliResponse(id=request.id, ok=True, result={"sent": True})
        else:
            return CliResponse(id=request.id, ok=False, error="Task not running or notification send failed")
