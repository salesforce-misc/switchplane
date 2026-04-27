"""Manages agent subprocess lifecycle.

Each agent gets a dedicated Unix socketpair for bidirectional IPC with
the control plane. The CP can send commands (cancel, shutdown) at any
time; the agent streams events back over the same socket. Both directions
use 4-byte big-endian length-prefixed JSON framing.
"""

import asyncio
import json
import socket
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from switchplane._util import deep_merge, read_frame, write_frame
from switchplane.agent import AgentRecord, AgentSpec, AgentStatus
from switchplane.config import AppConfig
from switchplane.persistence import Store
from switchplane.protocol import AgentCommand, AgentEvent
from switchplane.task import TaskRecord, TaskStatus

logger = structlog.get_logger()

# Backward-compatible aliases for tests that import these directly
_deep_merge = deep_merge
_read_message = read_frame
_write_message = write_frame


class _AgentHandle:
    """Tracks a running agent subprocess and its IPC channel."""

    __slots__ = (
        "agent_id",
        "proc",
        "reader",
        "reader_task",
        "sock",
        "stderr_task",
        "task_id",
        "writer",
    )

    def __init__(
        self,
        agent_id: str,
        task_id: str,
        proc: asyncio.subprocess.Process,
        sock: socket.socket,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        self.agent_id = agent_id
        self.task_id = task_id
        self.proc = proc
        self.sock = sock
        self.reader = reader
        self.writer = writer
        self.reader_task: asyncio.Task | None = None
        self.stderr_task: asyncio.Task | None = None


class SubprocessManager:
    """Manages agent subprocesses for task execution."""

    def __init__(
        self,
        store: Store,
        app=None,
        event_callback: Callable[[AgentEvent, int], Any] | None = None,
    ):
        self.store = store
        self._app = app
        self.event_callback = event_callback
        self._handles: dict[str, _AgentHandle] = {}  # agent_id -> handle
        self._task_to_agent: dict[str, str] = {}  # task_id -> agent_id

    async def launch_agent(self, agent_spec: AgentSpec, task: TaskRecord, config: AppConfig | None = None) -> str:
        """Launch an agent subprocess with a socketpair for IPC."""
        agent_id = uuid4().hex

        # Create a socketpair — cp_sock stays here, agent_sock is inherited by the child
        cp_sock, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        agent_fd = agent_sock.fileno()

        log_file: Path | None = None
        if self._app and hasattr(self._app, "runtime_dir"):
            log_file = self._app.runtime_dir / "logs" / "tasks" / f"{task.task_id}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "switchplane.agent_runtime",
            "--entry-point",
            agent_spec.module_path,
            "--ipc-fd",
            str(agent_fd),
        ]
        if log_file:
            cmd += ["--log-file", str(log_file)]

        stderr_dest = open(log_file, "ab") if log_file else asyncio.subprocess.PIPE  # noqa: SIM115
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                pass_fds=(agent_fd,),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr_dest,
            )
        finally:
            if log_file:
                stderr_dest.close()  # child has its own copy
            # Child has its own copy of agent_fd now; close ours
            agent_sock.close()

        # Wrap CP's end for async I/O
        reader, writer = await asyncio.open_connection(sock=cp_sock)

        handle = _AgentHandle(
            agent_id=agent_id,
            task_id=task.task_id,
            proc=proc,
            sock=cp_sock,
            reader=reader,
            writer=writer,
        )
        self._handles[agent_id] = handle
        self._task_to_agent[task.task_id] = agent_id

        # Record agent in DB
        await self.store.upsert_agent(
            AgentRecord(
                agent_id=agent_id,
                agent_name=agent_spec.agent_name,
                pid=proc.pid,
                status=AgentStatus.RUNNING,
                started_at=datetime.now(UTC),
                last_heartbeat=datetime.now(UTC),
            )
        )

        await self.store.update_task(task.task_id, status=TaskStatus.RUNNING)

        # Derive task module from agent module path
        base = agent_spec.module_path.rsplit(".", 1)[0]
        task_module = f"{base}.tasks.{task.task_name}"

        # Build per-agent config: global config with agent-specific overrides
        if config:
            agent_config = config.model_dump()
            overrides = agent_config.pop("agents", {}).get(agent_spec.agent_name, {})
            _deep_merge(agent_config, overrides)
        else:
            agent_config = {}

        # Resolve MCP server configs for this agent
        mcp_configs = []
        for server_name in agent_spec.mcp_servers:
            mcp_server = self._app.mcp_servers.get(server_name) if self._app else None
            if mcp_server:
                mcp_configs.append(mcp_server.model_dump(exclude_none=True))
            else:
                logger.warning("mcp_server_not_found", server=server_name, agent=agent_spec.agent_name)

        # Resolve db_path for agent-side checkpointing
        db_path = str(self._app.runtime_dir / "state.db") if self._app else None

        # Send the execute_task command over the socket with config
        command = AgentCommand(
            type="execute_task",
            task_id=task.task_id,
            payload={
                "task_name": task.task_name,
                "params": json.loads(task.input_json),
                "task_module": task_module,
                "config": agent_config,
                "mcp_servers": mcp_configs,
                "db_path": db_path,
                "log_level": config.logging.level if config else "debug",
            },
        )
        await _write_message(writer, command.model_dump_json().encode())

        # Start background readers
        handle.reader_task = asyncio.create_task(self._read_events(handle))
        handle.stderr_task = asyncio.create_task(self._read_stderr(handle))

        if log_file:
            logger.info("agent_launched", agent_id=agent_id, pid=proc.pid, task_id=task.task_id, log_file=str(log_file))
        else:
            logger.info("agent_launched", agent_id=agent_id, pid=proc.pid, task_id=task.task_id)
        return agent_id

    # -- Sending commands to agents ------------------------------------------

    async def send_cancel(self, task_id: str) -> bool:
        """Send a cancel command to the agent running a task."""
        agent_id = self._task_to_agent.get(task_id)
        if not agent_id:
            return False
        handle = self._handles.get(agent_id)
        if not handle:
            return False

        try:
            cmd = AgentCommand(type="cancel", task_id=task_id)
            await _write_message(handle.writer, cmd.model_dump_json().encode())
            return True
        except (ConnectionError, OSError) as e:
            logger.warning("cancel_send_failed", agent_id=agent_id, error=str(e))
            return False

    async def send_user_command(self, task_id: str, action: str, params: dict[str, Any] | None = None) -> bool:
        """Send a user command to the agent running a task."""
        agent_id = self._task_to_agent.get(task_id)
        if not agent_id:
            return False
        handle = self._handles.get(agent_id)
        if not handle:
            return False

        try:
            cmd = AgentCommand(type="user_command", task_id=task_id, payload={"action": action, "params": params or {}})
            await _write_message(handle.writer, cmd.model_dump_json().encode())
            return True
        except (ConnectionError, OSError) as e:
            logger.warning("user_command_send_failed", agent_id=agent_id, error=str(e))
            return False

    # -- Reading events from agents ------------------------------------------

    async def _read_events(self, handle: _AgentHandle) -> None:
        """Read length-prefixed events from the agent's IPC socket."""
        try:
            while True:
                try:
                    data = await _read_message(handle.reader)
                except asyncio.IncompleteReadError:
                    break

                try:
                    event = AgentEvent.model_validate_json(data)
                    event_id = await self._handle_event(event)

                    if self.event_callback:
                        if asyncio.iscoroutinefunction(self.event_callback):
                            await self.event_callback(event, event_id)
                        else:
                            self.event_callback(event, event_id)
                except Exception as e:
                    logger.error("event_processing_failed", agent_id=handle.agent_id, error=str(e))

            # Wait for process to finish
            await handle.proc.wait()

            rc = handle.proc.returncode
            task = await self.store.get_task(handle.task_id)
            if task and task.status in (TaskStatus.RUNNING, TaskStatus.INTERRUPTED):
                # Agent exited without emitting task.completed/task.failed
                logger.warning(
                    "agent_exited_task_still_running", agent_id=handle.agent_id, rc=rc, task_id=handle.task_id
                )
                await self.store.update_task(
                    handle.task_id,
                    status=TaskStatus.FAILED,
                    error_json=json.dumps({"error": f"Agent process exited unexpectedly (code {rc})"}),
                )
            elif rc and rc != 0:
                logger.warning("agent_nonzero_exit", agent_id=handle.agent_id, rc=rc)

        except Exception as e:
            logger.error("event_read_error", agent_id=handle.agent_id, error=str(e))
        finally:
            self._cleanup_handle(handle)

    async def _handle_event(self, event: AgentEvent) -> int:
        """Process an event from an agent and update persistence. Returns event_id."""
        event_id = await self.store.add_event(event.task_id, event.type, event.payload)

        match event.type:
            case "task.started":
                await self.store.update_task(event.task_id, status=TaskStatus.RUNNING)
            case "task.completed":
                result = event.payload.get("result")
                await self.store.update_task(
                    event.task_id,
                    status=TaskStatus.COMPLETED,
                    result_json=json.dumps(result) if result is not None else None,
                )
            case "task.failed":
                error = event.payload.get("error", "Unknown error")
                # Store both error and traceback if present
                error_data = {"error": error}
                if "traceback" in event.payload:
                    error_data["traceback"] = event.payload["traceback"]
                await self.store.update_task(
                    event.task_id,
                    status=TaskStatus.FAILED,
                    error_json=json.dumps(error_data),
                )
            case "task.cancelled":
                await self.store.update_task(event.task_id, status=TaskStatus.CANCELLED)
            case "task.interrupted":
                await self.store.update_task(event.task_id, status=TaskStatus.INTERRUPTED)
            case "task.resumed":
                await self.store.update_task(event.task_id, status=TaskStatus.RUNNING)
            case "task.progress":
                pass
            case "checkpoint.save":
                pass  # TODO: Forward to checkpoint saver
            case "llm.usage":
                pass  # Stored as regular event for task-level cost accounting
            case "task.command_result":
                pass  # Stored as regular event
            case "log":
                msg = event.payload.get("message", "")
                level = event.payload.get("level", "info")
                log_method = getattr(logger, level.lower(), logger.info)
                log_method("agent_log", task_id=event.task_id, message=msg)

        return event_id

    async def _read_stderr(self, handle: _AgentHandle) -> None:
        """Capture stderr from the agent subprocess for logging.

        Only active when stderr is a PIPE (no log file configured). When a
        per-task log file is used, stderr is redirected there directly and
        this coroutine exits immediately.
        """
        if handle.proc.stderr is None:
            return
        try:
            while True:
                line = await handle.proc.stderr.readline()
                if not line:
                    break
                logger.warning("agent_stderr", agent_id=handle.agent_id, line=line.decode().strip())
        except Exception:
            pass

    # -- Lifecycle ------------------------------------------------------------

    def _cleanup_handle(self, handle: _AgentHandle) -> None:
        """Remove a handle from tracking and close its socket."""
        self._handles.pop(handle.agent_id, None)
        self._task_to_agent.pop(handle.task_id, None)
        try:
            handle.writer.close()
        except Exception:
            pass
        try:
            handle.sock.close()
        except Exception:
            pass

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task by sending a cancel command over the IPC socket."""
        return await self.send_cancel(task_id)

    async def kill_all(self, timeout: float = 5.0) -> None:
        """Send cancel to all agents, then SIGTERM, then SIGKILL after timeout."""
        if not self._handles:
            return

        logger.info("shutting_down_agents", count=len(self._handles))

        # Try graceful cancel first
        for handle in list(self._handles.values()):
            try:
                cmd = AgentCommand(type="shutdown")
                await _write_message(handle.writer, cmd.model_dump_json().encode())
            except (ConnectionError, OSError):
                pass

        # Give agents a moment to finish
        procs = [h.proc for h in self._handles.values()]
        try:
            await asyncio.wait_for(
                asyncio.gather(*[p.wait() for p in procs], return_exceptions=True),
                timeout=min(timeout / 2, 2.0),
            )
        except TimeoutError:
            pass

        # SIGTERM anything still alive
        for handle in list(self._handles.values()):
            try:
                handle.proc.terminate()
            except ProcessLookupError:
                pass

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *[h.proc.wait() for h in self._handles.values()],
                    return_exceptions=True,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            for handle in list(self._handles.values()):
                try:
                    handle.proc.kill()
                except ProcessLookupError:
                    pass

        # Cancel background tasks and clean up
        for handle in list(self._handles.values()):
            if handle.reader_task:
                handle.reader_task.cancel()
            if handle.stderr_task:
                handle.stderr_task.cancel()
            self._cleanup_handle(handle)

        self._handles.clear()
        self._task_to_agent.clear()

    @property
    def active_count(self) -> int:
        return len(self._handles)
