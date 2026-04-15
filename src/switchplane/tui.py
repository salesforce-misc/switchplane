"""Switchplane TUI: prompt_toolkit-based interactive session.

Layout:
  ┌────────────────────────────────────────────────────────┐
  │ [0] system  [1] weather/watch ●  [2] joke ✓           │  TabBar
  ├────────────────────────────────────────────────────────┤
  │  [14:23:01] Task started                               │
  │  [14:23:35] Temp: 11°C                                 │  EventPane
  │                                                        │
  ├────────────────────────────────────────────────────────┤
  │ weather/watch [running] abc123  [Tab] switch ...       │  StatusBar
  │ [weather/watch] > _                                    │  InputBar
  └────────────────────────────────────────────────────────┘

Input model:
  - :cmd [args]  → daemon command (:run, :task, :runtime, :agent, :help)
  - /cmd [--key value ...]  → task command dispatched to focused task
  - Plain input  → freeform text sent to focused task (when task is interrupted/waiting for input)
"""

import asyncio
import json
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style

from switchplane import fmt
from switchplane.protocol import CliRequest, CliResponse, StreamEvent
from switchplane.transport import ControlPlaneClient

# ---------------------------------------------------------------------------
# Style class name constants
# ---------------------------------------------------------------------------

_S_ERROR = "class:event.error"
_S_INFO = "class:event.info"
_S_DIM = "class:event.dim"
_S_PROGRESS = "class:event.progress"
_S_SUCCESS = "class:event.success"
_S_WARN = "class:event.warn"
_S_LOG = "class:event.log"
_S_SYSTEM = "class:event.system"
_S_RESULT = "class:event.result"
_S_TS = "class:event.ts"

_STYLE_MAP: dict[str, str] = {
    fmt.TS: _S_TS,
    fmt.INFO: _S_INFO,
    fmt.DIM: _S_DIM,
    fmt.PROGRESS: _S_PROGRESS,
    fmt.SUCCESS: _S_SUCCESS,
    fmt.WARN: _S_WARN,
    fmt.ERROR: _S_ERROR,
    fmt.LOG: _S_LOG,
}

# ---------------------------------------------------------------------------
# Status icons (style class, character)
# ---------------------------------------------------------------------------

_STATUS: dict[str, tuple[str, str]] = {
    "running": ("class:status.running", "●"),
    "interrupted": ("class:status.interrupted", "⏸"),
    "completed": ("class:status.completed", "✓"),
    "failed": ("class:status.failed", "✗"),
    "cancelled": ("class:status.cancelled", "×"),
    "pending": ("class:status.pending", "○"),
}

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_HEARTBEAT_INTERVAL = 60  # seconds — must be well under daemon's IDLE_TIMEOUT (300s)
_SYSTEM_TAB_ID = "_system"
_DEFAULT_MAX_BUFFER_LINES = 10_000
_TS_COL_WIDTH = 11  # "[HH:MM:SS] " = 11 characters

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EventBuffer:
    """Per-task event log buffer."""

    task_id: str
    agent_name: str
    task_name: str
    status: str = "pending"
    last_event_id: int = 0
    lines: list[tuple[StyleAndTextTuples, StyleAndTextTuples]] = field(default_factory=list)
    auto_scroll: bool = True
    vertical_scroll: int = 0  # saved scroll position (physical rows) for tab restoration


# ---------------------------------------------------------------------------
# TUISession
# ---------------------------------------------------------------------------


class TUISession:
    """Manages TUI state: event buffers, background streams, and input dispatch."""

    def __init__(self, sock_path: Path, max_buffer_lines: int = _DEFAULT_MAX_BUFFER_LINES) -> None:
        self.sock_path = sock_path
        self.max_buffer_lines = max_buffer_lines
        self.buffers: dict[str, EventBuffer] = {}
        self.task_order: list[str] = []  # ordered task IDs for tab bar (excludes _system)
        self.focused_task_id: str | None = _SYSTEM_TAB_ID
        self.streams: dict[str, asyncio.Task] = {}
        self._app: Application | None = None
        self._heartbeat: asyncio.Task | None = None
        self._system_stream: asyncio.Task | None = None
        self._task_window: Window | None = None

        # Create the system tab buffer — always present at logical slot 0
        self.buffers[_SYSTEM_TAB_ID] = EventBuffer(
            task_id=_SYSTEM_TAB_ID,
            agent_name="",
            task_name="system",
        )

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    async def _request(self, method: str, params: dict | None = None) -> CliResponse:
        def _send() -> CliResponse:
            with ControlPlaneClient(self.sock_path) as c:
                return c.send(CliRequest(method=method, params=params or {}))

        try:
            return await asyncio.to_thread(_send)
        except OSError as exc:
            return CliResponse(id="error", ok=False, error=f"Daemon unreachable: {exc}")

    async def _heartbeat_loop(self) -> None:
        """Periodically ping the daemon to reset its idle timer.

        Without this, the daemon exits after IDLE_TIMEOUT seconds of inactivity
        even if the TUI session is still open.
        """
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            await self._request("status")

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def add_task(
        self,
        task_id: str,
        agent_name: str,
        task_name: str,
        status: str = "pending",
    ) -> None:
        if task_id not in self.buffers:
            self.buffers[task_id] = EventBuffer(
                task_id=task_id,
                agent_name=agent_name,
                task_name=task_name,
                status=status,
            )
            self.task_order.append(task_id)

    # ------------------------------------------------------------------
    # Focus / navigation
    # ------------------------------------------------------------------

    def focus_slot(self, slot: int) -> None:
        if slot == 0:
            self.focused_task_id = _SYSTEM_TAB_ID
            self._refresh()
            return
        idx = slot - 1
        if 0 <= idx < len(self.task_order):
            self.focused_task_id = self.task_order[idx]
            self._refresh()

    def _all_tab_ids(self) -> list[str]:
        """Return the full ordered list of tab IDs: system + tasks."""
        return [_SYSTEM_TAB_ID, *self.task_order]

    def focus_next(self) -> None:
        all_ids = self._all_tab_ids()
        if not all_ids:
            return
        if self.focused_task_id is None or self.focused_task_id not in all_ids:
            self.focused_task_id = all_ids[0]
        else:
            idx = (all_ids.index(self.focused_task_id) + 1) % len(all_ids)
            self.focused_task_id = all_ids[idx]
        self._refresh()

    def focus_prev(self) -> None:
        all_ids = self._all_tab_ids()
        if not all_ids:
            return
        if self.focused_task_id is None or self.focused_task_id not in all_ids:
            self.focused_task_id = all_ids[-1]
        else:
            idx = (all_ids.index(self.focused_task_id) - 1) % len(all_ids)
            self.focused_task_id = all_ids[idx]
        self._refresh()

    def detach_focused_task(self) -> None:
        """Remove the focused task from the TUI view without cancelling it.

        The task continues running in the daemon and can be re-attached with
        ``:task follow <id>`` or via ``:task list`` then ``:run``.
        """
        tid = self.focused_task_id
        if tid is None or tid == _SYSTEM_TAB_ID or tid not in self.buffers:
            return

        # Move focus to a neighbour before removing
        if tid in self.task_order:
            idx = self.task_order.index(tid)
            self.task_order.remove(tid)
            if self.task_order:
                self.focused_task_id = self.task_order[min(idx, len(self.task_order) - 1)]
            else:
                self.focused_task_id = _SYSTEM_TAB_ID

        del self.buffers[tid]

        # Cancel the background stream task if one is running
        stream = self.streams.pop(tid, None)
        if stream:
            stream.cancel()

        self._refresh()

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    def scroll_up(self, n: int = 1) -> None:
        buf = self._focused_buf()
        if not buf or not self._task_window:
            return
        # On first scroll-up from auto-scroll, seed from current max position.
        if buf.auto_scroll:
            max_phys = getattr(self._task_window, "_max_phys", 0)
            buf.vertical_scroll = max_phys
        buf.auto_scroll = False
        buf.vertical_scroll = max(0, buf.vertical_scroll - n)
        self._refresh()

    def scroll_down(self, n: int = 1) -> None:
        buf = self._focused_buf()
        if not buf or not self._task_window:
            return
        max_phys = getattr(self._task_window, "_max_phys", 0)
        if buf.vertical_scroll + n >= max_phys:
            buf.auto_scroll = True
        else:
            buf.vertical_scroll += n
        self._refresh()

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def _append_line(self, task_id: str, prefix: StyleAndTextTuples, content: StyleAndTextTuples) -> None:
        buf = self.buffers.get(task_id)
        if buf is None:
            return
        buf.lines.append((prefix, content))
        if len(buf.lines) > self.max_buffer_lines:
            trim = len(buf.lines) - self.max_buffer_lines
            del buf.lines[:trim]
            if not buf.auto_scroll:
                buf.vertical_scroll = max(0, buf.vertical_scroll - trim)
        self._refresh()

    def _append_text(self, task_id: str, style: str, text: str) -> None:
        self._append_line(task_id, [], [(style, text)])

    def _system_message(self, msg: str) -> None:
        """Append a message to the system tab's event buffer."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._append_line(_SYSTEM_TAB_ID, [(_S_TS, f"[{ts}] ")], [(_S_SYSTEM, msg)])

    def _system_messages(self, msgs: list[str]) -> None:
        """Append multiple lines to the system tab, timestamping only the first."""
        if not msgs:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self._append_line(_SYSTEM_TAB_ID, [(_S_TS, f"[{ts}] ")], [(_S_SYSTEM, msgs[0])])
        for msg in msgs[1:]:
            self._append_line(_SYSTEM_TAB_ID, [], [(_S_SYSTEM, msg)])

    def _refresh(self) -> None:
        if self._app is not None:
            self._app.invalidate()

    # ------------------------------------------------------------------
    # Background streaming
    # ------------------------------------------------------------------

    def start_stream(self, task_id: str) -> None:
        """Open a persistent streaming connection to the daemon for *task_id*."""
        if task_id in self.streams:
            return
        self.streams[task_id] = asyncio.create_task(self._stream_loop(task_id))

    async def _stream_loop(self, task_id: str) -> None:
        """Subscribe to the control plane's event push stream for a task.

        Replays historical events then receives new ones in real time.  Exits
        cleanly when the server sends ``stream.end`` or the connection closes.
        """
        buf = self.buffers.get(task_id)
        if buf is None:
            return

        try:
            reader, writer = await asyncio.open_unix_connection(str(self.sock_path))
        except OSError as exc:
            self._append_text(task_id, _S_ERROR, f"  [connect error: {exc}]")
            return

        try:
            # Send subscribe_task request
            req = CliRequest(
                method="subscribe_task",
                params={"task_id": task_id, "after_event_id": buf.last_event_id},
            )
            req_bytes = req.model_dump_json().encode()
            writer.write(struct.pack(">I", len(req_bytes)) + req_bytes)
            await writer.drain()

            # Read the ack (CliResponse)
            try:
                length_bytes = await reader.readexactly(4)
            except asyncio.IncompleteReadError:
                self._append_text(task_id, _S_ERROR, "  [stream: disconnected before ack]")
                return
            length = struct.unpack(">I", length_bytes)[0]
            ack_bytes = await reader.readexactly(length)
            ack = CliResponse.model_validate_json(ack_bytes)
            if not ack.ok:
                self._append_text(task_id, _S_ERROR, f"  [subscribe failed: {ack.error}]")
                return

            # Read pushed StreamEvent frames until stream.end or connection close
            while True:
                try:
                    length_bytes = await reader.readexactly(4)
                    length = struct.unpack(">I", length_bytes)[0]
                    data = await reader.readexactly(length)
                except asyncio.IncompleteReadError:
                    break  # daemon closed the connection

                event = StreamEvent.model_validate_json(data)

                if event.event_type == "stream.end":
                    if event.task_status:
                        buf.status = event.task_status
                    self._refresh()
                    await self._fetch_terminal_result(task_id, buf.status)
                    break

                # Update local tracking
                if event.event_id:
                    buf.last_event_id = event.event_id
                if event.task_status:
                    buf.status = event.task_status

                # Delegate to the existing renderer via a thin adapter dict
                self._render_event(
                    task_id,
                    {
                        "timestamp": event.ts.isoformat(),
                        "event_type": event.event_type,
                        "payload": event.payload,
                    },
                )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._append_text(task_id, _S_ERROR, f"  [stream error: {exc}]")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            self.streams.pop(task_id, None)

    async def _fetch_terminal_result(self, task_id: str, status: str) -> None:
        check = await self._request("get_task", {"task_id": task_id})
        if not check.ok:
            return
        t = check.result["task"]
        if status == "completed" and t.get("result_json"):
            for line in fmt.format_result(t["result_json"]):
                self._append_line(task_id, [], [(_S_RESULT, f"  {line}")])
        elif status == "failed" and t.get("error_json"):
            try:
                err = json.loads(t["error_json"])
                msg = err.get("error", str(err)) if isinstance(err, dict) else t["error_json"]
                err_parts = f"  Error: {msg}".split("\n")
                self._append_line(task_id, [], [(_S_ERROR, err_parts[0])])
                for cont in err_parts[1:]:
                    self._append_line(task_id, [], [(_S_ERROR, f"    {cont}")])
                if isinstance(err, dict) and "traceback" in err:
                    for line in err["traceback"].splitlines():
                        self._append_line(task_id, [], [(_S_ERROR, f"    {line}")])
            except (json.JSONDecodeError, TypeError):
                fallback_parts = f"  Error: {t['error_json']}".split("\n")
                self._append_line(task_id, [], [(_S_ERROR, fallback_parts[0])])
                for cont in fallback_parts[1:]:
                    self._append_line(task_id, [], [(_S_ERROR, f"    {cont}")])

    async def _system_stream_loop(self) -> None:
        """Subscribe to the control plane's system log push stream.

        Receives CP-level structlog output and renders it to the system tab.
        """
        try:
            reader, writer = await asyncio.open_unix_connection(str(self.sock_path))
        except OSError as exc:
            self._append_text(_SYSTEM_TAB_ID, _S_ERROR, f"  [system stream connect error: {exc}]")
            return

        try:
            req = CliRequest(
                method="subscribe_system",
                params={},
            )
            req_bytes = req.model_dump_json().encode()
            writer.write(struct.pack(">I", len(req_bytes)) + req_bytes)
            await writer.drain()

            # Read the ack
            try:
                length_bytes = await reader.readexactly(4)
            except asyncio.IncompleteReadError:
                self._append_text(_SYSTEM_TAB_ID, _S_ERROR, "  [system stream: disconnected before ack]")
                return
            length = struct.unpack(">I", length_bytes)[0]
            ack_bytes = await reader.readexactly(length)
            ack = CliResponse.model_validate_json(ack_bytes)
            if not ack.ok:
                self._append_text(_SYSTEM_TAB_ID, _S_ERROR, f"  [system subscribe failed: {ack.error}]")
                return

            # Read pushed StreamEvent frames
            while True:
                try:
                    length_bytes = await reader.readexactly(4)
                    length = struct.unpack(">I", length_bytes)[0]
                    data = await reader.readexactly(length)
                except asyncio.IncompleteReadError:
                    break

                event = StreamEvent.model_validate_json(data)
                self._render_event(
                    _SYSTEM_TAB_ID,
                    {
                        "timestamp": event.ts.isoformat(),
                        "event_type": event.event_type,
                        "payload": event.payload,
                    },
                )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._append_text(_SYSTEM_TAB_ID, _S_ERROR, f"  [system stream error: {exc}]")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _render_event(self, task_id: str, ev: dict) -> None:
        for line in fmt.render_event(ev):
            styled = [(_STYLE_MAP.get(s, _S_DIM), t) for s, t in line.segments]
            # Split timestamp prefix from content
            if styled and styled[0][0] == _S_TS:
                prefix = [styled[0]]
                content = styled[1:]
            else:
                prefix = []
                content = styled
            self._append_line(task_id, prefix, content)

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------

    async def handle_input(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if text.startswith(":"):
            await self._handle_daemon_command(text[1:].strip())
        elif text.startswith("/"):
            await self._handle_task_command(text[1:].strip())
        else:
            # Freeform text → send to focused task if it's interrupted (waiting for input)
            if self.focused_task_id and self.focused_task_id != _SYSTEM_TAB_ID:
                buf = self.buffers.get(self.focused_task_id)
                if buf and buf.status == "interrupted":
                    resp = await self._request(
                        "task_command",
                        {
                            "task_id": self.focused_task_id,
                            "action": "__input__",
                            "params": {"text": text},
                        },
                    )
                    if resp.ok:
                        self._append_text(self.focused_task_id, _S_DIM, f"  > {text}")
                    else:
                        self._append_text(self.focused_task_id, _S_ERROR, f"  > Error: {resp.error}")
                else:
                    self._append_text(
                        self.focused_task_id, _S_DIM,
                        "  Task is not waiting for input. Use /command for task commands.",
                    )
            else:
                self._system_message("Use /command for task commands, :command for daemon commands.")

    async def _handle_task_command(self, text: str) -> None:
        if not self.focused_task_id or self.focused_task_id == _SYSTEM_TAB_ID:
            self._system_message("No task focused. Use :run <agent> <task> to start one.")
            return

        if text.lower() in ("help", "?"):
            await self._cmd_task_help()
            return

        try:
            action, params = _parse_kv_args(text.split())
        except (IndexError, ValueError) as exc:
            self._append_text(self.focused_task_id, _S_ERROR, f"  Invalid command: {exc}")
            return

        resp = await self._request(
            "task_command",
            {
                "task_id": self.focused_task_id,
                "action": action,
                "params": params,
            },
        )
        if resp.ok:
            self._append_text(self.focused_task_id, _S_INFO, f"  > {action} sent")
        else:
            self._append_text(self.focused_task_id, _S_ERROR, f"  > Error: {resp.error}")

    async def _handle_daemon_command(self, text: str) -> None:
        parts = text.split()
        if not parts:
            return
        group = parts[0].lower()
        rest = parts[1:]

        # Auto-focus system tab for non-:run commands
        auto_focus_system = group not in ("run",)

        if group in ("help", "?"):
            if auto_focus_system:
                self.focused_task_id = _SYSTEM_TAB_ID
            self._cmd_help()
        elif group == "run":
            await self._cmd_run(rest)
        elif group == "task":
            if auto_focus_system:
                self.focused_task_id = _SYSTEM_TAB_ID
            await self._dispatch_task_group(rest)
        elif group == "runtime":
            if auto_focus_system:
                self.focused_task_id = _SYSTEM_TAB_ID
            await self._dispatch_runtime_group(rest)
        elif group == "agent":
            if auto_focus_system:
                self.focused_task_id = _SYSTEM_TAB_ID
            await self._dispatch_agent_group(rest)
        else:
            if auto_focus_system:
                self.focused_task_id = _SYSTEM_TAB_ID
            self._system_message(f"Unknown command: :{group}  (try :help)")

    async def _dispatch_task_group(self, args: list[str]) -> None:
        if not args:
            self._system_message("Usage: :task <follow|cancel|clear|purge|list|show|retry> [args]")
            return
        sub = args[0].lower()
        rest = args[1:]
        dispatch: dict[str, any] = {
            "follow": self._cmd_task_follow,
            "cancel": self._cmd_cancel,
            "clear": self._cmd_clear,
            "purge": self._cmd_task_purge,
            "list": self._cmd_list,
            "show": self._cmd_task_show,
            "retry": self._cmd_task_retry,
        }
        handler = dispatch.get(sub)
        if handler:
            await handler(rest)
        else:
            self._system_message(f"Unknown task command: {sub}  (try :help)")

    async def _dispatch_runtime_group(self, args: list[str]) -> None:
        if not args:
            self._system_message("Usage: :runtime <status>")
            return
        sub = args[0].lower()
        if sub == "status":
            await self._cmd_status(args[1:])
        else:
            self._system_message(f"Unknown runtime command: {sub}  (try :help)")

    async def _dispatch_agent_group(self, args: list[str]) -> None:
        if not args:
            self._system_message("Usage: :agent <list>")
            return
        sub = args[0].lower()
        if sub == "list":
            await self._cmd_agent_list(args[1:])
        else:
            self._system_message(f"Unknown agent command: {sub}  (try :help)")

    # ------------------------------------------------------------------
    # Daemon command handlers
    # ------------------------------------------------------------------

    async def _cmd_run(self, args: list[str]) -> None:
        if len(args) < 2:
            self._system_message("Usage: :run <agent> <task> [--key value ...]")
            return
        agent_name, task_name = args[0], args[1]
        try:
            _, params = _parse_kv_args(args[2:], start_index=0)
        except ValueError as exc:
            self._system_message(f"Invalid args: {exc}")
            return

        resp = await self._request(
            "submit_task",
            {
                "agent_name": agent_name,
                "task_name": task_name,
                "input": params,
            },
        )
        if not resp.ok:
            self._system_message(f"Error: {resp.error}")
            return

        task_id = resp.result["task_id"]
        self.add_task(task_id, agent_name, task_name, status="pending")
        self.focused_task_id = task_id
        self._append_text(task_id, _S_INFO, f"  Task submitted: {task_id[:8]}…")
        self.start_stream(task_id)
        self._refresh()

    async def _cmd_task_follow(self, args: list[str]) -> None:
        if not args:
            self._system_message("Usage: :task follow <task_id>")
            return
        task_id = args[0]
        if task_id in self.buffers:
            self.focused_task_id = task_id
            self._system_message(f"Already following {task_id}.")
            self._refresh()
            return
        resp = await self._request("get_task", {"task_id": task_id})
        if not resp.ok:
            self._system_message(f"Error: {resp.error}")
            return
        t = resp.result["task"]
        self.add_task(task_id, t["agent_name"], t["task_name"], status=t["status"])
        self.focused_task_id = task_id
        self.start_stream(task_id)
        self._system_message(f"Following {task_id} [{t['status']}].")
        self._refresh()

    async def _cmd_task_show(self, args: list[str]) -> None:
        if not args:
            self._system_message("Usage: :task show <task_id>")
            return
        task_id = args[0]
        resp = await self._request("get_task", {"task_id": task_id})
        if not resp.ok:
            self._system_message(f"Error: {resp.error}")
            return
        t = resp.result["task"]
        lines = [
            f"  Task ID:  {t['task_id']}",
            f"  Agent:    {t['agent_name']}",
            f"  Task:     {t['task_name']}",
            f"  Status:   {t['status']}",
            f"  Created:  {t['created_at']}",
            f"  Updated:  {t['updated_at']}",
        ]
        if t.get("input_json"):
            lines.append(f"  Input:    {t['input_json']}")
        if t.get("result_json"):
            for line in fmt.format_result(t["result_json"]):
                lines.append(f"  {line}")
        if t.get("error_json"):
            lines.append(f"  Error:    {t['error_json']}")
        self._system_messages(lines)

    async def _cmd_task_retry(self, args: list[str]) -> None:
        if not args:
            self._system_message("Usage: :task retry <task_id>")
            return
        task_id = args[0]
        resp = await self._request("retry_from_checkpoint", {"task_id": task_id})
        if not resp.ok:
            self._system_message(f"Error: {resp.error}")
            return
        if task_id in self.buffers:
            self.buffers[task_id].status = "pending"
        else:
            meta = await self._request("get_task", {"task_id": task_id})
            if meta.ok:
                t = meta.result["task"]
                self.add_task(task_id, t["agent_name"], t["task_name"], status="pending")
        self.focused_task_id = task_id
        self.start_stream(task_id)
        self._system_message(f"Retried task {task_id}.")
        self._refresh()

    async def _cmd_agent_list(self, _args: list[str] | None = None) -> None:
        resp = await self._request("list_agents")
        if not resp.ok:
            self._system_message(f"Error: {resp.error}")
            return
        agents = resp.result
        if not agents:
            self._system_message("No agents registered.")
            return
        lines: list[str] = []
        for ag in agents:
            lines.append(f"  {ag['name']}")
            tasks = ag.get("tasks", {})
            if not tasks:
                lines.append("    (no tasks)")
                continue
            for task_name, info in tasks.items():
                mode_tag = f" [{info['mode']}]" if info.get("mode") == "long_running" else ""
                desc = f" - {info['description']}" if info.get("description") else ""
                lines.append(f"    {task_name}{mode_tag}{desc}")
                params = info.get("parameters", {})
                if params:
                    for pname, pinfo in params.items():
                        req = "required" if pinfo.get("required") else f"default: {pinfo.get('default')}"
                        pdesc = f"  {pinfo['description']}" if pinfo.get("description") else ""
                        lines.append(f"      --{pname.replace('_', '-')}  ({req}){pdesc}")
                commands = info.get("commands", {})
                if commands:
                    lines.append(f"      commands: {', '.join(commands)}")
        self._system_messages(lines)

    async def _cmd_cancel(self, args: list[str]) -> None:
        task_id = args[0] if args else self.focused_task_id
        if not task_id or task_id == _SYSTEM_TAB_ID:
            self._system_message("No task to cancel. Specify a task ID or focus a running task.")
            return
        resp = await self._request("cancel_task", {"task_id": task_id})
        if resp.ok:
            msg = "Task cancelled." if resp.result.get("cancelled") else "Task not running."
            self._system_message(msg)
        else:
            self._system_message(f"Error: {resp.error}")

    def _remove_terminal_tasks(self) -> None:
        """Remove all terminal-status tasks from the local TUI view."""
        to_remove = [
            tid for tid, buf in self.buffers.items() if tid != _SYSTEM_TAB_ID and buf.status in _TERMINAL_STATUSES
        ]
        for tid in to_remove:
            self.task_order.remove(tid)
            del self.buffers[tid]
            stream = self.streams.pop(tid, None)
            if stream:
                stream.cancel()
        if self.focused_task_id not in self.buffers:
            self.focused_task_id = next(iter(self.task_order), _SYSTEM_TAB_ID)

    async def _cmd_clear(self, _args: list[str] | None = None) -> None:
        """Clear terminal-status tasks from daemon and remove them from the TUI."""
        resp = await self._request("clear_tasks")
        if not resp.ok:
            self._system_message(f"Error: {resp.error}")
            return
        count = resp.result.get("deleted", 0)
        self._remove_terminal_tasks()
        self._system_message(f"Cleared {count} task(s).")
        self._refresh()

    async def _cmd_task_purge(self, args: list[str] | None = None) -> None:
        """Purge all terminal tasks: database records, data files, and logs."""
        if not args or "--yes" not in args:
            self._system_messages([
                "This will delete all completed/failed/cancelled tasks and their data.",
                "Run :task purge --yes to confirm.",
            ])
            return
        resp = await self._request("purge_tasks")
        if not resp.ok:
            self._system_message(f"Error: {resp.error}")
            return
        count = resp.result.get("purged", 0)
        self._remove_terminal_tasks()
        self._system_message(f"Purged {count} task(s).")
        self._refresh()

    async def _cmd_list(self, args: list[str] | None = None) -> None:
        params = {}
        if args and len(args) >= 2 and args[0] == "--status":
            params["status"] = args[1]
        resp = await self._request("list_tasks", params)
        if not resp.ok:
            self._system_message(f"Error: {resp.error}")
            return
        tasks = resp.result
        if not tasks:
            self._system_message("No tasks found.")
            return
        self._system_messages([
            f"  {t['task_id']}  {t['agent_name']}/{t['task_name']}  {t['status']}"
            for t in tasks
        ])

    async def _cmd_status(self, _args: list[str] | None = None) -> None:
        resp = await self._request("status")
        if not resp.ok:
            self._system_message(f"Error: {resp.error}")
            return
        s = resp.result
        self._system_message(
            f"  running — agents: {s['active_agents']}  tasks: {s['running_tasks']}  conns: {s['active_connections']}"
        )

    async def _cmd_task_help(self) -> None:
        """Show help for the focused task: description and available /commands."""
        tid = self.focused_task_id
        buf = self.buffers.get(tid)
        if not buf:
            return

        # Look up task metadata from list_agents
        resp = await self._request("list_agents")
        if not resp.ok:
            self._append_text(tid, _S_ERROR, f"  Error: {resp.error}")
            return

        task_info = None
        for ag in resp.result:
            if ag["name"] == buf.agent_name:
                task_info = ag.get("tasks", {}).get(buf.task_name)
                break

        if not task_info:
            self._append_text(tid, _S_DIM, f"  {buf.agent_name}/{buf.task_name} — no metadata available")
            return

        # Task description
        desc = task_info.get("description", "")
        self._append_text(tid, _S_INFO, f"  {buf.agent_name}/{buf.task_name}" + (f" — {desc}" if desc else ""))

        # Commands
        commands = task_info.get("commands", {})
        if not commands:
            self._append_text(tid, _S_DIM, "  No commands available for this task.")
            return

        self._append_text(tid, _S_INFO, "  Commands:")
        for cmd_name, cmd_info in commands.items():
            params = cmd_info.get("parameters", {})
            if params:
                flags = " ".join(f"--{p.replace('_', '-')}" for p in params)
                self._append_text(tid, _S_INFO, f"    /{cmd_name} {flags}")
                for pname, pinfo in params.items():
                    ptype = pinfo.get("type", "str")
                    if pinfo.get("required"):
                        req = "required"
                    elif "default" in pinfo:
                        req = f"default: {pinfo['default']}"
                    else:
                        req = "optional"
                    self._append_text(tid, _S_DIM, f"      --{pname.replace('_', '-')}  ({ptype}, {req})")
            else:
                self._append_text(tid, _S_INFO, f"    /{cmd_name}")

    def _cmd_help(self) -> None:
        self._system_messages([
            "  Daemon commands (prefix with :):",
            "    :run <agent> <task> [--key value ...]  — start a task",
            "    :task follow <task_id>                 — follow an existing task",
            "    :task cancel [<task_id>]               — cancel focused or specified task",
            "    :task clear                            — delete completed/failed/cancelled tasks",
            "    :task purge --yes                      — purge terminal tasks, data files, and logs",
            "    :task list [--status <s>]               — list all tasks (optionally filter by status)",
            "    :task show <task_id>                   — show task details",
            "    :task retry <task_id>                  — retry a failed/cancelled task from checkpoint",
            "    :runtime status                        — show daemon status",
            "    :agent list                            — list agents and their tasks",
            "    :help                                  — show this help",
            "  Task commands (prefix with /):",
            "    /<command> [--key value ...]            — send command to focused task",
            "    /help                                  — show focused task's available commands",
            "  Plain text is reserved for future interactive LLM tasks.",
            "  Keys: Tab/Shift+Tab switch tab · 0 system · 1-9 task",
            "        PgUp/Dn scroll · Ctrl+X cancel · Ctrl+D detach · Ctrl+C quit",
        ])

    # ------------------------------------------------------------------
    # FormattedText providers (called by prompt_toolkit on each render)
    # ------------------------------------------------------------------

    def get_tab_bar_text(self) -> StyleAndTextTuples:
        result: StyleAndTextTuples = []

        # System tab [0]
        is_sys_focused = self.focused_task_id == _SYSTEM_TAB_ID
        sys_style = "class:tab.focused" if is_sys_focused else "class:tab.inactive"
        result.extend(
            [
                (sys_style, "  [0] system"),
                ("", "  "),
            ]
        )

        # Task tabs [1], [2], ...
        for i, tid in enumerate(self.task_order, 1):
            buf = self.buffers.get(tid)
            if buf is None:
                continue
            is_focused = tid == self.focused_task_id
            slot_style = "class:tab.focused" if is_focused else "class:tab.inactive"
            label = f"[{i}] {buf.agent_name}/{buf.task_name}" if buf.agent_name else f"[{i}] {buf.task_name}"
            icon_style, icon_char = _STATUS.get(buf.status, ("class:status.pending", "○"))
            result.extend(
                [
                    (slot_style, f"  {label} "),
                    (icon_style, icon_char),
                    ("", "  "),
                ]
            )
        return result

    def get_event_pane_text(self) -> StyleAndTextTuples:
        buf = self._focused_buf()
        if buf is None:
            return [(_S_DIM, "  No tab selected.\n")]

        if buf.task_id == _SYSTEM_TAB_ID and not buf.lines:
            return [(_S_DIM, "  :help for commands\n")]

        result: StyleAndTextTuples = []
        for _prefix, content in buf.lines:
            result.extend(content)
            result.append(("", "\n"))
        return result

    def _get_event_line_prefix(self, line_number: int, wrap_count: int) -> StyleAndTextTuples:
        """Return the timestamp prefix for a given line in the event pane.

        Called by prompt_toolkit's ``get_line_prefix`` on each visible line.
        ``line_number`` is the zero-based index within the full content (maps
        directly to ``buf.lines``); ``wrap_count`` indicates the soft-wrap
        continuation index (0 for the first physical row of a logical line).
        """
        buf = self._focused_buf()
        if buf is None:
            return [(_S_DIM, " " * _TS_COL_WIDTH)]

        if line_number < 0 or line_number >= len(buf.lines):
            return [(_S_DIM, " " * _TS_COL_WIDTH)]

        prefix, _content = buf.lines[line_number]

        if wrap_count > 0 or not prefix:
            # Continuation wrap or no timestamp — emit blank padding
            return [(_S_DIM, " " * _TS_COL_WIDTH)]

        # Build the prefix text, right-padded to fixed width
        prefix_text = "".join(t for _, t in prefix)
        result: StyleAndTextTuples = []
        for style, text in prefix:
            result.append((style, text))
        pad_len = _TS_COL_WIDTH - len(prefix_text)
        if pad_len > 0:
            result.append((_S_DIM, " " * pad_len))
        return result

    def get_status_bar_text(self) -> StyleAndTextTuples:
        buf = self._focused_buf()
        if buf and buf.task_id == _SYSTEM_TAB_ID:
            left = " system  "
        elif buf:
            label = f"{buf.agent_name}/{buf.task_name}" if buf.agent_name else buf.task_name
            left = f" {label} [{buf.status}] {buf.task_id}  "
        else:
            left = " switchplane  "
        hints = "[Tab] switch  [PgUp/Dn] scroll  [Ctrl+X] cancel  [Ctrl+D] detach  [Ctrl+C] quit"
        return [
            ("class:status.bar.label", left),
            ("class:status.bar.hint", hints),
        ]

    def get_prompt_text(self) -> StyleAndTextTuples:
        buf = self._focused_buf()
        if buf and buf.task_id == _SYSTEM_TAB_ID:
            label = "[system]"
        elif buf and buf.agent_name:
            label = f"[{buf.agent_name}/{buf.task_name}]"
        elif buf:
            label = f"[{buf.task_name}]"
        else:
            label = "[switchplane]"
        return [("class:prompt", f" {label} "), ("class:prompt.arrow", "> ")]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _focused_buf(self) -> EventBuffer | None:
        if self.focused_task_id and self.focused_task_id in self.buffers:
            return self.buffers[self.focused_task_id]
        return None


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------


def build_tui_app(session: TUISession) -> Application:
    """Construct the prompt_toolkit Application for a TUISession."""

    # Key bindings
    kb = KeyBindings()

    input_buf = Buffer(name="input", history=InMemoryHistory(), multiline=False)

    @Condition
    def input_is_empty() -> bool:
        return input_buf.text == ""

    @kb.add("tab")
    def _next_task(event) -> None:
        session.focus_next()

    @kb.add("s-tab")
    def _prev_task(event) -> None:
        session.focus_prev()

    for _slot in range(0, 10):
        # Capture slot in closure
        def _make_jump(s: int):
            @kb.add(str(s), filter=input_is_empty)
            def _jump(event) -> None:
                session.focus_slot(s)

        _make_jump(_slot)

    @kb.add("pageup")
    def _scroll_up(event) -> None:
        try:
            rows = max(1, event.app.output.get_size().rows - 3)
        except Exception:
            rows = 20
        session.scroll_up(rows)

    @kb.add("pagedown")
    def _scroll_down(event) -> None:
        try:
            rows = max(1, event.app.output.get_size().rows - 3)
        except Exception:
            rows = 20
        session.scroll_down(rows)


    @kb.add("c-x")
    def _cancel(event) -> None:
        if session.focused_task_id == _SYSTEM_TAB_ID:
            return
        event.app.create_background_task(session._cmd_cancel([]))

    @kb.add("c-d")
    def _detach(event) -> None:
        if session.focused_task_id == _SYSTEM_TAB_ID:
            return
        session.detach_focused_task()

    @kb.add("c-c")
    def _quit(event) -> None:
        event.app.exit()

    @kb.add("enter")
    def _submit(event) -> None:
        text = input_buf.text
        input_buf.reset(append_to_history=True)
        if text.strip():
            event.app.create_background_task(session.handle_input(text))

    # FormattedTextControl subclass that always reports preferred_width=0 so that
    # Dimension weights are the sole authority on column widths — content never
    # influences the layout calculation.  Also handles mouse scroll events so
    # that the wheel scrolls the event pane regardless of which window has focus.
    class _LayoutFixedControl(FormattedTextControl):
        def preferred_width(self, max_available_width: int) -> int:
            return 0

        def mouse_handler(self, mouse_event: MouseEvent):
            if mouse_event.event_type == MouseEventType.SCROLL_UP:
                session.scroll_up(3)
            elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                session.scroll_down(3)
            else:
                return NotImplemented

    class _ScrollableWindow(Window):
        """Window subclass that manages scrolling via a physical-row offset.

        prompt_toolkit's built-in scroll logic is cursor-driven, which doesn't
        support per-physical-row scrolling for non-focusable controls.  This
        subclass overrides ``_scroll`` to convert our physical-row offset
        (stored on each ``EventBuffer``) into the ``(vertical_scroll,
        vertical_scroll_2)`` pair that ``_copy_body`` expects.
        """

        def _scroll(self, ui_content, width: int, height: int) -> None:
            buf = session._focused_buf()
            if buf is None:
                self.vertical_scroll = 0
                self.vertical_scroll_2 = 0
                return

            line_count = ui_content.line_count

            # Compute total physical rows and per-line heights.
            line_heights: list[int] = []
            total_rows = 0
            for i in range(line_count):
                h = ui_content.get_height_for_line(i, width, self.get_line_prefix)
                line_heights.append(h)
                total_rows += h

            max_phys = max(0, total_rows - height)

            if buf.auto_scroll:
                phys = max_phys
            else:
                phys = min(buf.vertical_scroll, max_phys)
                buf.vertical_scroll = phys

            # Store total/max so scroll_up/scroll_down can use them.
            self._total_rows = total_rows
            self._max_phys = max_phys

            # Decompose physical row offset into (logical line, row-within-line).
            accumulated = 0
            for i, h in enumerate(line_heights):
                if accumulated + h > phys:
                    self.vertical_scroll = i
                    self.vertical_scroll_2 = phys - accumulated
                    return
                accumulated += h

            # Past end — show last line.
            self.vertical_scroll = max(0, line_count - 1)
            self.vertical_scroll_2 = 0

    # Layout windows
    tab_window = Window(
        content=FormattedTextControl(session.get_tab_bar_text),
        height=1,
        style="class:tab.bar",
    )
    task_window = _ScrollableWindow(
        content=_LayoutFixedControl(session.get_event_pane_text, focusable=False, show_cursor=False),
        wrap_lines=True,
        get_line_prefix=session._get_event_line_prefix,
    )
    session._task_window = task_window
    status_window = Window(
        content=FormattedTextControl(session.get_status_bar_text),
        height=1,
        style="class:status.bar",
    )
    input_window = Window(
        content=BufferControl(buffer=input_buf),
        get_line_prefix=lambda _lineno, _wc: session.get_prompt_text(),
        height=1,
    )

    layout = Layout(
        HSplit(
            [
                tab_window,
                task_window,
                status_window,
                input_window,
            ]
        ),
        focused_element=input_window,
    )

    app_style = Style.from_dict(
        {
            # Tab bar
            "tab.bar": "bg:#1a1a2e",
            "tab.focused": "bold #00ff88 bg:#1a1a2e",
            "tab.inactive": "#666688 bg:#1a1a2e",
            "tab.empty": "#555577 bg:#1a1a2e",
            # Status icons
            "status.running": "bold #00ff88",
            "status.interrupted": "bold #ffaa00",
            "status.completed": "#00ff88",
            "status.failed": "bold #ff5555",
            "status.cancelled": "#ffaa00",
            "status.pending": "#666688",
            # Status bar
            "status.bar": "bg:#2d2d4e",
            "status.bar.label": "bold #ccccff bg:#2d2d4e",
            "status.bar.hint": "#888899 bg:#2d2d4e",
            # Event pane
            "event.ts": "#555577",
            "event.info": "#aaaacc",
            "event.progress": "",
            "event.success": "#00ff88",
            "event.error": "bold #ff5555",
            "event.warn": "#ffaa00",
            "event.log": "#55aacc",
            "event.system": "italic #8888cc",
            "event.result": "#00ff88",
            "event.dim": "#444466",
            # Input bar
            "prompt": "bold #aaaaff",
            "prompt.arrow": "#666688",
        }
    )

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=app_style,
        full_screen=True,
        mouse_support=True,
    )
    session._app = app
    return app


async def run_tui(
    sock_path: Path,
    initial_tasks: list[tuple[str, str, str, str]] | None = None,
    max_buffer_lines: int = _DEFAULT_MAX_BUFFER_LINES,
) -> None:
    """Run the TUI session.

    Args:
        sock_path: Path to the daemon Unix socket.
        initial_tasks: Optional list of (task_id, agent_name, task_name, status) tuples
            to pre-populate the session with. Pass an empty list to auto-discover
            running tasks from the daemon.
        max_buffer_lines: Maximum lines retained per tab before oldest are trimmed.
    """
    session = TUISession(sock_path, max_buffer_lines=max_buffer_lines)

    if initial_tasks is None:
        # Auto-discover running tasks from the daemon
        try:

            def _list():
                with ControlPlaneClient(sock_path) as c:
                    return c.send(CliRequest(method="list_tasks", params={}))

            resp = await asyncio.to_thread(_list)
            if resp.ok:
                for t in resp.result:
                    session.add_task(
                        t["task_id"],
                        t["agent_name"],
                        t["task_name"],
                        status=t["status"],
                    )
        except Exception:
            pass
    else:
        for task_id, agent_name, task_name, status in initial_tasks:
            session.add_task(task_id, agent_name, task_name, status=status)

    # Start streams for all tasks so historical events are loaded into buffers.
    # The control plane replays stored events then sends stream.end for terminal tasks.
    for task_id, buf in session.buffers.items():
        if task_id != _SYSTEM_TAB_ID:
            session.start_stream(task_id)

    app = build_tui_app(session)
    session._heartbeat = asyncio.create_task(session._heartbeat_loop())
    session._system_stream = asyncio.create_task(session._system_stream_loop())

    try:
        await app.run_async()
    finally:
        if session._heartbeat:
            session._heartbeat.cancel()
        if session._system_stream:
            session._system_stream.cancel()
        for stream in session.streams.values():
            stream.cancel()
        await asyncio.gather(
            *([session._heartbeat] if session._heartbeat else []),
            *([session._system_stream] if session._system_stream else []),
            *session.streams.values(),
            return_exceptions=True,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _parse_kv_args(
    parts: list[str],
    start_index: int = 1,
) -> tuple[str, dict]:
    """Parse ``[action] --key value …`` into (action, params).

    When *start_index* is 1 (default) the first element of *parts* is the
    action name.  When *start_index* is 0 only params are parsed and action
    is returned as an empty string.
    """
    params: dict = {}
    if start_index == 1:
        if not parts:
            raise ValueError("Empty command")
        action = parts[0]
        rest = parts[1:]
    else:
        action = ""
        rest = parts

    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if "=" in key:
                k, v = key.split("=", 1)
                params[k] = v
            elif i + 1 < len(rest) and not rest[i + 1].startswith("--"):
                params[key] = rest[i + 1]
                i += 1
            else:
                params[key] = True
        i += 1
    return action, params
