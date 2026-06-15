import asyncio
import os
import socket
import struct

import pytest

from switchplane.agent_runtime import (
    AgentContext,
    _agent_resources,
    _execute_task,
    _import_task_class,
    _instantiate_task,
    _listen_for_commands,
    _read_message,
    _start_checkpointer,
    _start_mcp,
    _stop_checkpointer,
    _unwrap_cause,
    _write_message_sync,
    agent_main,
)
from switchplane.protocol import AgentCommand, AgentEvent


def _recv_message(sock: socket.socket) -> bytes:
    """Read a length-prefixed message from a socket."""
    length_bytes = b""
    while len(length_bytes) < 4:
        chunk = sock.recv(4 - len(length_bytes))
        if not chunk:
            raise ConnectionError("Socket closed")
        length_bytes += chunk
    length = struct.unpack(">I", length_bytes)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data


@pytest.fixture
def socketpair():
    cp_sock, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    yield cp_sock, agent_sock
    cp_sock.close()
    agent_sock.close()


@pytest.fixture
def ctx(socketpair):
    _, agent_sock = socketpair
    return AgentContext(
        task_id="task1",
        task_name="test_task",
        ipc_sock=agent_sock,
        config={"llm": {"model": "test"}},
    )


class TestWriteMessageSync:
    def test_sends_length_prefixed(self, socketpair):
        cp_sock, agent_sock = socketpair
        payload = b'{"type": "test"}'
        _write_message_sync(agent_sock, payload)

        raw = cp_sock.recv(4096)
        length = struct.unpack(">I", raw[:4])[0]
        assert length == len(payload)
        assert raw[4:] == payload


class TestAgentContextEmit:
    def test_emit_task_started(self, ctx, socketpair):
        cp_sock, _ = socketpair
        ctx.emit("task.started", {})

        data = _recv_message(cp_sock)
        event = AgentEvent.model_validate_json(data)
        assert event.type == "task.started"
        assert event.task_id == "task1"

    def test_emit_with_payload(self, ctx, socketpair):
        cp_sock, _ = socketpair
        ctx.emit("task.progress", {"message": "working", "pct": 50})

        data = _recv_message(cp_sock)
        event = AgentEvent.model_validate_json(data)
        assert event.payload["message"] == "working"
        assert event.payload["pct"] == 50


class TestAgentContextProgress:
    def test_progress(self, ctx, socketpair):
        cp_sock, _ = socketpair
        ctx.progress("Step 1 done", step=1)

        data = _recv_message(cp_sock)
        event = AgentEvent.model_validate_json(data)
        assert event.type == "task.progress"
        assert event.payload["message"] == "Step 1 done"
        assert event.payload["step"] == 1


class TestAgentContextComplete:
    def test_complete(self, ctx, socketpair):
        cp_sock, _ = socketpair
        ctx.complete({"answer": 42})

        data = _recv_message(cp_sock)
        event = AgentEvent.model_validate_json(data)
        assert event.type == "task.completed"
        assert event.payload["result"] == {"answer": 42}


class TestAgentContextFail:
    def test_fail_without_traceback(self, ctx, socketpair):
        cp_sock, _ = socketpair
        ctx.fail("something broke")

        data = _recv_message(cp_sock)
        event = AgentEvent.model_validate_json(data)
        assert event.type == "task.failed"
        assert event.payload["error"] == "something broke"
        assert "traceback" not in event.payload

    def test_fail_with_traceback(self, ctx, socketpair):
        cp_sock, _ = socketpair
        ctx.fail("error", "Traceback:\n  File...")

        data = _recv_message(cp_sock)
        event = AgentEvent.model_validate_json(data)
        assert event.payload["traceback"] == "Traceback:\n  File..."


class TestAgentContextCancellation:
    def test_not_cancelled_initially(self, ctx):
        assert ctx.is_cancelled is False

    def test_set_cancelled(self, ctx):
        ctx._cancelled.set()
        assert ctx.is_cancelled is True

    @pytest.mark.asyncio
    async def test_check_cancelled_raises(self, ctx):
        ctx._cancelled.set()
        with pytest.raises(asyncio.CancelledError):
            await ctx.check_cancelled()

    @pytest.mark.asyncio
    async def test_check_cancelled_passes(self, ctx):
        await ctx.check_cancelled()  # should not raise


class TestAgentContextCommands:
    def test_poll_empty(self, ctx):
        assert ctx.poll_command() is None

    @pytest.mark.asyncio
    async def test_poll_with_command(self, ctx):
        await ctx._command_queue.put({"action": "set_coords", "params": {}})
        cmd = ctx.poll_command()
        assert cmd["action"] == "set_coords"

    @pytest.mark.asyncio
    async def test_receive_command(self, ctx):
        await ctx._command_queue.put({"action": "test"})
        cmd = await ctx.receive_command()
        assert cmd["action"] == "test"

    def test_command_result(self, ctx, socketpair):
        cp_sock, _ = socketpair
        ctx.command_result("set_coords", {"lat": 1.0, "lon": 2.0})

        data = _recv_message(cp_sock)
        event = AgentEvent.model_validate_json(data)
        assert event.type == "task.command_result"
        assert event.payload["action"] == "set_coords"
        assert event.payload["result"]["lat"] == 1.0


class TestAgentContextProperties:
    def test_config(self, ctx):
        assert ctx.config == {"llm": {"model": "test"}}

    def test_task_id(self, ctx):
        assert ctx.task_id == "task1"

    def test_task_name(self, ctx):
        assert ctx.task_name == "test_task"

    def test_mcp_returns_empty_dict_when_none(self, ctx):
        """ctx.mcp returns an empty dict (not None) so callers can safely call .get()."""
        assert ctx._mcp is None
        result = ctx.mcp
        assert result == {}
        assert ctx.mcp.get("anything") is None

    @pytest.mark.asyncio
    async def test_mcp_tools_returns_empty_when_none(self, ctx):
        """mcp_tools() returns empty dict when no MCP is configured."""
        assert ctx._mcp is None
        tools = await ctx.mcp_tools()
        assert tools == {}

    def test_checkpointer_none_by_default(self, ctx):
        assert ctx.checkpointer is None


class TestReadMessage:
    @pytest.mark.asyncio
    async def test_reads_length_prefixed(self):
        reader = asyncio.StreamReader()
        payload = b'{"type": "cancel"}'
        reader.feed_data(struct.pack(">I", len(payload)) + payload)
        result = await _read_message(reader)
        assert result == payload

    @pytest.mark.asyncio
    async def test_incomplete_raises(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"\x00")
        reader.feed_eof()
        with pytest.raises(asyncio.IncompleteReadError):
            await _read_message(reader)


class TestListenForCommands:
    @pytest.mark.asyncio
    async def test_cancel_command(self):
        reader = asyncio.StreamReader()
        cmd = AgentCommand(type="cancel", task_id="t1")
        payload = cmd.model_dump_json().encode()
        reader.feed_data(struct.pack(">I", len(payload)) + payload)

        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="test", ipc_sock=agent_sock, config={})
        dummy_task = asyncio.create_task(asyncio.sleep(10))

        await _listen_for_commands(reader, ctx, dummy_task)

        assert ctx.is_cancelled
        # task.cancel() was called; wait for cancellation to propagate
        with pytest.raises(asyncio.CancelledError):
            await dummy_task
        agent_sock.close()

    @pytest.mark.asyncio
    async def test_shutdown_command(self):
        reader = asyncio.StreamReader()
        cmd = AgentCommand(type="shutdown")
        payload = cmd.model_dump_json().encode()
        reader.feed_data(struct.pack(">I", len(payload)) + payload)

        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="test", ipc_sock=agent_sock, config={})
        dummy_task = asyncio.create_task(asyncio.sleep(10))

        await _listen_for_commands(reader, ctx, dummy_task)

        assert ctx.is_cancelled
        agent_sock.close()

    @pytest.mark.asyncio
    async def test_user_command(self):
        reader = asyncio.StreamReader()
        cmd = AgentCommand(
            type="user_command",
            task_id="t1",
            payload={"action": "set_x", "params": {"x": 5}},
        )
        payload = cmd.model_dump_json().encode()
        reader.feed_data(struct.pack(">I", len(payload)) + payload)
        reader.feed_eof()

        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="test", ipc_sock=agent_sock, config={})
        dummy_task = asyncio.create_task(asyncio.sleep(10))

        await _listen_for_commands(reader, ctx, dummy_task)

        queued = ctx.poll_command()
        assert queued is not None
        assert queued["action"] == "set_x"
        dummy_task.cancel()
        agent_sock.close()

    @pytest.mark.asyncio
    async def test_connection_closed(self):
        reader = asyncio.StreamReader()
        reader.feed_eof()

        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="test", ipc_sock=agent_sock, config={})
        dummy_task = asyncio.create_task(asyncio.sleep(10))

        await _listen_for_commands(reader, ctx, dummy_task)
        dummy_task.cancel()
        agent_sock.close()


class TestStartStopCheckpointer:
    @pytest.mark.asyncio
    async def test_start_with_db_path(self, tmp_path):
        db_path = str(tmp_path / "cp_test.db")
        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(
            task_id="t1",
            task_name="test",
            ipc_sock=agent_sock,
            config={},
            db_path=db_path,
        )
        await _start_checkpointer(ctx)
        assert ctx.checkpointer is not None
        await _stop_checkpointer(ctx)
        agent_sock.close()

    @pytest.mark.asyncio
    async def test_start_without_db_path(self):
        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(
            task_id="t1",
            task_name="test",
            ipc_sock=agent_sock,
            config={},
        )
        await _start_checkpointer(ctx)
        assert ctx.checkpointer is None
        agent_sock.close()

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(
            task_id="t1",
            task_name="test",
            ipc_sock=agent_sock,
            config={},
        )
        await _stop_checkpointer(ctx)  # should not raise
        agent_sock.close()


class TestImportTaskClass:
    def test_finds_task_subclass(self, tmp_path, monkeypatch):
        pkg = tmp_path / "importpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mytask.py").write_text(
            "from switchplane.task import Task\n"
            "class MyTask(Task):\n"
            '    name = "mytask"\n'
            "    async def run(self, ctx):\n"
            "        pass\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        cls = _import_task_class("importpkg.mytask")
        assert cls.name == "mytask"

    def test_no_task_class_raises(self, tmp_path, monkeypatch):
        pkg = tmp_path / "noimportpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "empty.py").write_text("x = 1\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(RuntimeError, match="No Task subclass"):
            _import_task_class("noimportpkg.empty")


class TestRunTask:
    @pytest.mark.asyncio
    async def test_runs_task_from_class(self, tmp_path, monkeypatch):
        from switchplane.agent_runtime import _instantiate_task, _run_task

        pkg = tmp_path / "runtaskpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mytask.py").write_text(
            "from switchplane.task import Task\n"
            "class MyTask(Task):\n"
            '    name = "mytask"\n'
            "    async def run(self, ctx):\n"
            '        ctx.complete({"status": "done"})\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        cp_sock, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="mytask", ipc_sock=agent_sock, config={})

        task_class = _import_task_class("runtaskpkg.mytask")
        instance = _instantiate_task(ctx, task_class, {})
        await _run_task(ctx, instance)
        agent_sock.close()
        cp_sock.close()

    @pytest.mark.asyncio
    async def test_runs_task_with_params(self, tmp_path, monkeypatch):
        from switchplane.agent_runtime import _instantiate_task, _run_task

        pkg = tmp_path / "paramtaskpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "greet.py").write_text(
            "from switchplane.task import Task\n"
            "from pydantic import Field\n"
            "class GreetTask(Task):\n"
            '    name = "greet"\n'
            "    whom: str = Field(description='Who')\n"
            "    async def run(self, ctx):\n"
            '        ctx.complete({"greeting": f"Hello {self.whom}"})\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="greet", ipc_sock=agent_sock, config={})

        task_class = _import_task_class("paramtaskpkg.greet")
        instance = _instantiate_task(ctx, task_class, {"whom": "World"})
        assert instance.whom == "World"
        await _run_task(ctx, instance)
        agent_sock.close()


class TestInstantiateTaskAndStartupInfo:
    """`_instantiate_task` builds the Task subclass with parameters
    bound, so `agent_main` can call `startup_info()` on it before
    emitting `task.started`. Without this split, the lifecycle event
    couldn't carry param-derived metadata."""

    def test_default_startup_info_is_empty(self, ctx, tmp_path, monkeypatch):
        """Default `Task.startup_info` returns `{}` — preserves the
        historical empty-payload `task.started` shape for tasks that
        don't override."""
        from switchplane.agent_runtime import _instantiate_task

        pkg = tmp_path / "defaultinfo"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "t.py").write_text(
            'from switchplane.task import Task\nclass T(Task):\n    name = "t"\n    async def run(self, ctx): pass\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        task_class = _import_task_class("defaultinfo.t")

        instance = _instantiate_task(ctx, task_class, {})

        assert instance.startup_info() == {}

    def test_subclass_can_surface_payload(self, ctx, tmp_path, monkeypatch):
        """A subclass that overrides `startup_info` gets to read its
        own bound params (set by `_instantiate_task`) and ctx.config."""
        from switchplane.agent_runtime import _instantiate_task

        pkg = tmp_path / "infopkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "t.py").write_text(
            "from switchplane.task import Task\n"
            "from pydantic import Field\n"
            "class T(Task):\n"
            '    name = "t"\n'
            "    work_item: str = Field(description='wi')\n"
            "    def startup_info(self):\n"
            "        return {'work_item': self.work_item, "
            "'model': self._ctx.config['llm']['model']}\n"
            "    async def run(self, ctx): pass\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        task_class = _import_task_class("infopkg.t")

        instance = _instantiate_task(ctx, task_class, {"work_item": "W-1"})

        assert instance.startup_info() == {"work_item": "W-1", "model": "test"}


class TestWaitForInput:
    @pytest.mark.asyncio
    async def test_returns_text(self, socketpair, tmp_path):
        """wait_for_input emits interrupted/resumed events and returns the user text."""
        cp_sock, agent_sock = socketpair
        ctx = AgentContext(
            task_id="t1",
            task_name="test",
            ipc_sock=agent_sock,
            config={},
            db_path=str(tmp_path / "test.db"),
        )
        # Provide a fake checkpointer so the guard passes
        ctx._checkpointer = object()
        # Provide a mock task (wait_for_input dispatches non-input commands to it)
        ctx._task = None

        # Enqueue the __input__ command before calling wait_for_input
        await ctx._command_queue.put({"action": "__input__", "params": {"text": "hello"}})

        result = await ctx.wait_for_input("What is your name?")
        assert result == "hello"

        # Read the two events emitted: task.interrupted then task.resumed
        data1 = _recv_message(cp_sock)
        event1 = AgentEvent.model_validate_json(data1)
        assert event1.type == "task.interrupted"
        assert event1.payload["prompt"] == "What is your name?"

        data2 = _recv_message(cp_sock)
        event2 = AgentEvent.model_validate_json(data2)
        assert event2.type == "task.resumed"

    @pytest.mark.asyncio
    async def test_requires_checkpointer(self, socketpair):
        """wait_for_input raises RuntimeError when no checkpointer is set."""
        _, agent_sock = socketpair
        ctx = AgentContext(
            task_id="t1",
            task_name="test",
            ipc_sock=agent_sock,
            config={},
        )
        assert ctx.checkpointer is None

        with pytest.raises(RuntimeError, match="wait_for_input requires a checkpointer"):
            await ctx.wait_for_input("prompt")

    @pytest.mark.asyncio
    async def test_dispatches_other_commands(self, socketpair, tmp_path):
        """Non-__input__ commands are dispatched to the task while waiting."""
        from unittest.mock import AsyncMock, MagicMock

        _cp_sock, agent_sock = socketpair
        ctx = AgentContext(
            task_id="t1",
            task_name="test",
            ipc_sock=agent_sock,
            config={},
            db_path=str(tmp_path / "test.db"),
        )
        ctx._checkpointer = object()

        mock_task = MagicMock()
        mock_task._dispatch_command = AsyncMock()
        ctx._task = mock_task

        # First a non-input command, then the actual input
        other_cmd = {"action": "set_value", "params": {"x": 1}}
        input_cmd = {"action": "__input__", "params": {"text": "world"}}
        await ctx._command_queue.put(other_cmd)
        await ctx._command_queue.put(input_cmd)

        result = await ctx.wait_for_input()
        assert result == "world"

        # The non-input command should have been dispatched to the task
        mock_task._dispatch_command.assert_awaited_once_with(ctx, other_cmd)


class _FakeManager:
    """Stand-in for McpManager: start() returns canned (name, message) error
    tuples and records whether the started sessions were torn down."""

    def __init__(self, configs, runtime_dir=None):
        self.configs = configs
        self.stopped = False

    def set_errors(self, errors):
        self._errors = errors
        return self

    async def start(self):
        return getattr(self, "_errors", [])

    async def stop(self):
        self.stopped = True


class TestStartStopMcp:
    @pytest.fixture
    def _ctx(self):
        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="test", ipc_sock=agent_sock, config={})
        yield ctx
        agent_sock.close()

    @pytest.mark.asyncio
    async def test_start_empty_configs(self, _ctx):
        await _start_mcp(_ctx, [])
        assert _ctx._mcp is None  # internal state stays None
        assert _ctx.mcp == {}  # property returns empty dict for safe .get() access

    @pytest.mark.asyncio
    async def test_required_server_failure_aborts(self, _ctx, monkeypatch):
        import switchplane.mcp as mcp_mod

        captured = {}

        def _factory(configs, runtime_dir=None):
            mgr = _FakeManager(configs).set_errors([("dead", "Failed to start MCP server 'dead': boom")])
            captured["mgr"] = mgr
            return mgr

        monkeypatch.setattr(mcp_mod, "McpManager", _factory)

        with pytest.raises(RuntimeError, match="failed to start \\(dead\\)"):
            await _start_mcp(_ctx, [{"name": "dead", "url": "http://x/"}])
        assert captured["mgr"].stopped is True  # torn down on abort
        assert _ctx._mcp is None

    @pytest.mark.asyncio
    async def test_optional_server_failure_is_skipped(self, _ctx, monkeypatch):
        import switchplane.mcp as mcp_mod

        captured = {}

        def _factory(configs, runtime_dir=None):
            mgr = _FakeManager(configs).set_errors([("slack", "Failed to start MCP server 'slack': 429")])
            captured["mgr"] = mgr
            return mgr

        monkeypatch.setattr(mcp_mod, "McpManager", _factory)

        # An optional server's startup failure must not abort the task.
        await _start_mcp(_ctx, [{"name": "slack", "url": "http://x/", "optional": True}])
        assert captured["mgr"].stopped is False  # kept, not torn down
        assert _ctx._mcp is captured["mgr"]

    @pytest.mark.asyncio
    async def test_required_failure_aborts_even_with_optional_failure(self, _ctx, monkeypatch):
        import switchplane.mcp as mcp_mod

        captured = {}

        def _factory(configs, runtime_dir=None):
            mgr = _FakeManager(configs).set_errors([
                ("slack", "Failed to start MCP server 'slack': 429"),
                ("dxmcp", "Failed to start MCP server 'dxmcp': auth"),
            ])
            captured["mgr"] = mgr
            return mgr

        monkeypatch.setattr(mcp_mod, "McpManager", _factory)

        with pytest.raises(RuntimeError) as ei:
            await _start_mcp(_ctx, [
                {"name": "slack", "url": "http://x/", "optional": True},
                {"name": "dxmcp", "url": "http://y/"},
            ])
        # Only the required server is named in the abort message.
        assert "dxmcp" in str(ei.value)
        assert "slack" not in str(ei.value)
        assert captured["mgr"].stopped is True

class TestUnwrapCause:
    """`_unwrap_cause` recovers the real leaf from masked/wrapped errors."""

    def test_plain_exception_returned_as_is(self):
        err = ValueError("boom")
        assert _unwrap_cause(err) is err

    def test_bare_cancelled_returns_none(self):
        assert _unwrap_cause(asyncio.CancelledError()) is None

    def test_group_with_real_leaf_returns_leaf(self):
        leaf = RuntimeError("real cause")
        group = BaseExceptionGroup("wrapped", [asyncio.CancelledError(), leaf])
        assert _unwrap_cause(group) is leaf

    def test_group_of_only_cancellations_returns_none(self):
        group = BaseExceptionGroup("cancels", [asyncio.CancelledError(), asyncio.CancelledError()])
        assert _unwrap_cause(group) is None

    def test_nested_group_recurses_to_leaf(self):
        leaf = OSError("deep")
        inner = BaseExceptionGroup("inner", [asyncio.CancelledError(), leaf])
        outer = BaseExceptionGroup("outer", [asyncio.CancelledError(), inner])
        assert _unwrap_cause(outer) is leaf


class TestAgentResources:
    """`_agent_resources` owns the resource lifecycle inside the task scope.

    The defining behaviour of fix A: teardown propagates, so a fault trapped in
    a resource's async scope (the MCP transport task group) surfaces to the
    reporting boundary instead of being masked as a bare CancelledError.
    """

    @pytest.mark.asyncio
    async def test_teardown_propagates_trapped_cause(self):
        """A BaseExceptionGroup raised by mcp stop() escapes the context (not swallowed)."""
        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="test", ipc_sock=agent_sock, config={})

        leaf = RuntimeError("429 from transport")

        class _FaultedMgr:
            async def stop(self):
                raise BaseExceptionGroup("transport", [asyncio.CancelledError(), leaf])

        # _start_mcp([]) leaves _mcp None; simulate a started manager directly.
        with pytest.raises(BaseExceptionGroup) as ei:
            async with _agent_resources(ctx, []):
                ctx._mcp = _FaultedMgr()
        assert _unwrap_cause(ei.value) is leaf
        assert ctx._mcp is None  # cleared even though teardown raised
        agent_sock.close()

    @pytest.mark.asyncio
    async def test_clean_teardown_no_error(self):
        """With no MCP and a clean body, the context exits silently."""
        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="test", ipc_sock=agent_sock, config={})
        async with _agent_resources(ctx, []):
            pass
        assert ctx._mcp is None
        agent_sock.close()


class TestExecuteTask:
    """`_execute_task` emits task.started after resources are up, then runs."""

    @pytest.mark.asyncio
    async def test_emits_started_then_runs(self, tmp_path, monkeypatch):
        pkg = tmp_path / "exectaskpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "ok.py").write_text(
            "from switchplane.task import Task\n"
            "class OkTask(Task):\n"
            '    name = "ok"\n'
            "    async def run(self, ctx):\n"
            '        ctx.complete({"status": "done"})\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        cp_sock, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="ok", ipc_sock=agent_sock, config={})
        task_class = _import_task_class("exectaskpkg.ok")
        instance = _instantiate_task(ctx, task_class, {})

        await _execute_task(ctx, instance, [])

        types = []
        cp_sock.setblocking(False)
        while True:
            try:
                types.append(AgentEvent.model_validate_json(_recv_message(cp_sock)).type)
            except (BlockingIOError, ConnectionError):
                break
        assert types[0] == "task.started"
        assert "task.completed" in types
        agent_sock.close()
        cp_sock.close()

    @pytest.mark.asyncio
    async def test_wrapped_cause_propagates_to_boundary(self, tmp_path, monkeypatch):
        """A task whose body raises inside a TaskGroup surfaces the real leaf.

        This is the masking case made non-MCP: the BaseExceptionGroup must
        escape _execute_task so agent_main's boundary can unwrap it.
        """
        pkg = tmp_path / "wrappedpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "boom.py").write_text(
            "import asyncio\n"
            "from switchplane.task import Task\n"
            "class BoomTask(Task):\n"
            '    name = "boom"\n'
            "    async def run(self, ctx):\n"
            "        async with asyncio.TaskGroup() as tg:\n"
            "            tg.create_task(self._fault())\n"
            "    async def _fault(self):\n"
            '        raise ValueError("boom")\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(task_id="t1", task_name="boom", ipc_sock=agent_sock, config={})
        task_class = _import_task_class("wrappedpkg.boom")
        instance = _instantiate_task(ctx, task_class, {})

        with pytest.raises(BaseExceptionGroup) as ei:
            await _execute_task(ctx, instance, [])
        assert isinstance(_unwrap_cause(ei.value), ValueError)
        agent_sock.close()


class TestAgentMainBoundary:
    """End-to-end: drive the real `agent_main` entry point over an IPC fd and
    assert the terminal event it reports. This is the integration the masking
    fix rests on — that an exception raised during execution reaches
    `agent_main`'s reporting boundary and is reported with the real cause.
    """

    async def _drive(
        self,
        task_module: str,
        params: dict | None = None,
        mcp_servers: list | None = None,
    ) -> list[AgentEvent]:
        """Run agent_main against a task module, return all events it emitted."""
        cp_sock, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        # agent_main does socket.fromfd then os.close(fd); hand it a dup so our
        # agent_sock stays valid for cleanup. Own the dup in a finally so the
        # fd never leaks even if agent_main returns before its own os.close.
        ipc_fd = os.dup(agent_sock.fileno())
        closed_by_agent_main = False
        try:
            command = AgentCommand(
                type="execute_task",
                task_id="t1",
                payload={
                    "task_name": "x",
                    "params": params or {},
                    "task_module": task_module,
                    "config": {},
                    "mcp_servers": mcp_servers or [],
                    "db_path": None,
                    "log_level": "debug",
                },
            )
            # Send the initial command from the CP side before agent_main reads.
            _write_message_sync(cp_sock, command.model_dump_json().encode())

            await agent_main(ipc_fd, entry_point="test")
            closed_by_agent_main = True  # agent_main reached its own os.close(ipc_fd)

            events = []
            cp_sock.setblocking(False)
            while True:
                try:
                    events.append(AgentEvent.model_validate_json(_recv_message(cp_sock)))
                except (BlockingIOError, ConnectionError):
                    break
            return events
        finally:
            if not closed_by_agent_main:
                try:
                    os.close(ipc_fd)
                except OSError:
                    pass
            agent_sock.close()
            cp_sock.close()

    @pytest.mark.asyncio
    async def test_clean_task_completes(self, tmp_path, monkeypatch):
        pkg = tmp_path / "ampkg_ok"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "ok.py").write_text(
            "from switchplane.task import Task\n"
            "class OkTask(Task):\n"
            '    name = "ok"\n'
            "    async def run(self, ctx):\n"
            '        ctx.complete({"v": 1})\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        events = await self._drive("ampkg_ok.ok")
        types = [e.type for e in events]
        assert types[0] == "task.started"
        assert "task.completed" in types
        assert "task.failed" not in types

    @pytest.mark.asyncio
    async def test_wrapped_cause_reported_with_real_leaf(self, tmp_path, monkeypatch):
        """A TaskGroup fault must be reported as the real leaf, not ExceptionGroup
        and not 'CancelledError raised internally'."""
        pkg = tmp_path / "ampkg_boom"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "boom.py").write_text(
            "import asyncio\n"
            "from switchplane.task import Task\n"
            "class BoomTask(Task):\n"
            '    name = "boom"\n'
            "    async def run(self, ctx):\n"
            "        async with asyncio.TaskGroup() as tg:\n"
            "            tg.create_task(self._fault())\n"
            "    async def _fault(self):\n"
            '        raise ValueError("kaboom")\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        events = await self._drive("ampkg_boom.boom")
        failed = [e for e in events if e.type == "task.failed"]
        assert len(failed) == 1
        assert failed[0].payload["error"] == "ValueError: kaboom"
        assert "ExceptionGroup" not in failed[0].payload["error"]
        assert "CancelledError raised internally" not in failed[0].payload["error"]

    @pytest.mark.asyncio
    async def test_plain_exception_reported(self, tmp_path, monkeypatch):
        pkg = tmp_path / "ampkg_raise"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "r.py").write_text(
            "from switchplane.task import Task\n"
            "class RaiseTask(Task):\n"
            '    name = "r"\n'
            "    async def run(self, ctx):\n"
            '        raise RuntimeError("nope")\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        events = await self._drive("ampkg_raise.r")
        failed = [e for e in events if e.type == "task.failed"]
        assert len(failed) == 1
        assert failed[0].payload["error"] == "RuntimeError: nope"

    @pytest.mark.asyncio
    async def test_operator_cancel_reports_cancelled(self, tmp_path, monkeypatch):
        """A cancel command mid-execution yields task.cancelled, not a failure.

        Drives agent_main directly (not via _drive) because it must send a
        second command — `cancel` — while the task is running.
        """
        pkg = tmp_path / "ampkg_cancel"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "loop.py").write_text(
            "from switchplane.task import Task\n"
            "class LoopTask(Task):\n"
            '    name = "loop"\n'
            "    async def run(self, ctx):\n"
            "        while True:\n"
            "            await ctx.sleep(60)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        cp_sock, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ipc_fd = os.dup(agent_sock.fileno())
        try:
            execute = AgentCommand(
                type="execute_task",
                task_id="t1",
                payload={
                    "task_name": "loop",
                    "params": {},
                    "task_module": "ampkg_cancel.loop",
                    "config": {},
                    "mcp_servers": [],
                    "db_path": None,
                    "log_level": "debug",
                },
            )
            _write_message_sync(cp_sock, execute.model_dump_json().encode())

            main_task = asyncio.create_task(agent_main(ipc_fd, entry_point="test"))
            await asyncio.sleep(0.05)  # let the task start and enter its loop
            cancel = AgentCommand(type="cancel", task_id="t1")
            _write_message_sync(cp_sock, cancel.model_dump_json().encode())
            await asyncio.wait_for(main_task, timeout=5)

            events = []
            cp_sock.setblocking(False)
            while True:
                try:
                    events.append(AgentEvent.model_validate_json(_recv_message(cp_sock)))
                except (BlockingIOError, ConnectionError):
                    break
            types = [e.type for e in events]
            assert "task.cancelled" in types
            assert "task.failed" not in types
        finally:
            agent_sock.close()
            cp_sock.close()

    @pytest.mark.asyncio
    async def test_operator_cancel_wins_over_teardown_fault(self, tmp_path, monkeypatch):
        """Operator cancel is authoritative even if resource teardown raises.

        When a cancel coincides with a trapped fault that surfaces on teardown
        (the group *replaces* the in-flight CancelledError), classification must
        still key off the durable cancel signals and report task.cancelled —
        not task.failed.
        """
        pkg = tmp_path / "ampkg_cxl_fault"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        # The task installs a manager on ctx._mcp whose stop() raises a group,
        # mimicking a transport fault that only surfaces on teardown, then loops
        # until cancelled.
        (pkg / "loopfault.py").write_text(
            "import asyncio\n"
            "from switchplane.task import Task\n"
            "class _FaultMgr:\n"
            "    async def stop(self):\n"
            '        raise BaseExceptionGroup("t", [asyncio.CancelledError(), RuntimeError("429")])\n'
            "class LoopFaultTask(Task):\n"
            '    name = "loopfault"\n'
            "    async def run(self, ctx):\n"
            "        ctx._mcp = _FaultMgr()\n"
            "        while True:\n"
            "            await ctx.sleep(60)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        cp_sock, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ipc_fd = os.dup(agent_sock.fileno())
        try:
            execute = AgentCommand(
                type="execute_task",
                task_id="t1",
                payload={
                    "task_name": "loopfault",
                    "params": {},
                    "task_module": "ampkg_cxl_fault.loopfault",
                    "config": {},
                    "mcp_servers": [],
                    "db_path": None,
                    "log_level": "debug",
                },
            )
            _write_message_sync(cp_sock, execute.model_dump_json().encode())
            main_task = asyncio.create_task(agent_main(ipc_fd, entry_point="test"))
            await asyncio.sleep(0.05)
            _write_message_sync(cp_sock, AgentCommand(type="cancel", task_id="t1").model_dump_json().encode())
            await asyncio.wait_for(main_task, timeout=5)

            events = []
            cp_sock.setblocking(False)
            while True:
                try:
                    events.append(AgentEvent.model_validate_json(_recv_message(cp_sock)))
                except (BlockingIOError, ConnectionError):
                    break
            types = [e.type for e in events]
            assert "task.cancelled" in types
            assert "task.failed" not in types
        finally:
            agent_sock.close()
            cp_sock.close()

    @pytest.mark.asyncio
    async def test_teardown_fault_does_not_overwrite_completed(self, tmp_path, monkeypatch):
        """A teardown fault on the success path must not emit a second terminal
        event. The task completes, then resource teardown raises a group; the
        boundary must log it and leave the single task.completed standing —
        otherwise the control plane records the task as FAILED.
        """
        pkg = tmp_path / "ampkg_done_fault"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        # Task completes successfully, then its installed manager raises on stop.
        (pkg / "donefault.py").write_text(
            "import asyncio\n"
            "from switchplane.task import Task\n"
            "class _FaultMgr:\n"
            "    async def stop(self):\n"
            '        raise BaseExceptionGroup("t", [asyncio.CancelledError(), RuntimeError("429")])\n'
            "class DoneFaultTask(Task):\n"
            '    name = "donefault"\n'
            "    async def run(self, ctx):\n"
            "        ctx._mcp = _FaultMgr()\n"
            '        ctx.complete({"v": 1})\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        events = await self._drive("ampkg_done_fault.donefault")
        types = [e.type for e in events]
        assert types.count("task.completed") == 1
        assert "task.failed" not in types

    @pytest.mark.asyncio
    async def test_mcp_startup_failure_reported_before_started(self, tmp_path, monkeypatch):
        """An unreachable MCP server fails the task before task.started.

        Locks the prior contract: resource-startup failure surfaces as
        task.failed with no preceding task.started.
        """
        pkg = tmp_path / "ampkg_mcp"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "needs_mcp.py").write_text(
            "from typing import ClassVar\n"
            "from switchplane.task import Task\n"
            "class NeedsMcp(Task):\n"
            '    name = "needs_mcp"\n'
            '    mcp_servers: ClassVar[list[str]] = ["dead"]\n'
            "    async def run(self, ctx):\n"
            '        ctx.complete({"unreachable": True})\n'
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        # Force _start_mcp to fail as if the server were unreachable.
        import switchplane.agent_runtime as ar

        async def _boom_start_mcp(ctx, configs):
            if configs:
                raise RuntimeError("MCP server(s) failed to start (dead): unreachable")

        monkeypatch.setattr(ar, "_start_mcp", _boom_start_mcp)

        events = await self._drive(
            "ampkg_mcp.needs_mcp",
            mcp_servers=[{"name": "dead", "url": "http://127.0.0.1:0/"}],
        )
        types = [e.type for e in events]
        assert "task.started" not in types
        failed = [e for e in events if e.type == "task.failed"]
        assert len(failed) == 1
        assert "failed to start" in failed[0].payload["error"]


class TestTaskMcpServerFiltering:
    """Tests for task-level MCP server declarations in agent_main filtering logic."""

    def _agent_main_filter(self, task_class, mcp_configs):
        """Replicate the filtering logic from agent_main for unit testing."""
        if task_class.mcp_servers:
            available = {c["name"]: c for c in mcp_configs}
            filtered = [available[n] for n in task_class.mcp_servers if n in available]
            missing = [n for n in task_class.mcp_servers if n not in available]
            return filtered, missing
        return [], []

    def test_task_with_specific_servers(self):
        from switchplane.task import Task

        class MyTask(Task):
            name = "my"
            mcp_servers = ["server_a"]  # noqa: RUF012

            async def run(self, ctx):
                pass

        all_configs = [
            {"name": "server_a", "command": ["echo"]},
            {"name": "server_b", "command": ["echo"]},
        ]

        filtered, missing = self._agent_main_filter(MyTask, all_configs)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "server_a"
        assert missing == []

    def test_task_with_empty_mcp_servers(self):
        from switchplane.task import Task

        class MyTask(Task):
            name = "my"

            async def run(self, ctx):
                pass

        assert MyTask.mcp_servers == []

        all_configs = [
            {"name": "server_a", "command": ["echo"]},
            {"name": "server_b", "command": ["echo"]},
        ]

        filtered, missing = self._agent_main_filter(MyTask, all_configs)
        assert filtered == []
        assert missing == []

    def test_task_requests_unavailable_server(self):
        from switchplane.task import Task

        class MyTask(Task):
            name = "my"
            mcp_servers = ["server_a", "server_missing"]  # noqa: RUF012

            async def run(self, ctx):
                pass

        all_configs = [
            {"name": "server_a", "command": ["echo"]},
        ]

        filtered, missing = self._agent_main_filter(MyTask, all_configs)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "server_a"
        assert missing == ["server_missing"]

    def test_task_requests_multiple_servers(self):
        from switchplane.task import Task

        class MyTask(Task):
            name = "my"
            mcp_servers = ["server_b", "server_a"]  # noqa: RUF012

            async def run(self, ctx):
                pass

        all_configs = [
            {"name": "server_a", "command": ["echo"]},
            {"name": "server_b", "url": "http://x"},
            {"name": "server_c", "command": ["echo"]},
        ]

        filtered, missing = self._agent_main_filter(MyTask, all_configs)
        assert len(filtered) == 2
        # Order follows task_class.mcp_servers declaration order
        assert filtered[0]["name"] == "server_b"
        assert filtered[1]["name"] == "server_a"
        assert missing == []

    def test_mcp_servers_not_treated_as_parameter(self):
        from pydantic import Field

        from switchplane.task import Task

        class MyTask(Task):
            name = "my"
            mcp_servers = ["server_a"]  # noqa: RUF012
            custom: str = Field(description="a param")

            async def run(self, ctx):
                pass

        model = MyTask.parameters_model()
        assert model is not None
        assert "custom" in model.model_fields
        assert "mcp_servers" not in model.model_fields
