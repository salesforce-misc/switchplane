import asyncio
import socket
import struct

import pytest

from switchplane.agent_runtime import (
    AgentContext,
    _import_task_class,
    _listen_for_commands,
    _read_message,
    _start_checkpointer,
    _start_mcp,
    _stop_checkpointer,
    _stop_mcp,
    _write_message_sync,
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


class TestAgentContextLLMUsage:
    def test_record_llm_usage(self, ctx, socketpair):
        cp_sock, _ = socketpair
        record = ctx.record_llm_usage(
            model="gpt-4o-mini",
            node_name="summarize",
            prompt_tokens=100,
            completion_tokens=20,
            estimated_raw_prompt_tokens=500,
            estimated_tokens_saved=400,
            metadata={"rows_processed": 10},
        )

        data = _recv_message(cp_sock)
        event = AgentEvent.model_validate_json(data)
        assert event.type == "llm.usage"
        assert event.payload["task_id"] == "task1"
        assert event.payload["model"] == "gpt-4o-mini"
        assert event.payload["node_name"] == "summarize"
        assert event.payload["prompt_tokens"] == 100
        assert event.payload["completion_tokens"] == 20
        assert event.payload["total_tokens"] == 120
        assert event.payload["estimated_tokens_saved"] == 400
        assert event.payload["metadata"]["rows_processed"] == 10
        assert record.total_tokens == 120


class TestAgentContextProperties:
    def test_config(self, ctx):
        assert ctx.config == {"llm": {"model": "test"}}

    def test_task_id(self, ctx):
        assert ctx.task_id == "task1"

    def test_task_name(self, ctx):
        assert ctx.task_name == "test_task"

    def test_mcp_none_by_default(self, ctx):
        assert ctx.mcp is None

    @pytest.mark.asyncio
    async def test_mcp_tools_empty(self, ctx):
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
        from switchplane.agent_runtime import _run_task

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
        await _run_task(ctx, task_class, {})
        agent_sock.close()
        cp_sock.close()

    @pytest.mark.asyncio
    async def test_runs_task_with_params(self, tmp_path, monkeypatch):
        from switchplane.agent_runtime import _run_task

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
        await _run_task(ctx, task_class, {"whom": "World"})
        agent_sock.close()


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


class TestStartStopMcp:
    @pytest.mark.asyncio
    async def test_start_empty_configs(self):
        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(
            task_id="t1",
            task_name="test",
            ipc_sock=agent_sock,
            config={},
        )
        await _start_mcp(ctx, [])
        assert ctx.mcp is None
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
        await _stop_mcp(ctx)  # should not raise
        agent_sock.close()


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
