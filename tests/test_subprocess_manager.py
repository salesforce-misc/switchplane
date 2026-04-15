import asyncio
import json
import struct
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from switchplane.persistence import Store
from switchplane.protocol import AgentEvent
from switchplane.subprocess_manager import SubprocessManager, _deep_merge
from switchplane.task import TaskStatus


class TestDeepMerge:
    def test_flat(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested(self):
        base = {"llm": {"provider": "anthropic", "model": "claude"}}
        _deep_merge(base, {"llm": {"model": "gpt-4"}})
        assert base == {"llm": {"provider": "anthropic", "model": "gpt-4"}}

    def test_add_nested(self):
        base = {"a": 1}
        _deep_merge(base, {"b": {"x": 1}})
        assert base == {"a": 1, "b": {"x": 1}}

    def test_empty_override(self):
        base = {"a": 1}
        _deep_merge(base, {})
        assert base == {"a": 1}

    def test_replace_scalar_with_dict(self):
        base = {"a": "string"}
        _deep_merge(base, {"a": {"nested": True}})
        assert base == {"a": {"nested": True}}


class TestSubprocessManagerNoHandles:
    @pytest.fixture
    def mgr(self):
        store = MagicMock(spec=Store)
        return SubprocessManager(store)

    def test_active_count_empty(self, mgr):
        assert mgr.active_count == 0

    @pytest.mark.asyncio
    async def test_send_cancel_no_handle(self, mgr):
        result = await mgr.send_cancel("nonexistent_task")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_user_command_no_handle(self, mgr):
        result = await mgr.send_user_command("nonexistent", "action")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_task_delegates(self, mgr):
        result = await mgr.cancel_task("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_kill_all_empty(self, mgr):
        await mgr.kill_all()  # should not raise


class TestHandleEvent:
    @pytest_asyncio.fixture
    async def mgr(self, tmp_path):
        store = Store(tmp_path / "test.db")
        await store.initialize()
        from switchplane.checkpoint import setup_tables

        await setup_tables(store._db)
        mgr = SubprocessManager(store)
        yield mgr
        await store.close()

    @pytest.mark.asyncio
    async def test_task_started(self, mgr):
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t1",
            agent_name="a",
            task_name="t",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        event = AgentEvent(type="task.started", task_id="t1")
        await mgr._handle_event(event)

        result = await mgr.store.get_task("t1")
        assert result.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_task_completed(self, mgr):
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t2",
            agent_name="a",
            task_name="t",
            status=TaskStatus.RUNNING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        event = AgentEvent(
            type="task.completed",
            task_id="t2",
            payload={"result": {"answer": 42}},
        )
        await mgr._handle_event(event)

        result = await mgr.store.get_task("t2")
        assert result.status == TaskStatus.COMPLETED
        assert json.loads(result.result_json) == {"answer": 42}

    @pytest.mark.asyncio
    async def test_task_failed(self, mgr):
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t3",
            agent_name="a",
            task_name="t",
            status=TaskStatus.RUNNING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        event = AgentEvent(
            type="task.failed",
            task_id="t3",
            payload={"error": "boom", "traceback": "line 1\nline 2"},
        )
        await mgr._handle_event(event)

        result = await mgr.store.get_task("t3")
        assert result.status == TaskStatus.FAILED
        error = json.loads(result.error_json)
        assert error["error"] == "boom"
        assert "traceback" in error

    @pytest.mark.asyncio
    async def test_task_cancelled(self, mgr):
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t4",
            agent_name="a",
            task_name="t",
            status=TaskStatus.RUNNING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        event = AgentEvent(type="task.cancelled", task_id="t4")
        await mgr._handle_event(event)

        result = await mgr.store.get_task("t4")
        assert result.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_task_progress_stored_as_event(self, mgr):
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t5",
            agent_name="a",
            task_name="t",
            status=TaskStatus.RUNNING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        event = AgentEvent(
            type="task.progress",
            task_id="t5",
            payload={"message": "50% done"},
        )
        await mgr._handle_event(event)

        events = await mgr.store.get_events("t5")
        assert len(events) == 1
        assert events[0]["event_type"] == "task.progress"


class TestHandleEventInterruptedResumed:
    @pytest_asyncio.fixture
    async def mgr(self, tmp_path):
        store = Store(tmp_path / "test.db")
        await store.initialize()
        from switchplane.checkpoint import setup_tables

        await setup_tables(store._db)
        mgr = SubprocessManager(store)
        yield mgr
        await store.close()

    @pytest.mark.asyncio
    async def test_task_interrupted(self, mgr):
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t_int",
            agent_name="a",
            task_name="t",
            status=TaskStatus.RUNNING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        event = AgentEvent(
            type="task.interrupted",
            task_id="t_int",
            payload={"prompt": "Enter name"},
        )
        await mgr._handle_event(event)

        result = await mgr.store.get_task("t_int")
        assert result.status == TaskStatus.INTERRUPTED

    @pytest.mark.asyncio
    async def test_task_resumed(self, mgr):
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t_res",
            agent_name="a",
            task_name="t",
            status=TaskStatus.INTERRUPTED,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        event = AgentEvent(type="task.resumed", task_id="t_res")
        await mgr._handle_event(event)

        result = await mgr.store.get_task("t_res")
        assert result.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_orphan_detection_includes_interrupted(self, mgr):
        """Agent exits while task is INTERRUPTED — task should be marked FAILED."""
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t_int_orphan",
            agent_name="a",
            task_name="t",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        reader = asyncio.StreamReader()
        # Emit started then interrupted, then EOF (agent exits)
        for evt in [
            AgentEvent(type="task.started", task_id="t_int_orphan"),
            AgentEvent(type="task.interrupted", task_id="t_int_orphan", payload={"prompt": "?"}),
        ]:
            payload = evt.model_dump_json().encode()
            reader.feed_data(struct.pack(">I", len(payload)) + payload)
        reader.feed_eof()

        proc = AsyncMock()
        proc.wait = AsyncMock(return_value=None)
        proc.returncode = 0

        handle = MagicMock()
        handle.agent_id = "a_int_orphan"
        handle.task_id = "t_int_orphan"
        handle.reader = reader
        handle.proc = proc
        handle.writer = MagicMock()
        handle.writer.close = MagicMock()
        handle.sock = MagicMock()

        mgr._handles["a_int_orphan"] = handle
        mgr._task_to_agent["t_int_orphan"] = "a_int_orphan"

        await mgr._read_events(handle)

        result = await mgr.store.get_task("t_int_orphan")
        assert result.status == TaskStatus.FAILED


class TestReadEvents:
    @pytest_asyncio.fixture
    async def mgr_for_events(self, tmp_path):
        store = Store(tmp_path / "test.db")
        await store.initialize()
        from switchplane.checkpoint import setup_tables

        await setup_tables(store._db)
        mgr = SubprocessManager(store)
        yield mgr
        await store.close()

    @pytest.mark.asyncio
    async def test_reads_event_and_handles(self, mgr_for_events):
        mgr = mgr_for_events
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t1",
            agent_name="a",
            task_name="t",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        reader = asyncio.StreamReader()
        for evt in [
            AgentEvent(type="task.started", task_id="t1"),
            AgentEvent(type="task.completed", task_id="t1", payload={"result": "ok"}),
        ]:
            payload = evt.model_dump_json().encode()
            reader.feed_data(struct.pack(">I", len(payload)) + payload)
        reader.feed_eof()

        proc = AsyncMock()
        proc.wait = AsyncMock(return_value=None)
        proc.returncode = 0

        handle = MagicMock()
        handle.agent_id = "a1"
        handle.task_id = "t1"
        handle.reader = reader
        handle.proc = proc
        handle.writer = MagicMock()
        handle.writer.close = MagicMock()
        handle.sock = MagicMock()

        mgr._handles["a1"] = handle
        mgr._task_to_agent["t1"] = "a1"

        await mgr._read_events(handle)

        result = await mgr.store.get_task("t1")
        assert result.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_exit_while_running_marks_failed(self, mgr_for_events):
        """Agent exits (even with code 0) without emitting a terminal event — task is marked failed."""
        mgr = mgr_for_events
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t_orphan",
            agent_name="a",
            task_name="t",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        reader = asyncio.StreamReader()
        evt = AgentEvent(type="task.started", task_id="t_orphan")
        payload = evt.model_dump_json().encode()
        reader.feed_data(struct.pack(">I", len(payload)) + payload)
        reader.feed_eof()

        proc = AsyncMock()
        proc.wait = AsyncMock(return_value=None)
        proc.returncode = 0

        handle = MagicMock()
        handle.agent_id = "a_orphan"
        handle.task_id = "t_orphan"
        handle.reader = reader
        handle.proc = proc
        handle.writer = MagicMock()
        handle.writer.close = MagicMock()
        handle.sock = MagicMock()

        mgr._handles["a_orphan"] = handle
        mgr._task_to_agent["t_orphan"] = "a_orphan"

        await mgr._read_events(handle)

        result = await mgr.store.get_task("t_orphan")
        assert result.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_nonzero_exit_marks_failed(self, mgr_for_events):
        mgr = mgr_for_events
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t2",
            agent_name="a",
            task_name="t",
            status=TaskStatus.RUNNING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        reader = asyncio.StreamReader()
        reader.feed_eof()

        proc = AsyncMock()
        proc.wait = AsyncMock(return_value=None)
        proc.returncode = 1

        handle = MagicMock()
        handle.agent_id = "a2"
        handle.task_id = "t2"
        handle.reader = reader
        handle.proc = proc
        handle.writer = MagicMock()
        handle.writer.close = MagicMock()
        handle.sock = MagicMock()

        mgr._handles["a2"] = handle
        mgr._task_to_agent["t2"] = "a2"

        await mgr._read_events(handle)

        result = await mgr.store.get_task("t2")
        assert result.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_event_callback_called(self, mgr_for_events):
        mgr = mgr_for_events
        from datetime import datetime

        from switchplane.task import TaskRecord

        now = datetime.now(UTC)
        task = TaskRecord(
            task_id="t3",
            agent_name="a",
            task_name="t",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        await mgr.store.create_task(task)

        callback = AsyncMock()
        mgr.event_callback = callback

        event = AgentEvent(type="task.started", task_id="t3")
        payload = event.model_dump_json().encode()

        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", len(payload)) + payload)
        reader.feed_eof()

        proc = AsyncMock()
        proc.wait = AsyncMock(return_value=None)
        proc.returncode = 0

        handle = MagicMock()
        handle.agent_id = "a3"
        handle.task_id = "t3"
        handle.reader = reader
        handle.proc = proc
        handle.writer = MagicMock()
        handle.writer.close = MagicMock()
        handle.sock = MagicMock()

        mgr._handles["a3"] = handle
        mgr._task_to_agent["t3"] = "a3"

        await mgr._read_events(handle)
        callback.assert_awaited_once()


class TestReadStderr:
    @pytest.mark.asyncio
    async def test_reads_stderr_lines(self):
        store = MagicMock(spec=Store)
        mgr = SubprocessManager(store)

        stderr_reader = asyncio.StreamReader()
        stderr_reader.feed_data(b"warning: something\n")
        stderr_reader.feed_data(b"error: boom\n")
        stderr_reader.feed_eof()

        proc = MagicMock()
        proc.stderr = stderr_reader

        handle = MagicMock()
        handle.agent_id = "a1"
        handle.proc = proc

        await mgr._read_stderr(handle)


class TestSendWithHandle:
    @pytest.fixture
    def mgr_with_handle(self):
        store = MagicMock(spec=Store)
        mgr = SubprocessManager(store)

        handle = MagicMock()
        handle.agent_id = "a1"
        handle.task_id = "t1"
        handle.writer = MagicMock()
        handle.writer.write = MagicMock()
        handle.writer.drain = AsyncMock()
        handle.sock = MagicMock()

        mgr._handles["a1"] = handle
        mgr._task_to_agent["t1"] = "a1"
        return mgr, handle

    @pytest.mark.asyncio
    async def test_send_cancel_success(self, mgr_with_handle):
        mgr, handle = mgr_with_handle
        result = await mgr.send_cancel("t1")
        assert result is True
        handle.writer.write.assert_called_once()
        handle.writer.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_cancel_connection_error(self, mgr_with_handle):
        mgr, handle = mgr_with_handle
        handle.writer.drain = AsyncMock(side_effect=ConnectionError("broken"))
        result = await mgr.send_cancel("t1")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_user_command_success(self, mgr_with_handle):
        mgr, handle = mgr_with_handle
        result = await mgr.send_user_command("t1", "set_coords", {"lat": 1.0})
        assert result is True
        handle.writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_user_command_connection_error(self, mgr_with_handle):
        mgr, handle = mgr_with_handle
        handle.writer.drain = AsyncMock(side_effect=OSError("pipe broken"))
        result = await mgr.send_user_command("t1", "action")
        assert result is False


class TestKillAll:
    @pytest.mark.asyncio
    async def test_kill_all_with_handles(self):
        store = MagicMock(spec=Store)
        mgr = SubprocessManager(store)

        proc = AsyncMock()
        proc.wait = AsyncMock(return_value=0)
        proc.terminate = MagicMock()
        proc.kill = MagicMock()

        handle = MagicMock()
        handle.agent_id = "a1"
        handle.task_id = "t1"
        handle.proc = proc
        handle.writer = MagicMock()
        handle.writer.write = MagicMock()
        handle.writer.drain = AsyncMock()
        handle.writer.close = MagicMock()
        handle.sock = MagicMock()
        handle.reader_task = MagicMock()
        handle.reader_task.cancel = MagicMock()
        handle.stderr_task = MagicMock()
        handle.stderr_task.cancel = MagicMock()

        mgr._handles["a1"] = handle
        mgr._task_to_agent["t1"] = "a1"

        await mgr.kill_all(timeout=0.5)

        assert mgr.active_count == 0
        assert len(mgr._task_to_agent) == 0
        handle.reader_task.cancel.assert_called_once()
        handle.stderr_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_all_process_timeout(self):
        store = MagicMock(spec=Store)
        mgr = SubprocessManager(store)

        proc = AsyncMock()
        proc.wait = AsyncMock(side_effect=asyncio.TimeoutError)
        proc.terminate = MagicMock()
        proc.kill = MagicMock()

        handle = MagicMock()
        handle.agent_id = "a1"
        handle.task_id = "t1"
        handle.proc = proc
        handle.writer = MagicMock()
        handle.writer.write = MagicMock()
        handle.writer.drain = AsyncMock()
        handle.writer.close = MagicMock()
        handle.sock = MagicMock()
        handle.reader_task = None
        handle.stderr_task = None

        mgr._handles["a1"] = handle
        mgr._task_to_agent["t1"] = "a1"

        await mgr.kill_all(timeout=0.1)
        assert mgr.active_count == 0


class TestCleanupHandle:
    def test_cleanup_removes_tracking(self):
        store = MagicMock(spec=Store)
        mgr = SubprocessManager(store)

        mock_handle = MagicMock()
        mock_handle.agent_id = "a1"
        mock_handle.task_id = "t1"
        mock_handle.writer = MagicMock()
        mock_handle.sock = MagicMock()

        mgr._handles["a1"] = mock_handle
        mgr._task_to_agent["t1"] = "a1"

        mgr._cleanup_handle(mock_handle)

        assert "a1" not in mgr._handles
        assert "t1" not in mgr._task_to_agent

    def test_cleanup_handles_close_errors(self):
        store = MagicMock(spec=Store)
        mgr = SubprocessManager(store)

        mock_handle = MagicMock()
        mock_handle.agent_id = "a1"
        mock_handle.task_id = "t1"
        mock_handle.writer = MagicMock()
        mock_handle.writer.close = MagicMock(side_effect=OSError("already closed"))
        mock_handle.sock = MagicMock()
        mock_handle.sock.close = MagicMock(side_effect=OSError("already closed"))

        mgr._handles["a1"] = mock_handle
        mgr._task_to_agent["t1"] = "a1"

        mgr._cleanup_handle(mock_handle)  # should not raise
        assert "a1" not in mgr._handles
