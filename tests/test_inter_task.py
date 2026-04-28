"""Tests for inter-task orchestration: agent requests/responses, parent_task_id flow,
and the plumbing that connects agents to the control plane for task submission."""

import asyncio
import socket
import struct
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from switchplane.agent import AgentSpec
from switchplane.agent_runtime import AgentContext, _listen_for_commands
from switchplane.app import Application
from switchplane.control_plane import ControlPlane
from switchplane.daemon import RuntimePaths
from switchplane.persistence import Store
from switchplane.protocol import (
    AgentCommand,
    AgentEvent,
    AgentRequest,
    AgentResponse,
    CliRequest,
    CliResponse,
)
from switchplane.subprocess_manager import SubprocessManager, _AgentHandle
from switchplane.task import Task, TaskRecord, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recv_message(sock: socket.socket) -> bytes:
    """Read a length-prefixed message from a blocking socket."""
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


def _make_task(
    task_id="t1",
    agent_name="worker",
    task_name="hello",
    status=TaskStatus.PENDING,
    parent_task_id=None,
) -> TaskRecord:
    now = datetime.now(UTC)
    return TaskRecord(
        task_id=task_id,
        agent_name=agent_name,
        task_name=task_name,
        status=status,
        created_at=now,
        updated_at=now,
        parent_task_id=parent_task_id,
    )


class NoParamTask(Task):
    name = "noop"
    description = "No-op task for testing"

    async def run(self, ctx):
        pass


@pytest.fixture
def socketpair():
    cp_sock, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    yield cp_sock, agent_sock
    cp_sock.close()
    agent_sock.close()


@pytest_asyncio.fixture
async def cp(short_tmp):
    """Standalone ControlPlane fixture for inter-task tests."""
    paths = RuntimePaths(runtime_dir=short_tmp / "rt")
    paths.runtime_dir.mkdir(parents=True)

    app = Application(name="testapp", runtime_dir=paths.runtime_dir)
    spec = AgentSpec(agent_name="greeter", module_path="test.agents.greeter.agent")
    spec.tasks["noop"] = NoParamTask
    app.register_agent(spec)

    control_plane = ControlPlane(paths, app)
    await control_plane.start()

    control_plane.subprocess_mgr.launch_agent = AsyncMock(return_value="mock_agent_id")
    control_plane.subprocess_mgr.cancel_task = AsyncMock(return_value=True)

    yield control_plane
    await control_plane.shutdown()


async def _cp_request(cp, method, params=None):
    return await cp.handle_request(CliRequest(method=method, params=params or {}))


# ---------------------------------------------------------------------------
# 1. Protocol models
# ---------------------------------------------------------------------------


class TestAgentRequestResponse:
    def test_agent_request_defaults(self):
        req = AgentRequest(method="get_task", params={"task_id": "t1"})
        assert req.kind == "request"
        assert req.method == "get_task"
        assert req.params == {"task_id": "t1"}
        assert isinstance(req.request_id, str) and len(req.request_id) > 0

    def test_agent_request_unique_ids(self):
        r1 = AgentRequest(method="a")
        r2 = AgentRequest(method="b")
        assert r1.request_id != r2.request_id

    def test_agent_response_success(self):
        resp = AgentResponse(request_id="abc", ok=True, result={"task_id": "t1"})
        assert resp.kind == "response"
        assert resp.request_id == "abc"
        assert resp.ok is True
        assert resp.result == {"task_id": "t1"}
        assert resp.error is None

    def test_agent_response_error(self):
        resp = AgentResponse(request_id="abc", ok=False, error="not found")
        assert resp.ok is False
        assert resp.error == "not found"
        assert resp.result is None

    def test_agent_request_roundtrip(self):
        original = AgentRequest(method="submit_task", params={"agent_name": "greeter"})
        serialized = original.model_dump_json()
        restored = AgentRequest.model_validate_json(serialized)
        assert restored.kind == original.kind
        assert restored.request_id == original.request_id
        assert restored.method == original.method
        assert restored.params == original.params

    def test_agent_response_roundtrip(self):
        original = AgentResponse(
            request_id="xyz123",
            ok=True,
            result={"task_id": "new_task", "agent_id": "a1"},
        )
        serialized = original.model_dump_json()
        restored = AgentResponse.model_validate_json(serialized)
        assert restored.kind == original.kind
        assert restored.request_id == original.request_id
        assert restored.ok == original.ok
        assert restored.result == original.result
        assert restored.error == original.error


# ---------------------------------------------------------------------------
# 2. Persistence parent_task_id
# ---------------------------------------------------------------------------


class TestParentTaskId:
    @pytest.mark.asyncio
    async def test_create_with_parent_task_id(self, store):
        task = _make_task(task_id="child1", parent_task_id="parent1")
        await store.create_task(task)
        result = await store.get_task("child1")
        assert result is not None
        assert result.parent_task_id == "parent1"

    @pytest.mark.asyncio
    async def test_create_without_parent_task_id(self, store):
        task = _make_task(task_id="orphan1")
        await store.create_task(task)
        result = await store.get_task("orphan1")
        assert result is not None
        assert result.parent_task_id is None

    @pytest.mark.asyncio
    async def test_list_tasks_includes_parent_task_id(self, store):
        await store.create_task(_make_task("t1", parent_task_id="p1"))
        await store.create_task(_make_task("t2"))

        tasks = await store.list_tasks()
        by_id = {t.task_id: t for t in tasks}
        assert by_id["t1"].parent_task_id == "p1"
        assert by_id["t2"].parent_task_id is None

    @pytest.mark.asyncio
    async def test_update_task_can_set_parent_task_id(self, store):
        await store.create_task(_make_task("t1"))
        result_before = await store.get_task("t1")
        assert result_before.parent_task_id is None

        await store.update_task("t1", parent_task_id="new_parent")
        result_after = await store.get_task("t1")
        assert result_after.parent_task_id == "new_parent"


# ---------------------------------------------------------------------------
# 3. AgentContext._send_request
# ---------------------------------------------------------------------------


class TestSendRequest:
    @pytest.mark.asyncio
    async def test_send_request_success(self, socketpair):
        cp_sock, agent_sock = socketpair
        ctx = AgentContext(
            task_id="t1", task_name="test", ipc_sock=agent_sock, config={}
        )

        async def do_request():
            return await ctx._send_request("get_task", {"task_id": "abc"})

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.01)

        # Read the request from the CP side
        data = _recv_message(cp_sock)
        req = AgentRequest.model_validate_json(data)
        assert req.kind == "request"
        assert req.method == "get_task"
        assert req.params == {"task_id": "abc"}

        # Simulate control plane resolving the future
        response = AgentResponse(
            request_id=req.request_id,
            ok=True,
            result={"task": {"status": "completed"}},
        )
        future = ctx._pending_requests[req.request_id]
        future.set_result(response)

        result = await task
        assert result == {"task": {"status": "completed"}}
        # Pending request should be cleaned up
        assert req.request_id not in ctx._pending_requests

    @pytest.mark.asyncio
    async def test_send_request_error_raises(self, socketpair):
        cp_sock, agent_sock = socketpair
        ctx = AgentContext(
            task_id="t1", task_name="test", ipc_sock=agent_sock, config={}
        )

        async def do_request():
            return await ctx._send_request("get_task", {"task_id": "bad"})

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.01)

        data = _recv_message(cp_sock)
        req = AgentRequest.model_validate_json(data)

        response = AgentResponse(
            request_id=req.request_id,
            ok=False,
            error="task not found",
        )
        future = ctx._pending_requests[req.request_id]
        future.set_result(response)

        with pytest.raises(RuntimeError, match="Control plane error: task not found"):
            await task

        assert req.request_id not in ctx._pending_requests

    @pytest.mark.asyncio
    async def test_send_request_default_params(self, socketpair):
        """_send_request with no params sends an empty dict."""
        cp_sock, agent_sock = socketpair
        ctx = AgentContext(
            task_id="t1", task_name="test", ipc_sock=agent_sock, config={}
        )

        async def do_request():
            return await ctx._send_request("list_tasks")

        task = asyncio.create_task(do_request())
        await asyncio.sleep(0.01)

        data = _recv_message(cp_sock)
        req = AgentRequest.model_validate_json(data)
        assert req.params == {}

        future = ctx._pending_requests[req.request_id]
        future.set_result(
            AgentResponse(request_id=req.request_id, ok=True, result=[])
        )
        result = await task
        assert result == []


# ---------------------------------------------------------------------------
# 4. AgentContext.submit_task
# ---------------------------------------------------------------------------


class TestSubmitTask:
    @pytest.mark.asyncio
    async def test_submit_task_sends_parent_task_id(self, socketpair):
        cp_sock, agent_sock = socketpair
        ctx = AgentContext(
            task_id="parent1", task_name="orchestrator", ipc_sock=agent_sock, config={}
        )

        async def do_submit():
            return await ctx.submit_task("greeter", "greet", {"whom": "Alice"})

        task = asyncio.create_task(do_submit())
        await asyncio.sleep(0.01)

        data = _recv_message(cp_sock)
        req = AgentRequest.model_validate_json(data)
        assert req.method == "submit_task"
        assert req.params["agent_name"] == "greeter"
        assert req.params["task_name"] == "greet"
        assert req.params["input"] == {"whom": "Alice"}
        assert req.params["parent_task_id"] == "parent1"

        future = ctx._pending_requests[req.request_id]
        future.set_result(
            AgentResponse(
                request_id=req.request_id,
                ok=True,
                result={"task_id": "child1", "agent_id": "a1"},
            )
        )

        result = await task
        assert result == "child1"

    @pytest.mark.asyncio
    async def test_submit_task_default_params(self, socketpair):
        """submit_task with no params passes empty input dict."""
        cp_sock, agent_sock = socketpair
        ctx = AgentContext(
            task_id="parent2", task_name="orch", ipc_sock=agent_sock, config={}
        )

        async def do_submit():
            return await ctx.submit_task("worker", "noop")

        task = asyncio.create_task(do_submit())
        await asyncio.sleep(0.01)

        data = _recv_message(cp_sock)
        req = AgentRequest.model_validate_json(data)
        assert req.params["input"] == {}
        assert req.params["parent_task_id"] == "parent2"

        future = ctx._pending_requests[req.request_id]
        future.set_result(
            AgentResponse(
                request_id=req.request_id,
                ok=True,
                result={"task_id": "child2", "agent_id": "a2"},
            )
        )
        result = await task
        assert result == "child2"


# ---------------------------------------------------------------------------
# 5. _listen_for_commands handles AgentResponse
# ---------------------------------------------------------------------------


class TestListenForCommandsResponse:
    @pytest.mark.asyncio
    async def test_response_resolves_pending_future(self):
        reader = asyncio.StreamReader()

        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(
            task_id="t1", task_name="test", ipc_sock=agent_sock, config={}
        )

        # Set up a pending request
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        ctx._pending_requests["req123"] = future

        # Feed a response message into the reader
        response = AgentResponse(
            request_id="req123", ok=True, result={"data": "hello"}
        )
        payload = response.model_dump_json().encode()
        reader.feed_data(struct.pack(">I", len(payload)) + payload)
        # Feed EOF so _listen_for_commands exits after processing the response
        reader.feed_eof()

        dummy_task = asyncio.create_task(asyncio.sleep(10))
        await _listen_for_commands(reader, ctx, dummy_task)

        assert future.done()
        resolved = future.result()
        assert resolved.ok is True
        assert resolved.result == {"data": "hello"}

        dummy_task.cancel()
        agent_sock.close()

    @pytest.mark.asyncio
    async def test_response_for_unknown_request_id_ignored(self):
        """A response with no matching pending future is silently ignored."""
        reader = asyncio.StreamReader()

        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(
            task_id="t1", task_name="test", ipc_sock=agent_sock, config={}
        )

        response = AgentResponse(
            request_id="nonexistent", ok=True, result={}
        )
        payload = response.model_dump_json().encode()
        reader.feed_data(struct.pack(">I", len(payload)) + payload)
        reader.feed_eof()

        dummy_task = asyncio.create_task(asyncio.sleep(10))
        # Should not raise
        await _listen_for_commands(reader, ctx, dummy_task)

        dummy_task.cancel()
        agent_sock.close()

    @pytest.mark.asyncio
    async def test_response_then_command(self):
        """A response followed by a cancel command: both are handled correctly."""
        reader = asyncio.StreamReader()

        _, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        ctx = AgentContext(
            task_id="t1", task_name="test", ipc_sock=agent_sock, config={}
        )

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        ctx._pending_requests["req456"] = future

        # Feed response
        resp_payload = AgentResponse(
            request_id="req456", ok=True, result={"task_id": "child1"}
        ).model_dump_json().encode()
        reader.feed_data(struct.pack(">I", len(resp_payload)) + resp_payload)

        # Feed cancel command
        cmd_payload = AgentCommand(type="cancel", task_id="t1").model_dump_json().encode()
        reader.feed_data(struct.pack(">I", len(cmd_payload)) + cmd_payload)

        dummy_task = asyncio.create_task(asyncio.sleep(10))
        await _listen_for_commands(reader, ctx, dummy_task)

        # Response was resolved
        assert future.done()
        assert future.result().result == {"task_id": "child1"}

        # Cancel was processed
        assert ctx.is_cancelled

        with pytest.raises(asyncio.CancelledError):
            await dummy_task

        agent_sock.close()


# ---------------------------------------------------------------------------
# 6. SubprocessManager._handle_agent_request
# ---------------------------------------------------------------------------


class TestHandleAgentRequest:
    @pytest.mark.asyncio
    async def test_forwards_to_request_handler(self, tmp_path):
        """_handle_agent_request converts AgentRequest -> CliRequest, calls
        the handler, and writes back an AgentResponse."""
        store = Store(tmp_path / "test.db")
        await store.initialize()

        mock_handler = AsyncMock(
            return_value=CliResponse(
                id="req1", ok=True, result={"task_id": "new_child"}
            )
        )
        mgr = SubprocessManager(store, request_handler=mock_handler)

        # Set up a real socketpair to capture the response
        cp_sock, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        _reader, writer = await asyncio.open_connection(sock=cp_sock)

        handle = _AgentHandle(
            agent_id="a1",
            task_id="t1",
            proc=AsyncMock(),
            sock=agent_sock,
            reader=asyncio.StreamReader(),
            writer=writer,
        )

        raw = {
            "kind": "request",
            "request_id": "req1",
            "method": "submit_task",
            "params": {"agent_name": "greeter", "task_name": "greet"},
        }

        await mgr._handle_agent_request(handle, raw)

        # Verify the handler was called with a CliRequest
        mock_handler.assert_awaited_once()
        cli_req = mock_handler.call_args[0][0]
        assert isinstance(cli_req, CliRequest)
        assert cli_req.id == "req1"
        assert cli_req.method == "submit_task"
        assert cli_req.params == {"agent_name": "greeter", "task_name": "greet"}

        # Read the response from the agent side
        data = _recv_message(agent_sock)
        resp = AgentResponse.model_validate_json(data)
        assert resp.request_id == "req1"
        assert resp.ok is True
        assert resp.result == {"task_id": "new_child"}

        writer.close()
        cp_sock.close()
        agent_sock.close()
        await store.close()

    @pytest.mark.asyncio
    async def test_handler_exception_returns_error(self, tmp_path):
        """If the request handler raises, an error AgentResponse is sent back."""
        store = Store(tmp_path / "test.db")
        await store.initialize()

        mock_handler = AsyncMock(side_effect=RuntimeError("boom"))
        mgr = SubprocessManager(store, request_handler=mock_handler)

        cp_sock, agent_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        _reader, writer = await asyncio.open_connection(sock=cp_sock)

        handle = _AgentHandle(
            agent_id="a1",
            task_id="t1",
            proc=AsyncMock(),
            sock=agent_sock,
            reader=asyncio.StreamReader(),
            writer=writer,
        )

        raw = {
            "kind": "request",
            "request_id": "req2",
            "method": "bad_method",
            "params": {},
        }

        await mgr._handle_agent_request(handle, raw)

        data = _recv_message(agent_sock)
        resp = AgentResponse.model_validate_json(data)
        assert resp.request_id == "req2"
        assert resp.ok is False
        assert "boom" in resp.error

        writer.close()
        cp_sock.close()
        agent_sock.close()
        await store.close()

    @pytest.mark.asyncio
    async def test_no_handler_returns_early(self, tmp_path):
        """If no request_handler is configured, _handle_agent_request returns immediately."""
        store = Store(tmp_path / "test.db")
        await store.initialize()

        mgr = SubprocessManager(store, request_handler=None)

        handle = _AgentHandle(
            agent_id="a1",
            task_id="t1",
            proc=AsyncMock(),
            sock=AsyncMock(),
            reader=asyncio.StreamReader(),
            writer=AsyncMock(),
        )

        raw = {
            "kind": "request",
            "request_id": "req3",
            "method": "whatever",
            "params": {},
        }

        # Should not raise, just return
        await mgr._handle_agent_request(handle, raw)

        await store.close()


# ---------------------------------------------------------------------------
# 7. ControlPlane parent_task_id flow
# ---------------------------------------------------------------------------


class TestControlPlaneParentTaskId:
    @pytest.mark.asyncio
    async def test_submit_with_parent_task_id(self, cp):
        resp = await _cp_request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
                "parent_task_id": "parent_abc",
            },
        )
        assert resp.ok
        task_id = resp.result["task_id"]

        task = await cp.store.get_task(task_id)
        assert task is not None
        assert task.parent_task_id == "parent_abc"

    @pytest.mark.asyncio
    async def test_submit_without_parent_task_id(self, cp):
        resp = await _cp_request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
            },
        )
        assert resp.ok
        task_id = resp.result["task_id"]

        task = await cp.store.get_task(task_id)
        assert task is not None
        assert task.parent_task_id is None

    @pytest.mark.asyncio
    async def test_child_task_visible_in_list(self, cp):
        resp = await _cp_request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
                "parent_task_id": "parent_xyz",
            },
        )
        task_id = resp.result["task_id"]

        list_resp = await _cp_request(cp, "list_tasks")
        assert list_resp.ok
        tasks = list_resp.result
        matched = [t for t in tasks if t["task_id"] == task_id]
        assert len(matched) == 1
        assert matched[0].get("parent_task_id") == "parent_xyz"


# ---------------------------------------------------------------------------
# 8. Persistence: get_child_tasks
# ---------------------------------------------------------------------------


class TestGetChildTasks:
    @pytest.mark.asyncio
    async def test_returns_children(self, store):
        parent = _make_task(task_id="parent")
        child1 = _make_task(task_id="child1", parent_task_id="parent")
        child2 = _make_task(task_id="child2", parent_task_id="parent")
        await store.create_task(parent)
        await store.create_task(child1)
        await store.create_task(child2)

        children = await store.get_child_tasks("parent")
        child_ids = {c.task_id for c in children}
        assert child_ids == {"child1", "child2"}

    @pytest.mark.asyncio
    async def test_no_children(self, store):
        await store.create_task(_make_task(task_id="lonely"))
        children = await store.get_child_tasks("lonely")
        assert children == []

    @pytest.mark.asyncio
    async def test_only_direct_children(self, store):
        """get_child_tasks returns only direct children, not grandchildren."""
        await store.create_task(_make_task(task_id="grandparent"))
        await store.create_task(_make_task(task_id="parent", parent_task_id="grandparent"))
        await store.create_task(_make_task(task_id="grandchild", parent_task_id="parent"))

        children = await store.get_child_tasks("grandparent")
        assert len(children) == 1
        assert children[0].task_id == "parent"


# ---------------------------------------------------------------------------
# 9. Cascade cancellation
# ---------------------------------------------------------------------------


class TestCascadeCancellation:
    @pytest.mark.asyncio
    async def test_cancel_cascades_to_children(self, cp):
        # Submit parent and child
        parent_resp = await _cp_request(
            cp, "submit_task", {"agent_name": "greeter", "task_name": "noop"}
        )
        child_resp = await _cp_request(
            cp, "submit_task", {"agent_name": "greeter", "task_name": "noop"}
        )
        parent_id = parent_resp.result["task_id"]
        child_id = child_resp.result["task_id"]

        # Wire up the parent-child relationship
        await cp.store.update_task(child_id, parent_task_id=parent_id)

        # Cancel the parent
        cancel_resp = await _cp_request(cp, "cancel_task", {"task_id": parent_id})
        assert cancel_resp.ok
        assert cancel_resp.result["children_cancelled"] == 1

        # Verify child status
        child_task = await cp.store.get_task(child_id)
        assert child_task.status == TaskStatus.CANCELLED

        # Verify a task.cancelled event was recorded for the child
        events = await cp.store.get_events(child_id)
        cancel_events = [e for e in events if e["event_type"] == "task.cancelled"]
        assert len(cancel_events) == 1

    @pytest.mark.asyncio
    async def test_cancel_cascades_recursively(self, cp):
        # Submit parent, child, grandchild
        parent_resp = await _cp_request(
            cp, "submit_task", {"agent_name": "greeter", "task_name": "noop"}
        )
        child_resp = await _cp_request(
            cp, "submit_task", {"agent_name": "greeter", "task_name": "noop"}
        )
        grandchild_resp = await _cp_request(
            cp, "submit_task", {"agent_name": "greeter", "task_name": "noop"}
        )
        parent_id = parent_resp.result["task_id"]
        child_id = child_resp.result["task_id"]
        grandchild_id = grandchild_resp.result["task_id"]

        # Wire up relationships
        await cp.store.update_task(child_id, parent_task_id=parent_id)
        await cp.store.update_task(grandchild_id, parent_task_id=child_id)

        # Cancel the parent
        cancel_resp = await _cp_request(cp, "cancel_task", {"task_id": parent_id})
        assert cancel_resp.ok
        assert cancel_resp.result["children_cancelled"] == 2

        # Verify both descendants are cancelled
        child_task = await cp.store.get_task(child_id)
        grandchild_task = await cp.store.get_task(grandchild_id)
        assert child_task.status == TaskStatus.CANCELLED
        assert grandchild_task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_skips_terminal_children(self, cp):
        # Submit parent and two children
        parent_resp = await _cp_request(
            cp, "submit_task", {"agent_name": "greeter", "task_name": "noop"}
        )
        child_a_resp = await _cp_request(
            cp, "submit_task", {"agent_name": "greeter", "task_name": "noop"}
        )
        child_b_resp = await _cp_request(
            cp, "submit_task", {"agent_name": "greeter", "task_name": "noop"}
        )
        parent_id = parent_resp.result["task_id"]
        child_a_id = child_a_resp.result["task_id"]
        child_b_id = child_b_resp.result["task_id"]

        # Wire up relationships
        await cp.store.update_task(child_a_id, parent_task_id=parent_id)
        await cp.store.update_task(child_b_id, parent_task_id=parent_id)

        # Mark child_a as COMPLETED before the cascade
        await cp.store.update_task(child_a_id, status=TaskStatus.COMPLETED)

        # Cancel the parent
        cancel_resp = await _cp_request(cp, "cancel_task", {"task_id": parent_id})
        assert cancel_resp.ok
        assert cancel_resp.result["children_cancelled"] == 1

        # The completed child should remain COMPLETED
        child_a = await cp.store.get_task(child_a_id)
        assert child_a.status == TaskStatus.COMPLETED

        # The non-terminal child should be CANCELLED
        child_b = await cp.store.get_task(child_b_id)
        assert child_b.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_event_cascade_on_failure(self, cp):
        """Cascade triggers automatically when _on_agent_event receives a task.failed event."""
        # Submit parent and child
        parent_resp = await _cp_request(
            cp, "submit_task", {"agent_name": "greeter", "task_name": "noop"}
        )
        child_resp = await _cp_request(
            cp, "submit_task", {"agent_name": "greeter", "task_name": "noop"}
        )
        parent_id = parent_resp.result["task_id"]
        child_id = child_resp.result["task_id"]

        # Wire up the parent-child relationship
        await cp.store.update_task(child_id, parent_task_id=parent_id)

        # Simulate a task.failed event via the _on_agent_event hook
        event = AgentEvent(type="task.failed", task_id=parent_id)
        await cp._on_agent_event(event, event_id=1)

        # The child should have been cascade-cancelled
        child_task = await cp.store.get_task(child_id)
        assert child_task.status == TaskStatus.CANCELLED
