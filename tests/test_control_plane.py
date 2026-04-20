from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pydantic import Field

from switchplane.agent import AgentSpec
from switchplane.app import Application
from switchplane.control_plane import ControlPlane
from switchplane.daemon import RuntimePaths
from switchplane.protocol import CliRequest
from switchplane.task import Task, TaskStatus


class GreetTask(Task):
    name = "greet"
    description = "Say hello"
    whom: str = Field(description="Who to greet")

    async def run(self, ctx):
        ctx.complete(f"Hello {self.whom}")


class NoParamTask(Task):
    name = "noop"

    async def run(self, ctx):
        pass


@pytest_asyncio.fixture
async def cp(short_tmp):
    paths = RuntimePaths(runtime_dir=short_tmp / "rt")
    paths.runtime_dir.mkdir(parents=True)

    app = Application(name="testapp", runtime_dir=paths.runtime_dir)
    spec = AgentSpec(agent_name="greeter", module_path="test.agents.greeter.agent")
    spec.tasks["greet"] = GreetTask
    spec.tasks["noop"] = NoParamTask
    app.register_agent(spec)

    control_plane = ControlPlane(paths, app)
    await control_plane.start()

    # Mock launch_agent to avoid spawning real subprocesses
    control_plane.subprocess_mgr.launch_agent = AsyncMock(return_value="mock_agent_id")
    control_plane.subprocess_mgr.cancel_task = AsyncMock(return_value=True)
    control_plane.subprocess_mgr.send_user_command = AsyncMock(return_value=True)

    yield control_plane
    await control_plane.shutdown()


async def _request(cp, method, params=None):
    return await cp.handle_request(CliRequest(method=method, params=params or {}))


class TestSubmitTask:
    @pytest.mark.asyncio
    async def test_success(self, cp):
        resp = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "greet",
                "input": {"whom": "Alice"},
            },
        )
        assert resp.ok
        assert "task_id" in resp.result
        assert resp.result["agent_id"] == "mock_agent_id"

    @pytest.mark.asyncio
    async def test_missing_agent(self, cp):
        resp = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "nonexistent",
                "task_name": "greet",
            },
        )
        assert not resp.ok
        assert "not found" in resp.error

    @pytest.mark.asyncio
    async def test_missing_task(self, cp):
        resp = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "nonexistent",
            },
        )
        assert not resp.ok
        assert "not found" in resp.error

    @pytest.mark.asyncio
    async def test_missing_params(self, cp):
        resp = await _request(cp, "submit_task", {})
        assert not resp.ok
        assert "required" in resp.error

    @pytest.mark.asyncio
    async def test_invalid_task_params(self, cp):
        resp = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "greet",
                "input": {},  # missing required 'whom'
            },
        )
        assert not resp.ok
        assert "Invalid parameters" in resp.error

    @pytest.mark.asyncio
    async def test_task_without_params(self, cp):
        resp = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
            },
        )
        assert resp.ok


class TestListTasks:
    @pytest.mark.asyncio
    async def test_empty(self, cp):
        resp = await _request(cp, "list_tasks")
        assert resp.ok
        assert resp.result == []

    @pytest.mark.asyncio
    async def test_with_tasks(self, cp):
        await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
            },
        )
        resp = await _request(cp, "list_tasks")
        assert resp.ok
        assert len(resp.result) == 1

    @pytest.mark.asyncio
    async def test_filter_by_status(self, cp):
        await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
            },
        )
        resp = await _request(cp, "list_tasks", {"status": "completed"})
        assert resp.ok
        assert len(resp.result) == 0


class TestGetTask:
    @pytest.mark.asyncio
    async def test_existing(self, cp):
        submit = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
            },
        )
        task_id = submit.result["task_id"]

        resp = await _request(cp, "get_task", {"task_id": task_id})
        assert resp.ok
        assert resp.result["task"]["task_id"] == task_id
        assert "events" in resp.result

    @pytest.mark.asyncio
    async def test_nonexistent(self, cp):
        resp = await _request(cp, "get_task", {"task_id": "fake"})
        assert not resp.ok
        assert "not found" in resp.error

    @pytest.mark.asyncio
    async def test_missing_task_id(self, cp):
        resp = await _request(cp, "get_task", {})
        assert not resp.ok
        assert "required" in resp.error


class TestCancelTask:
    @pytest.mark.asyncio
    async def test_cancel(self, cp):
        submit = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
            },
        )
        task_id = submit.result["task_id"]

        resp = await _request(cp, "cancel_task", {"task_id": task_id})
        assert resp.ok

    @pytest.mark.asyncio
    async def test_cancel_missing_id(self, cp):
        resp = await _request(cp, "cancel_task", {})
        assert not resp.ok


class TestStatus:
    @pytest.mark.asyncio
    async def test_status(self, cp):
        resp = await _request(cp, "status")
        assert resp.ok
        assert "active_agents" in resp.result
        assert "running_tasks" in resp.result
        assert resp.result["app"] == "testapp"


class TestStop:
    @pytest.mark.asyncio
    async def test_sets_shutdown(self, cp):
        resp = await _request(cp, "stop")
        assert resp.ok
        assert cp._shutdown_event.is_set()


class TestUnknownMethod:
    @pytest.mark.asyncio
    async def test_returns_error(self, cp):
        resp = await _request(cp, "totally_fake_method")
        assert not resp.ok
        assert "Unknown method" in resp.error


class TestGetEventsSince:
    @pytest.mark.asyncio
    async def test_returns_events(self, cp):
        submit = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
            },
        )
        task_id = submit.result["task_id"]

        resp = await _request(
            cp,
            "get_events_since",
            {
                "task_id": task_id,
                "after_event_id": 0,
            },
        )
        assert resp.ok
        assert "events" in resp.result
        assert "status" in resp.result

    @pytest.mark.asyncio
    async def test_missing_task_id(self, cp):
        resp = await _request(cp, "get_events_since", {})
        assert not resp.ok


class TestTaskCommand:
    @pytest.mark.asyncio
    async def test_sends_command(self, cp):
        resp = await _request(
            cp,
            "task_command",
            {
                "task_id": "t1",
                "action": "set_coords",
                "params": {"lat": 1.0},
            },
        )
        assert resp.ok

    @pytest.mark.asyncio
    async def test_missing_fields(self, cp):
        resp = await _request(cp, "task_command", {})
        assert not resp.ok
        assert "required" in resp.error


class TestResumeTask:
    @pytest.mark.asyncio
    async def test_resume_failed_task(self, cp):
        submit = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
            },
        )
        task_id = submit.result["task_id"]

        await cp.store.update_task(task_id, status=TaskStatus.FAILED)

        resp = await _request(cp, "retry_from_checkpoint", {"task_id": task_id})
        assert resp.ok
        assert resp.result["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_retry_running_task_fails(self, cp):
        submit = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
            },
        )
        task_id = submit.result["task_id"]

        resp = await _request(cp, "retry_from_checkpoint", {"task_id": task_id})
        assert not resp.ok
        assert "running" in resp.error.lower() or "status" in resp.error.lower()

    @pytest.mark.asyncio
    async def test_retry_nonexistent(self, cp):
        resp = await _request(cp, "retry_from_checkpoint", {"task_id": "fake"})
        assert not resp.ok

    @pytest.mark.asyncio
    async def test_retry_missing_id(self, cp):
        resp = await _request(cp, "retry_from_checkpoint", {})
        assert not resp.ok


class TestClearTasks:
    @pytest.mark.asyncio
    async def test_clears(self, cp):
        submit = await _request(
            cp,
            "submit_task",
            {
                "agent_name": "greeter",
                "task_name": "noop",
            },
        )
        task_id = submit.result["task_id"]
        await cp.store.update_task(task_id, status=TaskStatus.RUNNING)
        await cp.store.update_task(task_id, status=TaskStatus.COMPLETED)

        resp = await _request(cp, "clear_tasks")
        assert resp.ok
        assert resp.result["deleted"] == 1


class TestListAgents:
    @pytest.mark.asyncio
    async def test_lists_with_task_info(self, cp):
        resp = await _request(cp, "list_agents")
        assert resp.ok
        agents = resp.result
        assert len(agents) == 1
        assert agents[0]["name"] == "greeter"
        assert "greet" in agents[0]["tasks"]
        assert "parameters" in agents[0]["tasks"]["greet"]
