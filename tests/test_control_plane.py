import asyncio
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


class TestConfigReload:
    @pytest.mark.asyncio
    async def test_config_mtime_none_path(self, cp):
        assert cp._config_mtime(None) == 0.0

    @pytest.mark.asyncio
    async def test_config_mtime_missing_file(self, cp):
        from pathlib import Path

        assert cp._config_mtime(Path("/nonexistent/config.toml")) == 0.0

    @pytest.mark.asyncio
    async def test_config_mtime_existing_file(self, cp):
        path = cp.paths.config_path
        path.write_text('[logging]\nlevel = "info"\n')
        mtime = cp._config_mtime(path)
        assert isinstance(mtime, float)
        assert mtime > 0.0

    @pytest.mark.asyncio
    async def test_reload_config_updates_config(self, cp):
        cp.paths.config_path.write_text('[logging]\nlevel = "warning"\n')
        cp._reload_config()
        assert cp.config.logging.level == "warning"

    @pytest.mark.asyncio
    async def test_reload_config_picks_up_new_content(self, cp):
        cp.paths.config_path.write_text('[logging]\nlevel = "error"\n')
        cp._reload_config()
        assert cp.config.logging.level == "error"

        cp.paths.config_path.write_text('[logging]\nlevel = "info"\n')
        cp._reload_config()
        assert cp.config.logging.level == "info"

    @pytest.mark.asyncio
    async def test_watch_task_running_after_start(self, cp):
        assert cp._config_watch_task is not None
        assert not cp._config_watch_task.done()

    @pytest.mark.asyncio
    async def test_watch_broadcasts_on_change(self, cp):
        from unittest.mock import patch

        cp._config_watch_task.cancel()
        try:
            await cp._config_watch_task
        except asyncio.CancelledError:
            pass

        cp.paths.config_path.write_text('[logging]\nlevel = "info"\n')

        broadcast_calls = []
        original_broadcast = cp._broadcast_system_event

        def capture_broadcast(event_type, payload):
            broadcast_calls.append((event_type, payload))
            original_broadcast(event_type, payload)

        cp._broadcast_system_event = capture_broadcast

        _real_sleep = asyncio.sleep

        async def instant_sleep(seconds):
            await _real_sleep(0)

        with patch("asyncio.sleep", side_effect=instant_sleep):
            cp._config_watch_task = asyncio.create_task(cp._watch_config())
            await _real_sleep(0.01)

            cp.paths.config_path.write_text('[logging]\nlevel = "warning"\n')
            await _real_sleep(0.01)

            cp._config_watch_task.cancel()
            try:
                await cp._config_watch_task
            except asyncio.CancelledError:
                pass

        reloaded = [c for c in broadcast_calls if c[0] == "config.reloaded"]
        assert len(reloaded) >= 1
        assert cp.config.logging.level == "warning"
