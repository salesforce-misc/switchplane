import json

import click
import pytest
from click.testing import CliRunner

from switchplane.agent import AgentSpec
from switchplane.app import Application
from switchplane.cli import _follow_task, _print_event, build_cli
from switchplane.protocol import CliResponse
from switchplane.task import Task
from switchplane.tui import _parse_kv_args


class TestParseKvArgs:
    def test_key_value(self):
        _, params = _parse_kv_args(["--name", "Alice"], start_index=0)
        assert params == {"name": "Alice"}

    def test_multiple_pairs(self):
        _, params = _parse_kv_args(["--name", "Alice", "--age", "30"], start_index=0)
        assert params == {"name": "Alice", "age": "30"}

    def test_flag_only(self):
        _, params = _parse_kv_args(["--verbose"], start_index=0)
        assert params == {"verbose": True}

    def test_hyphenated_key(self):
        _, params = _parse_kv_args(["--first-name", "Bob"], start_index=0)
        assert params == {"first_name": "Bob"}

    def test_empty(self):
        _, params = _parse_kv_args([], start_index=0)
        assert params == {}

    def test_flag_before_key_value(self):
        _, params = _parse_kv_args(["--flag", "--key", "val"], start_index=0)
        assert params == {"flag": True, "key": "val"}

    def test_consecutive_flags(self):
        _, params = _parse_kv_args(["--a", "--b", "--c"], start_index=0)
        assert params == {"a": True, "b": True, "c": True}

    def test_with_action(self):
        action, params = _parse_kv_args(["set_coords", "--lat", "1.0"])
        assert action == "set_coords"
        assert params == {"lat": "1.0"}

    def test_equals_syntax(self):
        action, params = _parse_kv_args(["cmd", "--key=val"])
        assert action == "cmd"
        assert params == {"key": "val"}

    def test_equals_syntax_preserves_dashes_in_value(self):
        _, params = _parse_kv_args(["--work-item=W-20948280"], start_index=0)
        assert params == {"work_item": "W-20948280"}


class TestPrintEvent:
    def test_progress(self, capsys):
        _print_event(
            {
                "timestamp": "2024-01-01T12:00:00.000Z",
                "event_type": "task.progress",
                "payload": {"message": "Working..."},
            }
        )
        assert "Working..." in capsys.readouterr().out

    def test_started(self, capsys):
        _print_event(
            {
                "timestamp": "2024-01-01T12:00:00.000Z",
                "event_type": "task.started",
                "payload": {},
            }
        )
        assert "started" in capsys.readouterr().out

    def test_completed(self, capsys):
        _print_event(
            {
                "timestamp": "2024-01-01T12:00:00.000Z",
                "event_type": "task.completed",
                "payload": {},
            }
        )
        assert "completed" in capsys.readouterr().out

    def test_cancelled(self, capsys):
        _print_event(
            {
                "timestamp": "2024-01-01T12:00:00.000Z",
                "event_type": "task.cancelled",
                "payload": {},
            }
        )
        assert "cancelled" in capsys.readouterr().out

    def test_failed(self, capsys):
        _print_event(
            {
                "timestamp": "2024-01-01T12:00:00.000Z",
                "event_type": "task.failed",
                "payload": {"error": "boom"},
            }
        )
        out = capsys.readouterr().out
        assert "failed" in out
        assert "boom" in out

    def test_failed_with_traceback(self, capsys):
        _print_event(
            {
                "timestamp": "2024-01-01T12:00:00.000Z",
                "event_type": "task.failed",
                "payload": {"error": "err", "traceback": "line 1\nline 2"},
            }
        )
        out = capsys.readouterr().out
        assert "line 1" in out

    def test_log(self, capsys):
        _print_event(
            {
                "timestamp": "2024-01-01T12:00:00.000Z",
                "event_type": "log",
                "payload": {"level": "info", "message": "hello"},
            }
        )
        out = capsys.readouterr().out
        assert "info" in out
        assert "hello" in out

    def test_command_result(self, capsys):
        _print_event(
            {
                "timestamp": "2024-01-01T12:00:00.000Z",
                "event_type": "task.command_result",
                "payload": {"action": "set_x", "result": {"x": 1}},
            }
        )
        out = capsys.readouterr().out
        assert "set_x" in out

    def test_unknown_type(self, capsys):
        _print_event(
            {
                "timestamp": "2024-01-01T12:00:00.000Z",
                "event_type": "custom.event",
                "payload": {"data": 1},
            }
        )
        out = capsys.readouterr().out
        assert "custom.event" in out

    def test_no_timestamp_in_output(self, capsys):
        _print_event(
            {
                "timestamp": "2024-01-15T14:30:45.123Z",
                "event_type": "task.started",
                "payload": {},
            }
        )
        out = capsys.readouterr().out
        # Timestamps are no longer rendered
        assert "14:30:45" not in out
        assert "Task started" in out


class TestBuildCli:
    @pytest.fixture
    def cli(self, tmp_path):
        app = Application(name="testcli", runtime_dir=tmp_path / ".testcli")
        return build_cli(app)

    def test_returns_click_group(self, cli):
        assert isinstance(cli, click.Group)

    def test_has_subcommands(self, cli):
        assert "run" in cli.commands
        assert "runtime" in cli.commands
        assert "task" in cli.commands
        assert "agent" in cli.commands

    def test_runtime_subcommands(self, cli):
        runtime = cli.commands["runtime"]
        assert "start" in runtime.commands
        assert "stop" in runtime.commands
        assert "status" in runtime.commands

    def test_task_subcommands(self, cli):
        task = cli.commands["task"]
        assert "list" in task.commands
        assert "show" in task.commands
        assert "cancel" in task.commands
        assert "follow" in task.commands
        assert "retry" in task.commands
        assert "clear" in task.commands

    def test_runtime_status_not_running(self, cli, monkeypatch):
        monkeypatch.setattr("switchplane.cli.is_alive", lambda path: False)
        runner = CliRunner()
        result = runner.invoke(cli, ["runtime", "status"])
        assert "not running" in result.output

    def test_help(self, cli):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_no_subcommand_shows_help(self, cli):
        runner = CliRunner()
        result = runner.invoke(cli, [])
        assert result.exit_code == 0


class _MockClient:
    """Mock ControlPlaneClient that returns pre-configured responses."""

    def __init__(self, responses: dict):
        self._responses = responses

    def __call__(self, sock_path):
        self._sock_path = sock_path
        return self

    def connect(self):
        pass

    def close(self):
        pass

    def send(self, request):
        method = request.method
        if method in self._responses:
            resp_data = self._responses[method]
            if callable(resp_data):
                return resp_data(request)
            return CliResponse(id=request.id, ok=True, result=resp_data)
        return CliResponse(id=request.id, ok=True, result={})

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()


class TestCliWithMockedDaemon:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        app = Application(name="testcli", runtime_dir=tmp_path / ".testcli")

        class SimpleTask(Task):
            name = "hello"
            description = "Says hello"

            async def run(self, ctx):
                pass

        spec = AgentSpec(agent_name="greeter", module_path="test.agent")
        spec.tasks["hello"] = SimpleTask
        app.register_agent(spec)

        monkeypatch.setattr("switchplane.cli.is_alive", lambda path: True)
        monkeypatch.setattr("switchplane.cli.start_daemon", lambda paths, app: None)

        responses = {}
        mock_client = _MockClient(responses)
        monkeypatch.setattr("switchplane.cli.ControlPlaneClient", mock_client)

        cli = build_cli(app)
        return cli, responses

    def test_runtime_status_running(self, setup):
        cli, responses = setup
        responses["status"] = {
            "active_agents": 1,
            "active_connections": 2,
            "running_tasks": 3,
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["runtime", "status"])
        assert "running" in result.output
        assert "Active agents: 1" in result.output

    def test_agent_list(self, setup):
        cli, responses = setup
        responses["list_agents"] = [
            {
                "name": "greeter",
                "tasks": {
                    "hello": {
                        "description": "Says hello",
                        "mode": "ephemeral",
                        "parameters": {
                            "name": {
                                "type": "str",
                                "required": True,
                                "description": "Who to greet",
                            }
                        },
                    }
                },
            }
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["agent", "list"])
        assert "greeter" in result.output
        assert "hello" in result.output

    def test_agent_list_empty(self, setup):
        cli, responses = setup
        responses["list_agents"] = []
        runner = CliRunner()
        result = runner.invoke(cli, ["agent", "list"])
        assert "No agents" in result.output

    def test_agent_list_with_commands(self, setup):
        cli, responses = setup
        responses["list_agents"] = [
            {
                "name": "worker",
                "tasks": {
                    "watch": {
                        "description": "Watch stuff",
                        "mode": "long_running",
                        "commands": ["set_coords", "pause"],
                    }
                },
            }
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["agent", "list"])
        assert "[long_running]" in result.output
        assert "set_coords" in result.output

    def test_task_list(self, setup):
        cli, responses = setup
        responses["list_tasks"] = [
            {
                "task_id": "abc123",
                "agent_name": "greeter",
                "task_name": "hello",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "list"])
        assert "abc123" in result.output
        assert "completed" in result.output

    def test_task_list_empty(self, setup):
        cli, responses = setup
        responses["list_tasks"] = []
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "list"])
        assert "No tasks" in result.output

    def test_task_list_with_filter(self, setup):
        cli, responses = setup
        responses["list_tasks"] = []
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "list", "--status", "running"])
        assert result.exit_code == 0

    def test_task_list_error(self, setup):
        cli, responses = setup
        responses["list_tasks"] = lambda req: CliResponse(id=req.id, ok=False, error="db error")
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "list"])
        assert "Error" in result.output

    def test_task_show(self, setup):
        cli, responses = setup
        responses["get_task"] = {
            "task": {
                "task_id": "t1",
                "agent_name": "greeter",
                "task_name": "hello",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:01Z",
                "input_json": '{"name": "Alice"}',
                "result_json": '{"greeting": "Hello Alice"}',
            },
            "events": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "event_type": "task.started",
                    "payload": {},
                }
            ],
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "show", "t1"])
        assert "t1" in result.output
        assert "greeter" in result.output
        assert "Alice" in result.output

    def test_task_show_with_error(self, setup):
        cli, responses = setup
        responses["get_task"] = {
            "task": {
                "task_id": "t1",
                "agent_name": "greeter",
                "task_name": "hello",
                "status": "failed",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:01Z",
                "error_json": '{"error": "boom"}',
            },
            "events": [],
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "show", "t1"])
        assert "boom" in result.output

    def test_task_show_not_found(self, setup):
        cli, responses = setup
        responses["get_task"] = lambda req: CliResponse(id=req.id, ok=False, error="not found")
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "show", "fake"])
        assert "Error" in result.output

    def test_task_cancel(self, setup):
        cli, responses = setup
        responses["cancel_task"] = {"cancelled": True}
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "cancel", "t1"])
        assert "cancelled" in result.output

    def test_task_cancel_not_found(self, setup):
        cli, responses = setup
        responses["cancel_task"] = {"cancelled": False}
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "cancel", "t1"])
        assert "not running" in result.output.lower() or "not found" in result.output.lower()

    def test_task_clear(self, setup):
        cli, responses = setup
        responses["clear_tasks"] = {"deleted": 5}
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "clear"])
        assert "5" in result.output

    def test_run_task_detached(self, setup):
        cli, responses = setup
        responses["submit_task"] = {"task_id": "new_task_123", "agent_id": "a1"}
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "greeter", "hello", "-d"])
        assert "new_task_123" in result.output

    def test_run_task_error(self, setup):
        cli, responses = setup
        responses["submit_task"] = lambda req: CliResponse(id=req.id, ok=False, error="bad params")
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "greeter", "hello", "-d"])
        assert result.exit_code != 0

    def test_run_task_with_params(self, setup):
        cli, responses = setup
        responses["submit_task"] = {"task_id": "t99", "agent_id": "a1"}
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "greeter", "hello", "-d", "--name", "Bob"])
        assert "t99" in result.output

    def test_task_dispatch_command(self, setup):
        cli, responses = setup
        responses["task_command"] = {"sent": True}
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "t1", "set_coords", "--lat", "1.0"])
        assert "sent" in result.output.lower()

    def test_task_dispatch_error(self, setup):
        cli, responses = setup
        responses["task_command"] = lambda req: CliResponse(id=req.id, ok=False, error="not running")
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "t1", "do_thing"])
        assert result.exit_code != 0

    def test_task_dispatch_with_equals(self, setup):
        cli, responses = setup
        responses["task_command"] = {"sent": True}
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "t1", "cmd", "--key=val"])
        assert "sent" in result.output.lower()

    def test_task_retry_detached(self, setup):
        cli, responses = setup
        responses["retry_from_checkpoint"] = {"task_id": "t1", "agent_id": "a1"}
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "retry", "t1", "-d"])
        assert "retried" in result.output

    def test_task_retry_error(self, setup):
        cli, responses = setup
        responses["retry_from_checkpoint"] = lambda req: CliResponse(id=req.id, ok=False, error="cannot retry")
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "retry", "t1"])
        assert result.exit_code != 0


class TestFollowTask:
    def test_follow_until_complete(self):
        call_count = 0

        def mock_send(method, params=None):
            nonlocal call_count
            call_count += 1
            if method == "get_events_since":
                if call_count == 1:
                    return CliResponse(
                        id="1",
                        ok=True,
                        result={
                            "events": [
                                {
                                    "event_id": 1,
                                    "timestamp": "2024-01-01T12:00:00Z",
                                    "event_type": "task.started",
                                    "payload": {},
                                }
                            ],
                            "status": "running",
                        },
                    )
                return CliResponse(
                    id="2",
                    ok=True,
                    result={
                        "events": [
                            {
                                "event_id": 2,
                                "timestamp": "2024-01-01T12:00:01Z",
                                "event_type": "task.completed",
                                "payload": {},
                            }
                        ],
                        "status": "completed",
                    },
                )
            elif method == "get_task":
                return CliResponse(
                    id="3",
                    ok=True,
                    result={
                        "task": {
                            "task_id": "t1",
                            "status": "completed",
                            "result_json": '{"answer": 42}',
                        }
                    },
                )
            return CliResponse(id="0", ok=True, result={})

        _follow_task("t1", mock_send)

    def test_follow_failed_task(self):
        call_count = 0

        def mock_send(method, params=None):
            nonlocal call_count
            call_count += 1
            if method == "get_events_since":
                return CliResponse(
                    id="1",
                    ok=True,
                    result={
                        "events": [],
                        "status": "failed",
                    },
                )
            elif method == "get_task":
                return CliResponse(
                    id="2",
                    ok=True,
                    result={
                        "task": {
                            "task_id": "t1",
                            "status": "failed",
                            "error_json": json.dumps({"error": "boom", "traceback": "tb"}),
                        }
                    },
                )
            return CliResponse(id="0", ok=True, result={})

        _follow_task("t1", mock_send)

    def test_follow_error_response(self):
        def mock_send(method, params=None):
            return CliResponse(id="1", ok=False, error="connection lost")

        _follow_task("t1", mock_send)
