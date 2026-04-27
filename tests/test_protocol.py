from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from switchplane.protocol import (
    AgentCommand,
    AgentEvent,
    CliRequest,
    CliResponse,
    StreamEvent,
)


class TestCliRequest:
    def test_auto_generated_id(self):
        req = CliRequest(method="test")
        assert req.id
        assert len(req.id) == 32

    def test_unique_ids(self):
        r1 = CliRequest(method="a")
        r2 = CliRequest(method="b")
        assert r1.id != r2.id

    def test_default_params(self):
        req = CliRequest(method="test")
        assert req.params == {}

    def test_custom_params(self):
        req = CliRequest(method="submit_task", params={"key": "value"})
        assert req.params == {"key": "value"}

    def test_serialization_round_trip(self):
        req = CliRequest(method="test", params={"x": 1})
        data = req.model_dump_json()
        req2 = CliRequest.model_validate_json(data)
        assert req2.method == "test"
        assert req2.params == {"x": 1}


class TestCliResponse:
    def test_success(self):
        resp = CliResponse(id="abc", ok=True, result={"data": 1})
        assert resp.ok
        assert resp.result == {"data": 1}
        assert resp.error is None

    def test_error(self):
        resp = CliResponse(id="abc", ok=False, error="something failed")
        assert not resp.ok
        assert resp.error == "something failed"
        assert resp.result is None

    def test_serialization_round_trip(self):
        resp = CliResponse(id="xyz", ok=True, result=[1, 2, 3])
        data = resp.model_dump_json()
        resp2 = CliResponse.model_validate_json(data)
        assert resp2.id == "xyz"
        assert resp2.result == [1, 2, 3]


class TestStreamEvent:
    def test_fields(self):
        ts = datetime.now(UTC)
        ev = StreamEvent(task_id="t1", event_type="progress", ts=ts)
        assert ev.task_id == "t1"
        assert ev.event_type == "progress"
        assert ev.payload == {}

    def test_with_payload(self):
        ts = datetime.now(UTC)
        ev = StreamEvent(task_id="t1", event_type="log", payload={"msg": "hi"}, ts=ts)
        assert ev.payload == {"msg": "hi"}


class TestAgentEvent:
    def test_valid_types(self):
        valid_types = [
            "task.started",
            "task.progress",
            "task.completed",
            "task.failed",
            "task.cancelled",
            "checkpoint.save",
            "llm.usage",
            "log",
            "task.command_result",
        ]
        for t in valid_types:
            ev = AgentEvent(type=t, task_id="t1")
            assert ev.type == t

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            AgentEvent(type="invalid", task_id="t1")

    def test_default_timestamp(self):
        ev = AgentEvent(type="task.started", task_id="t1")
        assert ev.ts is not None
        assert ev.ts.tzinfo is not None

    def test_default_payload(self):
        ev = AgentEvent(type="task.started", task_id="t1")
        assert ev.payload == {}

    def test_serialization_round_trip(self):
        ev = AgentEvent(type="task.progress", task_id="t1", payload={"msg": "hi"})
        data = ev.model_dump_json()
        ev2 = AgentEvent.model_validate_json(data)
        assert ev2.type == ev.type
        assert ev2.task_id == ev.task_id
        assert ev2.payload == ev.payload


class TestAgentCommand:
    def test_valid_types(self):
        for t in ["execute_task", "cancel", "shutdown", "user_command"]:
            cmd = AgentCommand(type=t)
            assert cmd.type == t

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            AgentCommand(type="invalid")

    def test_defaults(self):
        cmd = AgentCommand(type="cancel")
        assert cmd.task_id is None
        assert cmd.payload == {}

    def test_with_payload(self):
        cmd = AgentCommand(
            type="user_command",
            task_id="t1",
            payload={"action": "set_coords", "params": {"lat": 1.0}},
        )
        assert cmd.task_id == "t1"
        assert cmd.payload["action"] == "set_coords"
