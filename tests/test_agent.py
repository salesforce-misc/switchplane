from datetime import UTC, datetime

from switchplane.agent import AgentRecord, AgentSpec, AgentStatus


class TestAgentRecord:
    def test_defaults(self):
        rec = AgentRecord(agent_id="a1", agent_name="myagent")
        assert rec.agent_id == "a1"
        assert rec.agent_name == "myagent"
        assert rec.pid is None
        assert rec.status == AgentStatus.IDLE
        assert rec.status == "idle"  # StrEnum is also a str
        assert rec.capabilities_json == "{}"
        assert rec.started_at is None
        assert rec.last_heartbeat is None

    def test_full_record(self):
        now = datetime.now(UTC)
        rec = AgentRecord(
            agent_id="a2",
            agent_name="worker",
            pid=12345,
            status=AgentStatus.RUNNING,
            capabilities_json='{"tools": ["grep"]}',
            started_at=now,
            last_heartbeat=now,
        )
        assert rec.pid == 12345
        assert rec.status == AgentStatus.RUNNING
        assert rec.started_at == now

    def test_serialization_round_trip(self):
        rec = AgentRecord(agent_id="a1", agent_name="test", pid=100, status=AgentStatus.RUNNING)
        data = rec.model_dump_json()
        rec2 = AgentRecord.model_validate_json(data)
        assert rec2.agent_id == rec.agent_id
        assert rec2.pid == rec.pid

    def test_status_from_string(self):
        rec = AgentRecord(agent_id="a1", agent_name="test", status="running")
        assert rec.status == AgentStatus.RUNNING


class TestAgentSpec:
    def test_defaults(self):
        spec = AgentSpec(agent_name="myagent")
        assert spec.agent_name == "myagent"
        assert spec.module_path == ""
        assert spec.mcp_servers == []
        assert spec.tasks == {}

    def test_with_module_path_and_mcp(self):
        spec = AgentSpec(
            agent_name="worker",
            module_path="myapp.agents.worker.agent",
            mcp_servers=["filesystem", "git"],
        )
        assert spec.module_path == "myapp.agents.worker.agent"
        assert spec.mcp_servers == ["filesystem", "git"]

    def test_tasks_dict(self):
        spec = AgentSpec(agent_name="test")
        spec.tasks["hello"] = "SomeTaskClass"
        assert "hello" in spec.tasks

    def test_strip_whitespace(self):
        spec = AgentSpec(agent_name="  myagent  ")
        assert spec.agent_name == "myagent"
