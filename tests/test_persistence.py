from datetime import UTC, datetime

import pytest

from switchplane.agent import AgentRecord, AgentStatus
from switchplane.persistence import Store
from switchplane.task import TaskRecord, TaskStatus


def _make_task(task_id="t1", agent_name="worker", task_name="hello", status=TaskStatus.PENDING) -> TaskRecord:
    now = datetime.now(UTC)
    return TaskRecord(
        task_id=task_id,
        agent_name=agent_name,
        task_name=task_name,
        status=status,
        created_at=now,
        updated_at=now,
    )


def _make_agent(agent_id="a1", agent_name="worker") -> AgentRecord:
    return AgentRecord(
        agent_id=agent_id,
        agent_name=agent_name,
        pid=1234,
        status=AgentStatus.RUNNING,
        started_at=datetime.now(UTC),
        last_heartbeat=datetime.now(UTC),
    )


class TestStoreNotInitialized:
    @pytest.mark.asyncio
    async def test_create_task_raises(self, tmp_path):
        s = Store(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await s.create_task(_make_task())

    @pytest.mark.asyncio
    async def test_get_task_raises(self, tmp_path):
        s = Store(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await s.get_task("t1")

    @pytest.mark.asyncio
    async def test_list_tasks_raises(self, tmp_path):
        s = Store(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await s.list_tasks()

    @pytest.mark.asyncio
    async def test_update_task_raises(self, tmp_path):
        s = Store(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await s.update_task("t1", status=TaskStatus.RUNNING)


class TestTaskCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get(self, store):
        task = _make_task()
        await store.create_task(task)
        result = await store.get_task("t1")
        assert result is not None
        assert result.task_id == "t1"
        assert result.agent_name == "worker"
        assert result.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        result = await store.get_task("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_status(self, store):
        await store.create_task(_make_task())
        await store.update_task("t1", status=TaskStatus.RUNNING)
        result = await store.get_task("t1")
        assert result.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_update_result(self, store):
        await store.create_task(_make_task())
        await store.update_task("t1", status=TaskStatus.COMPLETED, result_json='{"answer": 42}')
        result = await store.get_task("t1")
        assert result.status == TaskStatus.COMPLETED
        assert result.result_json == '{"answer": 42}'

    @pytest.mark.asyncio
    async def test_update_auto_sets_updated_at(self, store):
        task = _make_task()
        await store.create_task(task)
        original_updated = task.updated_at
        await store.update_task("t1", status=TaskStatus.RUNNING)
        result = await store.get_task("t1")
        assert result.updated_at >= original_updated

    @pytest.mark.asyncio
    async def test_list_all(self, store):
        await store.create_task(_make_task("t1"))
        await store.create_task(_make_task("t2"))
        await store.create_task(_make_task("t3"))
        tasks = await store.list_tasks()
        assert len(tasks) == 3

    @pytest.mark.asyncio
    async def test_list_by_status(self, store):
        await store.create_task(_make_task("t1", status=TaskStatus.PENDING))
        await store.create_task(_make_task("t2", status=TaskStatus.RUNNING))
        await store.create_task(_make_task("t3", status=TaskStatus.COMPLETED))

        pending = await store.list_tasks(status=TaskStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].task_id == "t1"

        running = await store.list_tasks(status=TaskStatus.RUNNING)
        assert len(running) == 1


class TestAgentCRUD:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self, store):
        agent = _make_agent()
        await store.upsert_agent(agent)
        result = await store.get_agent("a1")
        assert result is not None
        assert result.agent_name == "worker"
        assert result.pid == 1234

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        result = await store.get_agent("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert_replaces(self, store):
        await store.upsert_agent(_make_agent())
        updated = AgentRecord(
            agent_id="a1",
            agent_name="worker",
            pid=5678,
            status=AgentStatus.IDLE,
        )
        await store.upsert_agent(updated)
        result = await store.get_agent("a1")
        assert result.pid == 5678
        assert result.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_list_agents(self, store):
        await store.upsert_agent(_make_agent("a1", "alpha"))
        await store.upsert_agent(_make_agent("a2", "beta"))
        agents = await store.list_agents()
        assert len(agents) == 2
        names = {a.agent_name for a in agents}
        assert names == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_delete_agent(self, store):
        await store.upsert_agent(_make_agent())
        await store.delete_agent("a1")
        result = await store.get_agent("a1")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_all_agents(self, store):
        await store.upsert_agent(_make_agent("a1", "alpha"))
        await store.upsert_agent(_make_agent("a2", "beta"))
        count = await store.clear_all_agents()
        assert count == 2
        assert await store.list_agents() == []

    @pytest.mark.asyncio
    async def test_clear_all_agents_when_empty(self, store):
        assert await store.clear_all_agents() == 0


class TestEvents:
    @pytest.mark.asyncio
    async def test_add_and_get(self, store):
        await store.create_task(_make_task())
        await store.add_event("t1", "task.started", {})
        await store.add_event("t1", "task.progress", {"message": "working"})

        events = await store.get_events("t1")
        assert len(events) == 2
        assert events[0]["event_type"] == "task.started"
        assert events[1]["event_type"] == "task.progress"
        assert events[1]["payload"] == {"message": "working"}

    @pytest.mark.asyncio
    async def test_events_ordered_by_timestamp(self, store):
        await store.create_task(_make_task())
        await store.add_event("t1", "first", {})
        await store.add_event("t1", "second", {})
        await store.add_event("t1", "third", {})

        events = await store.get_events("t1")
        types = [e["event_type"] for e in events]
        assert types == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_get_events_since(self, store):
        await store.create_task(_make_task())
        await store.add_event("t1", "first", {})
        await store.add_event("t1", "second", {})
        await store.add_event("t1", "third", {})

        events = await store.get_events("t1")
        first_id = events[0]["event_id"]

        since = await store.get_events_since("t1", after_event_id=first_id)
        assert len(since) == 2
        assert since[0]["event_type"] == "second"

    @pytest.mark.asyncio
    async def test_empty_events(self, store):
        events = await store.get_events("nonexistent")
        assert events == []


class TestRecoverOrphanedTasks:
    @pytest.mark.asyncio
    async def test_recovers_running_and_pending(self, store):
        await store.create_task(_make_task("t1", status=TaskStatus.RUNNING))
        await store.create_task(_make_task("t2", status=TaskStatus.PENDING))
        await store.create_task(_make_task("t3", status=TaskStatus.COMPLETED))

        count = await store.recover_orphaned_tasks()
        assert count == 2

        t1 = await store.get_task("t1")
        assert t1.status == TaskStatus.FAILED
        t2 = await store.get_task("t2")
        assert t2.status == TaskStatus.FAILED
        t3 = await store.get_task("t3")
        assert t3.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_no_orphans(self, store):
        await store.create_task(_make_task("t1", status=TaskStatus.COMPLETED))
        count = await store.recover_orphaned_tasks()
        assert count == 0

    @pytest.mark.asyncio
    async def test_recovers_interrupted(self, store):
        await store.create_task(_make_task("t_int", status=TaskStatus.INTERRUPTED))
        await store.create_task(_make_task("t_done", status=TaskStatus.COMPLETED))

        count = await store.recover_orphaned_tasks()
        assert count == 1

        t_int = await store.get_task("t_int")
        assert t_int.status == TaskStatus.FAILED
        t_done = await store.get_task("t_done")
        assert t_done.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_adds_failure_events(self, store):
        await store.create_task(_make_task("t1", status=TaskStatus.RUNNING))
        await store.recover_orphaned_tasks()
        events = await store.get_events("t1")
        assert len(events) == 1
        assert events[0]["event_type"] == "task.failed"


class TestClearTerminalTasks:
    @pytest.mark.asyncio
    async def test_soft_clears_completed_failed_cancelled(self, store):
        """clear marks terminal tasks CLEARED; rows and data are preserved."""
        await store.create_task(_make_task("t1", status=TaskStatus.COMPLETED))
        await store.create_task(_make_task("t2", status=TaskStatus.FAILED))
        await store.create_task(_make_task("t3", status=TaskStatus.CANCELLED))
        await store.create_task(_make_task("t4", status=TaskStatus.RUNNING))

        await store.add_event("t1", "task.completed", {})

        count = await store.clear_terminal_tasks()
        assert count == 3

        # Rows still exist, just flipped to CLEARED.
        t1 = await store.get_task("t1")
        assert t1 is not None
        assert t1.status == TaskStatus.CLEARED
        assert (await store.get_task("t4")).status == TaskStatus.RUNNING

        # Data preserved until purge.
        assert len(await store.get_events("t1")) == 1

    @pytest.mark.asyncio
    async def test_clear_is_idempotent(self, store):
        """Already-cleared tasks are not re-cleared."""
        await store.create_task(_make_task("t1", status=TaskStatus.COMPLETED))
        assert await store.clear_terminal_tasks() == 1
        assert await store.clear_terminal_tasks() == 0

    @pytest.mark.asyncio
    async def test_nothing_to_clear(self, store):
        await store.create_task(_make_task("t1", status=TaskStatus.RUNNING))
        count = await store.clear_terminal_tasks()
        assert count == 0


class TestPurgeTerminalTasks:
    @pytest.mark.asyncio
    async def test_purges_terminal_and_cleared(self, store):
        await store.create_task(_make_task("t1", status=TaskStatus.COMPLETED))
        await store.create_task(_make_task("t2", status=TaskStatus.FAILED))
        await store.create_task(_make_task("t3", status=TaskStatus.CLEARED))
        await store.create_task(_make_task("t4", status=TaskStatus.RUNNING))
        await store.add_event("t1", "task.completed", {})

        count = await store.purge_terminal_tasks()
        assert count == 3

        assert await store.get_task("t1") is None
        assert await store.get_task("t3") is None
        assert await store.get_task("t4") is not None
        assert await store.get_events("t1") == []

    @pytest.mark.asyncio
    async def test_nothing_to_purge(self, store):
        await store.create_task(_make_task("t1", status=TaskStatus.RUNNING))
        assert await store.purge_terminal_tasks() == 0

    @pytest.mark.asyncio
    async def test_purges_checkpoints_by_recorded_thread(self, tmp_path):
        """Checkpoints written under a non-task_id thread are still purged."""
        from switchplane.checkpoint import setup_tables

        s = Store(tmp_path / "cp.db")
        await s.initialize()
        try:
            await setup_tables(s.connection)
            await s.create_task(_make_task("task-abc", status=TaskStatus.COMPLETED))
            # Task checkpointed under a work-item thread id, not its task_id.
            await s.connection.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) VALUES (?, '', 'c1')",
                ("W-123",),
            )
            await s.connection.commit()
            await s.record_checkpoint_thread("W-123", "task-abc")

            await s.purge_terminal_tasks()

            cursor = await s.connection.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'W-123'")
            assert (await cursor.fetchone())[0] == 0
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_orphan_sweep_removes_unmapped_checkpoints(self, tmp_path):
        """Checkpoints with no mapping to any surviving task are swept on purge."""
        from switchplane.checkpoint import setup_tables

        s = Store(tmp_path / "orphan.db")
        await s.initialize()
        try:
            await setup_tables(s.connection)
            # A terminal task to trigger purge, unrelated to the orphan.
            await s.create_task(_make_task("t1", status=TaskStatus.COMPLETED))
            # Legacy checkpoint with no checkpoint_threads mapping at all.
            await s.connection.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) VALUES (?, '', 'c1')",
                ("W-orphan",),
            )
            await s.connection.commit()

            await s.purge_terminal_tasks()

            cursor = await s.connection.execute("SELECT COUNT(*) FROM checkpoints")
            assert (await cursor.fetchone())[0] == 0
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_keeps_checkpoints_of_surviving_tasks(self, tmp_path):
        """A checkpoint mapped to a still-running task is not swept."""
        from switchplane.checkpoint import setup_tables

        s = Store(tmp_path / "survive.db")
        await s.initialize()
        try:
            await setup_tables(s.connection)
            await s.create_task(_make_task("done", status=TaskStatus.COMPLETED))
            await s.create_task(_make_task("live", status=TaskStatus.RUNNING))
            await s.connection.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) VALUES (?, '', 'c1')",
                ("live-thread",),
            )
            await s.connection.commit()
            await s.record_checkpoint_thread("live-thread", "live")

            await s.purge_terminal_tasks()

            cursor = await s.connection.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'live-thread'")
            assert (await cursor.fetchone())[0] == 1
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_purge_without_checkpoint_tables(self, tmp_path):
        """purge works even if checkpoint tables were never created."""
        s = Store(tmp_path / "no_cp.db")
        await s.initialize()
        try:
            await s.create_task(_make_task("t1", status=TaskStatus.COMPLETED))
            count = await s.purge_terminal_tasks()
            assert count == 1
        finally:
            await s.close()


class TestStoreConnection:
    def test_connection_raises_when_not_initialized(self, tmp_path):
        s = Store(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = s.connection

    @pytest.mark.asyncio
    async def test_connection_returns_db(self, store):
        conn = store.connection
        assert conn is not None


class TestStoreClose:
    @pytest.mark.asyncio
    async def test_close(self, tmp_path):
        s = Store(tmp_path / "test.db")
        await s.initialize()
        await s.close()
        assert s._db is None

    @pytest.mark.asyncio
    async def test_double_close(self, tmp_path):
        s = Store(tmp_path / "test.db")
        await s.initialize()
        await s.close()
        await s.close()  # should not raise
