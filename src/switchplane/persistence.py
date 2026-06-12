"""SQLite persistence layer for the control plane."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from switchplane.agent import AgentRecord
from switchplane.task import TaskRecord, TaskStatus

# Statuses eligible to be cleared from view (terminal, but not already cleared).
_CLEARABLE_STATUSES = (
    TaskStatus.COMPLETED.value,
    TaskStatus.FAILED.value,
    TaskStatus.CANCELLED.value,
)

# Statuses eligible for hard deletion by purge — clearable plus already-cleared.
_PURGEABLE_STATUSES = (*_CLEARABLE_STATUSES, TaskStatus.CLEARED.value)


class Store:
    """Async SQLite store for control plane persistence."""

    def __init__(self, db_path: Path):
        """Initialize the store with a database path."""
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def connection(self) -> aiosqlite.Connection:
        """Return the active database connection.

        Raises RuntimeError if the store has not been initialized.
        """
        if self._db is None:
            raise RuntimeError("Store not initialized")
        return self._db

    async def initialize(self) -> None:
        """Open connection and create tables if needed."""
        self._db = await aiosqlite.connect(self.db_path)
        try:
            # Enable WAL mode for better concurrency
            await self._db.execute("PRAGMA journal_mode=WAL")

            # Create tables if they don't exist
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    pid INTEGER,
                    status TEXT NOT NULL DEFAULT 'idle',
                    capabilities_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT,
                    last_heartbeat TEXT
                )
            """)

            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    input_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    workflow_identity_json TEXT,
                    checkpoint_metadata_json TEXT,
                    parent_task_id TEXT
                )
            """)

            # Migration: add parent_task_id if upgrading from older schema
            try:
                await self._db.execute("ALTER TABLE tasks ADD COLUMN parent_task_id TEXT")
            except Exception:
                pass  # Column already exists

            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                )
            """)

            # Maps a checkpointer thread_id to the task that wrote it. Tasks may
            # use any thread_id (LangGraph's, e.g. a stable work-item key), so we
            # cannot assume thread_id == task_id when purging checkpoints.
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_threads (
                    thread_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    PRIMARY KEY (thread_id, task_id)
                )
            """)

            await self._db.commit()
        except BaseException:
            await self._db.close()
            self._db = None
            raise

    # Task methods

    async def create_task(self, task: TaskRecord) -> None:
        """Insert a new task record."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        await self._db.execute(
            """
            INSERT INTO tasks (
                task_id, agent_name, task_name, status,
                input_json, result_json, error_json,
                created_at, updated_at,
                workflow_identity_json, checkpoint_metadata_json,
                parent_task_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task.task_id,
                task.agent_name,
                task.task_name,
                task.status.value,
                task.input_json,
                task.result_json,
                task.error_json,
                task.created_at.isoformat(),
                task.updated_at.isoformat(),
                task.workflow_identity_json,
                task.checkpoint_metadata_json,
                task.parent_task_id,
            ),
        )
        await self._db.commit()

    async def update_task(self, task_id: str, **fields: Any) -> None:
        """Update specific fields of a task, auto-setting updated_at."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        # Build the SET clause dynamically
        set_parts = []
        values = []

        for key, value in fields.items():
            if key == "status" and isinstance(value, TaskStatus):
                set_parts.append(f"{key} = ?")
                values.append(value.value)
            elif key in (
                "input_json",
                "result_json",
                "error_json",
                "workflow_identity_json",
                "checkpoint_metadata_json",
            ):
                # These are already JSON strings, don't double-encode
                set_parts.append(f"{key} = ?")
                values.append(value)
            elif key in ("created_at", "updated_at") and isinstance(value, datetime):
                set_parts.append(f"{key} = ?")
                values.append(value.isoformat())
            else:
                set_parts.append(f"{key} = ?")
                values.append(value)

        # Always update updated_at
        set_parts.append("updated_at = ?")
        values.append(datetime.now(UTC).isoformat())

        # Add task_id for WHERE clause
        values.append(task_id)

        query = f"UPDATE tasks SET {', '.join(set_parts)} WHERE task_id = ?"
        await self._db.execute(query, values)
        await self._db.commit()

    async def get_task(self, task_id: str) -> TaskRecord | None:
        """Retrieve a task by ID, returning a hydrated model."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        cursor = await self._db.execute(
            """
            SELECT task_id, agent_name, task_name, status,
                   input_json, result_json, error_json,
                   created_at, updated_at,
                   workflow_identity_json, checkpoint_metadata_json,
                   parent_task_id
            FROM tasks WHERE task_id = ?
        """,
            (task_id,),
        )

        row = await cursor.fetchone()
        if not row:
            return None

        return TaskRecord(
            task_id=row[0],
            agent_name=row[1],
            task_name=row[2],
            status=TaskStatus(row[3]),
            input_json=row[4],
            result_json=row[5],
            error_json=row[6],
            created_at=datetime.fromisoformat(row[7]),
            updated_at=datetime.fromisoformat(row[8]),
            workflow_identity_json=row[9],
            checkpoint_metadata_json=row[10],
            parent_task_id=row[11],
        )

    async def list_tasks(self, status: TaskStatus | None = None) -> list[TaskRecord]:
        """List tasks with optional status filter."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        if status:
            cursor = await self._db.execute(
                """
                SELECT task_id, agent_name, task_name, status,
                       input_json, result_json, error_json,
                       created_at, updated_at,
                       workflow_identity_json, checkpoint_metadata_json,
                       parent_task_id
                FROM tasks WHERE status = ?
                ORDER BY created_at DESC
            """,
                (status.value,),
            )
        else:
            cursor = await self._db.execute("""
                SELECT task_id, agent_name, task_name, status,
                       input_json, result_json, error_json,
                       created_at, updated_at,
                       workflow_identity_json, checkpoint_metadata_json,
                       parent_task_id
                FROM tasks ORDER BY created_at DESC
            """)

        rows = await cursor.fetchall()
        tasks = []
        for row in rows:
            tasks.append(
                TaskRecord(
                    task_id=row[0],
                    agent_name=row[1],
                    task_name=row[2],
                    status=TaskStatus(row[3]),
                    input_json=row[4],
                    result_json=row[5],
                    error_json=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                    updated_at=datetime.fromisoformat(row[8]),
                    workflow_identity_json=row[9],
                    checkpoint_metadata_json=row[10],
                    parent_task_id=row[11],
                )
            )

        return tasks

    async def get_child_tasks(self, parent_task_id: str) -> list[TaskRecord]:
        """Get all direct children of a task."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        cursor = await self._db.execute(
            """
            SELECT task_id, agent_name, task_name, status,
                   input_json, result_json, error_json,
                   created_at, updated_at,
                   workflow_identity_json, checkpoint_metadata_json,
                   parent_task_id
            FROM tasks WHERE parent_task_id = ?
            ORDER BY created_at DESC
        """,
            (parent_task_id,),
        )

        rows = await cursor.fetchall()
        return [
            TaskRecord(
                task_id=row[0],
                agent_name=row[1],
                task_name=row[2],
                status=TaskStatus(row[3]),
                input_json=row[4],
                result_json=row[5],
                error_json=row[6],
                created_at=datetime.fromisoformat(row[7]),
                updated_at=datetime.fromisoformat(row[8]),
                workflow_identity_json=row[9],
                checkpoint_metadata_json=row[10],
                parent_task_id=row[11],
            )
            for row in rows
        ]

    # Agent methods

    async def upsert_agent(self, agent: AgentRecord) -> None:
        """Insert or replace an agent record."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        await self._db.execute(
            """
            INSERT OR REPLACE INTO agents (
                agent_id, agent_name, pid, status,
                capabilities_json, started_at, last_heartbeat
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                agent.agent_id,
                agent.agent_name,
                agent.pid,
                agent.status,
                agent.capabilities_json,
                agent.started_at.isoformat() if agent.started_at else None,
                agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            ),
        )
        await self._db.commit()

    async def get_agent(self, agent_id: str) -> AgentRecord | None:
        """Retrieve an agent by ID."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        cursor = await self._db.execute(
            """
            SELECT agent_id, agent_name, pid, status,
                   capabilities_json, started_at, last_heartbeat
            FROM agents WHERE agent_id = ?
        """,
            (agent_id,),
        )

        row = await cursor.fetchone()
        if not row:
            return None

        return AgentRecord(
            agent_id=row[0],
            agent_name=row[1],
            pid=row[2],
            status=row[3],
            capabilities_json=row[4],
            started_at=datetime.fromisoformat(row[5]) if row[5] else None,
            last_heartbeat=datetime.fromisoformat(row[6]) if row[6] else None,
        )

    async def list_agents(self) -> list[AgentRecord]:
        """List all agents."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        cursor = await self._db.execute("""
            SELECT agent_id, agent_name, pid, status,
                   capabilities_json, started_at, last_heartbeat
            FROM agents ORDER BY agent_name
        """)

        rows = await cursor.fetchall()
        agents = []
        for row in rows:
            agents.append(
                AgentRecord(
                    agent_id=row[0],
                    agent_name=row[1],
                    pid=row[2],
                    status=row[3],
                    capabilities_json=row[4],
                    started_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    last_heartbeat=datetime.fromisoformat(row[6]) if row[6] else None,
                )
            )

        return agents

    async def delete_agent(self, agent_id: str) -> None:
        """Delete an agent record."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        await self._db.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
        await self._db.commit()

    # Event methods

    async def add_event(self, task_id: str, event_type: str, payload: dict) -> int:
        """Add an event with current timestamp. Returns the assigned event_id."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        cursor = await self._db.execute(
            """
            INSERT INTO events (task_id, timestamp, event_type, payload_json)
            VALUES (?, ?, ?, ?)
        """,
            (
                task_id,
                datetime.now(UTC).isoformat(),
                event_type,
                json.dumps(payload),
            ),
        )
        await self._db.commit()
        return cursor.lastrowid or 0

    async def get_events(self, task_id: str) -> list[dict]:
        """Get all events for a task, ordered by timestamp."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        cursor = await self._db.execute(
            """
            SELECT event_id, task_id, timestamp, event_type, payload_json
            FROM events WHERE task_id = ?
            ORDER BY timestamp ASC
        """,
            (task_id,),
        )

        rows = await cursor.fetchall()
        events = []
        for row in rows:
            events.append(
                {
                    "event_id": row[0],
                    "task_id": row[1],
                    "timestamp": row[2],
                    "event_type": row[3],
                    "payload": json.loads(row[4]),
                }
            )

        return events

    async def get_events_since(self, task_id: str, after_event_id: int = 0) -> list[dict]:
        """Get events for a task with event_id > after_event_id."""
        if not self._db:
            raise RuntimeError("Store not initialized")

        cursor = await self._db.execute(
            """
            SELECT event_id, task_id, timestamp, event_type, payload_json
            FROM events WHERE task_id = ? AND event_id > ?
            ORDER BY event_id ASC
        """,
            (task_id, after_event_id),
        )

        rows = await cursor.fetchall()
        return [
            {
                "event_id": row[0],
                "task_id": row[1],
                "timestamp": row[2],
                "event_type": row[3],
                "payload": json.loads(row[4]),
            }
            for row in rows
        ]

    async def recover_orphaned_tasks(self) -> int:
        """Mark any running/pending tasks as failed after a restart.

        Returns the number of tasks recovered.
        """
        if not self._db:
            raise RuntimeError("Store not initialized")

        # Find all tasks with non-terminal status
        cursor = await self._db.execute(
            "SELECT task_id FROM tasks WHERE status IN (?, ?, ?)",
            (TaskStatus.RUNNING.value, TaskStatus.PENDING.value, TaskStatus.INTERRUPTED.value),
        )
        task_ids = [row[0] for row in await cursor.fetchall()]

        if not task_ids:
            return 0

        # Update each task to failed status
        error_json = json.dumps({"error": "Task orphaned by runtime restart"})
        for task_id in task_ids:
            await self.update_task(task_id, status=TaskStatus.FAILED, error_json=error_json)
            # Add an event for the failure
            await self.add_event(task_id, "task.failed", {"error": "Task orphaned by runtime restart"})

        return len(task_ids)

    async def clear_all_agents(self) -> int:
        """Delete all agent rows. Agent rows track live subprocesses; after a
        daemon restart none of the recorded agents are alive, so any rows left
        behind by a crash are stale. Returns the number of rows deleted.
        """
        if not self._db:
            raise RuntimeError("Store not initialized")
        cursor = await self._db.execute("DELETE FROM agents")
        await self._db.commit()
        return cursor.rowcount

    async def record_checkpoint_thread(self, thread_id: str, task_id: str) -> None:
        """Record that ``task_id`` writes checkpoints under ``thread_id``.

        Tasks may use any thread_id (e.g. a stable work-item key), so purge
        relies on this mapping rather than assuming thread_id == task_id.
        """
        if not self._db:
            raise RuntimeError("Store not initialized")
        await self._db.execute(
            "INSERT OR IGNORE INTO checkpoint_threads (thread_id, task_id) VALUES (?, ?)",
            (thread_id, task_id),
        )
        await self._db.commit()

    async def get_terminal_task_ids(self) -> list[str]:
        if not self._db:
            raise RuntimeError("Store not initialized")
        placeholders = ",".join("?" * len(_PURGEABLE_STATUSES))
        cursor = await self._db.execute(
            f"SELECT task_id FROM tasks WHERE status IN ({placeholders})",
            _PURGEABLE_STATUSES,
        )
        return [row[0] for row in await cursor.fetchall()]

    async def clear_terminal_tasks(self) -> int:
        """Soft-delete terminal tasks by marking them CLEARED.

        Cleared tasks are hidden from view but their data (events, checkpoints,
        files) is preserved until an explicit purge. Already-cleared tasks are
        left untouched.

        Returns:
            Number of tasks cleared.
        """
        if not self._db:
            raise RuntimeError("Store not initialized")

        placeholders = ",".join("?" * len(_CLEARABLE_STATUSES))
        now = datetime.now(UTC).isoformat()
        cursor = await self._db.execute(
            f"UPDATE tasks SET status = ?, updated_at = ? WHERE status IN ({placeholders})",
            (TaskStatus.CLEARED.value, now, *_CLEARABLE_STATUSES),
        )
        await self._db.commit()
        return cursor.rowcount

    async def purge_terminal_tasks(self) -> int:
        """Hard-delete terminal/cleared tasks and all their associated rows.

        Deletes the tasks plus their events and checkpoint data. Checkpoints are
        removed by the thread_ids recorded for those tasks (see
        ``record_checkpoint_thread``); any checkpoint thread that no longer maps
        to a surviving task is also swept, reclaiming orphaned rows written
        before the mapping existed.

        Returns:
            Number of tasks deleted.
        """
        if not self._db:
            raise RuntimeError("Store not initialized")

        status_placeholders = ",".join("?" * len(_PURGEABLE_STATUSES))
        cursor = await self._db.execute(
            f"SELECT task_id FROM tasks WHERE status IN ({status_placeholders})",
            _PURGEABLE_STATUSES,
        )
        task_ids = [row[0] for row in await cursor.fetchall()]

        if not task_ids:
            return 0

        placeholders = ",".join("?" * len(task_ids))

        await self._db.execute(
            f"DELETE FROM events WHERE task_id IN ({placeholders})",
            task_ids,
        )

        # Resolve the thread_ids these tasks wrote checkpoints under.
        cursor = await self._db.execute(
            f"SELECT DISTINCT thread_id FROM checkpoint_threads WHERE task_id IN ({placeholders})",
            task_ids,
        )
        thread_ids = {row[0] for row in await cursor.fetchall()}

        # Orphan sweep: any checkpoint thread no longer referenced by a surviving
        # task (e.g. rows written before the mapping table existed) is dead data.
        cursor = await self._db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
        has_checkpoints = await cursor.fetchone() is not None
        if has_checkpoints:
            survivor_placeholders = ",".join("?" * len(task_ids))
            cursor = await self._db.execute(
                f"""
                SELECT DISTINCT thread_id FROM checkpoints
                WHERE thread_id NOT IN (
                    SELECT thread_id FROM checkpoint_threads
                    WHERE task_id NOT IN ({survivor_placeholders})
                )
                """,
                task_ids,
            )
            thread_ids.update(row[0] for row in await cursor.fetchall())

        # Checkpoint tables are created by checkpoint.py, not by Store.initialize().
        # Guard with IF EXISTS so purge works even if checkpointing was never used.
        if thread_ids:
            thread_list = list(thread_ids)
            thread_placeholders = ",".join("?" * len(thread_list))
            if has_checkpoints:
                await self._db.execute(
                    f"DELETE FROM checkpoints WHERE thread_id IN ({thread_placeholders})",
                    thread_list,
                )
            cursor = await self._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoint_writes'"
            )
            if await cursor.fetchone():
                await self._db.execute(
                    f"DELETE FROM checkpoint_writes WHERE thread_id IN ({thread_placeholders})",
                    thread_list,
                )

        await self._db.execute(
            f"DELETE FROM checkpoint_threads WHERE task_id IN ({placeholders})",
            task_ids,
        )
        await self._db.execute(
            f"DELETE FROM tasks WHERE task_id IN ({placeholders})",
            task_ids,
        )
        await self._db.commit()

        return len(task_ids)

    # Cleanup

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
