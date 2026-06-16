"""Custom LangGraph checkpoint saver backed by SQLite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    JsonPlusSerializer,
)


async def setup_tables(db: aiosqlite.Connection) -> None:
    """Create checkpoint tables in the SQLite database."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata BLOB,
            metadata_type TEXT,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_writes (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            channel TEXT NOT NULL,
            type TEXT,
            blob BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        )
    """)

    # Maps a checkpointer thread_id to the switchplane task that wrote it, so
    # purge can find checkpoints even when thread_id != task_id. Also created by
    # Store.initialize(); duplicated here because the agent subprocess writes via
    # its own connection.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_threads (
            thread_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            PRIMARY KEY (thread_id, task_id)
        )
    """)

    await db.commit()


class SqliteCheckpointSaver(BaseCheckpointSaver):
    """LangGraph checkpoint saver backed by SQLite."""

    serde = JsonPlusSerializer()

    def __init__(self, db: aiosqlite.Connection, task_id: str | None = None):
        """Initialize with an existing SQLite connection.

        ``task_id`` is the owning switchplane task. When set, each saved
        checkpoint records a thread_id -> task_id mapping so purge can later
        delete this task's checkpoints regardless of the thread_id chosen.
        """
        super().__init__()
        self.db = db
        self.task_id = task_id

    async def setup(self) -> None:
        """Create checkpoint tables if they don't exist."""
        await setup_tables(self.db)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple for the given config."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        # Query for the checkpoint
        if checkpoint_id:
            # Get specific checkpoint
            cursor = await self.db.execute(
                """
                SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, metadata_type
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            """,
                (thread_id, checkpoint_ns, checkpoint_id),
            )
        else:
            # Get latest checkpoint
            cursor = await self.db.execute(
                """
                SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, metadata_type
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """,
                (thread_id, checkpoint_ns),
            )

        row = await cursor.fetchone()
        if not row:
            return None

        checkpoint_id, parent_checkpoint_id, checkpoint_type, checkpoint_blob, metadata_blob, metadata_type = row

        # Deserialize checkpoint and metadata
        checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_blob)) if checkpoint_blob else {}
        metadata = self.serde.loads_typed((metadata_type or checkpoint_type, metadata_blob)) if metadata_blob else {}

        # Query for pending writes
        writes_cursor = await self.db.execute(
            """
            SELECT task_id, channel, type, blob
            FROM checkpoint_writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            ORDER BY task_id, idx
        """,
            (thread_id, checkpoint_ns, checkpoint_id),
        )

        writes_rows = await writes_cursor.fetchall()
        pending_writes = []
        for write_row in writes_rows:
            task_id, channel, write_type, write_blob = write_row
            value = self.serde.loads_typed((write_type, write_blob)) if write_blob else None
            pending_writes.append((task_id, channel, value))

        # Build parent config if there's a parent checkpoint
        parent_config = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int = 10,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints for a thread."""
        if config is None:
            return
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Build query
        query = """
            SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, metadata_type
            FROM checkpoints
            WHERE thread_id = ? AND checkpoint_ns = ?
        """
        params = [thread_id, checkpoint_ns]

        # Add before filter if specified
        if before:
            before_checkpoint_id = before["configurable"].get("checkpoint_id")
            if before_checkpoint_id:
                query += " AND checkpoint_id < ?"
                params.append(before_checkpoint_id)

        query += " ORDER BY checkpoint_id DESC LIMIT ?"
        params.append(limit)

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()

        for row in rows:
            checkpoint_id, parent_checkpoint_id, checkpoint_type, checkpoint_blob, metadata_blob, metadata_type = row

            # Deserialize
            checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_blob)) if checkpoint_blob else {}
            metadata = (
                self.serde.loads_typed((metadata_type or checkpoint_type, metadata_blob)) if metadata_blob else {}
            )

            # Apply filter if provided
            if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                continue

            # Query for pending writes
            writes_cursor = await self.db.execute(
                """
                SELECT task_id, channel, type, blob
                FROM checkpoint_writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                ORDER BY task_id, idx
            """,
                (thread_id, checkpoint_ns, checkpoint_id),
            )

            writes_rows = await writes_cursor.fetchall()
            pending_writes = []
            for write_row in writes_rows:
                task_id, channel, write_type, write_blob = write_row
                value = self.serde.loads_typed((write_type, write_blob)) if write_blob else None
                pending_writes.append((task_id, channel, value))

            # Build parent config
            parent_config = None
            if parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        # Generate new checkpoint ID from checkpoint data
        checkpoint_id = checkpoint["id"]

        # Serialize checkpoint and metadata
        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(checkpoint)
        metadata_type, metadata_blob = self.serde.dumps_typed(metadata)

        # Insert or replace checkpoint
        await self.db.execute(
            """
            INSERT OR REPLACE INTO checkpoints
            (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, metadata_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                checkpoint_type,
                checkpoint_blob,
                metadata_blob,
                metadata_type,
            ),
        )

        if self.task_id is not None:
            await self.db.execute(
                "INSERT OR IGNORE INTO checkpoint_threads (thread_id, task_id) VALUES (?, ?)",
                (thread_id, self.task_id),
            )

        await self.db.commit()

        # Return updated config with new checkpoint_id
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save pending writes for a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        # Insert writes
        rows = []
        for idx, (channel, value) in enumerate(writes):
            write_type, write_blob = self.serde.dumps_typed(value)
            rows.append(
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    idx,
                    channel,
                    write_type,
                    write_blob,
                )
            )

        await self.db.executemany(
            """
            INSERT OR REPLACE INTO checkpoint_writes
            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            rows,
        )

        await self.db.commit()

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint, pending write, and thread mapping for
        ``thread_id``.

        Overrides ``BaseCheckpointSaver.adelete_thread`` (which raises
        ``NotImplementedError``). LangGraph keys all checkpoint state on
        ``thread_id``, so callers that reuse a thread id across runs (e.g. a
        task whose thread is its work item) use this to guarantee a clean
        slate before a fresh invoke. Mirrors the per-thread deletes in
        ``Store.purge_terminal_tasks`` — same three tables, same key — but
        scoped to a single thread and without touching the ``tasks`` row.

        Drops the ``checkpoint_threads`` mapping too, so a later orphan sweep
        in purge doesn't resurrect this thread as dangling data.
        """
        await self.db.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
        await self.db.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        await self.db.execute("DELETE FROM checkpoint_threads WHERE thread_id = ?", (thread_id,))
        await self.db.commit()

    # Sync methods - not implemented

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Sync version not implemented - use async version."""
        raise NotImplementedError("Use async version: aget_tuple")

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int = 10,
    ):
        """Sync version not implemented - use async version."""
        raise NotImplementedError("Use async version: alist")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Sync version not implemented - use async version."""
        raise NotImplementedError("Use async version: aput")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Sync version not implemented - use async version."""
        raise NotImplementedError("Use async version: aput_writes")

    def delete_thread(self, thread_id: str) -> None:
        """Sync version not implemented - use async version."""
        raise NotImplementedError("Use async version: adelete_thread")
