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

    await db.commit()


class SqliteCheckpointSaver(BaseCheckpointSaver):
    """LangGraph checkpoint saver backed by SQLite."""

    serde = JsonPlusSerializer()

    def __init__(self, db: aiosqlite.Connection):
        """Initialize with an existing SQLite connection."""
        super().__init__()
        self.db = db

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
                SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            """,
                (thread_id, checkpoint_ns, checkpoint_id),
            )
        else:
            # Get latest checkpoint
            cursor = await self.db.execute(
                """
                SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
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

        checkpoint_id, parent_checkpoint_id, checkpoint_type, checkpoint_blob, metadata_blob = row

        # Deserialize checkpoint and metadata
        checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_blob)) if checkpoint_blob else {}
        metadata = self.serde.loads_typed((checkpoint_type, metadata_blob)) if metadata_blob else {}

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
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Build query
        query = """
            SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
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
            checkpoint_id, parent_checkpoint_id, checkpoint_type, checkpoint_blob, metadata_blob = row

            # Deserialize
            checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_blob)) if checkpoint_blob else {}
            metadata = self.serde.loads_typed((checkpoint_type, metadata_blob)) if metadata_blob else {}

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
        _, metadata_blob = self.serde.dumps_typed(metadata)

        # Insert or replace checkpoint
        await self.db.execute(
            """
            INSERT OR REPLACE INTO checkpoints
            (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                checkpoint_type,
                checkpoint_blob,
                metadata_blob,
            ),
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
        for idx, (channel, value) in enumerate(writes):
            write_type, write_blob = self.serde.dumps_typed(value)

            await self.db.execute(
                """
                INSERT OR REPLACE INTO checkpoint_writes
                (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    idx,
                    channel,
                    write_type,
                    write_blob,
                ),
            )

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
