import aiosqlite
import pytest
import pytest_asyncio

from switchplane.checkpoint import SqliteCheckpointSaver, setup_tables


@pytest_asyncio.fixture
async def db(tmp_path):
    conn = await aiosqlite.connect(tmp_path / "test.db")
    await setup_tables(conn)
    yield conn
    await conn.close()


@pytest_asyncio.fixture
async def saver(db):
    return SqliteCheckpointSaver(db)


def _config(thread_id="thread1", checkpoint_ns="", checkpoint_id=None):
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
    if checkpoint_id:
        cfg["configurable"]["checkpoint_id"] = checkpoint_id
    return cfg


def _checkpoint(cp_id="cp1"):
    return {
        "id": cp_id,
        "ts": "2024-01-01T00:00:00Z",
        "channel_values": {"messages": []},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


class TestSetupTables:
    @pytest.mark.asyncio
    async def test_creates_tables(self, tmp_path):
        conn = await aiosqlite.connect(tmp_path / "fresh.db")
        await setup_tables(conn)

        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert "checkpoints" in tables
        assert "checkpoint_writes" in tables
        await conn.close()

    @pytest.mark.asyncio
    async def test_idempotent(self, tmp_path):
        conn = await aiosqlite.connect(tmp_path / "idem.db")
        await setup_tables(conn)
        await setup_tables(conn)  # should not raise
        await conn.close()


class TestAput:
    @pytest.mark.asyncio
    async def test_saves_checkpoint(self, saver):
        config = _config()
        cp = _checkpoint("cp1")
        metadata = {"source": "test", "step": 1}

        result_config = await saver.aput(config, cp, metadata, {})

        assert result_config["configurable"]["checkpoint_id"] == "cp1"

    @pytest.mark.asyncio
    async def test_overwrites_existing(self, saver):
        config = _config()
        cp1 = _checkpoint("cp1")
        await saver.aput(config, cp1, {"step": 1}, {})

        cp1_updated = _checkpoint("cp1")
        cp1_updated["channel_values"] = {"messages": ["updated"]}
        await saver.aput(config, cp1_updated, {"step": 2}, {})

        result = await saver.aget_tuple(_config(checkpoint_id="cp1"))
        assert result is not None
        assert result.metadata.get("step") == 2


class TestAgetTuple:
    @pytest.mark.asyncio
    async def test_get_latest(self, saver):
        config = _config()
        await saver.aput(config, _checkpoint("cp1"), {"step": 1}, {})

        config_with_cp1 = _config(checkpoint_id="cp1")
        await saver.aput(config_with_cp1, _checkpoint("cp2"), {"step": 2}, {})

        result = await saver.aget_tuple(_config())
        assert result is not None
        assert result.config["configurable"]["checkpoint_id"] == "cp2"

    @pytest.mark.asyncio
    async def test_get_specific(self, saver):
        config = _config()
        await saver.aput(config, _checkpoint("cp1"), {"step": 1}, {})

        result = await saver.aget_tuple(_config(checkpoint_id="cp1"))
        assert result is not None
        assert result.config["configurable"]["checkpoint_id"] == "cp1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, saver):
        result = await saver.aget_tuple(_config(checkpoint_id="nope"))
        assert result is None

    @pytest.mark.asyncio
    async def test_includes_parent_config(self, saver):
        await saver.aput(_config(), _checkpoint("cp1"), {}, {})
        await saver.aput(_config(checkpoint_id="cp1"), _checkpoint("cp2"), {}, {})

        result = await saver.aget_tuple(_config(checkpoint_id="cp2"))
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == "cp1"


class TestAlist:
    @pytest.mark.asyncio
    async def test_lists_checkpoints(self, saver):
        await saver.aput(_config(), _checkpoint("cp1"), {"step": 1}, {})
        await saver.aput(_config(checkpoint_id="cp1"), _checkpoint("cp2"), {"step": 2}, {})
        await saver.aput(_config(checkpoint_id="cp2"), _checkpoint("cp3"), {"step": 3}, {})

        results = []
        async for cp_tuple in saver.alist(_config()):
            results.append(cp_tuple)

        assert len(results) == 3
        # Should be ordered descending by checkpoint_id
        ids = [r.config["configurable"]["checkpoint_id"] for r in results]
        assert ids == ["cp3", "cp2", "cp1"]

    @pytest.mark.asyncio
    async def test_list_with_limit(self, saver):
        await saver.aput(_config(), _checkpoint("cp1"), {}, {})
        await saver.aput(_config(checkpoint_id="cp1"), _checkpoint("cp2"), {}, {})

        results = []
        async for cp_tuple in saver.alist(_config(), limit=1):
            results.append(cp_tuple)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_with_before(self, saver):
        await saver.aput(_config(), _checkpoint("cp1"), {}, {})
        await saver.aput(_config(checkpoint_id="cp1"), _checkpoint("cp2"), {}, {})
        await saver.aput(_config(checkpoint_id="cp2"), _checkpoint("cp3"), {}, {})

        before_config = _config(checkpoint_id="cp3")
        results = []
        async for cp_tuple in saver.alist(_config(), before=before_config):
            results.append(cp_tuple)

        ids = [r.config["configurable"]["checkpoint_id"] for r in results]
        assert "cp3" not in ids

    @pytest.mark.asyncio
    async def test_list_with_filter(self, saver):
        await saver.aput(_config(), _checkpoint("cp1"), {"source": "input"}, {})
        await saver.aput(_config(checkpoint_id="cp1"), _checkpoint("cp2"), {"source": "loop"}, {})

        results = []
        async for cp_tuple in saver.alist(_config(), filter={"source": "input"}):
            results.append(cp_tuple)

        assert len(results) == 1


class TestAputWrites:
    @pytest.mark.asyncio
    async def test_saves_writes(self, saver):
        await saver.aput(_config(), _checkpoint("cp1"), {}, {})

        writes = [("messages", ["hello"]), ("counter", 1)]
        await saver.aput_writes(
            _config(checkpoint_id="cp1"),
            writes,
            task_id="task1",
        )

        result = await saver.aget_tuple(_config(checkpoint_id="cp1"))
        assert len(result.pending_writes) == 2


class TestSyncMethodsRaise:
    def test_get_tuple(self, saver):
        with pytest.raises(NotImplementedError, match="async"):
            saver.get_tuple(_config())

    def test_list(self, saver):
        with pytest.raises(NotImplementedError, match="async"):
            saver.list(_config())

    def test_put(self, saver):
        with pytest.raises(NotImplementedError, match="async"):
            saver.put(_config(), _checkpoint(), {}, {})

    def test_put_writes(self, saver):
        with pytest.raises(NotImplementedError, match="async"):
            saver.put_writes(_config(), [], "task1")

    @pytest_asyncio.fixture
    async def saver(self, tmp_path):
        conn = await aiosqlite.connect(tmp_path / "sync_test.db")
        await setup_tables(conn)
        s = SqliteCheckpointSaver(conn)
        yield s
        await conn.close()
