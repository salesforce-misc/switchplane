import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from switchplane.agent import AgentSpec
from switchplane.app import Application
from switchplane.checkpoint import setup_tables
from switchplane.daemon import RuntimePaths
from switchplane.persistence import Store
from switchplane.task import Task


class DummyTask(Task):
    name = "dummy"
    description = "A dummy task for testing"

    async def run(self, ctx):
        pass


@pytest.fixture
def runtime_paths(tmp_path):
    rd = tmp_path / "runtime"
    rd.mkdir()
    return RuntimePaths(runtime_dir=rd)


@pytest.fixture
def app(tmp_path):
    return Application(name="testapp", runtime_dir=tmp_path / ".testapp")


@pytest.fixture
def app_with_agent(app):
    spec = AgentSpec(agent_name="test_agent", module_path="test.module.agent")
    spec.tasks["dummy"] = DummyTask
    app.register_agent(spec)
    return app


@pytest.fixture
def short_tmp():
    """Short temp dir for Unix sockets (which have a ~104 char limit).

    Uses a path inside the workspace to satisfy sandbox restrictions.
    """
    base = Path(__file__).resolve().parent.parent / ".tmp"
    base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base, prefix="t_") as d:
        yield Path(d)


@pytest_asyncio.fixture
async def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = Store(db_path)
    await s.initialize()
    await setup_tables(s._db)
    yield s
    await s.close()
