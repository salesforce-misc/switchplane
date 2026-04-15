import sys

from switchplane.agent import AgentSpec
from switchplane.app import Application
from switchplane.discovery import (
    _discover_agent,
    _discover_from_root,
    _discover_task,
    discover_agents_for_app,
)


class TestDiscoverFromRoot:
    def test_full_discovery(self, tmp_path, app, monkeypatch):
        pkg = tmp_path / "testpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")

        agent_dir = pkg / "myagent"
        agent_dir.mkdir()
        (agent_dir / "__init__.py").write_text("")
        (agent_dir / "agent.py").write_text(
            'from switchplane.agent import AgentSpec\nagent_spec = AgentSpec(agent_name="myagent")\n'
        )

        tasks_dir = agent_dir / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "__init__.py").write_text("")
        (tasks_dir / "hello.py").write_text(
            "from switchplane.task import Task\n"
            "class HelloTask(Task):\n"
            '    name = "hello"\n'
            "    async def run(self, ctx): pass\n"
        )

        monkeypatch.syspath_prepend(str(tmp_path))

        _discover_from_root(app, "testpkg")

        assert "myagent" in app.agents
        assert "hello" in app.agents["myagent"].tasks

    def test_missing_root(self, app):
        _discover_from_root(app, "nonexistent.package.xyz")
        assert len(app.agents) == 0

    def test_root_without_path(self, app, monkeypatch):
        """A root module that isn't a package (no __path__)."""
        import types

        mod = types.ModuleType("flat_module")
        monkeypatch.setitem(sys.modules, "flat_module", mod)
        _discover_from_root(app, "flat_module")
        assert len(app.agents) == 0


class TestDiscoverAgent:
    def test_no_agent_py(self, app, tmp_path, monkeypatch):
        pkg = tmp_path / "pkg2"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        agent_dir = pkg / "empty_agent"
        agent_dir.mkdir()
        (agent_dir / "__init__.py").write_text("")

        monkeypatch.syspath_prepend(str(tmp_path))
        _discover_agent(app, "pkg2.empty_agent", "empty_agent")
        assert "empty_agent" not in app.agents

    def test_no_agent_spec_attr(self, app, tmp_path, monkeypatch):
        pkg = tmp_path / "pkg3"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        agent_dir = pkg / "nospec"
        agent_dir.mkdir()
        (agent_dir / "__init__.py").write_text("")
        (agent_dir / "agent.py").write_text("x = 42\n")

        monkeypatch.syspath_prepend(str(tmp_path))
        _discover_agent(app, "pkg3.nospec", "nospec")
        assert "nospec" not in app.agents

    def test_agent_spec_wrong_type(self, app, tmp_path, monkeypatch):
        pkg = tmp_path / "pkg4"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        agent_dir = pkg / "wrongtype"
        agent_dir.mkdir()
        (agent_dir / "__init__.py").write_text("")
        (agent_dir / "agent.py").write_text('agent_spec = "not an AgentSpec"\n')

        monkeypatch.syspath_prepend(str(tmp_path))
        _discover_agent(app, "pkg4.wrongtype", "wrongtype")
        assert "wrongtype" not in app.agents


class TestDiscoverTask:
    def test_module_with_task(self, tmp_path, monkeypatch):
        pkg = tmp_path / "taskpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "greeting.py").write_text(
            "from switchplane.task import Task\n"
            "class GreetTask(Task):\n"
            '    name = "greet"\n'
            "    async def run(self, ctx): pass\n"
        )

        monkeypatch.syspath_prepend(str(tmp_path))
        spec = AgentSpec(agent_name="test")
        _discover_task(spec, "taskpkg.greeting")
        # Registered under Task.name, not the module filename
        assert "greet" in spec.tasks

    def test_module_without_task(self, tmp_path, monkeypatch):
        pkg = tmp_path / "notaskpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "util.py").write_text("def helper(): pass\n")

        monkeypatch.syspath_prepend(str(tmp_path))
        spec = AgentSpec(agent_name="test")
        _discover_task(spec, "notaskpkg.util")
        assert len(spec.tasks) == 0

    def test_import_error(self):
        spec = AgentSpec(agent_name="test")
        _discover_task(spec, "totally.fake.module")
        assert len(spec.tasks) == 0


class TestDiscoverAgentsForApp:
    def test_processes_all_roots(self, tmp_path, monkeypatch):
        for name in ["root1", "root2"]:
            pkg = tmp_path / name
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            agent_dir = pkg / "agent1"
            agent_dir.mkdir()
            (agent_dir / "__init__.py").write_text("")
            (agent_dir / "agent.py").write_text(
                f'from switchplane.agent import AgentSpec\nagent_spec = AgentSpec(agent_name="{name}_agent")\n'
            )
            tasks_dir = agent_dir / "tasks"
            tasks_dir.mkdir()
            (tasks_dir / "__init__.py").write_text("")

        monkeypatch.syspath_prepend(str(tmp_path))

        app = Application(name="test", runtime_dir=tmp_path / ".test")
        app.discover_agents("root1")
        app.discover_agents("root2")
        discover_agents_for_app(app)

        assert "root1_agent" in app.agents
        assert "root2_agent" in app.agents

    def test_bad_root_does_not_crash(self, tmp_path):
        app = Application(name="test", runtime_dir=tmp_path / ".test")
        app.discover_agents("nonexistent.root")
        discover_agents_for_app(app)
        assert len(app.agents) == 0
