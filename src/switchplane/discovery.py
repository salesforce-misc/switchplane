"""Application and agent discovery system for Switchplane.

This module provides mechanisms to discover and load user-defined agents
and tasks through module imports.
"""

import importlib
import pkgutil

import structlog

from switchplane.agent import AgentSpec
from switchplane.app import Application
from switchplane.task import Task

logger = structlog.get_logger()


def discover_agents_for_app(app: Application) -> None:
    """Process discovery roots and populate app.agents.

    Args:
        app: Application instance to discover agents for.
    """
    for root in app._discovery_roots:
        try:
            _discover_from_root(app, root)
        except Exception as e:
            logger.error("discovery_root_failed", root=root, error=str(e))


def _discover_from_root(app: Application, root: str) -> None:
    """Import agent modules from a root package.

    Expected structure:
        root/
            <agent_name>/
                agent.py    -> contains agent_spec: AgentSpec
                tasks/
                    <task>.py -> contains a Task subclass

    For each subpackage of root:
    1. Import <root>.<agent_name>.agent
    2. Look for 'agent_spec' attribute (AgentSpec instance)
    3. Import <root>.<agent_name>.tasks.*
    4. Look for Task subclass in each task module
    5. Add Task class to agent_spec.tasks
    6. Register agent in app

    Args:
        app: Application to register discovered agents with.
        root: Root package path to discover from.
    """
    try:
        root_module = importlib.import_module(root)
    except ImportError as e:
        logger.warning("root_module_import_failed", root=root, error=str(e))
        return

    # Walk through subpackages of root
    if hasattr(root_module, "__path__"):
        for _importer, modname, ispkg in pkgutil.iter_modules(root_module.__path__, prefix=f"{root}."):
            if ispkg:
                agent_name = modname.split(".")[-1]
                _discover_agent(app, modname, agent_name)


def _discover_agent(app: Application, agent_package: str, agent_name: str) -> None:
    """Discover a single agent and its tasks.

    Args:
        app: Application to register the agent with.
        agent_package: Full package path of the agent.
        agent_name: Name of the agent.
    """
    # Try to import agent.py module
    agent_module_path = f"{agent_package}.agent"
    try:
        agent_module = importlib.import_module(agent_module_path)
    except ImportError as e:
        logger.warning("no_agent_module", package=agent_package, error=str(e))
        return

    # Look for agent_spec attribute
    if not hasattr(agent_module, "agent_spec"):
        logger.debug("no_agent_spec", module=agent_module_path)
        return

    agent_spec = agent_module.agent_spec
    if not isinstance(agent_spec, AgentSpec):
        logger.warning("invalid_agent_spec_type", module=agent_module_path)
        return

    # Set module_path if not already specified
    if not agent_spec.module_path:
        agent_spec.module_path = agent_module_path

    # Discover tasks for this agent
    _discover_tasks_for_agent(agent_spec, f"{agent_package}.tasks")

    # Register the agent with the app
    app.register_agent(agent_spec)
    logger.info("agent_discovered", agent=agent_name, task_count=len(agent_spec.tasks))


def _discover_tasks_for_agent(agent_spec: AgentSpec, tasks_package: str) -> None:
    """Discover and register tasks for an agent.

    Args:
        agent_spec: AgentSpec to add discovered tasks to.
        tasks_package: Package path containing task modules.
    """
    try:
        tasks_module = importlib.import_module(tasks_package)
    except ImportError as e:
        logger.debug("no_tasks_package", package=tasks_package, error=str(e))
        return

    # Walk through task modules and packages. A task can be a flat module
    # (`tasks/hello.py`) or a package (`tasks/hello/__init__.py`); the
    # latter lets a task split its implementation across submodules
    # without leaking those into discovery — `_discover_task` only picks
    # up Task subclasses whose `__module__` matches the package's own
    # `__init__.py`.
    if hasattr(tasks_module, "__path__"):
        for _importer, modname, _ispkg in pkgutil.iter_modules(tasks_module.__path__, prefix=f"{tasks_package}."):
            _discover_task(agent_spec, modname)


def _discover_task(agent_spec: AgentSpec, task_module_path: str) -> None:
    """Discover task(s) in a module and add them to the agent.

    All ``Task`` subclasses in the module are registered. The task's ``name``
    class attribute is used as the registration key; if unset, the module
    filename is used as a fallback.

    Args:
        agent_spec: AgentSpec to add the task to.
        task_module_path: Full module path of the task.
    """
    try:
        task_module = importlib.import_module(task_module_path)
    except ImportError as e:
        logger.warning("task_module_import_failed", module=task_module_path, error=str(e))
        return

    found = False

    for attr_name in dir(task_module):
        obj = getattr(task_module, attr_name)
        if isinstance(obj, type) and issubclass(obj, Task) and obj is not Task:
            # Skip classes imported from other modules (e.g. base classes)
            if getattr(obj, "__module__", None) != task_module.__name__:
                continue
            # Skip base classes that don't declare a task name
            task_name = getattr(obj, "name", None)
            if not task_name:
                continue
            agent_spec.tasks[task_name] = obj
            logger.debug("task_discovered", task=task_name, cls=obj.__name__)
            found = True

    if not found:
        logger.debug("no_task_class", module=task_module_path)
