"""End-to-end tests for the hello example app."""

import os
import re
import time

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("E2E") != "1", reason="Set E2E=1 to run e2e tests")


class TestRunHello:
    """Tests that use a running daemon."""

    def test_run_hello(self, hello_daemon):
        """Run the hello task attached and verify greeting output."""
        run_cli = hello_daemon
        result = run_cli("run", "example", "hello", "--user-name", "Alice")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Hello, Alice" in result.stdout

    def test_run_detached_and_follow(self, hello_daemon):
        """Submit detached, extract task ID, then follow to see output."""
        run_cli = hello_daemon

        # Submit detached
        result = run_cli("run", "example", "hello", "--user-name", "Bob", "-d")
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Extract task ID from "Task submitted: <id>" line
        match = re.search(r"Task submitted:\s+(\S+)", result.stdout)
        assert match, f"Could not find task ID in output: {result.stdout}"
        task_id = match.group(1)

        # Wait briefly for the task to complete (it's fast)
        time.sleep(1)

        # Follow the task
        result = run_cli("task", "follow", task_id)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # The follow output should contain the greeting or the completed result
        assert "Hello, Bob" in result.stdout or "completed" in result.stdout.lower()

    def test_task_list(self, hello_daemon):
        """Run a task to completion, then verify it appears in task list."""
        run_cli = hello_daemon

        # Run a task to completion
        result = run_cli("run", "example", "hello", "--user-name", "Charlie")
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # List tasks
        result = run_cli("task", "list")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "completed" in result.stdout

    def test_task_show(self, hello_daemon):
        """Run a task, extract ID, then show its details."""
        run_cli = hello_daemon

        # Run detached to get the task ID easily
        result = run_cli("run", "example", "hello", "--user-name", "Diana", "-d")
        assert result.returncode == 0, f"stderr: {result.stderr}"

        match = re.search(r"Task submitted:\s+(\S+)", result.stdout)
        assert match, f"Could not find task ID in output: {result.stdout}"
        task_id = match.group(1)

        # Wait for task to complete
        time.sleep(1)

        # Show task details
        result = run_cli("task", "show", task_id)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Task ID:" in result.stdout
        assert task_id in result.stdout
        assert "example" in result.stdout
        assert "hello" in result.stdout


class TestRuntimeLifecycle:
    """Tests for daemon start/stop without the hello_daemon fixture."""

    def test_runtime_lifecycle(self, run_cli, e2e_home):
        """Start, check status, stop, and verify stopped status."""
        # Start the daemon (start_daemon polls up to 10s internally)
        result = run_cli("runtime", "start", timeout=15)
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Wait for socket
        sock_path = e2e_home / ".hello" / "runtime.sock"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if sock_path.exists():
                break
            time.sleep(0.05)
        else:
            pytest.fail("Daemon socket not found within 5s")

        # Check status — should report running
        result = run_cli("runtime", "status")
        assert result.returncode == 0
        assert "running" in result.stdout.lower()

        # Stop the daemon
        result = run_cli("runtime", "stop", timeout=15)
        assert result.returncode == 0

        # Give it a moment to shut down
        time.sleep(0.5)

        # Check status again — should report not running
        result = run_cli("runtime", "status")
        # Either non-zero exit or "not running" in output
        assert "not running" in result.stdout.lower() or result.returncode != 0
