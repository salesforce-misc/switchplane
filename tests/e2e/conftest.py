"""End-to-end test fixtures for subprocess-based CLI testing."""

import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("E2E") != "1", reason="Set E2E=1 to run e2e tests")


@pytest.fixture
def e2e_home():
    """Isolated HOME directory kept short for Unix socket path limits (~104 chars on macOS).

    Uses a short temp dir under the workspace .tmp/ directory to satisfy
    sandbox restrictions while staying well under the limit.
    """
    base = Path(__file__).resolve().parent.parent.parent / ".tmp"
    base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base, prefix="e2e_") as d:
        yield Path(d)


@pytest.fixture
def cli_env(e2e_home):
    """Environment dict with HOME pointed at the temp dir."""
    env = os.environ.copy()
    env["HOME"] = str(e2e_home)
    return env


@pytest.fixture
def run_cli(cli_env):
    """Helper that wraps subprocess.run for the hello CLI."""
    hello_bin = shutil.which("hello")
    if hello_bin is None:
        pytest.skip("'hello' CLI not installed (run: uv pip install -e examples/hello)")

    def _run(*args: str, timeout: int = 10) -> subprocess.CompletedProcess:
        return subprocess.run(
            [hello_bin, *args],
            env=cli_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    return _run


@pytest.fixture
def hello_daemon(run_cli, e2e_home):
    """Start the hello daemon, wait for socket, yield run_cli, then tear down."""
    # start_daemon polls up to 10s internally; give the subprocess enough headroom
    result = run_cli("runtime", "start", timeout=15)
    assert result.returncode == 0, f"Failed to start daemon: {result.stderr}"

    sock_path = e2e_home / ".hello" / "runtime.sock"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if sock_path.exists():
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"Daemon socket not found at {sock_path} within 5s")

    yield run_cli

    # Teardown: stop the daemon gracefully
    stop_result = run_cli("runtime", "stop", timeout=15)
    if stop_result.returncode != 0:
        # Fallback: read PID file and SIGKILL
        pid_path = e2e_home / ".hello" / "runtime.pid"
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
                os.kill(pid, signal.SIGKILL)
            except (ValueError, ProcessLookupError, OSError):
                pass
