"""Daemonization and lifecycle management for the control plane."""

import asyncio
import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import structlog

from switchplane.transport import is_alive

# Constants
IDLE_TIMEOUT = 300  # 5 minutes


@dataclass
class RuntimePaths:
    """Paths for application runtime files."""

    runtime_dir: Path

    @property
    def sock_path(self) -> Path:
        return self.runtime_dir / "runtime.sock"

    @property
    def db_path(self) -> Path:
        return self.runtime_dir / "state.db"

    @property
    def pid_path(self) -> Path:
        return self.runtime_dir / "runtime.pid"

    @property
    def data_dir(self) -> Path:
        return self.runtime_dir / "data"

    @property
    def log_dir(self) -> Path:
        return self.runtime_dir / "logs"

    @property
    def ca_bundle_path(self) -> Path:
        return self.runtime_dir / "ca-bundle.pem"

    @property
    def config_path(self) -> Path:
        return self.runtime_dir / "config.toml"


def daemonize(paths: RuntimePaths) -> None:
    """Double-fork to detach from terminal and become a daemon.

    This follows the standard Unix daemon creation process:
    1. First fork to run in background
    2. Create new session with setsid()
    3. Second fork to prevent reacquiring a terminal
    4. Redirect standard file descriptors

    Args:
        paths: Runtime paths for this application
    """
    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process exits
            sys.exit(0)
    except OSError as e:
        print(f"First fork failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0o077)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process of second fork exits
            sys.exit(0)
    except OSError as e:
        print(f"Second fork failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Create runtime directories with restricted permissions
    paths.runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(paths.runtime_dir, 0o700)
    paths.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    # Open /dev/null for stdin
    with open("/dev/null") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())

    # Redirect stdout and stderr to log file
    log_file = paths.log_dir / "control_plane.log"
    with open(log_file, "a") as logf:
        os.dup2(logf.fileno(), sys.stdout.fileno())
        os.dup2(logf.fileno(), sys.stderr.fileno())

    import logging as _stdlib_logging

    from switchplane import logging

    logging.configure(log_file=log_file, level=_stdlib_logging.INFO)

    # Write PID file
    with open(paths.pid_path, "w") as pidf:
        pidf.write(str(os.getpid()))


class IdleTimer:
    """Timer that shuts down the control plane after a period of inactivity."""

    def __init__(self, timeout: float, callback: Callable[[], None]):
        """Initialize idle timer.

        Args:
            timeout: Seconds of inactivity before triggering callback
            callback: Function to call when timer expires
        """
        self.timeout = timeout
        self.callback = callback
        self._handle: asyncio.TimerHandle | None = None
        self._loop = asyncio.get_running_loop()

    def reset(self) -> None:
        """Cancel existing timer and start a new one."""
        self.cancel()
        self._handle = self._loop.call_later(self.timeout, self.callback)

    def cancel(self) -> None:
        """Cancel the timer if it's running."""
        if self._handle and not self._handle.cancelled():
            self._handle.cancel()
        self._handle = None


async def run_control_plane(paths: RuntimePaths, app) -> None:
    """Main entry point for the control plane after daemonization.

    This function:
    1. Creates and initializes the ControlPlane
    2. Sets up signal handlers
    3. Runs until shutdown

    Args:
        paths: Runtime paths for this application
        app: Application instance to run
    """
    logger = structlog.get_logger()

    # If a custom CA bundle exists, set it for all subprocesses
    if paths.ca_bundle_path.exists() and "SSL_CERT_FILE" not in os.environ:
        os.environ["SSL_CERT_FILE"] = str(paths.ca_bundle_path)
        logger.info("using_ca_bundle", path=str(paths.ca_bundle_path))

    logger.info("starting_control_plane", app=app.name)

    shutdown_event = asyncio.Event()

    loop = asyncio.get_event_loop()
    _setup_signals(loop, shutdown_event)

    from switchplane.control_plane import ControlPlane

    cp = ControlPlane(paths, app)
    cp._shutdown_event = shutdown_event

    await cp.start()
    logger.info("control_plane_started")

    try:
        await shutdown_event.wait()
        logger.info("shutdown_signal_received")
    finally:
        await cp.shutdown()
        cleanup(paths)
        logger.info("control_plane_shutdown_complete")


def _setup_signals(loop: asyncio.AbstractEventLoop, shutdown_event: asyncio.Event) -> None:
    """Register signal handlers for graceful shutdown.

    Uses loop.add_signal_handler so the event is set safely within
    the asyncio event loop context (signal.signal doesn't work
    reliably with asyncio).
    """
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: (
                structlog.get_logger().info("received_signal", signal=str(s)),
                shutdown_event.set(),
            ),
        )


def start_daemon(paths: RuntimePaths, app) -> None:
    """Start the control plane daemon if not already running.

    This function is called by the CLI to start the control plane.
    It checks if the daemon is already running, and if not, forks
    and starts it.

    Args:
        paths: Runtime paths for this application
        app: Application instance to run
    """
    # Check if already running
    if paths.pid_path.exists():
        try:
            with open(paths.pid_path) as f:
                pid = int(f.read().strip())

            # Check if process is actually running
            os.kill(pid, 0)  # Signal 0 just checks if process exists

            # Process exists, check if socket is available
            if is_alive(paths.sock_path):
                print(f"Control plane already running (PID {pid})")
                return
            else:
                # Process exists but socket not available, might be starting up
                for _ in range(10):  # Poll for up to 2 seconds
                    time.sleep(0.2)
                    if is_alive(paths.sock_path):
                        print(f"Control plane already running (PID {pid})")
                        return

                # Process exists but socket never became available — kill it
                print(f"Warning: Found stale PID {pid}, terminating", file=sys.stderr)
                os.kill(pid, signal.SIGTERM)
                for _ in range(25):  # Wait up to 5 seconds
                    time.sleep(0.2)
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        break
                else:
                    # Still alive after SIGTERM, force kill
                    try:
                        os.kill(pid, signal.SIGKILL)
                        time.sleep(0.5)
                    except OSError:
                        pass
                cleanup(paths)

        except (OSError, ValueError):
            # Process doesn't exist or PID file is invalid
            cleanup(paths)

    # Fork to start daemon
    pid = os.fork()
    if pid == 0:
        # Child process - become daemon
        daemonize(paths)

        # Run the async control plane
        try:
            asyncio.run(run_control_plane(paths, app))
        except Exception as e:
            structlog.get_logger().error("control_plane_crashed", error=str(e), exc_info=True)
            sys.exit(1)
    else:
        # Parent process - wait for daemon to be ready
        print("Starting control plane daemon...")

        # Poll for socket availability
        for _ in range(50):  # Poll for up to 10 seconds
            time.sleep(0.2)
            if is_alive(paths.sock_path):
                # Read the daemon PID
                if paths.pid_path.exists():
                    with open(paths.pid_path) as f:
                        daemon_pid = f.read().strip()
                    print(f"Control plane started (PID {daemon_pid})")
                else:
                    print("Control plane started")
                return

        # Timeout waiting for daemon
        print("Error: Control plane failed to start", file=sys.stderr)
        sys.exit(1)


def stop_daemon(paths: RuntimePaths) -> None:
    """Stop the control plane daemon if running.

    This function is called by the CLI to stop the control plane.
    It sends SIGTERM to the daemon process and waits for it to exit.

    Args:
        paths: Runtime paths for this application
    """
    if not paths.pid_path.exists():
        print("Control plane is not running")
        return

    try:
        with open(paths.pid_path) as f:
            pid = int(f.read().strip())

        # Check if process exists
        os.kill(pid, 0)

        print(f"Stopping control plane (PID {pid})...")

        # Send SIGTERM
        os.kill(pid, signal.SIGTERM)

        # Wait for process to exit (with timeout)
        for _ in range(50):  # Wait up to 10 seconds
            time.sleep(0.2)
            try:
                os.kill(pid, 0)  # Check if still running
            except OSError:
                # Process has exited
                print("Control plane stopped")
                cleanup(paths)
                return

        # Process didn't exit gracefully, try SIGKILL
        print("Warning: Control plane did not stop gracefully, forcing shutdown", file=sys.stderr)
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        cleanup(paths)
        print("Control plane stopped (forced)")

    except (OSError, ValueError) as e:
        # Process doesn't exist or PID file is invalid
        print(f"Error: Could not stop control plane: {e}", file=sys.stderr)
        cleanup(paths)


def cleanup(paths: RuntimePaths) -> None:
    """Remove socket and PID files during shutdown.

    This ensures a clean state for the next daemon startup.

    Args:
        paths: Runtime paths for this application
    """
    if paths.sock_path.exists():
        try:
            paths.sock_path.unlink()
        except OSError:
            pass

    if paths.pid_path.exists():
        try:
            paths.pid_path.unlink()
        except OSError:
            pass
