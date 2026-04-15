import asyncio

import pytest

from switchplane.daemon import IDLE_TIMEOUT, IdleTimer, RuntimePaths, cleanup


class TestRuntimePaths:
    def test_all_paths(self, tmp_path):
        paths = RuntimePaths(runtime_dir=tmp_path / "myapp")
        assert paths.sock_path == tmp_path / "myapp" / "runtime.sock"
        assert paths.db_path == tmp_path / "myapp" / "state.db"
        assert paths.pid_path == tmp_path / "myapp" / "runtime.pid"
        assert paths.log_dir == tmp_path / "myapp" / "logs"
        assert paths.ca_bundle_path == tmp_path / "myapp" / "ca-bundle.pem"
        assert paths.config_path == tmp_path / "myapp" / "config.toml"


class TestIdleTimer:
    @pytest.mark.asyncio
    async def test_callback_fires(self):
        fired = asyncio.Event()

        def on_idle():
            fired.set()

        timer = IdleTimer(timeout=0.05, callback=on_idle)
        timer.reset()

        await asyncio.wait_for(fired.wait(), timeout=1.0)
        assert fired.is_set()

    @pytest.mark.asyncio
    async def test_reset_restarts_timer(self):
        call_count = 0

        def on_idle():
            nonlocal call_count
            call_count += 1

        timer = IdleTimer(timeout=0.1, callback=on_idle)
        timer.reset()
        await asyncio.sleep(0.05)
        timer.reset()  # restart before it fires
        await asyncio.sleep(0.05)
        timer.reset()  # restart again
        assert call_count == 0
        timer.cancel()

    @pytest.mark.asyncio
    async def test_cancel(self):
        fired = False

        def on_idle():
            nonlocal fired
            fired = True

        timer = IdleTimer(timeout=0.05, callback=on_idle)
        timer.reset()
        timer.cancel()
        await asyncio.sleep(0.1)
        assert not fired

    @pytest.mark.asyncio
    async def test_cancel_idempotent(self):
        timer = IdleTimer(timeout=1.0, callback=lambda: None)
        timer.cancel()
        timer.cancel()  # should not raise


class TestCleanup:
    def test_removes_sock_and_pid(self, runtime_paths):
        runtime_paths.sock_path.write_text("sock")
        runtime_paths.pid_path.write_text("123")

        cleanup(runtime_paths)

        assert not runtime_paths.sock_path.exists()
        assert not runtime_paths.pid_path.exists()

    def test_handles_missing_files(self, runtime_paths):
        cleanup(runtime_paths)  # should not raise

    def test_partial_cleanup(self, runtime_paths):
        runtime_paths.sock_path.write_text("sock")
        cleanup(runtime_paths)
        assert not runtime_paths.sock_path.exists()


class TestConstants:
    def test_idle_timeout(self):
        assert IDLE_TIMEOUT == 300
