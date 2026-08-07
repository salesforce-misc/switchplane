"""Tests for quality/_concurrency.py — async file lock and retry primitives.

The file_lock non-blocking test (test_file_lock_yields_via_asyncio_sleep) is the
critical one: it proves the lock polls rather than blocks, which is what keeps
concurrent tasks from freezing the event loop while waiting for lock contention.
"""

import asyncio
import fcntl
import multiprocessing
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


def _hold_lock_subprocess(lock_path_str: str, hold_seconds: float, ready_evt, done_evt):
    """Hold a file lock for *hold_seconds*, signaling ready/done via events.

    Runs in a separate process to exercise real cross-process lock contention.
    """
    # Import inside the subprocess to avoid pickling issues
    import asyncio
    import fcntl

    async def _hold():
        lock_path = Path(lock_path_str)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = lock_path.open("w")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ready_evt.set()
            await asyncio.sleep(hold_seconds)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()
            done_evt.set()

    asyncio.run(_hold())


class TestFileLock:
    """Tests for file_lock context manager."""

    @pytest.mark.asyncio
    async def test_creates_parent_directory(self, tmp_path):
        """file_lock must create parent directories if they don't exist.

        Pins mkdir(parents=True, exist_ok=True) behavior.
        """
        lock_path = tmp_path / "subdir" / "nested" / "file.lock"
        # Import after conftest injects path
        from quality._concurrency import file_lock

        async with file_lock(lock_path):
            assert lock_path.exists()
            assert lock_path.parent.exists()

    @pytest.mark.asyncio
    async def test_yields_control_to_caller(self, tmp_path):
        """The context manager must yield, allowing caller code to run."""
        lock_path = tmp_path / "test.lock"
        from quality._concurrency import file_lock

        entered = False
        async with file_lock(lock_path):
            entered = True

        assert entered, "Context manager did not yield control"

    @pytest.mark.asyncio
    async def test_lock_released_on_normal_exit(self, tmp_path):
        """Lock must be released after normal exit, allowing re-acquisition."""
        lock_path = tmp_path / "test.lock"
        from quality._concurrency import file_lock

        async with file_lock(lock_path):
            pass

        # Should be able to acquire again immediately
        async with file_lock(lock_path):
            pass

    @pytest.mark.asyncio
    async def test_lock_released_on_exception(self, tmp_path):
        """Lock must be released even when the body raises an exception.

        This pins the finally: block behavior that prevents deadlock on error.
        """
        lock_path = tmp_path / "test.lock"
        from quality._concurrency import file_lock

        class SentinelError(Exception):
            pass

        with pytest.raises(SentinelError):
            async with file_lock(lock_path):
                raise SentinelError("body failed")

        # Lock should be released, allowing re-acquisition
        async with file_lock(lock_path):
            pass

    @pytest.mark.asyncio
    async def test_cross_process_contention_blocks_until_release(self, tmp_path):
        """A second process must wait for the first to release the lock.

        This is the real lock semantics test: spawn a separate process that holds
        the lock, then try to acquire in this process and verify we wait.
        """
        lock_path = tmp_path / "contested.lock"
        from quality._concurrency import file_lock

        ctx = multiprocessing.get_context("spawn")
        ready = ctx.Event()
        done = ctx.Event()

        holder = ctx.Process(target=_hold_lock_subprocess, args=(str(lock_path), 0.4, ready, done))
        holder.start()
        try:
            assert ready.wait(timeout=5), "Holder process never acquired the lock"

            # Now try to acquire in our process — should block until holder releases
            start = time.monotonic()
            async with file_lock(lock_path):
                elapsed = time.monotonic() - start

            # We must have waited at least most of the hold time
            assert elapsed >= 0.3, f"Acquired lock too quickly ({elapsed:.3f}s); lock did not block"
            assert done.is_set(), "Holder should have released by now"
        finally:
            holder.join(timeout=5)

    @pytest.mark.asyncio
    async def test_file_lock_yields_via_asyncio_sleep(self, tmp_path, monkeypatch):
        """The critical test: while waiting for a contended lock, file_lock must
        yield control via asyncio.sleep, NOT block the thread.

        This is what makes concurrent tasks contend for the lock without freezing
        the event loop — including the agent's IPC command listener, so cancel
        still works mid-lock-wait.

        Mutation detected: if you replace the asyncio.sleep loop with blocking
        fcntl.flock(fd, LOCK_EX), this test fails because no asyncio.sleep calls
        occur.
        """
        lock_path = tmp_path / "test.lock"
        from quality import _concurrency

        # Simulate contention: make LOCK_NB raise BlockingIOError twice before succeeding
        real_flock = fcntl.flock
        call_count = {"n": 0}

        def flaky_flock(fd, op):
            if op & fcntl.LOCK_NB and call_count["n"] < 2:
                call_count["n"] += 1
                raise BlockingIOError("lock held by another process")
            # On the third attempt (or any unlock), call the real flock without LOCK_NB
            return real_flock(fd, op & ~fcntl.LOCK_NB)

        monkeypatch.setattr(_concurrency.fcntl, "flock", flaky_flock)
        sleep_mock = AsyncMock()
        monkeypatch.setattr(_concurrency.asyncio, "sleep", sleep_mock)

        async with _concurrency.file_lock(lock_path):
            pass

        # We raised BlockingIOError twice, so asyncio.sleep must have been awaited twice
        assert call_count["n"] == 2, "Did not retry twice as expected"
        assert sleep_mock.await_count == 2, (
            f"asyncio.sleep called {sleep_mock.await_count} times, expected 2. "
            "If this is 0, file_lock is blocking the thread instead of polling."
        )


class TestRetryAsync:
    """Tests for retry_async — predicate-based retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_first_call_success_no_sleep(self, monkeypatch):
        """When the function succeeds immediately, no retries or sleeps occur.

        Mutation detected: accidentally always sleeping once would fail this.
        """
        from quality import _concurrency

        sleep_mock = AsyncMock()
        monkeypatch.setattr(_concurrency.asyncio, "sleep", sleep_mock)

        async def success():
            return "success"

        result = await _concurrency.retry_async(
            success,
            should_retry=lambda exc: True,
            delays=[0.1, 0.2],
        )

        assert result == "success"
        sleep_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_matching_predicate_raises_immediately(self, monkeypatch):
        """When should_retry returns False, the exception is raised immediately
        with zero sleeps.

        This pins the early-exit behavior that prevents wasting retries on
        non-transient errors (e.g. 404 vs 429).
        """
        from quality._concurrency import retry_async

        sleep_mock = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep_mock)

        class PermanentError(Exception):
            pass

        async def always_fails():
            raise PermanentError("not transient")

        with pytest.raises(PermanentError, match="not transient"):
            await retry_async(always_fails, should_retry=lambda exc: False, delays=[1.0, 2.0])

        sleep_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_matching_predicate_recovers_on_second_attempt(self, monkeypatch):
        """When should_retry returns True, the function is retried after the
        specified delay, and recovers on the second attempt."""
        from quality._concurrency import retry_async

        sleep_mock = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep_mock)

        call_count = {"n": 0}

        async def flaky():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("transient")
            return "recovered"

        result = await retry_async(flaky, should_retry=lambda exc: isinstance(exc, ValueError), delays=[0.5, 1.0])

        assert result == "recovered"
        assert call_count["n"] == 2
        sleep_mock.assert_awaited_once_with(0.5)

    @pytest.mark.asyncio
    async def test_exhaustion_re_raises_after_all_delays(self, monkeypatch):
        """After exhausting all delays, the last exception is re-raised.

        Pins that we retry exactly len(delays) times, then give up.
        """
        from quality._concurrency import retry_async

        sleep_mock = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep_mock)

        async def always_fails():
            raise RuntimeError("persistent failure")

        with pytest.raises(RuntimeError, match="persistent failure"):
            await retry_async(always_fails, should_retry=lambda exc: True, delays=[0.1, 0.2, 0.3])

        # Should have slept exactly three times (once per delay)
        assert sleep_mock.await_count == 3
        assert sleep_mock.await_args_list[0].args == (0.1,)
        assert sleep_mock.await_args_list[1].args == (0.2,)
        assert sleep_mock.await_args_list[2].args == (0.3,)

    @pytest.mark.asyncio
    async def test_on_retry_callback_invoked(self, monkeypatch):
        """The optional on_retry callback is called on each retry with
        (exc, attempt, delay), allowing logging without coupling this module
        to a specific log vocabulary."""
        from quality._concurrency import retry_async

        sleep_mock = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep_mock)

        call_count = {"n": 0}
        retry_log = []

        async def flaky():
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise ValueError(f"attempt {call_count['n']}")
            return "ok"

        def log_retry(exc, attempt, delay):
            retry_log.append((str(exc), attempt, delay))

        result = await retry_async(
            flaky,
            should_retry=lambda exc: isinstance(exc, ValueError),
            delays=[0.5, 1.0],
            on_retry=log_retry,
        )

        assert result == "ok"
        assert len(retry_log) == 2
        assert retry_log[0] == ("attempt 1", 1, 0.5)
        assert retry_log[1] == ("attempt 2", 2, 1.0)

    @pytest.mark.asyncio
    async def test_empty_delays_raises_immediately(self, monkeypatch):
        """When delays is empty, the first exception is raised without retries.

        This documents that delays=[] means "never retry".
        """
        from quality._concurrency import retry_async

        sleep_mock = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep_mock)

        async def always_fails():
            raise RuntimeError("no retries")

        with pytest.raises(RuntimeError, match="no retries"):
            await retry_async(always_fails, should_retry=lambda exc: True, delays=[])

        sleep_mock.assert_not_awaited()
