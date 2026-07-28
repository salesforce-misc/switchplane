"""Generic async concurrency primitives — file locking and retry-on-predicate.

Deliberately kept in a separate module (no GitHub, SSH, or git dependencies)
so promoting them into src/switchplane/ later is a file move rather than
an untangle.
"""

from __future__ import annotations

import asyncio
import fcntl
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

_LOCK_POLL_SECONDS = 0.25


@asynccontextmanager
async def file_lock(lock_path: Path):
    """Acquire an exclusive file lock asynchronously, polling non-blockingly.

    **Critical design note:** The obvious implementation — opening a file and
    calling ``fcntl.flock(fd, LOCK_EX)`` — blocks the *thread* until the lock
    is granted. Inside an asyncio event loop, that freezes *every other coroutine*
    including the agent's IPC command listener, so the task stops responding to
    cancellation while waiting for the lock.

    This implementation polls ``LOCK_EX | LOCK_NB`` and yields the event loop via
    ``asyncio.sleep`` on contention. That keeps concurrent tasks able to progress
    and allows cancel signals to arrive mid-wait.

    Args:
        lock_path: Path to the lock file (created if it doesn't exist).

    Yields:
        None (context manager yields control to the caller).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = lock_path.open("w")
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(_LOCK_POLL_SECONDS)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


async def retry_async[T](
    fn: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    delays: list[float],
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> T:
    """Retry an async function on transient failures with exponential backoff.

    Calls ``await fn()``; on exception, if ``should_retry(exc)`` is False,
    re-raises immediately. Otherwise, sleeps ``delays[attempt]`` and retries.
    After exhausting all delays, re-raises the last exception.

    Taking a predicate rather than an exception type allows domain-specific
    retry decisions (e.g. SSH key errors vs 429 rate limits) to live with
    their domain code.

    Args:
        fn: Async callable to retry.
        should_retry: Predicate taking an exception and returning True if
            the error is transient.
        delays: List of sleep durations (in seconds) between retries.
            An empty list means "never retry".
        on_retry: Optional callback invoked on each retry with
            (exc, attempt, delay) for logging.

    Returns:
        The result of ``fn()`` on success.

    Raises:
        The last exception after exhausting retries, or the first exception
        if ``should_retry`` returns False.
    """
    last_exc: Exception | None = None

    for attempt in range(len(delays) + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if not should_retry(exc):
                raise
            if attempt < len(delays):
                delay = delays[attempt]
                if on_retry is not None:
                    on_retry(exc, attempt + 1, delay)
                await asyncio.sleep(delay)

    # Exhausted all retries
    if last_exc is not None:
        raise last_exc
    # Should never reach here (fn must have raised or returned by now)
    raise RuntimeError("retry_async: unreachable")
