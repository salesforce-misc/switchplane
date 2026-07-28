"""Rate-limit retry wrapper for LLM API calls.

This example fans out N domains x M models concurrently against the same API key,
which is exactly the shape that trips a 429. The rate-limit retry wrapper prevents
a single 429 from silently halving the review by catching and retrying rate-limit
errors.

Must be applied AFTER ``bind_tools`` / ``with_structured_output`` so the retry sits
on the outermost ``ainvoke`` that callers (including ``run_tool_loop``) invoke.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

_RATE_LIMIT_MAX_ATTEMPTS = 6
_RATE_LIMIT_MAX_WAIT = 90.0  # cap any single sleep (seconds)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Whether *exc* is an HTTP 429 rate-limit error.

    Checks both a ``status_code`` attribute (if present) and string representation
    for rate-limit indicators.

    Args:
        exc: Exception to check.

    Returns:
        True if the exception appears to be a rate-limit error.
    """
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract retry wait time from exception response headers.

    Checks for a ``Retry-After`` header (both casings) in the response.

    Args:
        exc: Exception that may carry response headers.

    Returns:
        Seconds to wait, or None if no hint is present or invalid.
    """
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return None


def _retry_wait(exc: Exception, attempt: int) -> float:
    """Compute how long to sleep before retry *attempt* (1-based) for *exc*.

    Prefers a ``Retry-After`` header value if present, otherwise falls back to
    exponential backoff (2^(attempt-1)). The result is capped at 90 seconds and
    jitter in [0, 1) is added.

    **Jitter is critical:** without it, concurrent branches that all hit a 429
    at the same time would all retry at exactly the same moment, causing a
    thundering herd. Jitter spreads them out.

    Args:
        exc: The rate-limit exception.
        attempt: Retry attempt number (1-based).

    Returns:
        Seconds to sleep (base wait + jitter, capped at 90s + jitter).
    """
    wait = _retry_after_seconds(exc)
    if wait is None:
        wait = 2.0 ** (attempt - 1)  # exponential backoff: 1, 2, 4, 8, ...
    # Jitter spreads the concurrent branches so they don't all retry in lockstep.
    return min(wait, _RATE_LIMIT_MAX_WAIT) + random.uniform(0, 1)


class RateLimitRetry:
    """Wraps a LangChain Runnable to retry ``ainvoke`` on rate-limit errors.

    Transparent for all other attributes via ``__getattr__``, so it stays
    compatible with ``run_tool_loop`` and other LangChain patterns.

    **Must be applied AFTER** ``bind_tools`` / ``with_structured_output`` so
    the retry logic wraps the actual API call, not an intermediate chain step.

    Example:
        llm = build_llm("claude-sonnet-4-20250514", api_key=key)
        llm = llm.bind_tools([search_tool, calculator_tool])
        llm = with_rate_limit_retry(llm)  # Wrap AFTER bind_tools
        await run_tool_loop(llm, messages, tools, ctx, model_name)

    Args:
        inner: The runnable to wrap (typically a LangChain chat model).
    """

    def __init__(self, inner: Any):
        self._inner = inner

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the inner runnable, retrying on rate-limit errors.

        Retries up to ``_RATE_LIMIT_MAX_ATTEMPTS`` times with exponential
        backoff and jitter. Non-rate-limit errors are raised immediately.

        Args:
            *args: Positional arguments passed to the inner runnable.
            **kwargs: Keyword arguments passed to the inner runnable.

        Returns:
            The result from the inner runnable's ``ainvoke``.

        Raises:
            Exception: The last exception after exhausting retries, or any
                non-rate-limit exception immediately.
        """
        attempt = 0
        while True:
            try:
                return await self._inner.ainvoke(*args, **kwargs)
            except Exception as exc:
                attempt += 1
                if not _is_rate_limit_error(exc) or attempt >= _RATE_LIMIT_MAX_ATTEMPTS:
                    raise
                wait = _retry_wait(exc, attempt)
                await asyncio.sleep(wait)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the inner runnable.

        This is what keeps RateLimitRetry transparent to LangChain's tool loop,
        which may call methods like ``.bind_tools()`` or ``.with_structured_output()``.
        """
        return getattr(self._inner, name)


def with_rate_limit_retry(runnable: Any) -> RateLimitRetry:
    """Wrap a runnable with rate-limit retry logic.

    Args:
        runnable: The runnable to wrap (typically a LangChain chat model).

    Returns:
        A RateLimitRetry wrapper around the runnable.
    """
    return RateLimitRetry(runnable)
