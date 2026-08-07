"""Tests for quality/ratelimit.py — 429 retry wrapper for LLM fan-out.

All tests monkeypatch asyncio.sleep so the suite never actually sleeps. The
jitter test is critical: it proves concurrent branches don't retry in lockstep,
which is what prevents a thundering herd after a 429.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class Mock429Error(Exception):
    """Fake HTTP 429 error with optional status_code and response.headers."""

    def __init__(self, message="Rate limit exceeded", status_code=429, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        if retry_after is not None:
            self.response = MagicMock()
            self.response.headers = {"Retry-After": str(retry_after)}
        else:
            self.response = None


class TestIsRateLimitError:
    """Tests for _is_rate_limit_error predicate."""

    def test_status_code_429_recognized(self):
        """An exception with status_code == 429 must be recognized."""
        from quality.ratelimit import _is_rate_limit_error

        exc = Mock429Error(status_code=429)
        assert _is_rate_limit_error(exc) is True

    def test_429_in_string_lowercase_recognized(self):
        """An exception containing '429' in its string representation is recognized."""
        from quality.ratelimit import _is_rate_limit_error

        exc = Exception("HTTP error 429: too many requests")
        assert _is_rate_limit_error(exc) is True

    def test_rate_limit_in_string_recognized(self):
        """An exception containing 'rate limit' (case-insensitive) is recognized."""
        from quality.ratelimit import _is_rate_limit_error

        exc = Exception("Rate Limit Exceeded")
        assert _is_rate_limit_error(exc) is True

    def test_non_rate_limit_error_not_recognized(self):
        """A generic error without 429 or rate-limit indicators is not recognized.

        This pins the early-exit behavior that prevents wasting retries on
        non-transient errors like 404 or 500.
        """
        from quality.ratelimit import _is_rate_limit_error

        exc = ValueError("Invalid input")
        assert _is_rate_limit_error(exc) is False


class TestRetryAfterSeconds:
    """Tests for _retry_after_seconds — extracting wait time from response headers."""

    def test_retry_after_header_with_seconds(self):
        """Retry-After header with integer seconds is parsed correctly."""
        from quality.ratelimit import _retry_after_seconds

        exc = Mock429Error(retry_after=12)
        assert _retry_after_seconds(exc) == 12.0

    def test_retry_after_header_case_insensitive(self):
        """Both 'Retry-After' and 'retry-after' header casings are checked."""
        from quality.ratelimit import _retry_after_seconds

        exc = Mock429Error(retry_after=5)
        # Manually set lowercase variant
        exc.response.headers = {"retry-after": "5"}
        assert _retry_after_seconds(exc) == 5.0

    def test_no_retry_after_header_returns_none(self):
        """When no Retry-After header is present, returns None."""
        from quality.ratelimit import _retry_after_seconds

        exc = Mock429Error(status_code=429)
        assert _retry_after_seconds(exc) is None

    def test_invalid_retry_after_header_returns_none(self):
        """Non-numeric Retry-After header is ignored (returns None)."""
        from quality.ratelimit import _retry_after_seconds

        exc = Mock429Error(retry_after="invalid")
        assert _retry_after_seconds(exc) is None

    def test_no_response_attribute_returns_none(self):
        """An exception without a response attribute returns None."""
        from quality.ratelimit import _retry_after_seconds

        exc = Exception("Generic error")
        assert _retry_after_seconds(exc) is None


class TestRetryWait:
    """Tests for _retry_wait — computing wait time with jitter."""

    def test_header_value_preferred_over_exponential(self, monkeypatch):
        """When Retry-After header is present, it is used instead of exponential backoff."""
        from quality.ratelimit import _retry_wait

        exc = Mock429Error(retry_after=12)
        # Patch random to return 0.5 for predictable jitter
        import random

        monkeypatch.setattr(random, "uniform", lambda a, b: 0.5)

        wait = _retry_wait(exc, attempt=3)
        # Should be ~12 + 0.5 jitter, not 2^(3-1) = 4
        assert 12.0 <= wait < 13.0

    def test_exponential_backoff_when_no_header(self, monkeypatch):
        """Without a Retry-After header, uses 2^(attempt-1) exponential backoff."""
        from quality.ratelimit import _retry_wait

        exc = Mock429Error(status_code=429)
        import random

        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)

        # attempt=1 → 2^0 = 1, attempt=2 → 2^1 = 2, attempt=3 → 2^2 = 4
        assert _retry_wait(exc, attempt=1) == 1.0
        assert _retry_wait(exc, attempt=2) == 2.0
        assert _retry_wait(exc, attempt=3) == 4.0

    def test_wait_capped_at_90_seconds(self, monkeypatch):
        """Long waits (from header or exponential) are capped at 90 seconds."""
        from quality.ratelimit import _retry_wait

        exc = Mock429Error(retry_after=200)
        import random

        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)

        wait = _retry_wait(exc, attempt=1)
        assert wait == 90.0

    def test_jitter_added_to_wait(self, monkeypatch):
        """Jitter is added to prevent concurrent branches from retrying in lockstep.

        This is the key property: without jitter, N branches that all hit a 429
        at the same time would all retry at exactly the same moment, causing a
        thundering herd.
        """
        from quality.ratelimit import _retry_wait

        exc = Mock429Error(status_code=429)
        import random

        # First call returns 0.3, second returns 0.7
        jitter_values = iter([0.3, 0.7])
        monkeypatch.setattr(random, "uniform", lambda a, b: next(jitter_values))

        wait1 = _retry_wait(exc, attempt=1)  # 1.0 + 0.3
        wait2 = _retry_wait(exc, attempt=1)  # 1.0 + 0.7

        assert 1.0 <= wait1 < 2.0
        assert 1.0 <= wait2 < 2.0
        assert wait1 != wait2  # Different jitter

    def test_jitter_range(self, monkeypatch):
        """Jitter is in [0, 1), keeping wait within [base, base+1)."""
        from quality.ratelimit import _retry_wait

        exc = Mock429Error(status_code=429)
        import random

        # Test both extremes
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
        wait_min = _retry_wait(exc, attempt=2)  # 2.0 + 0.0
        assert wait_min == 2.0

        monkeypatch.setattr(random, "uniform", lambda a, b: 0.999)
        wait_max = _retry_wait(exc, attempt=2)  # 2.0 + 0.999
        assert 2.0 <= wait_max < 3.0


class TestRateLimitRetry:
    """Tests for RateLimitRetry wrapper class."""

    @pytest.mark.asyncio
    async def test_non_429_error_raises_immediately(self, monkeypatch):
        """Non-rate-limit errors are raised immediately without retries.

        This pins the predicate guard that prevents wasting retries on errors
        like 404, 500, or ValueError.
        """
        from quality.ratelimit import RateLimitRetry

        from quality import ratelimit

        sleep_mock = AsyncMock()
        monkeypatch.setattr(ratelimit.asyncio, "sleep", sleep_mock)

        # Error message must not contain "429" or "rate limit" — those trigger retry
        async def fails(*args, **kwargs):
            raise ValueError("Invalid input parameter")

        inner = MagicMock()
        inner.ainvoke = fails
        wrapped = RateLimitRetry(inner)

        with pytest.raises(ValueError, match="Invalid input parameter"):
            await wrapped.ainvoke("arg")

        sleep_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_429_recovers_on_retry(self, monkeypatch):
        """A 429 error is retried and recovers on the second attempt."""
        from quality.ratelimit import RateLimitRetry

        from quality import ratelimit

        sleep_mock = AsyncMock()
        monkeypatch.setattr(ratelimit.asyncio, "sleep", sleep_mock)

        call_count = {"n": 0}

        async def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Mock429Error("Rate limited")
            return "success"

        inner = MagicMock()
        inner.ainvoke = flaky
        wrapped = RateLimitRetry(inner)

        result = await wrapped.ainvoke("arg")

        assert result == "success"
        assert call_count["n"] == 2
        sleep_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exhaustion_after_6_attempts_re_raises(self, monkeypatch):
        """After 6 failed attempts, the last exception is re-raised.

        Pins the max-attempts limit that prevents infinite retry loops.
        """
        from quality.ratelimit import RateLimitRetry

        from quality import ratelimit

        sleep_mock = AsyncMock()
        monkeypatch.setattr(ratelimit.asyncio, "sleep", sleep_mock)

        call_count = {"n": 0}

        async def always_fails_counting(*args, **kwargs):
            call_count["n"] += 1
            raise Mock429Error("Persistent 429")

        inner = MagicMock()
        inner.ainvoke = always_fails_counting
        wrapped = RateLimitRetry(inner)

        with pytest.raises(Mock429Error, match="Persistent 429"):
            await wrapped.ainvoke("arg")

        # 6 attempts means 5 waits between them; no sleep before the final re-raise
        assert call_count["n"] == 6, "Should attempt exactly _RATE_LIMIT_MAX_ATTEMPTS times"
        assert sleep_mock.await_count == 5

    @pytest.mark.asyncio
    async def test_retry_after_header_produces_correct_wait(self, monkeypatch):
        """A Retry-After: 12 header produces a ~12s wait."""
        from quality.ratelimit import RateLimitRetry

        from quality import ratelimit

        sleep_mock = AsyncMock()
        monkeypatch.setattr(ratelimit.asyncio, "sleep", sleep_mock)

        call_count = {"n": 0}

        async def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Mock429Error(retry_after=12)
            return "ok"

        inner = MagicMock()
        inner.ainvoke = flaky
        wrapped = RateLimitRetry(inner)

        await wrapped.ainvoke("arg")

        # Should have slept once for ~12 seconds (plus jitter)
        assert sleep_mock.await_count == 1
        slept = sleep_mock.await_args.args[0]
        assert 12.0 <= slept < 13.0  # 12 + jitter in [0, 1)

    @pytest.mark.asyncio
    async def test_getattr_delegates_to_inner(self):
        """__getattr__ must pass through arbitrary attributes to the inner runnable.

        This is what keeps RateLimitRetry transparent to LangChain's run_tool_loop,
        which may call .bind_tools(), .with_structured_output(), etc.
        """
        from quality.ratelimit import RateLimitRetry

        inner = MagicMock()
        inner.some_method = MagicMock(return_value="delegated")
        inner.some_attr = "attr_value"

        wrapped = RateLimitRetry(inner)

        assert wrapped.some_attr == "attr_value"
        assert wrapped.some_method() == "delegated"

    @pytest.mark.asyncio
    async def test_jitter_keeps_wait_in_range(self, monkeypatch):
        """Jitter keeps the wait within [base, base+1) as documented.

        This pins the jitter formula that prevents lockstep retries.
        """
        from quality.ratelimit import RateLimitRetry

        from quality import ratelimit

        sleep_mock = AsyncMock()
        monkeypatch.setattr(ratelimit.asyncio, "sleep", sleep_mock)

        # Make the first call fail with a 429, second call succeed
        call_count = {"n": 0}

        async def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Mock429Error(status_code=429)
            return "ok"

        inner = MagicMock()
        inner.ainvoke = flaky
        wrapped = RateLimitRetry(inner)

        await wrapped.ainvoke("arg")

        # First retry (attempt=1) → base wait is 2^0 = 1.0, jitter in [0, 1)
        slept = sleep_mock.await_args.args[0]
        assert 1.0 <= slept < 2.0, f"Wait {slept} outside [1.0, 2.0)"

    @pytest.mark.asyncio
    async def test_forwards_args_and_kwargs_to_inner(self, monkeypatch):
        """RateLimitRetry.ainvoke must forward positional and keyword arguments
        to the inner runnable unchanged.

        This pins the pass-through behavior that keeps the wrapper transparent
        to callers.
        """
        from quality.ratelimit import RateLimitRetry

        from quality import ratelimit

        sleep_mock = AsyncMock()
        monkeypatch.setattr(ratelimit.asyncio, "sleep", sleep_mock)

        captured = {}

        async def capture(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "result"

        inner = MagicMock()
        inner.ainvoke = capture
        wrapped = RateLimitRetry(inner)

        result = await wrapped.ainvoke("pos1", "pos2", key1="val1", key2="val2")

        assert result == "result"
        assert captured["args"] == ("pos1", "pos2")
        assert captured["kwargs"] == {"key1": "val1", "key2": "val2"}


class TestWithRateLimitRetry:
    """Tests for the with_rate_limit_retry factory function."""

    @pytest.mark.asyncio
    async def test_returns_wrapped_runnable(self):
        """with_rate_limit_retry must return a RateLimitRetry wrapper."""
        from quality.ratelimit import RateLimitRetry, with_rate_limit_retry

        inner = MagicMock()
        wrapped = with_rate_limit_retry(inner)

        assert isinstance(wrapped, RateLimitRetry)
        assert wrapped._inner is inner

    @pytest.mark.asyncio
    async def test_wrapped_runnable_retries_429(self, monkeypatch):
        """The wrapped runnable retries 429 errors as expected."""
        from quality.ratelimit import with_rate_limit_retry

        from quality import ratelimit

        sleep_mock = AsyncMock()
        monkeypatch.setattr(ratelimit.asyncio, "sleep", sleep_mock)

        call_count = {"n": 0}

        async def flaky():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Mock429Error("Rate limited")
            return "recovered"

        inner = MagicMock()
        inner.ainvoke = flaky
        wrapped = with_rate_limit_retry(inner)

        result = await wrapped.ainvoke()

        assert result == "recovered"
        assert call_count["n"] == 2
        sleep_mock.assert_awaited_once()
