"""Tests for switchplane.tui — TUISession state, input dispatch, UI rendering.

prompt_toolkit's Application/key-binding machinery is excluded from unit tests
(it requires a real PTY and running event loop integration); everything that
can be exercised without a terminal is covered here.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from switchplane.protocol import CliResponse
from switchplane.tui import (
    _S_ERROR,
    _S_INFO,
    _SYSTEM_TAB_ID,
    EventBuffer,
    TUISession,
    _parse_kv_args,
    build_tui_app,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(result=None) -> CliResponse:
    return CliResponse(id="x", ok=True, result=result)


def _err(error="something went wrong") -> CliResponse:
    return CliResponse(id="x", ok=False, error=error)


def _line_segments(line: tuple) -> list[tuple[str, str]]:
    """Flatten a (prefix, content) line into a single list of (style, text) tuples."""
    prefix, content = line
    return list(prefix) + list(content)


def _system_lines(session: TUISession) -> list[str]:
    """Return all text fragments in the system tab buffer."""
    buf = session.buffers[_SYSTEM_TAB_ID]
    return [text for line in buf.lines for _style, text in _line_segments(line)]


@pytest.fixture
def session(tmp_path):
    return TUISession(sock_path=tmp_path / "test.sock")


# ---------------------------------------------------------------------------
# EventBuffer
# ---------------------------------------------------------------------------


class TestEventBuffer:
    def test_defaults(self):
        buf = EventBuffer(task_id="t1", agent_name="ag", task_name="task")
        assert buf.status == "pending"
        assert buf.last_event_id == 0
        assert buf.lines == []
        assert buf.vertical_scroll == 0
        assert buf.auto_scroll is True

    def test_custom_status(self):
        buf = EventBuffer(task_id="t1", agent_name="ag", task_name="task", status="running")
        assert buf.status == "running"


# ---------------------------------------------------------------------------
# TUISession.__init__
# ---------------------------------------------------------------------------


class TestTUISessionInit:
    def test_system_tab_created(self, session):
        assert _SYSTEM_TAB_ID in session.buffers
        assert session.buffers[_SYSTEM_TAB_ID].task_name == "system"

    def test_focused_on_system_at_start(self, session):
        assert session.focused_task_id == _SYSTEM_TAB_ID

    def test_task_order_empty(self, session):
        assert session.task_order == []

    def test_streams_dict_empty(self, session):
        assert session.streams == {}

    def test_app_none(self, session):
        assert session._app is None

    def test_custom_max_buffer_lines(self, tmp_path):
        s = TUISession(tmp_path / "s.sock", max_buffer_lines=500)
        assert s.max_buffer_lines == 500

    def test_default_spinner_interval(self, session):
        # Default lives on the TuiConfig but is mirrored as
        # `_DEFAULT_SPINNER_INTERVAL` in tui.py for ad-hoc launches.
        assert session.spinner_interval == 0.5

    def test_custom_spinner_interval(self, tmp_path):
        s = TUISession(tmp_path / "s.sock", spinner_interval=0.25)
        assert s.spinner_interval == 0.25


# ---------------------------------------------------------------------------
# add_task
# ---------------------------------------------------------------------------


class TestAddTask:
    def test_adds_buffer_and_order(self, session):
        session.add_task("t1", "agent", "task1")
        assert "t1" in session.buffers
        assert "t1" in session.task_order

    def test_no_duplicate(self, session):
        session.add_task("t1", "agent", "task1")
        session.add_task("t1", "agent", "task1")
        assert session.task_order.count("t1") == 1

    def test_status_stored(self, session):
        session.add_task("t1", "agent", "task1", status="running")
        assert session.buffers["t1"].status == "running"

    def test_multiple_tasks_ordered(self, session):
        session.add_task("t1", "a", "x")
        session.add_task("t2", "a", "y")
        assert session.task_order == ["t1", "t2"]


# ---------------------------------------------------------------------------
# _all_tab_ids
# ---------------------------------------------------------------------------


class TestAllTabIds:
    def test_just_system(self, session):
        assert session._all_tab_ids() == [_SYSTEM_TAB_ID]

    def test_system_plus_tasks(self, session):
        session.add_task("t1", "a", "x")
        session.add_task("t2", "a", "y")
        assert session._all_tab_ids() == [_SYSTEM_TAB_ID, "t1", "t2"]


# ---------------------------------------------------------------------------
# focus_slot
# ---------------------------------------------------------------------------


class TestFocusSlot:
    def test_slot_0_focuses_system(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session.focus_slot(0)
        assert session.focused_task_id == _SYSTEM_TAB_ID

    def test_slot_1_focuses_first_task(self, session):
        session.add_task("t1", "a", "t")
        session.focus_slot(1)
        assert session.focused_task_id == "t1"

    def test_slot_2_focuses_second_task(self, session):
        session.add_task("t1", "a", "t1")
        session.add_task("t2", "a", "t2")
        session.focus_slot(2)
        assert session.focused_task_id == "t2"

    def test_out_of_bounds_is_ignored(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session.focus_slot(5)
        assert session.focused_task_id == "t1"  # unchanged

    def test_slot_0_with_no_tasks(self, session):
        session.focus_slot(0)
        assert session.focused_task_id == _SYSTEM_TAB_ID


# ---------------------------------------------------------------------------
# focus_next / focus_prev
# ---------------------------------------------------------------------------


class TestFocusNavigation:
    def test_focus_next_from_system(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = _SYSTEM_TAB_ID
        session.focus_next()
        assert session.focused_task_id == "t1"

    def test_focus_next_wraps_around(self, session):
        session.add_task("t1", "a", "t1")
        session.add_task("t2", "a", "t2")
        session.focused_task_id = "t2"
        session.focus_next()
        assert session.focused_task_id == _SYSTEM_TAB_ID

    def test_focus_next_only_system_tab(self, session):
        session.focused_task_id = _SYSTEM_TAB_ID
        session.focus_next()
        assert session.focused_task_id == _SYSTEM_TAB_ID

    def test_focus_next_unknown_current_goes_to_first(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "unknown"
        session.focus_next()
        assert session.focused_task_id == _SYSTEM_TAB_ID

    def test_focus_prev_from_first_task_wraps(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = _SYSTEM_TAB_ID
        session.focus_prev()
        assert session.focused_task_id == "t1"

    def test_focus_prev_from_system(self, session):
        session.add_task("t1", "a", "t1")
        session.add_task("t2", "a", "t2")
        session.focused_task_id = "t1"
        session.focus_prev()
        assert session.focused_task_id == _SYSTEM_TAB_ID

    def test_focus_prev_only_system(self, session):
        session.focused_task_id = _SYSTEM_TAB_ID
        session.focus_prev()
        assert session.focused_task_id == _SYSTEM_TAB_ID

    def test_focus_prev_unknown_current_goes_to_last(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "unknown"
        session.focus_prev()
        assert session.focused_task_id == "t1"


# ---------------------------------------------------------------------------
# detach_focused_task
# ---------------------------------------------------------------------------


class TestDetachFocusedTask:
    def test_removes_from_buffers_and_order(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session.detach_focused_task()
        assert "t1" not in session.buffers
        assert "t1" not in session.task_order

    def test_focuses_system_when_last_task_removed(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session.detach_focused_task()
        assert session.focused_task_id == _SYSTEM_TAB_ID

    def test_focuses_successor_when_not_last(self, session):
        session.add_task("t1", "a", "t1")
        session.add_task("t2", "a", "t2")
        session.focused_task_id = "t1"
        session.detach_focused_task()
        assert session.focused_task_id == "t2"

    def test_focuses_predecessor_when_last_in_list(self, session):
        session.add_task("t1", "a", "t1")
        session.add_task("t2", "a", "t2")
        session.focused_task_id = "t2"
        session.detach_focused_task()
        assert session.focused_task_id == "t1"

    def test_cancels_background_stream(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        mock_stream = MagicMock()
        session.streams["t1"] = mock_stream
        session.detach_focused_task()
        mock_stream.cancel.assert_called_once()
        assert "t1" not in session.streams

    def test_system_tab_is_ignored(self, session):
        session.focused_task_id = _SYSTEM_TAB_ID
        session.detach_focused_task()
        assert _SYSTEM_TAB_ID in session.buffers

    def test_none_focused_is_ignored(self, session):
        session.focused_task_id = None
        session.detach_focused_task()  # must not raise


# ---------------------------------------------------------------------------
# scroll_up / scroll_down
# ---------------------------------------------------------------------------


class TestScrolling:
    def _add_lines(self, session: TUISession, task_id: str, count: int) -> None:
        buf = session.buffers[task_id]
        for i in range(count):
            buf.lines.append(([], [(_S_INFO, f"line {i}")]))

    def _setup_task_window(self, session: TUISession, max_phys: int) -> None:
        """Attach a mock _task_window with the given _max_phys value."""
        window = MagicMock()
        window._max_phys = max_phys
        session._task_window = window

    def test_scroll_up_sets_vertical_scroll(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        buf = session.buffers["t1"]
        buf.auto_scroll = False
        buf.vertical_scroll = 50
        self._setup_task_window(session, max_phys=100)
        session.scroll_up(5)
        assert buf.vertical_scroll == 45

    def test_scroll_up_from_auto_scroll_seeds_from_max(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        buf = session.buffers["t1"]
        # auto_scroll is True by default
        self._setup_task_window(session, max_phys=100)
        session.scroll_up(5)
        assert buf.vertical_scroll == 95
        assert buf.auto_scroll is False

    def test_scroll_up_clamps_to_zero(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        buf = session.buffers["t1"]
        buf.auto_scroll = False
        buf.vertical_scroll = 3
        self._setup_task_window(session, max_phys=100)
        session.scroll_up(10)
        assert buf.vertical_scroll == 0

    def test_scroll_up_noop_with_no_focused_buf(self, session):
        session.focused_task_id = None
        session.scroll_up()  # must not raise

    def test_scroll_up_noop_without_task_window(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session._task_window = None
        session.scroll_up(5)  # noop — no _task_window
        assert session.buffers["t1"].auto_scroll is True
        assert session.buffers["t1"].vertical_scroll == 0

    def test_scroll_down_sets_vertical_scroll(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        buf = session.buffers["t1"]
        buf.auto_scroll = False
        buf.vertical_scroll = 50
        self._setup_task_window(session, max_phys=180)
        session.scroll_down(10)
        assert buf.vertical_scroll == 60

    def test_scroll_down_enables_auto_scroll_at_bottom(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        buf = session.buffers["t1"]
        buf.auto_scroll = False
        buf.vertical_scroll = 170
        self._setup_task_window(session, max_phys=180)
        session.scroll_down(15)
        assert buf.auto_scroll is True

    def test_scroll_down_noop_with_no_focused_buf(self, session):
        session.focused_task_id = None
        session.scroll_down()  # must not raise

    def test_scroll_down_noop_without_task_window(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session._task_window = None
        session.scroll_down(5)  # noop — no _task_window
        assert session.buffers["t1"].auto_scroll is True
        assert session.buffers["t1"].vertical_scroll == 0


# ---------------------------------------------------------------------------
# _append_line (buffer management)
# ---------------------------------------------------------------------------


class TestAppendLine:
    def test_appends_to_buffer(self, session):
        session._append_line(_SYSTEM_TAB_ID, [], [(_S_INFO, "hello")])
        assert session.buffers[_SYSTEM_TAB_ID].lines == [([], [(_S_INFO, "hello")])]

    def test_trims_oldest_when_over_limit(self, tmp_path):
        s = TUISession(tmp_path / "s.sock", max_buffer_lines=5)
        for i in range(7):
            s._append_line(_SYSTEM_TAB_ID, [], [(_S_INFO, f"line {i}")])
        buf = s.buffers[_SYSTEM_TAB_ID]
        assert len(buf.lines) == 5
        # Lines 0 and 1 were trimmed
        assert buf.lines[0] == ([], [(_S_INFO, "line 2")])
        assert buf.lines[-1] == ([], [(_S_INFO, "line 6")])

    def test_trim_adjusts_vertical_scroll(self, tmp_path):
        s = TUISession(tmp_path / "s.sock", max_buffer_lines=5)
        buf = s.buffers[_SYSTEM_TAB_ID]
        buf.auto_scroll = False
        for i in range(5):
            buf.lines.append(([], [(_S_INFO, f"line {i}")]))
        buf.vertical_scroll = 3
        # Adding one line → 6 > 5 → trim=1 → vertical_scroll becomes max(0, 3-1)=2
        s._append_line(_SYSTEM_TAB_ID, [], [(_S_INFO, "new")])
        assert buf.vertical_scroll == 2

    def test_unknown_task_id_is_noop(self, session):
        session._append_line("nonexistent", [], [(_S_INFO, "x")])  # must not raise

    def test_calls_refresh_with_app_set(self, session):
        mock_app = MagicMock()
        # See TestRefreshDebounce / test_refresh_calls_invalidate_when_app_set:
        # `loop=None` + no running loop steers `_refresh` to the
        # direct-invalidate fallback (legacy path). The timer-armed
        # path is covered separately under TestRefreshDebounce.
        mock_app.loop = None
        session._app = mock_app
        session._append_line(_SYSTEM_TAB_ID, [], [(_S_INFO, "x")])
        mock_app.invalidate.assert_called()


class TestRefreshDebounce:
    """`_refresh()` coalesces multiple rapid invalidates into a single
    redraw on a `_REFRESH_DEBOUNCE_SECONDS` timer.

    Background: high event rate (e.g. an LLM tool loop firing dozens
    of `tool.invoke` events per second) was triggering one
    `Application.invalidate()` per `_append_line`, pinning the
    prompt_toolkit renderer at 100% CPU re-rendering the whole
    scrollback. py-spy dump on a wedged session showed the main
    thread spending 100% CPU in `split_lines / create_content`. The
    debounce caps effective redraw rate so the renderer can drain
    the event queue between frames.
    """

    async def test_coalesces_burst_into_single_invalidate(self, session):
        from switchplane.tui import _REFRESH_DEBOUNCE_SECONDS

        mock_app = MagicMock()
        # `_refresh` follows prompt_toolkit's convention:
        # `self._app.loop or asyncio.get_running_loop()`. Force the
        # `or` branch by setting `loop = None` so the test exercises
        # the path most production callers take pre-`run_async`.
        # `MagicMock().loop` is itself a (truthy) MagicMock by
        # default, so without this the test would call
        # `mock_loop.call_later(...)` and never schedule a real timer.
        mock_app.loop = None
        session._app = mock_app

        # 10 rapid appends within one event-loop iteration. Inside an
        # event loop, `_refresh` arms a timer instead of calling
        # `invalidate` directly.
        for i in range(10):
            session._append_line(_SYSTEM_TAB_ID, [], [(_S_INFO, f"line {i}")])

        # Burst phase: timer pending, no invalidate yet.
        assert mock_app.invalidate.call_count == 0
        assert session._refresh_timer is not None

        # Let the timer fire — wait slightly longer than the debounce.
        await asyncio.sleep(_REFRESH_DEBOUNCE_SECONDS + 0.01)

        # Exactly one redraw for the whole burst.
        assert mock_app.invalidate.call_count == 1
        assert session._refresh_timer is None

    async def test_subsequent_burst_after_fire_arms_new_timer(self, session):
        from switchplane.tui import _REFRESH_DEBOUNCE_SECONDS

        mock_app = MagicMock()
        mock_app.loop = None  # see comment in coalesces test above
        session._app = mock_app

        session._append_line(_SYSTEM_TAB_ID, [], [(_S_INFO, "first burst")])
        await asyncio.sleep(_REFRESH_DEBOUNCE_SECONDS + 0.01)
        assert mock_app.invalidate.call_count == 1

        session._append_line(_SYSTEM_TAB_ID, [], [(_S_INFO, "second burst")])
        await asyncio.sleep(_REFRESH_DEBOUNCE_SECONDS + 0.01)
        # Timer re-armed and fired again — debounce doesn't permanently
        # gate redraws, just throttles rate.
        assert mock_app.invalidate.call_count == 2

    def test_refresh_outside_event_loop_falls_back_to_direct_invalidate(self, session):
        """If `_refresh` is called outside a running event loop (test
        fixtures, pre-startup paths, the existing
        `test_calls_refresh_with_app_set` shape), arm-a-timer can't
        work — fall back to a direct `invalidate()` so legacy callers
        and tests don't silently lose their redraw."""
        mock_app = MagicMock()
        # `loop=None` forces the get_running_loop fallback; combined
        # with no running loop in this synchronous test, both branches
        # fail and we land on the direct-invalidate fallback.
        mock_app.loop = None
        session._app = mock_app
        # Synchronous (no event loop): direct invalidate, no timer.
        session._refresh()
        assert mock_app.invalidate.call_count == 1
        assert session._refresh_timer is None

    def test_refresh_uses_app_loop_when_set(self, session):
        """When `self._app.loop` is set (i.e. `Application.run_async`
        is in flight), `_refresh` schedules on that loop directly
        instead of falling through to `asyncio.get_running_loop()`.
        This matches prompt_toolkit's own
        `self.loop or get_running_loop()` convention from
        `Application.create_background_task`."""
        mock_app = MagicMock()
        mock_loop = MagicMock()
        mock_app.loop = mock_loop
        session._app = mock_app

        session._refresh()
        # Timer scheduled on the app's loop, not via get_running_loop.
        mock_loop.call_later.assert_called_once()
        # No direct invalidate — the timer is the path.
        mock_app.invalidate.assert_not_called()


# ---------------------------------------------------------------------------
# _focused_buf / _system_message / _append_text
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    def test_focused_buf_returns_system(self, session):
        assert session._focused_buf() == session.buffers[_SYSTEM_TAB_ID]

    def test_focused_buf_returns_task_buf(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        assert session._focused_buf() == session.buffers["t1"]

    def test_focused_buf_none_when_no_focus(self, session):
        session.focused_task_id = None
        assert session._focused_buf() is None

    def test_focused_buf_none_when_id_not_in_buffers(self, session):
        session.focused_task_id = "ghost"
        assert session._focused_buf() is None

    def test_system_message_appends_to_system_tab(self, session):
        session._system_message("hello world")
        lines = _system_lines(session)
        assert "hello world" in lines

    def test_append_text_appends_styled(self, session):
        session._append_text(_SYSTEM_TAB_ID, _S_ERROR, "oops")
        buf = session.buffers[_SYSTEM_TAB_ID]
        assert buf.lines[-1] == ([], [(_S_ERROR, "oops")])

    def test_refresh_noop_when_app_is_none(self, session):
        session._refresh()  # must not raise

    def test_refresh_calls_invalidate_when_app_set(self, session):
        mock_app = MagicMock()
        # `loop=None` + no running loop → direct invalidate fallback,
        # which is what this legacy test exercises. With a real loop
        # the path arms a timer instead; the dedicated burst tests
        # in TestRefreshDebounce cover that branch.
        mock_app.loop = None
        session._app = mock_app
        session._refresh()
        mock_app.invalidate.assert_called_once()


# ---------------------------------------------------------------------------
# start_stream
# ---------------------------------------------------------------------------


class TestStartStream:
    @pytest.mark.asyncio
    async def test_creates_asyncio_task(self, session):
        session.add_task("t1", "a", "t")
        # _stream_loop will immediately fail (no socket) – that is fine for
        # this test, we just want to confirm a task is created.
        session.start_stream("t1")
        assert "t1" in session.streams
        # clean up
        session.streams["t1"].cancel()
        try:
            await session.streams["t1"]
        except (asyncio.CancelledError, Exception):
            pass

    @pytest.mark.asyncio
    async def test_does_not_duplicate_stream(self, session):
        session.add_task("t1", "a", "t")
        session.start_stream("t1")
        first = session.streams["t1"]
        session.start_stream("t1")
        assert session.streams["t1"] is first
        first.cancel()
        try:
            await first
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# UI text providers
# ---------------------------------------------------------------------------


class TestGetTabBarText:
    def test_system_tab_focused(self, session):
        text = session.get_tab_bar_text()
        styles = {s for s, _ in text}
        assert "class:tab.focused" in styles

    def test_system_tab_inactive_when_task_focused(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        text = session.get_tab_bar_text()
        # System entry uses class:tab.inactive
        assert any(t == "  [0] system" and s == "class:tab.inactive" for s, t in text)

    def test_task_tab_shows_label_and_icon(self, session):
        session.add_task("t1", "agent", "task")
        session.buffers["t1"].status = "running"
        text = session.get_tab_bar_text()
        flat = [(s, t) for s, t in text]
        labels = [t for _s, t in flat]
        assert any("agent/task" in lbl for lbl in labels)
        # Running tasks show a spinner frame instead of a static icon
        from switchplane.tui import _SPINNER_FRAMES

        assert any(any(c in lbl for c in _SPINNER_FRAMES) for lbl in labels)

    def test_task_tab_without_agent_name(self, session):
        session.add_task("t1", "", "task")
        text = session.get_tab_bar_text()
        labels = [t for _s, t in text]
        assert any("task" in lbl for lbl in labels)
        assert not any("/" in lbl for lbl in labels if "task" in lbl)

    def test_completed_task_shows_checkmark(self, session):
        session.add_task("t1", "a", "t")
        session.buffers["t1"].status = "completed"
        text = session.get_tab_bar_text()
        assert any("✓" in t for _s, t in text)

    def test_failed_task_shows_cross(self, session):
        session.add_task("t1", "a", "t")
        session.buffers["t1"].status = "failed"
        text = session.get_tab_bar_text()
        assert any("✗" in t for _s, t in text)

    def test_unknown_status_falls_back_to_pending_icon(self, session):
        session.add_task("t1", "a", "t")
        session.buffers["t1"].status = "nonexistent_status"
        text = session.get_tab_bar_text()
        assert any("○" in t for _s, t in text)


class TestGetEventPaneText:
    def test_empty_system_tab_shows_help(self, session):
        result = session.get_event_pane_text()
        combined = "".join(t for _s, t in result)
        assert ":help" in combined

    def test_no_focused_buf_shows_default(self, session):
        session.focused_task_id = None
        result = session.get_event_pane_text()
        combined = "".join(t for _s, t in result)
        assert "No tab selected" in combined

    def test_renders_lines(self, session):
        session._append_line(_SYSTEM_TAB_ID, [], [(_S_INFO, "event text")])
        result = session.get_event_pane_text()
        texts = [t for _s, t in result]
        assert any("event text" in t for t in texts)

    def test_renders_all_lines(self, session):
        # get_event_pane_text now renders ALL lines; scroll is handled by
        # _ScrollableWindow, not by the text provider.
        for i in range(20):
            session._append_line(_SYSTEM_TAB_ID, [], [(_S_INFO, f"line {i}")])
        result = session.get_event_pane_text()
        texts = [t for _s, t in result]
        # All lines should be present in the rendered output
        assert any("line 0" in t for t in texts)
        assert any("line 19" in t for t in texts)


class TestGetStatusBarText:
    def test_system_tab(self, session):
        result = session.get_status_bar_text()
        texts = [t for _s, t in result]
        assert any("system" in t for t in texts)

    def test_task_tab_with_agent(self, session):
        session.add_task("t1", "myagent", "mytask", status="running")
        session.focused_task_id = "t1"
        result = session.get_status_bar_text()
        combined = "".join(t for _s, t in result)
        assert "myagent/mytask" in combined
        assert "running" in combined
        assert "t1" in combined

    def test_task_tab_without_agent(self, session):
        session.add_task("t1", "", "mytask", status="pending")
        session.focused_task_id = "t1"
        result = session.get_status_bar_text()
        # The label fragment (class:status.bar.label) must not include a "/"
        label = next((t for s, t in result if s == "class:status.bar.label"), "")
        assert "mytask" in label
        assert "/" not in label

    def test_no_focused_buf_fallback(self, session):
        session.focused_task_id = None
        result = session.get_status_bar_text()
        combined = "".join(t for _s, t in result)
        assert "switchplane" in combined

    def test_hints_present(self, session):
        result = session.get_status_bar_text()
        combined = "".join(t for _s, t in result)
        assert "Tab" in combined


class TestGetPromptText:
    def test_system_tab(self, session):
        result = session.get_prompt_text()
        combined = "".join(t for _s, t in result)
        assert "[system]" in combined

    def test_task_with_agent(self, session):
        session.add_task("t1", "myagent", "mytask")
        session.focused_task_id = "t1"
        result = session.get_prompt_text()
        combined = "".join(t for _s, t in result)
        assert "[myagent/mytask]" in combined

    def test_task_without_agent(self, session):
        session.add_task("t1", "", "mytask")
        session.focused_task_id = "t1"
        result = session.get_prompt_text()
        combined = "".join(t for _s, t in result)
        assert "[mytask]" in combined

    def test_no_focused_buf_fallback(self, session):
        session.focused_task_id = None
        result = session.get_prompt_text()
        combined = "".join(t for _s, t in result)
        assert "[switchplane]" in combined

    def test_prompt_arrow_present(self, session):
        result = session.get_prompt_text()
        assert any(">" in t for _s, t in result)


# ---------------------------------------------------------------------------
# handle_input routing
# ---------------------------------------------------------------------------


class TestHandleInput:
    @pytest.mark.asyncio
    async def test_empty_input_is_noop(self, session):
        session._handle_daemon_command = AsyncMock()
        session._handle_task_command = AsyncMock()
        await session.handle_input("   ")
        session._handle_daemon_command.assert_not_called()
        session._handle_task_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_colon_routes_to_daemon_command(self, session):
        session._handle_daemon_command = AsyncMock()
        await session.handle_input(":runtime status")
        session._handle_daemon_command.assert_awaited_once_with("runtime status")

    @pytest.mark.asyncio
    async def test_slash_routes_to_task_command(self, session):
        session._handle_task_command = AsyncMock()
        await session.handle_input("/set_coords --lat 1.0")
        session._handle_task_command.assert_awaited_once_with("set_coords --lat 1.0")


# ---------------------------------------------------------------------------
# _handle_task_command
# ---------------------------------------------------------------------------


class TestHandleTaskCommand:
    @pytest.mark.asyncio
    async def test_no_focused_task_shows_system_message(self, session):
        session.focused_task_id = None
        await session._handle_task_command("go")
        assert any("No task" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_system_tab_focused_shows_message(self, session):
        session.focused_task_id = _SYSTEM_TAB_ID
        await session._handle_task_command("go")
        assert any("No task" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_empty_command_raises_and_shows_error(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session._request = AsyncMock()
        await session._handle_task_command("")
        # ValueError from _parse_kv_args("Empty command") — shown in task buffer
        buf = session.buffers["t1"]
        lines = [text for line in buf.lines for _style, text in _line_segments(line)]
        assert any("Invalid" in line for line in lines)

    @pytest.mark.asyncio
    async def test_sends_command_and_shows_confirmation(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session._request = AsyncMock(return_value=_ok({}))
        await session._handle_task_command("set_speed --value 10")
        session._request.assert_awaited_once()
        buf = session.buffers["t1"]
        assert any("set_speed sent" in t for _s, t in _line_segments(buf.lines[-1]))

    @pytest.mark.asyncio
    async def test_error_response_shows_error_in_task_tab(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session._request = AsyncMock(return_value=_err("task not running"))
        await session._handle_task_command("do_thing")
        buf = session.buffers["t1"]
        assert any("task not running" in t for _s, t in _line_segments(buf.lines[-1]))


# ---------------------------------------------------------------------------
# _handle_daemon_command routing
# ---------------------------------------------------------------------------


class TestHandleDaemonCommandRouting:
    @pytest.mark.asyncio
    async def test_empty_parts_is_noop(self, session):
        await session._handle_daemon_command("   ")  # must not raise

    @pytest.mark.asyncio
    async def test_help_routes_to_cmd_help(self, session):
        session._cmd_help = MagicMock()
        await session._handle_daemon_command("help")
        session._cmd_help.assert_called_once()

    @pytest.mark.asyncio
    async def test_question_mark_routes_to_help(self, session):
        session._cmd_help = MagicMock()
        await session._handle_daemon_command("?")
        session._cmd_help.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_routes_correctly(self, session):
        session._cmd_run = AsyncMock()
        await session._handle_daemon_command("run agent task --x y")
        session._cmd_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_task_routes_to_dispatch(self, session):
        session._dispatch_task_group = AsyncMock()
        await session._handle_daemon_command("task list")
        session._dispatch_task_group.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_runtime_routes_to_dispatch(self, session):
        session._dispatch_runtime_group = AsyncMock()
        await session._handle_daemon_command("runtime status")
        session._dispatch_runtime_group.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_routes_to_dispatch(self, session):
        session._dispatch_agent_group = AsyncMock()
        await session._handle_daemon_command("agent list")
        session._dispatch_agent_group.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_command_shows_error(self, session):
        await session._handle_daemon_command("gobbledygook")
        assert any("Unknown command" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_non_run_commands_auto_focus_system(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session._cmd_help = MagicMock()
        await session._handle_daemon_command("help")
        assert session.focused_task_id == _SYSTEM_TAB_ID

    @pytest.mark.asyncio
    async def test_run_does_not_auto_focus_system(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session._cmd_run = AsyncMock()
        await session._handle_daemon_command("run")
        # focus should NOT have been changed to system
        assert session.focused_task_id == "t1"


# ---------------------------------------------------------------------------
# _dispatch_task_group
# ---------------------------------------------------------------------------


class TestDispatchTaskGroup:
    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self, session):
        await session._dispatch_task_group([])
        assert any("Usage" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_follow_routed(self, session):
        session._cmd_task_follow = AsyncMock()
        await session._dispatch_task_group(["follow", "t1"])
        session._cmd_task_follow.assert_awaited_once_with(["t1"])

    @pytest.mark.asyncio
    async def test_cancel_routed(self, session):
        session._cmd_cancel = AsyncMock()
        await session._dispatch_task_group(["cancel"])
        session._cmd_cancel.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_clear_routed(self, session):
        session._cmd_clear = AsyncMock()
        await session._dispatch_task_group(["clear"])
        session._cmd_clear.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_routed(self, session):
        session._cmd_list = AsyncMock()
        await session._dispatch_task_group(["list"])
        session._cmd_list.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_show_routed(self, session):
        session._cmd_task_show = AsyncMock()
        await session._dispatch_task_group(["show", "t1"])
        session._cmd_task_show.assert_awaited_once_with(["t1"])

    @pytest.mark.asyncio
    async def test_retry_routed(self, session):
        session._cmd_task_retry = AsyncMock()
        await session._dispatch_task_group(["retry", "t1"])
        session._cmd_task_retry.assert_awaited_once_with(["t1"])

    @pytest.mark.asyncio
    async def test_unknown_subcommand_shows_error(self, session):
        await session._dispatch_task_group(["bogus"])
        assert any("Unknown task command" in line for line in _system_lines(session))


# ---------------------------------------------------------------------------
# _dispatch_runtime_group
# ---------------------------------------------------------------------------


class TestDispatchRuntimeGroup:
    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self, session):
        await session._dispatch_runtime_group([])
        assert any("Usage" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_status_routed(self, session):
        session._cmd_status = AsyncMock()
        await session._dispatch_runtime_group(["status"])
        session._cmd_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_subcommand_shows_error(self, session):
        await session._dispatch_runtime_group(["restart"])
        assert any("Unknown runtime command" in line for line in _system_lines(session))


# ---------------------------------------------------------------------------
# _dispatch_agent_group
# ---------------------------------------------------------------------------


class TestDispatchAgentGroup:
    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self, session):
        await session._dispatch_agent_group([])
        assert any("Usage" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_list_routed(self, session):
        session._cmd_agent_list = AsyncMock()
        await session._dispatch_agent_group(["list"])
        session._cmd_agent_list.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_subcommand_shows_error(self, session):
        await session._dispatch_agent_group(["show"])
        assert any("Unknown agent command" in line for line in _system_lines(session))


# ---------------------------------------------------------------------------
# _cmd_help
# ---------------------------------------------------------------------------


class TestCmdHelp:
    def test_appends_help_lines_to_system(self, session):
        session._cmd_help()
        lines = _system_lines(session)
        assert any(":run" in line for line in lines)
        assert any(":task" in line for line in lines)
        assert any(":help" in line for line in lines)
        assert any("Tab" in line for line in lines)


# ---------------------------------------------------------------------------
# _cmd_run
# ---------------------------------------------------------------------------


class TestCmdRun:
    @pytest.mark.asyncio
    async def test_missing_args_shows_usage(self, session):
        await session._cmd_run([])
        assert any("Usage" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_missing_task_shows_usage(self, session):
        await session._cmd_run(["agent"])
        assert any("Usage" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_invalid_args_shows_error(self, session):
        session._request = AsyncMock()
        # Provide a dangling --flag that would parse but cause issues isn't
        # straightforward — instead pass something that triggers the ValueError
        # by patching _parse_kv_args to raise
        import switchplane.tui as tui_mod

        original = tui_mod._parse_kv_args
        tui_mod._parse_kv_args = lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("bad"))
        try:
            await session._cmd_run(["agent", "task", "--x"])
        finally:
            tui_mod._parse_kv_args = original
        assert any("Invalid args" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_api_error_shows_message(self, session):
        session._request = AsyncMock(return_value=_err("unknown task"))
        await session._cmd_run(["agent", "task"])
        assert any("unknown task" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_success_adds_task_and_starts_stream(self, session):
        session._request = AsyncMock(return_value=_ok({"task_id": "abc123"}))
        session.start_stream = MagicMock()
        await session._cmd_run(["agent", "task"])
        assert "abc123" in session.buffers
        assert session.focused_task_id == "abc123"
        session.start_stream.assert_called_once_with("abc123")

    @pytest.mark.asyncio
    async def test_success_with_params(self, session):
        session._request = AsyncMock(return_value=_ok({"task_id": "t99"}))
        session.start_stream = MagicMock()
        await session._cmd_run(["agent", "task", "--name", "Alice"])
        call_params = session._request.call_args[0]
        assert call_params[1]["input"]["name"] == "Alice"


# ---------------------------------------------------------------------------
# _cmd_task_follow
# ---------------------------------------------------------------------------


class TestCmdTaskFollow:
    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self, session):
        await session._cmd_task_follow([])
        assert any("Usage" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_already_following_focuses_and_messages(self, session):
        session.add_task("t1", "a", "t")
        session._request = AsyncMock()
        await session._cmd_task_follow(["t1"])
        assert session.focused_task_id == "t1"
        assert any("Already following" in line for line in _system_lines(session))
        session._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_error_shows_message(self, session):
        session._request = AsyncMock(return_value=_err("not found"))
        await session._cmd_task_follow(["t99"])
        assert any("not found" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_success_adds_task_and_starts_stream(self, session):
        session._request = AsyncMock(
            return_value=_ok({"task": {"agent_name": "ag", "task_name": "wk", "status": "running"}})
        )
        session.start_stream = MagicMock()
        await session._cmd_task_follow(["t99"])
        assert "t99" in session.buffers
        assert session.focused_task_id == "t99"
        session.start_stream.assert_called_once_with("t99")
        assert any("Following" in line for line in _system_lines(session))


# ---------------------------------------------------------------------------
# _cmd_task_show
# ---------------------------------------------------------------------------


class TestCmdTaskShow:
    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self, session):
        await session._cmd_task_show([])
        assert any("Usage" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_api_error_shows_message(self, session):
        session._request = AsyncMock(return_value=_err("not found"))
        await session._cmd_task_show(["t99"])
        assert any("not found" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_displays_task_fields(self, session):
        session._request = AsyncMock(
            return_value=_ok(
                {
                    "task": {
                        "task_id": "t1",
                        "agent_name": "ag",
                        "task_name": "wk",
                        "status": "completed",
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-01T00:01:00Z",
                        "input_json": '{"x": 1}',
                        "result_json": '{"y": 2}',
                        "error_json": None,
                    }
                }
            )
        )
        await session._cmd_task_show(["t1"])
        lines = _system_lines(session)
        assert any("t1" in line for line in lines)
        assert any("ag" in line for line in lines)
        assert any("completed" in line for line in lines)
        assert any('{"x": 1}' in line for line in lines)
        assert any('"y"' in line for line in lines)

    @pytest.mark.asyncio
    async def test_displays_error_json_when_present(self, session):
        session._request = AsyncMock(
            return_value=_ok(
                {
                    "task": {
                        "task_id": "t1",
                        "agent_name": "ag",
                        "task_name": "wk",
                        "status": "failed",
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-01T00:01:00Z",
                        "error_json": '{"error": "boom"}',
                    }
                }
            )
        )
        await session._cmd_task_show(["t1"])
        assert any("boom" in line for line in _system_lines(session))


# ---------------------------------------------------------------------------
# _cmd_task_retry
# ---------------------------------------------------------------------------


class TestCmdTaskRetry:
    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self, session):
        await session._cmd_task_retry([])
        assert any("Usage" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_api_error_shows_message(self, session):
        session._request = AsyncMock(return_value=_err("cannot retry"))
        await session._cmd_task_retry(["t1"])
        assert any("cannot retry" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_task_in_buffers_resets_status(self, session):
        session.add_task("t1", "a", "t", status="failed")
        session._request = AsyncMock(return_value=_ok({}))
        session.start_stream = MagicMock()
        await session._cmd_task_retry(["t1"])
        assert session.buffers["t1"].status == "pending"
        assert session.focused_task_id == "t1"
        session.start_stream.assert_called_once_with("t1")

    @pytest.mark.asyncio
    async def test_task_not_in_buffers_fetches_metadata(self, session):
        def side_effect(method, params=None):
            if method == "retry_from_checkpoint":
                return _ok({})
            if method == "get_task":
                return _ok({"task": {"agent_name": "ag", "task_name": "wk"}})

        session._request = AsyncMock(side_effect=side_effect)
        session.start_stream = MagicMock()
        await session._cmd_task_retry(["newt"])
        assert "newt" in session.buffers

    @pytest.mark.asyncio
    async def test_shows_retried_message(self, session):
        session.add_task("t1", "a", "t", status="failed")
        session._request = AsyncMock(return_value=_ok({}))
        session.start_stream = MagicMock()
        await session._cmd_task_retry(["t1"])
        assert any("Retried" in line for line in _system_lines(session))


# ---------------------------------------------------------------------------
# _cmd_agent_list
# ---------------------------------------------------------------------------


class TestCmdAgentList:
    @pytest.mark.asyncio
    async def test_api_error_shows_message(self, session):
        session._request = AsyncMock(return_value=_err("db error"))
        await session._cmd_agent_list()
        assert any("db error" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_empty_list_shows_message(self, session):
        session._request = AsyncMock(return_value=_ok([]))
        await session._cmd_agent_list()
        assert any("No agents" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_displays_agents_and_tasks(self, session):
        session._request = AsyncMock(
            return_value=_ok(
                [
                    {
                        "name": "myagent",
                        "tasks": {"hello": {"description": "Says hello", "mode": "ephemeral"}},
                    }
                ]
            )
        )
        await session._cmd_agent_list()
        lines = _system_lines(session)
        assert any("myagent" in line for line in lines)
        assert any("hello" in line for line in lines)

    @pytest.mark.asyncio
    async def test_long_running_tasks_show_mode_tag(self, session):
        session._request = AsyncMock(
            return_value=_ok(
                [
                    {
                        "name": "watcher",
                        "tasks": {"watch": {"mode": "long_running", "description": "Watches stuff"}},
                    }
                ]
            )
        )
        await session._cmd_agent_list()
        lines = _system_lines(session)
        assert any("[long_running]" in line for line in lines)

    @pytest.mark.asyncio
    async def test_agent_with_no_tasks(self, session):
        session._request = AsyncMock(return_value=_ok([{"name": "idle", "tasks": {}}]))
        await session._cmd_agent_list()
        lines = _system_lines(session)
        assert any("no tasks" in line for line in lines)


# ---------------------------------------------------------------------------
# _cmd_cancel
# ---------------------------------------------------------------------------


class TestCmdCancel:
    @pytest.mark.asyncio
    async def test_no_focused_task_shows_message(self, session):
        session.focused_task_id = None
        await session._cmd_cancel([])
        assert any("No task" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_system_tab_focused_shows_message(self, session):
        session.focused_task_id = _SYSTEM_TAB_ID
        await session._cmd_cancel([])
        assert any("No task" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_cancels_focused_task(self, session):
        session.add_task("t1", "a", "t", status="running")
        session.focused_task_id = "t1"
        session._request = AsyncMock(return_value=_ok({"cancelled": True}))
        await session._cmd_cancel([])
        assert any("cancelled" in line.lower() for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_cancels_specific_task_by_arg(self, session):
        session._request = AsyncMock(return_value=_ok({"cancelled": True}))
        await session._cmd_cancel(["some-id"])
        call_params = session._request.call_args[0]
        assert call_params[1]["task_id"] == "some-id"

    @pytest.mark.asyncio
    async def test_not_running_shows_appropriate_message(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session._request = AsyncMock(return_value=_ok({"cancelled": False}))
        await session._cmd_cancel([])
        assert any("not running" in line.lower() for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_api_error_shows_message(self, session):
        session.add_task("t1", "a", "t")
        session.focused_task_id = "t1"
        session._request = AsyncMock(return_value=_err("already done"))
        await session._cmd_cancel([])
        assert any("already done" in line for line in _system_lines(session))


# ---------------------------------------------------------------------------
# _cmd_clear
# ---------------------------------------------------------------------------


class TestCmdClear:
    @pytest.mark.asyncio
    async def test_api_error_shows_message(self, session):
        session._request = AsyncMock(return_value=_err("db error"))
        await session._cmd_clear()
        assert any("db error" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_removes_terminal_tasks(self, session):
        for tid, status in [("t1", "completed"), ("t2", "failed"), ("t3", "running")]:
            session.add_task(tid, "a", "t", status=status)
        session._request = AsyncMock(return_value=_ok({"deleted": 2}))
        await session._cmd_clear()
        assert "t1" not in session.buffers
        assert "t2" not in session.buffers
        assert "t3" in session.buffers  # running — not cleared

    @pytest.mark.asyncio
    async def test_cancels_streams_for_removed_tasks(self, session):
        session.add_task("t1", "a", "t", status="completed")
        mock_stream = MagicMock()
        session.streams["t1"] = mock_stream
        session._request = AsyncMock(return_value=_ok({"deleted": 1}))
        await session._cmd_clear()
        mock_stream.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_refocuses_when_focused_task_cleared(self, session):
        session.add_task("t1", "a", "t", status="completed")
        session.focused_task_id = "t1"
        session._request = AsyncMock(return_value=_ok({"deleted": 1}))
        await session._cmd_clear()
        assert session.focused_task_id != "t1"
        assert session.focused_task_id == _SYSTEM_TAB_ID

    @pytest.mark.asyncio
    async def test_shows_count_message(self, session):
        session._request = AsyncMock(return_value=_ok({"deleted": 3}))
        await session._cmd_clear()
        assert any("3" in line for line in _system_lines(session))


# ---------------------------------------------------------------------------
# _cmd_list
# ---------------------------------------------------------------------------


class TestCmdList:
    @pytest.mark.asyncio
    async def test_api_error_shows_message(self, session):
        session._request = AsyncMock(return_value=_err("db error"))
        await session._cmd_list()
        assert any("db error" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_empty_list_shows_message(self, session):
        session._request = AsyncMock(return_value=_ok([]))
        await session._cmd_list()
        assert any("No tasks" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_displays_tasks(self, session):
        session._request = AsyncMock(
            return_value=_ok([{"task_id": "abc", "agent_name": "ag", "task_name": "tk", "status": "running"}])
        )
        await session._cmd_list()
        lines = _system_lines(session)
        assert any("abc" in line for line in lines)
        assert any("running" in line for line in lines)

    @pytest.mark.asyncio
    async def test_status_filter_is_passed(self, session):
        session._request = AsyncMock(return_value=_ok([]))
        await session._cmd_list(["--status", "running"])
        call_params = session._request.call_args[0]
        assert call_params[1].get("status") == "running"

    @pytest.mark.asyncio
    async def test_no_filter_without_args(self, session):
        session._request = AsyncMock(return_value=_ok([]))
        await session._cmd_list([])
        call_params = session._request.call_args[0]
        assert "status" not in call_params[1]


# ---------------------------------------------------------------------------
# _cmd_status
# ---------------------------------------------------------------------------


class TestCmdStatus:
    @pytest.mark.asyncio
    async def test_api_error_shows_message(self, session):
        session._request = AsyncMock(return_value=_err("unreachable"))
        await session._cmd_status()
        assert any("unreachable" in line for line in _system_lines(session))

    @pytest.mark.asyncio
    async def test_displays_status_fields(self, session):
        session._request = AsyncMock(
            return_value=_ok({"active_agents": 2, "running_tasks": 3, "active_connections": 1})
        )
        await session._cmd_status()
        lines = _system_lines(session)
        combined = " ".join(lines)
        assert "2" in combined
        assert "3" in combined
        assert "1" in combined


# ---------------------------------------------------------------------------
# _fetch_terminal_result
# ---------------------------------------------------------------------------


class TestFetchTerminalResult:
    @pytest.mark.asyncio
    async def test_api_error_returns_early(self, session):
        session.add_task("t1", "a", "t")
        session._request = AsyncMock(return_value=_err("not found"))
        before = len(session.buffers["t1"].lines)
        await session._fetch_terminal_result("t1", "completed")
        assert len(session.buffers["t1"].lines) == before

    @pytest.mark.asyncio
    async def test_completed_no_longer_dumps_result_json(self, session):
        """`completed` doesn't print `result_json` any more — agents
        render their own summary via `stream.flush` events earlier in
        the live stream. The structured dict is still in sqlite for
        `:task show <id>` to surface on demand. Regression guard for
        the change that moved this responsibility back to the agent."""
        session.add_task("t1", "a", "t")
        session._request = AsyncMock(return_value=_ok({"task": {"result_json": '{"answer": 42}', "error_json": None}}))
        before = len(session.buffers["t1"].lines)
        await session._fetch_terminal_result("t1", "completed")
        # No new buffer lines — completed renders nothing here.
        assert len(session.buffers["t1"].lines) == before

    @pytest.mark.asyncio
    async def test_completed_without_result_json_no_output(self, session):
        session.add_task("t1", "a", "t")
        session._request = AsyncMock(return_value=_ok({"task": {"result_json": None, "error_json": None}}))
        before = len(session.buffers["t1"].lines)
        await session._fetch_terminal_result("t1", "completed")
        assert len(session.buffers["t1"].lines) == before

    @pytest.mark.asyncio
    async def test_failed_with_error_json_dict(self, session):
        session.add_task("t1", "a", "t")
        err_data = json.dumps({"error": "kaboom"})
        session._request = AsyncMock(return_value=_ok({"task": {"result_json": None, "error_json": err_data}}))
        await session._fetch_terminal_result("t1", "failed")
        assert any("kaboom" in t for line in session.buffers["t1"].lines for _s, t in _line_segments(line))

    @pytest.mark.asyncio
    async def test_failed_with_traceback_appended(self, session):
        session.add_task("t1", "a", "t")
        err_data = json.dumps({"error": "err", "traceback": "  File foo.py\n  line 2"})
        session._request = AsyncMock(return_value=_ok({"task": {"result_json": None, "error_json": err_data}}))
        await session._fetch_terminal_result("t1", "failed")
        texts = [t for line in session.buffers["t1"].lines for _s, t in _line_segments(line)]
        assert any("foo.py" in t for t in texts)

    @pytest.mark.asyncio
    async def test_failed_with_invalid_json_falls_back(self, session):
        session.add_task("t1", "a", "t")
        session._request = AsyncMock(return_value=_ok({"task": {"result_json": None, "error_json": "not json {{{"}}))
        await session._fetch_terminal_result("t1", "failed")
        assert any("not json" in t for line in session.buffers["t1"].lines for _s, t in _line_segments(line))


# ---------------------------------------------------------------------------
# _render_event
# ---------------------------------------------------------------------------


class TestRenderEvent:
    def test_renders_event_and_appends_lines(self, session):
        session.add_task("t1", "a", "t")
        ev = {
            "timestamp": "2024-01-01T12:00:00.000Z",
            "event_type": "task.progress",
            "payload": {"message": "Working..."},
        }
        before = len(session.buffers["t1"].lines)
        session._render_event("t1", ev)
        assert len(session.buffers["t1"].lines) > before

    def test_applies_style_mapping(self, session):
        session.add_task("t1", "a", "t")
        ev = {
            "timestamp": "2024-01-01T12:00:00.000Z",
            "event_type": "task.started",
            "payload": {},
        }
        session._render_event("t1", ev)
        buf = session.buffers["t1"]
        styles = {s for line in buf.lines for s, _t in _line_segments(line)}
        # At least one style class should be used
        assert len(styles) > 0


# ---------------------------------------------------------------------------
# _request
# ---------------------------------------------------------------------------


class TestRequest:
    @pytest.mark.asyncio
    async def test_oserror_returns_error_response(self, session):
        import switchplane.transport as transport_mod

        original = transport_mod.ControlPlaneClient
        transport_mod.ControlPlaneClient = lambda *_: (_ for _ in ()).throw(OSError("refused"))
        try:
            resp = await session._request("status")
        finally:
            transport_mod.ControlPlaneClient = original
        assert not resp.ok
        assert "Daemon unreachable" in resp.error

    @pytest.mark.asyncio
    async def test_successful_request_returns_response(self, session, monkeypatch):
        expected = CliResponse(id="r1", ok=True, result={"pong": True})

        async def fake_to_thread(fn, *args, **kwargs):
            return expected

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        resp = await session._request("ping")
        assert resp.ok
        assert resp.result["pong"] is True


# ---------------------------------------------------------------------------
# build_tui_app
# ---------------------------------------------------------------------------


class TestBuildTuiApp:
    def test_returns_prompt_toolkit_application(self, session):
        from prompt_toolkit import Application

        app = build_tui_app(session)
        assert isinstance(app, Application)

    def test_sets_session_app(self, session):
        build_tui_app(session)
        assert session._app is not None

    def test_app_is_full_screen(self, session):
        app = build_tui_app(session)
        assert app.full_screen is True


# ---------------------------------------------------------------------------
# _parse_kv_args (edge cases beyond test_cli.py)
# ---------------------------------------------------------------------------


class TestParseKvArgsEdgeCases:
    def test_inline_equals_syntax(self):
        action, params = _parse_kv_args(["cmd", "--key=value"])
        assert action == "cmd"
        assert params == {"key": "value"}

    def test_empty_parts_with_start_index_1_raises(self):
        with pytest.raises(ValueError, match="Empty command"):
            _parse_kv_args([])

    def test_start_index_0_returns_empty_action(self):
        action, params = _parse_kv_args(["--x", "1"], start_index=0)
        assert action == ""
        assert params == {"x": "1"}

    def test_flag_followed_by_another_flag(self):
        _, params = _parse_kv_args(["cmd", "--verbose", "--debug"])
        assert params == {"verbose": True, "debug": True}

    def test_hyphen_normalised_to_underscore(self):
        _, params = _parse_kv_args(["cmd", "--dry-run"])
        assert "dry_run" in params
