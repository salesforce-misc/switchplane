"""Formatting utilities for event rendering and progress details."""

import json
from dataclasses import dataclass, field

# Semantic style names — consumers map these to their output format
TS = "ts"
INFO = "info"
DIM = "dim"
PROGRESS = "progress"
SUCCESS = "success"
WARN = "warn"
ERROR = "error"
LOG = "log"


@dataclass
class EventLine:
    """A single rendered line: a list of (style, text) segments."""

    segments: list[tuple[str, str]] = field(default_factory=list)


def render_event(event: dict) -> list[EventLine]:
    """Render an event dict into a list of styled lines.

    Each line contains (style_name, text) segments using the semantic
    style constants defined in this module. Consumers map these to
    their output format (click.echo, prompt_toolkit styled tuples, etc.).
    """
    raw_ts = event.get("timestamp", "")
    ts = raw_ts.split("T")[1].split(".")[0] if "T" in raw_ts else raw_ts
    etype = event.get("event_type", "")
    payload = event.get("payload", {})

    lines: list[EventLine] = []

    def main_line(style: str, msg: str) -> None:
        lines.append(EventLine([(TS, f"[{ts}] "), (style, msg)]))

    def continuation(style: str, text: str) -> None:
        lines.append(EventLine([(style, f"  {text}")]))

    if etype == "task.started":
        main_line(INFO, "Task started")
    elif etype == "task.progress":
        msg = payload.get("message", json.dumps(payload))
        parts = msg.split("\n")
        main_line(PROGRESS, parts[0])
        for cont in parts[1:]:
            continuation(PROGRESS, cont)
        for det in tree(payload.get("detail", [])):
            continuation(DIM, det)
    elif etype == "task.completed":
        main_line(SUCCESS, "Task completed")
    elif etype == "task.cancelled":
        main_line(WARN, "Task cancelled")
    elif etype == "task.interrupted":
        prompt = payload.get("prompt")
        msg = f"Waiting for input: {prompt}" if prompt else "Waiting for input..."
        main_line(WARN, msg)
    elif etype == "task.resumed":
        main_line(INFO, "Resumed")
    elif etype == "task.failed":
        main_line(ERROR, f"Task failed: {payload.get('error', '')}")
        if "traceback" in payload:
            for tb_line in payload["traceback"].splitlines():
                lines.append(EventLine([(ERROR, f"    {tb_line}")]))
    elif etype == "log":
        level = payload.get("level", "info")
        logger_name = payload.get("logger", "")
        style = {"warning": WARN, "error": ERROR, "debug": DIM}.get(level, LOG)
        prefix = f"[{logger_name}] " if logger_name else ""
        msg = f"[{level}] {prefix}{payload.get('message', '')}"
        parts = msg.split("\n")
        main_line(style, parts[0])
        for cont in parts[1:]:
            continuation(style, cont)
    elif etype == "system.log":
        level = payload.get("level", "info")
        logger_name = payload.get("logger", "")
        style = {"warning": WARN, "error": ERROR, "debug": DIM}.get(level, LOG)
        prefix = f"[{logger_name}] " if logger_name else ""
        msg = f"{prefix}{payload.get('message', '')}"
        parts = msg.split("\n")
        main_line(style, parts[0])
        for cont in parts[1:]:
            continuation(style, cont)
    elif etype == "task.command_result":
        action = payload.get("action", "unknown")
        result = payload.get("result", {})
        msg = f"\u21a9 {action}: {json.dumps(result)}"
        parts = msg.split("\n")
        main_line(INFO, parts[0])
        for cont in parts[1:]:
            continuation(INFO, cont)
    else:
        msg = f"{etype}: {json.dumps(payload)}"
        parts = msg.split("\n")
        main_line(DIM, parts[0])
        for cont in parts[1:]:
            continuation(DIM, cont)

    return lines


def format_result(result_json: str) -> list[str]:
    """Format a result_json string for display (no prefix).

    - str value  → output as-is, split on newlines
    - dict/list  → pretty-printed JSON
    - other      → str() representation
    """
    try:
        value = json.loads(result_json)
    except (ValueError, TypeError):
        return [result_json]

    if isinstance(value, str):
        return value.splitlines() or [""]
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2).splitlines()
    return [str(value)]


def tree(items: list[str]) -> list[str]:
    """Format a list of strings as a tree with box-drawing prefixes."""
    lines = []
    for i, item in enumerate(items):
        prefix = "\u2514\u2500\u2500" if i == len(items) - 1 else "\u251c\u2500\u2500"
        lines.append(f"{prefix} {item}")
    return lines
