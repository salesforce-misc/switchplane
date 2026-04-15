"""Shared structlog configuration for control plane and agent subprocesses.

Call ``configure`` once at process startup before any loggers are used.
File destinations get JSON output (machine-parseable); stderr gets the
human-readable ConsoleRenderer.
"""

import logging
import sys
from pathlib import Path

import structlog

_SKIP_FIELDS = frozenset({"event", "level", "logger", "timestamp", "_logger", "_name", "_record"})


def format_record(record: logging.LogRecord) -> tuple[str, str]:
    """Extract a human-readable (message, logger_name) from a log record.

    Can't use a standard ConsoleRenderer here: logs are transmitted over IPC
    and rendered by the TUI/CLI, which add their own timestamp and level styling.
    Running a full renderer would produce double timestamps and double level
    prefixes on the receiving end.  Instead we extract just the content —
    ``event  key=value ...`` — and let the display layer handle presentation.

    structlog's wrap_for_formatter stores the event dict as record.msg (a dict);
    _record is only set in a secondary code path.  For plain stdlib records
    (e.g. httpx) record.msg is already a string, so we fall back to getMessage().

    Returns:
        (message, logger_name)
    """
    logger_name: str = record.name
    event_dict = getattr(record, "_record", None)
    if event_dict is None and isinstance(record.msg, dict):
        event_dict = record.msg
    if event_dict and isinstance(event_dict, dict):
        event = str(event_dict.get("event", ""))
        extras = {k: v for k, v in event_dict.items() if k not in _SKIP_FIELDS}
        if extras:
            fields_str = "  ".join(f"{k}={v}" for k, v in extras.items())
            message = f"{event}  {fields_str}"
        else:
            message = event
        logger_name = str(event_dict.get("logger", record.name))
    else:
        message = record.getMessage()
    return message, logger_name


class StreamMessageFormatter(logging.Formatter):
    """Formatter for log records streamed to TUI/CLI over IPC.

    Produces just the message content (event + fields) without presentation
    chrome.  Attach to a handler via setFormatter() so the format can be
    swapped without touching handler code.
    """

    def format(self, record: logging.LogRecord) -> str:
        message, _ = format_record(record)
        return message


def configure(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure structlog with stdlib integration.

    Args:
        log_file: Write structured JSON log lines here when set.
                  Falls back to stderr with ConsoleRenderer when ``None``.
        level: Root log level (default: INFO).
    """
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer: structlog.types.Processor
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_file)
        renderer = structlog.processors.JSONRenderer()
    else:
        handler = logging.StreamHandler(sys.stderr)
        renderer = structlog.dev.ConsoleRenderer()

    formatter = structlog.stdlib.ProcessorFormatter(processor=renderer)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)
