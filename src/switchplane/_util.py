"""Shared internal utilities."""

import asyncio
import struct
from typing import Any

MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64 MB


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Merge *overrides* into *base* in place. Nested dicts are merged recursively."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def encode_frame(data: bytes) -> bytes:
    """Prepend a 4-byte big-endian length header to *data*."""
    return struct.pack(">I", len(data)) + data


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read a length-prefixed frame, enforcing MAX_MESSAGE_SIZE."""
    length_bytes = await reader.readexactly(4)
    length = struct.unpack(">I", length_bytes)[0]
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message size {length} exceeds limit of {MAX_MESSAGE_SIZE}")
    return await reader.readexactly(length)


async def write_frame(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Write a length-prefixed frame and drain."""
    writer.write(encode_frame(data))
    await writer.drain()
