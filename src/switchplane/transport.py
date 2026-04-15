"""Unix domain socket transport layer for CLI ↔ Control Plane communication."""

import asyncio
import json
import os
import socket
import struct
from collections.abc import Callable
from pathlib import Path

from switchplane._util import MAX_MESSAGE_SIZE, read_frame, write_frame
from switchplane.protocol import CliRequest, CliResponse

# Server side (async, used by control plane)

# Re-export for backward compatibility
read_message = read_frame
write_message = write_frame


class SocketServer:
    """Async Unix domain socket server for control plane."""

    def __init__(
        self,
        sock_path: Path,
        handler: Callable[[CliRequest], CliResponse],
        stream_handler: Callable | None = None,
        system_stream_handler: Callable | None = None,
    ):
        """Initialize server with socket path and request handler.

        Args:
            sock_path: Path to Unix domain socket file
            handler: Async function that processes CliRequest and returns CliResponse
            stream_handler: Optional async function (request, writer) -> None called for
                ``subscribe_task`` requests. It takes ownership of the writer for the
                lifetime of the stream and must close it when done.
            system_stream_handler: Optional async function (request, writer) -> None
                called for ``subscribe_system`` requests. Same ownership semantics.
        """
        self.sock_path = sock_path
        self.handler = handler
        self.stream_handler = stream_handler
        self.system_stream_handler = system_stream_handler
        self.server: asyncio.Server | None = None
        self._active_connections: set[asyncio.StreamWriter] = set()

    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return len(self._active_connections)

    async def start(self) -> None:
        """Start the Unix domain socket server."""
        # Remove stale socket file if it exists
        if self.sock_path.exists():
            self.sock_path.unlink()

        # Ensure parent directory exists
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)

        self.server = await asyncio.start_unix_server(self._handle_connection, path=str(self.sock_path))

        # Restrict socket to owner only
        os.chmod(str(self.sock_path), 0o600)

    async def stop(self) -> None:
        """Stop the server and clean up."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        # Close all active connections
        for writer in list(self._active_connections):
            writer.close()
            await writer.wait_closed()
        self._active_connections.clear()

        # Remove socket file
        if self.sock_path.exists():
            self.sock_path.unlink()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a client connection."""
        self._active_connections.add(writer)

        try:
            while True:
                request_id = "unknown"
                try:
                    # Read message
                    message_bytes = await read_message(reader)

                    # Deserialize request
                    request_data = json.loads(message_bytes.decode("utf-8"))
                    request = CliRequest.model_validate(request_data)
                    request_id = request.id

                    # subscribe_task upgrades this connection to a push stream
                    if request.method == "subscribe_task" and self.stream_handler:
                        ack = CliResponse(id=request.id, ok=True, result="subscribed")
                        await write_message(writer, json.dumps(ack.model_dump()).encode("utf-8"))
                        await self.stream_handler(request, writer)
                        return  # writer cleanup handled in finally block below

                    # subscribe_system upgrades this connection to a system log stream
                    if request.method == "subscribe_system" and self.system_stream_handler:
                        ack = CliResponse(id=request.id, ok=True, result="subscribed")
                        await write_message(writer, json.dumps(ack.model_dump()).encode("utf-8"))
                        await self.system_stream_handler(request, writer)
                        return

                    # Normal request/response
                    response = await self.handler(request)

                    # Serialize and send response
                    response_bytes = json.dumps(response.model_dump()).encode("utf-8")
                    await write_message(writer, response_bytes)

                except asyncio.IncompleteReadError:
                    # Client disconnected
                    break
                except Exception as e:
                    # Send error response
                    error_response = CliResponse(id=request_id, ok=False, error=str(e))
                    response_bytes = json.dumps(error_response.model_dump()).encode("utf-8")
                    await write_message(writer, response_bytes)

        finally:
            self._active_connections.discard(writer)
            writer.close()
            await writer.wait_closed()


# Client side (synchronous, used by CLI)


class ControlPlaneClient:
    """Synchronous Unix domain socket client for CLI."""

    def __init__(self, sock_path: Path):
        """Initialize client with socket path.

        Args:
            sock_path: Path to Unix domain socket file
        """
        self.sock_path = sock_path
        self.socket: socket.socket | None = None

    def connect(self) -> None:
        """Connect to the control plane server."""
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(str(self.sock_path))

    def close(self) -> None:
        """Close the socket connection."""
        if self.socket:
            self.socket.close()
            self.socket = None

    def send(self, request: CliRequest) -> CliResponse:
        """Send a request and receive a response.

        Args:
            request: The CLI request to send

        Returns:
            The CLI response from the server

        Raises:
            RuntimeError: If not connected
        """
        if not self.socket:
            raise RuntimeError("Not connected to control plane")

        # Serialize request
        request_bytes = json.dumps(request.model_dump()).encode("utf-8")

        # Send with length prefix
        self.socket.sendall(struct.pack(">I", len(request_bytes)) + request_bytes)

        # Read response length
        length_bytes = self._recv_exactly(4)
        length = struct.unpack(">I", length_bytes)[0]
        if length > MAX_MESSAGE_SIZE:
            raise ValueError(f"Message size {length} exceeds limit of {MAX_MESSAGE_SIZE}")

        # Read response data
        response_bytes = self._recv_exactly(length)

        # Deserialize response
        response_data = json.loads(response_bytes.decode("utf-8"))
        return CliResponse.model_validate(response_data)

    def _recv_exactly(self, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = self.socket.recv(remaining)
            if not chunk:
                raise ConnectionError("Socket closed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def is_alive(sock_path: Path) -> bool:
    """Check if control plane is alive by attempting to connect.

    Args:
        sock_path: Path to Unix domain socket file

    Returns:
        True if control plane is listening, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(sock_path))
        sock.close()
        return True
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        return False
