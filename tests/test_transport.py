import asyncio
import struct

import pytest

from switchplane.protocol import CliRequest, CliResponse
from switchplane.transport import (
    ControlPlaneClient,
    SocketServer,
    is_alive,
    read_message,
)


class TestReadWriteMessage:
    @pytest.mark.asyncio
    async def test_round_trip(self):
        """Test read/write with connected asyncio streams."""
        reader = asyncio.StreamReader()
        payload = b'{"method": "test"}'
        framed = struct.pack(">I", len(payload)) + payload
        reader.feed_data(framed)

        result = await read_message(reader)
        assert result == payload

    @pytest.mark.asyncio
    async def test_empty_message(self):
        reader = asyncio.StreamReader()
        payload = b""
        framed = struct.pack(">I", 0) + payload
        reader.feed_data(framed)

        result = await read_message(reader)
        assert result == b""

    @pytest.mark.asyncio
    async def test_incomplete_read(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"\x00\x00")
        reader.feed_eof()

        with pytest.raises(asyncio.IncompleteReadError):
            await read_message(reader)


class TestSocketServerAndClient:
    @pytest.mark.asyncio
    async def test_round_trip(self, short_tmp):
        sock_path = short_tmp / "test.sock"

        async def handler(req: CliRequest) -> CliResponse:
            return CliResponse(id=req.id, ok=True, result={"echo": req.method})

        server = SocketServer(sock_path, handler)
        await server.start()
        assert server.server is not None

        def client_fn():
            client = ControlPlaneClient(sock_path)
            client.connect()
            resp = client.send(CliRequest(method="ping"))
            client.close()
            return resp

        resp = await asyncio.to_thread(client_fn)

        await server.stop()

        assert resp.ok
        assert resp.result["echo"] == "ping"

    @pytest.mark.asyncio
    async def test_multiple_requests(self, short_tmp):
        sock_path = short_tmp / "test.sock"

        async def handler(req: CliRequest) -> CliResponse:
            return CliResponse(id=req.id, ok=True, result=req.method)

        server = SocketServer(sock_path, handler)
        await server.start()

        def client_fn():
            client = ControlPlaneClient(sock_path)
            client.connect()
            r1 = client.send(CliRequest(method="first"))
            r2 = client.send(CliRequest(method="second"))
            client.close()
            return r1, r2

        r1, r2 = await asyncio.to_thread(client_fn)
        await server.stop()

        assert r1.result == "first"
        assert r2.result == "second"

    @pytest.mark.asyncio
    async def test_handler_error(self, short_tmp):
        sock_path = short_tmp / "test.sock"

        async def handler(req: CliRequest) -> CliResponse:
            raise ValueError("boom")

        server = SocketServer(sock_path, handler)
        await server.start()

        def client_fn():
            client = ControlPlaneClient(sock_path)
            client.connect()
            resp = client.send(CliRequest(method="fail"))
            client.close()
            return resp

        resp = await asyncio.to_thread(client_fn)
        await server.stop()

        assert not resp.ok
        assert "boom" in resp.error

    @pytest.mark.asyncio
    async def test_server_stop_removes_socket(self, short_tmp):
        sock_path = short_tmp / "test.sock"

        async def handler(req):
            return CliResponse(id=req.id, ok=True)

        server = SocketServer(sock_path, handler)
        await server.start()
        assert sock_path.exists()
        await server.stop()
        assert not sock_path.exists()

    @pytest.mark.asyncio
    async def test_connection_count(self, short_tmp):
        sock_path = short_tmp / "test.sock"

        async def handler(req):
            return CliResponse(id=req.id, ok=True)

        server = SocketServer(sock_path, handler)
        await server.start()
        assert server.connection_count == 0
        await server.stop()

    @pytest.mark.asyncio
    async def test_removes_stale_socket(self, short_tmp):
        sock_path = short_tmp / "test.sock"
        sock_path.write_text("stale")

        async def handler(req):
            return CliResponse(id=req.id, ok=True)

        server = SocketServer(sock_path, handler)
        await server.start()
        assert server.server is not None
        await server.stop()


class TestControlPlaneClient:
    def test_context_manager(self, tmp_path):
        sock_path = tmp_path / "test.sock"
        client = ControlPlaneClient(sock_path)
        assert client.socket is None
        client.close()

    def test_send_without_connect(self, tmp_path):
        sock_path = tmp_path / "test.sock"
        client = ControlPlaneClient(sock_path)
        with pytest.raises(RuntimeError, match="Not connected"):
            client.send(CliRequest(method="test"))


class TestIsAlive:
    def test_no_socket(self, tmp_path):
        assert is_alive(tmp_path / "nonexistent.sock") is False

    @pytest.mark.asyncio
    async def test_with_running_server(self, short_tmp):
        sock_path = short_tmp / "test.sock"

        async def handler(req):
            return CliResponse(id=req.id, ok=True)

        server = SocketServer(sock_path, handler)
        await server.start()

        result = await asyncio.to_thread(is_alive, sock_path)
        assert result is True

        await server.stop()

    def test_stale_socket_file(self, tmp_path):
        sock_path = tmp_path / "test.sock"
        sock_path.write_text("not a real socket")
        assert is_alive(sock_path) is False
