"""Owned Docker and native HTTPS peers for the same host security campaign."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import uuid

from tests.utils.tls import make_tls_contexts


@dataclass
class MCPTestPeer:
    kind: str
    selector: str
    witness_file: Path | None = None

    async def witness(self, feature, name):
        if self.kind == "remote":
            return self.witness_file.read_text().splitlines() if self.witness_file.exists() else []
        container = feature.manager.owned_containers[name]["container"]
        result = await asyncio.to_thread(container.exec_run, ["python", "-c",
            'from pathlib import Path; p=Path("/tmp/permission-witness"); '
            'print(p.read_text() if p.exists() else "", end="")'])
        assert result.exit_code == 0, result.output
        return result.output.decode().splitlines()

    def connection(self, feature, name):
        if self.kind == "remote":
            return feature._remote_connections[name].connection
        return feature.manager.owned_containers[name]["connection"]


@asynccontextmanager
async def docker_peer(tmp_path):
    import docker

    build = tmp_path / "peer"
    build.mkdir()
    shutil.copyfile(Path(__file__).with_name("mcp_permission_peer.py"), build / "peer.py")
    (build / "Dockerfile").write_text(
        'FROM python:3.11-slim\nWORKDIR /app\nRUN pip install "mcp>=2.3,<3"\n'
        'COPY peer.py .\nCMD ["python", "peer.py"]\n')
    image = f"mcp-{uuid.uuid4().hex}:test"
    async with AsyncExitStack() as cleanup:
        client = await asyncio.to_thread(docker.from_env)
        cleanup.push_async_callback(asyncio.to_thread, client.close)
        await asyncio.to_thread(client.images.build, path=str(build), tag=image, rm=True)
        cleanup.push_async_callback(asyncio.to_thread, client.images.remove, image)
        yield MCPTestPeer("container", image)


@asynccontextmanager
async def https_peer(tmp_path, monkeypatch):
    import httpcore2
    import docker

    hostname = "remote-permission.example"
    _, client_context, certificate = make_tls_contexts(tmp_path, hostname, ip_addresses=())
    ready, witness = tmp_path / "peer-ready.json", tmp_path / "remote-witness"
    log = tmp_path / "remote-peer.log"
    with log.open("wb") as output:
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).with_name("mcp_permission_peer.py")),
            "--tls-cert", str(certificate), "--tls-key", str(tmp_path / "private-key.pem"),
            "--witness", str(witness), "--ready", str(ready),
            env={"PATH": os.environ["PATH"], "PYTHONUNBUFFERED": "1"},
            stdout=output, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            async def await_ready():
                while True:
                    if process.returncode is not None:
                        raise RuntimeError("Private MCP peer exited: " + log.read_text())
                    if ready.exists():
                        port = json.loads(ready.read_text())["port"]
                        try:
                            _, writer = await asyncio.open_connection(
                                "127.0.0.1", port, ssl=client_context, server_hostname=hostname)
                        except ConnectionRefusedError:
                            pass
                        else:
                            writer.close()
                            await writer.wait_closed()
                            return port
                    await asyncio.sleep(0.01)
            port = await asyncio.wait_for(await_ready(), 10)
            original = httpcore2.AnyIOBackend.connect_tcp

            def resolve(host, requested_port, *, type, proto):
                assert host == hostname and requested_port == port
                return [(socket.AF_INET, type, proto, "", ("8.8.8.8", port))]

            async def dial(backend, host, port, **kwargs):
                assert host == "8.8.8.8"
                return await original(backend, "127.0.0.1", port, **kwargs)

            def refuse_docker(*args, **kwargs):
                raise AssertionError("Remote MCP path attempted to acquire Docker")

            with monkeypatch.context() as network:
                network.setattr(socket, "getaddrinfo", resolve)
                network.setattr(httpcore2.AnyIOBackend, "connect_tcp", dial)
                network.setattr(docker, "from_env", refuse_docker)
                network.setenv("SSL_CERT_FILE", str(certificate))
                network.delenv("SSL_CERT_DIR", raising=False)
                network.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
                network.setenv("ALL_PROXY", "http://127.0.0.1:1")
                network.setenv("NO_PROXY", "")
                yield MCPTestPeer("remote", f"https://{hostname}:{port}/mcp", witness)
        finally:
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), 5)
                raise
