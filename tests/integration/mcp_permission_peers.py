"""Owned Docker and native HTTPS peers for the same host security campaign."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
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
    provider_url: str | None = None
    provider_witness: Path | None = None
    tls_certificate: Path | None = None
    local_ports: set[int] = field(default_factory=set)

    def provider_requests(self):
        if self.provider_witness is None:
            raise ValueError("This peer did not start an owned provider")
        return [json.loads(line) for line in self.provider_witness.read_text().splitlines()] if self.provider_witness.exists() else []

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


@contextmanager
def owned_provider_network(monkeypatch, peer):
    """Permit only the exact owned TLS listener for provider and MCP traffic."""
    import socket
    from urllib.parse import urlsplit

    endpoint = urlsplit(peer.provider_url or peer.selector)
    with monkeypatch.context() as network:
        network.setenv("NO_PROXY", endpoint.hostname)
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex
        dials = []

        def require_owned(address):
            assert isinstance(address, tuple) and address[0] == "127.0.0.1" and address[1] in {endpoint.port, *peer.local_ports}
            dials.append(address)

        def connect(sock, address):
            require_owned(address)
            return original_connect(sock, address)

        def connect_ex(sock, address):
            require_owned(address)
            return original_connect_ex(sock, address)

        network.setattr(socket.socket, "connect", connect)
        network.setattr(socket.socket, "connect_ex", connect_ex)
        yield dials



@contextmanager
def owned_https_network(monkeypatch, peer):
    """Pin public-looking fixture DNS to one real, certificate-verified listener."""
    from urllib.parse import urlsplit
    import docker
    import httpcore2

    endpoint = urlsplit(peer.selector)
    assert endpoint.scheme == "https" and endpoint.hostname and endpoint.port
    assert peer.tls_certificate is not None
    hostname, owned_port = endpoint.hostname, endpoint.port
    original = httpcore2.AnyIOBackend.connect_tcp

    def resolve(host, requested_port, *, type, proto):
        if host == "127.0.0.1" and requested_port in peer.local_ports:
            address = "127.0.0.1"
        else:
            assert host == hostname and requested_port == owned_port
            address = "8.8.8.8"
        return [(socket.AF_INET, type, proto, "", (address, requested_port))]

    async def dial(backend, host, port, **kwargs):
        if host == "127.0.0.1" and port in peer.local_ports:
            return await original(backend, host, port, **kwargs)
        allowed_hosts = {"8.8.8.8", hostname} if peer.provider_url else {"8.8.8.8"}
        assert host in allowed_hosts and port == owned_port
        return await original(backend, "127.0.0.1", port, **kwargs)

    def refuse_docker(*args, **kwargs):
        raise AssertionError("Remote MCP path attempted to acquire Docker")

    with monkeypatch.context() as network:
        network.setattr(socket, "getaddrinfo", resolve)
        network.setattr(httpcore2.AnyIOBackend, "connect_tcp", dial)
        network.setattr(docker, "from_env", refuse_docker)
        network.setenv("SSL_CERT_FILE", str(peer.tls_certificate))
        network.delenv("SSL_CERT_DIR", raising=False)
        network.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        network.setenv("ALL_PROXY", "http://127.0.0.1:1")
        network.setenv("NO_PROXY", "")
        yield


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
async def https_peer(tmp_path, monkeypatch, *, provider_model: str | None = None, provider_fault: str | None = None):
    hostname = "remote-permission.example"
    _, client_context, certificate = make_tls_contexts(tmp_path, hostname, ip_addresses=())
    ready, witness = tmp_path / "peer-ready.json", tmp_path / "remote-witness"
    log = tmp_path / "remote-peer.log"
    with log.open("wb") as output:
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).with_name("mcp_permission_peer.py")),
            "--tls-cert", str(certificate), "--tls-key", str(tmp_path / "private-key.pem"),
            "--witness", str(witness), "--ready", str(ready),
            *(["--provider-model", provider_model] if provider_model is not None else []),
            *(["--provider-fault", provider_fault] if provider_fault is not None else []),
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
            peer = MCPTestPeer("remote", f"https://{hostname}:{port}/mcp", witness,
                f"https://{hostname}:{port}/v1" if provider_model else None,
                witness.with_suffix(".provider.jsonl") if provider_model else None,
                tls_certificate=certificate)
            with owned_https_network(monkeypatch, peer):
                yield peer
        finally:
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), 5)
                raise
