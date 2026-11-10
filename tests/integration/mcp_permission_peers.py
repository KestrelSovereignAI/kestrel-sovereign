"""Owned Docker and native HTTPS peers for the same host security campaign."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
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

    async def load(self, feature):
        return await feature.load_tool(self.selector)

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
async def owned_peer_image(tmp_path):
    """One native image, shared by direct HTTP and actual Docker Gateway peers."""
    import docker
    from docker.errors import ImageNotFound
    from kestrel_sovereign._bounded_subprocess import run_bounded_subprocess

    build = tmp_path / "peer"
    build.mkdir()
    shutil.copyfile(Path(__file__).with_name("mcp_permission_peer.py"), build / "peer.py")
    (build / "Dockerfile").write_text(
        'FROM python:3.11-slim\nWORKDIR /app\nRUN pip install "mcp>=2.3,<3"\n'
        'COPY peer.py .\nCMD ["python", "peer.py"]\n')
    identity = uuid.uuid4().hex
    image = f"mcp-{identity}:test"
    async with AsyncExitStack() as cleanup:
        client = await asyncio.to_thread(docker.from_env)
        cleanup.push_async_callback(asyncio.to_thread, client.close)

        async def remove_owned_image():
            try:
                owned = await asyncio.to_thread(client.images.get, image)
            except ImageNotFound:
                return  # A failed build never published this owned image.
            assert owned.attrs["Config"]["Labels"]["kestrel.mcp.permission.image-owner"] == identity
            await asyncio.to_thread(client.images.remove, owned.id, noprune=True)

        cleanup.push_async_callback(remove_owned_image)
        result = await run_bounded_subprocess([
            "docker", "buildx", "build", "--load", "--tag", image,
            "--label", "kestrel.mcp.permission.image-owner=" + identity,
            "--iidfile", str(build / "image-id"), str(build),
        ], timeout=90)
        (tmp_path / "native-peer-image-build.log").write_bytes(result.stdout + result.stderr)
        assert not result.timed_out and result.returncode == 0, result.stderr.decode(errors="replace")
        assert not result.stdout_truncated and not result.stderr_truncated
        owned = await asyncio.to_thread(client.images.get, image)
        assert (build / "image-id").read_text().strip() == owned.id
        yield client, image


@asynccontextmanager
async def docker_peer(tmp_path):
    async with owned_peer_image(tmp_path) as (_, image):
        yield MCPTestPeer("container", image)


@dataclass
class GatewayTestPeer(MCPTestPeer):
    client: object = None
    image: str = ""
    image_id: str = ""
    volume: str = ""
    owner: str = ""
    gateways: list = field(default_factory=list)
    container_server_names: set = field(default_factory=set)

    async def witness(self, feature, name):
        return await self._read_volume_lines("permission-witness")

    async def held_call_witness(self):
        return await self._read_volume_lines("permission-witness.held")

    async def _read_volume_lines(self, filename):
        from docker.types import LogConfig

        assert filename in {"permission-witness", "permission-witness.held"}
        # A separate read-only native container sees the same owned volume even
        # after gateway replacement. No host bind mount or operator data enters.
        output = await asyncio.to_thread(self.client.containers.run, self.image,
            command=["python", "-c", 'import sys; from pathlib import Path; p=Path("/tmp") / sys.argv[1]; '
                'print(p.read_text() if p.exists() else "", end="")', filename],
            volumes={self.volume: {"bind": "/tmp", "mode": "ro"}},
            network_mode="none", log_config=LogConfig(type="json-file"),
            labels={"kestrel.mcp.permission.owner": self.owner}, remove=True)
        return output.decode().splitlines()

    def connection(self, feature, name):
        return feature.gateway_manager.connection

    async def load(self, feature):
        result = await feature.gateway_start(self.selector)
        if feature.gateway_manager is not None:
            self.remember_gateway(feature.gateway_manager.gateway)
        return result

    def remember_gateway(self, gateway):
        self.gateways.append(gateway)
        for name in gateway.requested_servers:
            self.container_server_names.add(name)
            if gateway._profile_id is not None:
                self.container_server_names.add(gateway._profile_id + "-" + name)

    async def containers(self):
        containers = []
        for name in self.container_server_names:
            containers.extend(await asyncio.to_thread(self.client.containers.list, all=True,
                filters={"label": ["docker-mcp=true", "docker-mcp-name=" + name]}))
        for container in containers:
            await asyncio.to_thread(container.reload)
            labels = container.attrs["Config"]["Labels"]
            assert labels["docker-mcp-name"] in self.container_server_names
            assert labels["docker-mcp-tool-type"] == "mcp" and labels["docker-mcp-transport"] == "stdio"
            assert container.attrs["Image"] == self.image_id
            assert container.attrs["HostConfig"]["NetworkMode"] == "none"
            assert any(mount["Name"] == self.volume and mount["Destination"] == "/tmp"
                       for mount in container.attrs["Mounts"] if mount["Type"] == "volume")
        return containers


@asynccontextmanager
async def gateway_peer(tmp_path, monkeypatch, *, linger_after_stdio=False, configured=False, hold_tool_reply=False):
    """Run the installed Docker MCP CLI with private configuration and resources."""
    plugin = Path(os.environ["MCP_TEST_DOCKER_PLUGIN"]).resolve(strict=True)
    docker_host = os.environ["MCP_TEST_DOCKER_HOST"]
    assert docker_host.startswith("unix://")
    identity = uuid.uuid4().hex
    home = tmp_path / "gateway-home"
    config = home / ".docker"
    (config / "cli-plugins").mkdir(parents=True)
    (config / "cli-plugins/docker-mcp").symlink_to(plugin)
    original_configuration = json.dumps({
        "cliPluginsExtraDirs": [str(plugin.parent)],
        "features": {"dynamic-tools": "enabled"},
    })
    (config / "config.json").write_text(original_configuration)
    (config / "mcp/catalogs").mkdir(parents=True)
    with monkeypatch.context() as runtime:
        runtime.setenv("HOME", str(home))
        runtime.setenv("DOCKER_CONFIG", str(config))
        runtime.setenv("DOCKER_HOST", docker_host)
        # Official standalone mode uses the explicit Engine socket and leaves
        # Desktop's backend, settings, and OAuth store outside this fixture.
        # Current Gateway profiles remain enabled in this mode.
        runtime.setenv("DOCKER_MCP_IN_CONTAINER", "1")
        runtime.setenv("XDG_CACHE_HOME", str(home / ".cache"))
        runtime.delenv("DOCKER_CONTEXT", raising=False)
        async with owned_peer_image(tmp_path) as (client, image):
            volume = await asyncio.to_thread(client.volumes.create, name="mcp-permission-" + identity,
                labels={"kestrel.mcp.permission.owner": identity})
            try:
                server = "permission-" + identity
                catalog = {"version": 2, "name": "docker-mcp", "displayName": "Owned permission peer",
                    "registry": {server: {"name": server, "type": "server", "image": image,
                        "description": "Owned permission effect witness", "disableNetwork": True,
                        "command": ["python", "peer.py", "--transport", "stdio", "--witness", "/tmp/permission-witness",
                                    *(["--linger-after-stdio"] if linger_after_stdio else []),
                                    *(["--hold-tool-reply"] if hold_tool_reply else [])],
                        "volumes": [volume.name + ":/tmp"], "tools": [{"name": "record_marker"}]}}}
                if configured:
                    definition = catalog["registry"][server]
                    definition["config"] = [{"name": server, "description": "Owned required configuration",
                        "type": "object", "properties": {"mode": {"type": "string"}}, "required": ["mode"]}]
                    definition["env"] = [{"name": "OWNED_PROFILE_MODE", "value": "{{" + server + ".mode}}"}]
                    definition["command"].extend(["--require-profile-mode", "owned-configured-mode"])
                # Discovery reads the current native OCI catalog, while the
                # gateway's explicit server selection reads the same definition.
                # Import through the actual CLI; do not fabricate its database.
                reference = "kestrel-permission/catalog:" + identity
                await create_owned_mcp_catalog(config / "mcp/catalogs/docker-mcp.yaml", catalog, reference)
                runtime.setenv("KESTREL_MCP_DOCKER_CATALOG", reference)
                if configured:
                    for command in (["create", "--id", "default", "--name", "Owned configured source",
                                     "--server", "catalog://" + reference + "/" + server],
                                    ["config", "default", "--set", server + '.mode="owned-configured-mode"'],
                                    ["tools", "default", "--disable-all", server],
                                    ["tools", "default", "--enable", server + ".record_marker"]):
                        await run_owned_mcp_cli(["profile", *command])
                    exported_source = config / "owned-source-profile.json"
                    exported_source.touch(mode=0o600)
                    await run_owned_mcp_cli(["profile", "export", "default", str(exported_source)])
                    source_profile = json.loads(exported_source.read_text())
                    source_profile["servers"][0]["secrets"] = "owned-source-binding"
                    source_profile["secrets"] = {"owned-source-binding": {"provider": "docker-desktop-store"}}
                    exported_source.write_text(json.dumps(source_profile))
                    await run_owned_mcp_cli(["profile", "import", str(exported_source)])
                    runtime.setenv("KESTREL_MCP_DOCKER_PROFILE", "default")
                image_id = (await asyncio.to_thread(client.images.get, image)).id
                peer = GatewayTestPeer(kind="gateway", selector=server, client=client, image=image,
                    image_id=image_id, volume=volume.name, owner=identity)
                try:
                    yield peer
                finally:
                    assert (config / "config.json").read_text() == original_configuration
                    diagnostics = [{"profile": gateway._profile_id,
                        "pending_profile": gateway._profile_pending,
                        "process_retired": gateway.process is None,
                        "stdout": gateway._recent_output()} for gateway in peer.gateways]
                    (tmp_path / "native-gateway-diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
                    (tmp_path / "native-gateway-external-witness.json").write_text(
                        json.dumps(await peer.witness(None, None)))
                    remaining = await peer.containers()
                    for container in remaining:
                        await asyncio.to_thread(container.remove, force=True)
                    assert not remaining, "Feature shutdown left owned Gateway containers running"
            finally:
                await asyncio.to_thread(volume.reload)
                assert volume.attrs["Labels"]["kestrel.mcp.permission.owner"] == identity
                await asyncio.to_thread(volume.remove)


async def run_owned_mcp_cli(arguments):
    result = await asyncio.to_thread(subprocess.run, ["docker", "mcp", *arguments],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


async def create_owned_mcp_catalog(path, definition, reference):
    path.write_text(json.dumps(definition))
    return await run_owned_mcp_cli(["catalog", "create", reference, "--from-legacy-catalog", str(path)])


@asynccontextmanager
async def https_peer(tmp_path, monkeypatch, *, provider_model: str | None = None,
                     provider_fault: str | None = None, drop_first_tool_reply: bool = False):
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
            *(["--drop-first-tool-reply"] if drop_first_tool_reply else []),
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
