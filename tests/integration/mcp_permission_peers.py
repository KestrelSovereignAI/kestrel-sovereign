"""Owned Docker and native HTTPS peers for the same host security campaign."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager, nullcontext
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


async def retire_original_gateway_process(pid, wait, exited):
    """Retire one actual admitted session; native and orphan waiters stay original."""
    import signal
    from kestrel_sovereign._async_ownership import await_owned_operation, raise_owned_outcome

    async def join():
        raise_owned_outcome(await await_owned_operation(asyncio.to_thread(wait, 5)),
            operation='original native Gateway retirement')

    def signal_original_group(value):
        try:
            assert os.getpgid(pid) == pid
            os.killpg(pid, value)
        except ProcessLookupError:
            # A zombie may no longer have a process group. Its original wait
            # still has to reap/join it; a missing group is not that receipt.
            pass

    async def force_if_live():
        if not exited():
            signal_original_group(signal.SIGKILL)
            await join()

    if exited():
        return
    async with AsyncExitStack() as retirement:
        retirement.push_async_callback(force_if_live)
        signal_original_group(signal.SIGTERM)
        await join()


@asynccontextmanager
async def owned_gateway_network(monkeypatch, peer, settings):
    """Own original Gateway processes and admit only their real TCP listeners."""
    if settings is None:
        yield None
        return
    import hashlib
    import psutil
    import re

    home = Path(settings['home']).resolve(strict=True)
    plugin = Path(settings['plugin']).resolve(strict=True)
    docker_socket = Path(settings['docker_socket']).resolve(strict=True)
    assert hashlib.sha256(plugin.read_bytes()).hexdigest() == settings['plugin_sha256']
    accepted = []
    native_popen = subprocess.Popen

    class GatewayNetwork:
        def owns_docker(self, address):
            return isinstance(address, (str, bytes)) and Path(os.fsdecode(address)).resolve() == docker_socket

        def owns_listener(self, address):
            if not isinstance(address, tuple) or address[0] not in {'127.0.0.1', '::1', 'localhost'} or address[1] == 11434:
                return False
            for process, port in accepted:
                if process.poll() is not None or port != address[1]:
                    continue
                try:
                    original = psutil.Process(process.pid)
                    candidates = [original, *original.children()]
                    for candidate in candidates:
                        if Path(candidate.exe()).resolve() != plugin:
                            continue
                        if any(connection.status == psutil.CONN_LISTEN
                               and connection.laddr.ip in {'127.0.0.1', '::1'}
                               and connection.laddr.port == port
                               for connection in candidate.net_connections(kind='tcp')):
                            return process.poll() is None
                except psutil.NoSuchProcess:
                    continue
            return False

    class OwnedGatewayPopen(native_popen):
        def __init__(self, arguments, *args, **kwargs):
            gateway = isinstance(arguments, (list, tuple)) and list(arguments[:4]) == ['docker', 'mcp', 'gateway', 'run']
            if gateway:
                environment = kwargs.get('env') or os.environ
                assert Path(environment['HOME']).resolve() == home
                configuration = Path(environment['DOCKER_CONFIG']).resolve()
                assert configuration.parent == home / '.docker/mcp/owned-gateways'
                assert configuration.name.startswith('kestrel-'), 'Native Gateway configuration must be its owned mkdtemp directory'
                assert json.loads((configuration / 'config.json').read_text())['features']['dynamic-tools'] == 'disabled'
                assert environment['DOCKER_HOST'] == 'unix://' + settings['docker_socket']
                assert re.fullmatch('[a-f0-9]{48}', environment['MCP_GATEWAY_AUTH_TOKEN'])
                assert len(arguments) == 9 and arguments[4] == '--transport=streaming'
                assert arguments[6] == '--host=127.0.0.1' and arguments[8] == '--long-lived'
                assert re.fullmatch('--profile=kestrel-[a-f0-9]{32}', arguments[7])
                assert re.fullmatch('--port=[0-9]+', arguments[5])
                port = int(arguments[5].split('=', 1)[1])
                assert 0 < port <= 65535 and port != 11434
                assert kwargs.get('start_new_session') is True
                admission = None
                if settings.get('admission_directory') is not None:
                    host = psutil.Process()
                    profile = arguments[7].split('=', 1)[1]
                    admission = dict(host_pid=host.pid, host_created=host.create_time(),
                        profile=profile, configuration=str(configuration),
                        executables=[str(plugin), str(Path(shutil.which('docker')).resolve())])
                    receipt = Path(settings['admission_directory']) / (profile + '.json')
                    assert not receipt.exists()
                    intent = receipt.with_suffix('.writing')
                    intent.write_text(json.dumps(admission))
                    intent.replace(receipt)
            super().__init__(arguments, *args, **kwargs)
            if gateway:
                accepted.append((self, port))
                if admission is not None:
                    admission.update(pid=self.pid, created=psutil.Process(self.pid).create_time())
                    temporary = receipt.with_suffix('.writing')
                    temporary.write_text(json.dumps(admission))
                    temporary.replace(receipt)

    async def retire(process):
        await retire_original_gateway_process(process.pid, process.wait,
            lambda: process.poll() is not None)

    with monkeypatch.context() as ownership:
        ownership.setattr(subprocess, 'Popen', OwnedGatewayPopen)
        try:
            yield GatewayNetwork()
        finally:
            # A fixture reconciler is never credited as the Core close ACK.
            # Independent original processes still retire if another fails.
            try:
                async with AsyncExitStack() as cleanup:
                    for process, _port in accepted:
                        cleanup.push_async_callback(retire, process)
                    assert all(process.poll() is not None for process, _port in accepted), (
                        'Core host shutdown left an original native Gateway process alive')
            finally:
                if settings.get('reconciliation_receipt') is not None:
                    Path(settings['reconciliation_receipt']).write_text(json.dumps(dict(
                        original_processes=len(accepted),
                        all_retired=all(process.poll() is not None for process, _port in accepted))))


@contextmanager
def owned_provider_network(monkeypatch, peer, *, gateway=None):
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
            if gateway is not None and (gateway.owns_docker(address) or gateway.owns_listener(address)):
                return
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
def owned_https_network(monkeypatch, peer, *, gateway=None):
    """Pin public-looking fixture DNS to one real, certificate-verified listener."""
    from urllib.parse import urlsplit
    import docker
    import httpcore2

    endpoint = urlsplit(peer.provider_url or peer.selector)
    assert endpoint.scheme == "https" and endpoint.hostname and endpoint.port
    assert peer.tls_certificate is not None
    hostname, owned_port = endpoint.hostname, endpoint.port
    original = httpcore2.AnyIOBackend.connect_tcp

    def resolve(host, requested_port, family=socket.AF_UNSPEC, type=0, proto=0, flags=0):
        # asyncio/AnyIO invoke the standard resolver with encoded hostnames.
        # Normalize that public call contract before checking either owner.
        host = os.fsdecode(host)
        if (host == "127.0.0.1" and requested_port in peer.local_ports) or (
                gateway is not None and gateway.owns_listener((host, requested_port))):
            address = "127.0.0.1"
        else:
            assert host == hostname and requested_port == owned_port, (host, requested_port)
            address = "8.8.8.8"
        return [(socket.AF_INET, type, proto, "", (address, requested_port))]

    async def dial(backend, host, port, **kwargs):
        if (host == "127.0.0.1" and port in peer.local_ports) or (
                gateway is not None and gateway.owns_listener((host, port))):
            return await original(backend, host, port, **kwargs)
        allowed_hosts = {"8.8.8.8", hostname} if peer.provider_url else {"8.8.8.8"}
        assert host in allowed_hosts and port == owned_port
        return await original(backend, "127.0.0.1", port, **kwargs)

    def refuse_docker(*args, **kwargs):
        raise AssertionError("Remote MCP path attempted to acquire Docker")

    with monkeypatch.context() as network:
        network.setattr(socket, "getaddrinfo", resolve)
        network.setattr(httpcore2.AnyIOBackend, "connect_tcp", dial)
        if gateway is None:
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
    gateway_home: Path | None = None
    host_observations: dict = field(default_factory=dict)
    host_admissions: dict = field(default_factory=dict)
    witness_readers: set = field(default_factory=set)

    async def witness(self, feature, name):
        return await self._read_volume_lines("permission-witness")

    async def held_call_witness(self):
        return await self._read_volume_lines("permission-witness.held")

    async def _read_volume_lines(self, filename):
        from docker.types import LogConfig

        assert filename in {"permission-witness", "permission-witness.held"}
        # A separate read-only native container sees the same owned volume even
        # after gateway replacement. No host bind mount or operator data enters.
        from kestrel_sovereign._async_ownership import OwnedOperation, raise_owned_outcome

        reader = OwnedOperation(asyncio.to_thread(self.client.containers.run, self.image,
            command=["python", "-c", 'import sys; from pathlib import Path; p=Path("/tmp") / sys.argv[1]; '
                'print(p.read_text() if p.exists() else "", end="")', filename],
            volumes={self.volume: {"bind": "/tmp", "mode": "ro"}},
            network_mode="none", log_config=LogConfig(type="json-file"),
            labels={"kestrel.mcp.permission.owner": self.owner}, remove=True),
            retirement_resource=self, name='owned-native-witness-reader')
        self.witness_readers.add(reader)
        try:
            output = raise_owned_outcome(await reader.observe(), operation='native witness reader')
            return output.decode().splitlines()
        finally:
            self.witness_readers.remove(reader)

    async def join_witness_readers(self):
        from kestrel_sovereign._async_ownership import raise_owned_outcome

        async def join(reader):
            raise_owned_outcome(await reader.observe(), operation='native witness inventory join')

        async with AsyncExitStack() as joins:
            for reader in tuple(self.witness_readers):
                joins.push_async_callback(join, reader)


    def connection(self, feature, name):
        return feature.gateway_manager.connection

    async def load(self, feature):
        result = await feature.gateway_start(self.selector)
        if feature.gateway_manager is not None:
            self.remember_gateway(feature.gateway_manager.gateway)
        return result

    def remember_gateway(self, gateway):
        if all(known is not gateway for known in self.gateways):
            self.gateways.append(gateway)

    def observe_host_gateway(self, host_pid, snapshot):
        import psutil
        import re
        import sqlite3

        assert self.gateway_home is not None and snapshot is not None
        assert re.fullmatch('kestrel-[a-f0-9]{32}', snapshot['profile'])
        assert snapshot['servers'] == [self.selector]
        original = psutil.Process(snapshot['pid'])
        assert host_pid in {parent.pid for parent in original.parents()}, (
            'Native Gateway receipt does not belong to the original host process')
        assert '--profile=' + snapshot['profile'] in original.cmdline()
        configuration = Path(snapshot['configuration']).resolve(strict=True)
        assert configuration.parent == self.gateway_home / '.docker/mcp/owned-gateways'
        database = self.gateway_home / '.docker/mcp/mcp-toolkit.db'
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
            assert connection.execute('SELECT id FROM working_set WHERE id=?',
                (snapshot['profile'],)).fetchone() == (snapshot['profile'],), (
                'Native CLI profile receipt is missing before host shutdown')
        key = host_pid, original.pid, snapshot['profile']
        self.host_observations.setdefault(key, dict(process=original,
            profile=snapshot['profile'], configuration=configuration, database=database))

    def read_host_admissions(self, host, directory):
        import re

        key = host.pid, host.create_time(), directory
        self.host_admissions[key] = host
        records = []
        for receipt in sorted(directory.glob('kestrel-*.json')):
            record = json.loads(receipt.read_text())
            assert record['host_pid'] == host.pid and record['host_created'] == host.create_time()
            assert re.fullmatch('kestrel-[a-f0-9]{32}', record['profile'])
            assert receipt.name == record['profile'] + '.json'
            configuration = Path(record['configuration'])
            assert configuration.parent == self.gateway_home / '.docker/mcp/owned-gateways'
            assert configuration.name.startswith('kestrel-')
            record['configuration'] = configuration
            records.append(record)
        return records

    def admitted_processes(self, record):
        import psutil

        def matches(process):
            return (process.create_time() >= record['host_created']
                and '--profile=' + record['profile'] in process.cmdline()
                and str(Path(process.exe()).resolve()) in record['executables']
                and os.getpgid(process.pid) == process.pid)

        if 'pid' in record:
            try:
                original = psutil.Process(record['pid'])
            except psutil.NoSuchProcess:
                return []
            assert original.create_time() == record['created'], 'Original native Gateway PID was reused'
            try:
                if original.status() != psutil.STATUS_ZOMBIE:
                    assert matches(original), 'Original native Gateway admission no longer matches its receipt'
            except (psutil.ZombieProcess, ProcessLookupError):
                # The verified original can exit at any attestation step,
                # including the final native group lookup. Keep its exact
                # waiter until physical retirement is observed, never adopt
                # another process or infer a join from missing metadata.
                pass
            except psutil.NoSuchProcess:
                return []  # The original was already reaped during observation.
            return [original]
        # Intent is durable before Popen. A forced host exit between actual
        # creation and PID publication must not orphan that admitted process.
        found = []
        for candidate in psutil.process_iter():
            try:
                if '--profile=' + record['profile'] in candidate.cmdline() and matches(candidate):
                    found.append(candidate)
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                continue  # Never adopt or signal an unobserved candidate.
        assert len(found) <= 1, 'Native launch intent has multiple original process candidates'
        return found

    async def reconcile_host_gateways(self, host, directory):
        import sqlite3

        records = self.read_host_admissions(host, directory)

        async def reconcile(record):
            for original in self.admitted_processes(record):
                await retire_original_gateway_process(original.pid, original.wait,
                    lambda: not original.is_running())
            assert not self.admitted_processes(record), 'Original native Gateway is still alive'
            assert await self.containers() == [], 'Retain native profile until its owned peers retire'
            database = self.gateway_home / '.docker/mcp/mcp-toolkit.db'
            with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
                present = connection.execute('SELECT id FROM working_set WHERE id=?',
                    (record['profile'],)).fetchone()
            if present is not None:
                await run_owned_mcp_cli(['profile', 'remove', record['profile']])
            if record['configuration'].exists():
                shutil.rmtree(record['configuration'])

        async with AsyncExitStack() as retirement:
            for record in records:
                retirement.push_async_callback(reconcile, record)


    async def assert_host_retired(self, host_pid):
        import sqlite3

        for (owner_pid, _pid, _profile), record in self.host_observations.items():
            if owner_pid != host_pid:
                continue
            assert not record['process'].is_running(), (
                'Core host acknowledged shutdown with its original native Gateway alive')
            assert not record['configuration'].exists(), (
                'Core host shutdown retained its acknowledged private Gateway configuration')
            with sqlite3.connect(record['database'].as_uri() + '?mode=ro', uri=True) as connection:
                assert connection.execute('SELECT id FROM working_set WHERE id=?',
                    (record['profile'],)).fetchone() is None, (
                    'Core host shutdown retained its acknowledged native CLI profile')
        assert await self.containers() == [], (
            'Core host shutdown left an actual peer attached to its owned witness volume')

    def assert_restart_observed(self):
        assert len(self.host_observations) == 2, 'Both original host Gateway generations must be observed'
        assert len({key[0] for key in self.host_observations}) == 2
        assert len({key[1] for key in self.host_observations}) == 2
        assert len({key[2] for key in self.host_observations}) == 2, (
            'A restarted host reused a retired native Gateway profile')

    async def containers(self):
        await self.join_witness_readers()
        return await self._container_inventory()

    async def _container_inventory(self):
        # The unique fixture volume and immutable image identify our peers
        # across real host processes. CLI name labels are descriptive only.
        volume = await asyncio.to_thread(self.client.volumes.get, self.volume)
        assert volume.attrs['Labels']['kestrel.mcp.permission.owner'] == self.owner
        containers = await asyncio.to_thread(self.client.containers.list, all=True,
            filters={'volume': self.volume})
        for container in containers:
            await asyncio.to_thread(container.reload)
            labels = container.attrs['Config']['Labels']
            if labels.get('kestrel.mcp.permission.owner') == self.owner:
                # A failed removal can retain our own read-only witness reader.
                # It is a cleanup resource, never evidence of a live MCP server.
                assert container.attrs['Config']['Cmd'][:2] == ['python', '-c']
                assert any(mount['Name'] == self.volume and mount['RW'] is False
                           for mount in container.attrs['Mounts'] if mount['Type'] == 'volume')
            else:
                assert labels['docker-mcp'] == 'true'
                assert labels['docker-mcp-tool-type'] == 'mcp' and labels['docker-mcp-transport'] == 'stdio'
            assert container.attrs['Image'] == self.image_id
            assert container.attrs['HostConfig']['NetworkMode'] == 'none'
            assert any(mount['Name'] == self.volume and mount['Destination'] == '/tmp'
                       for mount in container.attrs['Mounts'] if mount['Type'] == 'volume')
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
                    image_id=image_id, volume=volume.name, owner=identity, gateway_home=home.resolve())
                try:
                    yield peer
                finally:
                    # Diagnostics are observations, never prerequisites for
                    # retiring independent original fixture resources. Preserve
                    # their pre-reconciliation oracles and every cleanup error.
                    async with AsyncExitStack() as retirement:
                        retirement.push_async_callback(_reconcile_admitted_host_gateways, peer)
                        retirement.push_async_callback(_retire_remembered_gateways, peer)
                        retirement.push_async_callback(_remove_owned_gateway_containers, peer)
                        retirement.push_async_callback(_retire_remembered_gateways, peer)
                        retirement.push_async_callback(_reconcile_admitted_host_gateways, peer)
                        assert (config / "config.json").read_text() == original_configuration
                        diagnostics = [{"profile": gateway._profile_id,
                            "pending_profile": gateway._profile_pending,
                            "process_retired": gateway.process is None,
                            "stdout": gateway._recent_output()} for gateway in peer.gateways]
                        (tmp_path / "native-gateway-diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
                        (tmp_path / "native-gateway-external-witness.json").write_text(
                            json.dumps(await peer.witness(None, None)))
                        remaining = await peer.containers()
                        assert not remaining, "Feature shutdown left owned Gateway containers running"
            finally:
                await asyncio.to_thread(volume.reload)
                assert volume.attrs["Labels"]["kestrel.mcp.permission.owner"] == identity
                await asyncio.to_thread(volume.remove)


async def _reconcile_admitted_host_gateways(peer):
    async with AsyncExitStack() as retirement:
        for (pid, created, directory) in tuple(peer.host_admissions):
            # Retain the original birth receipt even after the host has exited.
            host = peer.host_admissions[(pid, created, directory)]
            retirement.push_async_callback(peer.reconcile_host_gateways, host, directory)


async def _retire_remembered_gateways(peer):
    async with AsyncExitStack() as retirement:
        for gateway in tuple(peer.gateways):
            retirement.push_async_callback(gateway.stop)


async def _remove_owned_gateway_containers(peer):
    async def remove_inventory():
        async with AsyncExitStack() as retirement:
            for container in await peer._container_inventory():
                retirement.push_async_callback(asyncio.to_thread, container.remove, force=True)

    async with AsyncExitStack() as retirement:
        retirement.push_async_callback(remove_inventory)
        retirement.push_async_callback(peer.join_witness_readers)


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
                     provider_fault: str | None = None, drop_first_tool_reply: bool = False,
                     provider_only: bool = False):
    if provider_only:
        assert provider_model is not None, 'Provider-only fixture requires its finite provider model'
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
            # A provider-only listener is independent of the selected MCP
            # transport. Its caller installs the explicit combined network
            # owner; a remote MCP peer keeps its Docker-refusal contract.
            with nullcontext() if provider_only else owned_https_network(monkeypatch, peer):
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
