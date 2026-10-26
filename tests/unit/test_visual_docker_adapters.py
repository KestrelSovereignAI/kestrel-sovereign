"""SDK adapter authority/identity regression checks; real Docker is separate."""
import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import SecretBytes, SecretStr
from kestrel_sdk.computer_use import DriverConnection, EnvironmentRequest, SessionRequest
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceScope, ResourceUnsupported
from kestrel_sovereign.visual_computer import docker
from kestrel_sovereign.visual_computer import docker_policy as policy

SCOPE = ResourceScope(tenant_id="fixture", principal_id="owner")
IMAGE = "local-worker@sha256:" + "a" * 64
CID = "c" * 64


def provider():
    return docker.environment_provider_factory(settings={"docker_socket": "/tmp/fixture.sock", "image": IMAGE,
                                                         "credential_key": SecretBytes(b"k" * 32)})


def request(**changes):
    now = datetime.now(UTC)
    fields = dict(request_id="fixture", scope=SCOPE, provider_id=docker.PROVIDER, profile_id=docker.PROFILE,
                  image=IMAGE, surface="browser", required_capabilities=frozenset({"screenshot"}), location="local",
                  requested_at=now, expires_at=now + timedelta(seconds=60), idle_ttl_seconds=30,
                  ready_timeout_seconds=20, max_hourly_cost_usd=Decimal(0), max_total_cost_usd=Decimal(0))
    return EnvironmentRequest(**(fields | changes))


def record(identity, *, running=True):
    host = policy.verified_host_configuration()
    host["SecurityOpt"] = ["no-new-privileges", "seccomp=" + (policy.POLICY_ROOT / "seccomp.json").read_text()]
    return {"Id": CID, "Image": "sha256:" + "a" * 64, "Mounts": [], "RestartCount": 0,
            "Config": {"Labels": {docker.LABEL: identity}, "User": "pwuser", "OpenStdin": False,
                       "Tty": False, "Cmd": ["python", "/opt/kestrel-browser/worker.py"], "Entrypoint": None},
            "HostConfig": host, "NetworkSettings": {"Ports": {}},
            "State": {"Running": running, "Pid": 900, "StartedAt": "2026-10-26T00:00:00Z", "Status": "running"}}


@pytest.mark.asyncio
async def test_capabilities_quote_and_refused_advanced_profiles():
    p = provider()
    capabilities = await p.capabilities()
    assert "takeover" in capabilities.operations
    assert "files" in capabilities.operations
    assert not {"navigate", "exec", "clipboard"} & capabilities.operations
    req = request()
    quote = await p.quote(req)
    assert quote.request_sha256 == req.digest()
    assert quote == await p.quote(req)
    for changes in ({"required_capabilities": frozenset({"screenshot", "navigate"})}, {"location": "remote"},
                    {"profile_id": "network"}, {"image": "other@sha256:" + "b" * 64}, {"region": "local"}):
        with pytest.raises(ResourceUnsupported):
            await p.quote(request(**changes))


@pytest.mark.parametrize("settings", [{}, {"docker_socket": "tcp://localhost"},
                                      {"docker_socket": "/tmp/fixture.sock", "extra": "secret"}])
def test_driver_factory_refuses_implicit_or_remote_configuration(settings):
    with pytest.raises(ResourceUnsupported):
        docker.computer_driver_factory(settings=settings)


@pytest.mark.asyncio
async def test_stable_private_credentials_bind_exact_process_incarnation(monkeypatch):
    p, req = provider(), request()
    identity = docker._identity(SCOPE, req.request_id)
    current = record(identity)
    lease = docker.EnvironmentLease(lease_id="docker-" + identity, request_id=req.request_id,
        quote_id="quote", scope=SCOPE, provider_id=docker.PROVIDER, location="local", state="pending",
        expires_at=req.expires_at, provider_resource_id=CID)
    async def inspect(target):
        assert target == CID
        return current
    incarnation = [b"d" * 64]
    async def invoke(arguments, **kwargs):
        if arguments[0] == "exec":
            return incarnation[0]
        assert arguments == ["image", "inspect", "sha256:" + "a" * 64]
        return json.dumps([{"Id": "sha256:" + "a" * 64, "Config": {"Labels": {docker.LABEL + ".worker-contract": docker.WORKER_CONTRACT}}}]).encode()
    monkeypatch.setattr(p.boundary, "inspect", inspect)
    monkeypatch.setattr(p.boundary, "invoke", invoke)
    one = await p.connect(SCOPE, lease)
    assert one == await p.connect(SCOPE, lease)
    assert one.credential.get_secret_value() not in json.dumps(one.model_dump(mode="json"))
    assert CID not in repr(one)
    current["State"]["StartedAt"] = "2026-10-26T00:00:01Z"
    two = await p.connect(SCOPE, lease)
    assert two.credential != one.credential
    driver = docker.computer_driver_factory(settings={"docker_socket": "/tmp/fixture.sock"})
    monkeypatch.setattr(driver.boundary, "inspect", inspect)
    monkeypatch.setattr(driver.boundary, "invoke", invoke)
    with pytest.raises(ResourceNotFound, match="incarnation"):
        await driver._bound(one)
    assert (await driver._bound(two))["container_id"] == CID
    incarnation[0] = b"e" * 64
    with pytest.raises(ResourceNotFound, match="incarnation"):
        await driver._bound(two)
    current["HostConfig"]["NetworkMode"] = "host"
    with pytest.raises(RuntimeError, match="isolated"):
        await p.connect(SCOPE, lease)


@pytest.mark.asyncio
async def test_open_validates_identity_and_expiry_before_sending_credentials(monkeypatch):
    driver = docker.computer_driver_factory(settings={"docker_socket": "/tmp/fixture.sock"})
    now = datetime.now(UTC)
    connection = DriverConnection(lease_id="lease", driver_id=docker.PROVIDER,
        endpoint=SecretStr("PRIVATE"), credential=SecretStr("SECRET"), expires_at=now+timedelta(seconds=30))
    req = SessionRequest(session_id="session", scope=SCOPE, lease_id="lease", surface="browser",
                         required_capabilities=frozenset({"screenshot"}), expires_at=now+timedelta(seconds=40))
    calls = []
    async def invoke(*args, **kwargs):
        calls.append(args)
        raise AssertionError("connection validation reached daemon")
    monkeypatch.setattr(driver.boundary, "invoke", invoke)
    with pytest.raises(ResourceUnsupported):
        await driver.open(req, connection)
    assert calls == []


@pytest.mark.asyncio
async def test_release_does_not_turn_owner_mismatch_into_physical_absence(monkeypatch):
    p, req = provider(), request()
    identity = docker._identity(SCOPE, req.request_id)
    lease = docker.EnvironmentLease(lease_id="docker-" + identity, request_id=req.request_id,
        quote_id="quote", scope=SCOPE, provider_id=docker.PROVIDER, location="local", state="pending",
        expires_at=req.expires_at, provider_resource_id=CID)
    async def inspect(target):
        return record("different-owner")
    monkeypatch.setattr(p.boundary, "inspect", inspect)
    with pytest.raises(ResourceConflict, match="identity"):
        await p.release(SCOPE, lease)
    with pytest.raises(ResourceConflict, match="identity"):
        await p.inspect(SCOPE, lease)


@pytest.mark.asyncio
async def test_priority_control_and_reconciliation_use_dedicated_transport(monkeypatch):
    driver = docker.computer_driver_factory(settings={"docker_socket": "/tmp/fixture.sock"})
    connection = DriverConnection(lease_id="lease", driver_id=docker.PROVIDER,
        endpoint=SecretStr("private"), credential=SecretStr("a"*64), expires_at=datetime.now(UTC)+timedelta(seconds=60))
    driver._connections["session"] = connection
    async def bound(value, **options):
        return {"container_id": CID}
    calls = []
    async def rpc(cid, value, *, mode=None, **options):
        calls.append((mode, value["operation"]))
        return {"id": value["id"], "result": {}}
    monkeypatch.setattr(driver, "_bound", bound)
    monkeypatch.setattr(driver.boundary, "rpc", rpc)
    await driver._rpc("session", "control", {}, priority=True)
    await driver._rpc("session", "status", {}, priority=True)
    await driver._rpc("session", "act", {})
    assert calls == [("priority", "control"), ("priority", "status"), (None, "act")]


def test_injected_container_environment_and_exec_entrypoint_are_refused():
    value = record("fixture")
    image = {"Id": "sha256:" + "a" * 64, "Config": {"Env": ["PATH=/usr/bin"], "Labels": {docker.LABEL + ".worker-contract": docker.WORKER_CONTRACT}}}
    value["Config"]["Env"] = ["PATH=/usr/bin"]
    docker._admit(value, image, "fixture")
    value["Config"]["Env"].append("CLOUD_CREDENTIAL=operator")
    with pytest.raises(ResourceUnsupported):
        docker._admit(value, image, "fixture")
    value["Config"]["Env"] = ["PATH=/usr/bin"]
    value["Config"]["Entrypoint"] = ["sh", "-c"]
    with pytest.raises(ResourceUnsupported):
        docker._admit(value, image, "fixture")


@pytest.mark.asyncio
async def test_acquire_admits_actual_container_before_start(monkeypatch):
    p, req = provider(), request()
    quote = await p.quote(req)
    current = record(docker._identity(SCOPE, req.request_id))
    current["State"]["Status"] = "created"
    current["State"]["Running"] = False
    current["Config"]["Labels"].update({docker.LABEL+".request": req.digest(),
                                          docker.LABEL+".quote": docker._quote_hash(quote)})
    current["HostConfig"]["NetworkMode"] = "host"
    calls = []
    async def platform():
        return None
    async def invoke(arguments, **options):
        calls.append(arguments)
        if arguments[:2] == ["image", "inspect"]:
            return json.dumps([{"Id": "sha256:" + "a" * 64, "Os": "linux", "Architecture": "arm64", "Config": {"Labels": {docker.LABEL + ".worker-contract": docker.WORKER_CONTRACT}}}]).encode()
        raise AssertionError("unsafe container reached start")
    async def inspect(name):
        return current
    monkeypatch.setattr(p.boundary, "platform", platform)
    monkeypatch.setattr(p.boundary, "invoke", invoke)
    monkeypatch.setattr(p.boundary, "inspect", inspect)
    with pytest.raises(RuntimeError, match="isolated"):
        await p.acquire(req, quote)
    assert not any(arguments[0] == "start" for arguments in calls)


def test_private_daemon_diagnostics_do_not_enter_exception_repr_or_message():
    error = docker.DockerCommandError("bounded_failure", diagnostics=b"PRIVATE DATA")
    assert "PRIVATE" not in repr(error) and "PRIVATE" not in str(error)
    assert error.diagnostics.get_secret_value() == b"PRIVATE DATA"
