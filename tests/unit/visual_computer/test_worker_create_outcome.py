"""Interrupted create receipts through real Host/SQLite and original closers.

Only daemon replies are scripted. No Docker process or resource is created,
and current absence is deliberately not evidence of native retirement.
"""
import json

import pytest
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown

from kestrel_sovereign.visual_computer.local_worker import _LocalWorkerResources
from kestrel_sovereign.visual_computer.repository import CustodyRecord
from kestrel_sovereign.visual_computer.runtime import VisualAction
from kestrel_sovereign.visual_computer.worker_profile import CONTRACT, LABEL, LocalWorkerProfile
from tests.unit.visual_computer.test_setup_cleanup_sweeps import controlled_setup as controlled_setup, leaves


_CREATE = _LocalWorkerResources.create
_RETIRE = _LocalWorkerResources._retire


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["volume", "network", "browser"])
@pytest.mark.parametrize("reply", ["interrupted", "malformed-ack"])
async def test_unknown_original_create_current_absence_keeps_real_capacity_charged(
    controlled_setup, monkeypatch, kind, reply,
):
    setup, custody = controlled_setup
    setup.runtime.repository.max_requests = 1
    facts = {"commands": [], "volume": False, "network": False, "resource": None}
    failure = ResourceOutcomeUnknown("SCRIPTED original create transport interrupted")
    network_id = "d" * 64

    async def create(resource):
        facts["resource"] = resource
        custody["resources"].append(resource)
        return await _CREATE(resource)

    async def invoke(argv, *, data=None, current=None, timeout=None):
        if current is not None:
            current()
        facts["commands"].append(tuple(argv))
        resource = facts["resource"]
        operation = "browser" if argv[0] == "create" else argv[0]
        is_create = argv[0] == "create" or argv[:2] in (["volume", "create"], ["network", "create"])
        if is_create and operation == kind:
            if reply == "interrupted":
                raise failure
            return b"SCRIPTED-missing-original-create-receipt\n"
        if argv == ["info", "--format", "{{json .}}"]:
            value = {"ServerVersion": "29.8.2", "OSType": "linux", "Architecture": "aarch64", "DefaultRuntime": "runc"}
        elif argv[:2] == ["image", "inspect"]:
            value = [{"Id": argv[2], "Os": "linux", "Architecture": "arm64",
                      "Config": {"Labels": {LABEL + ".contract": CONTRACT}}}]
        elif argv[:2] == ["volume", "create"]:
            facts["volume"] = True
            return (resource.names["volume"] + "\n").encode()
        elif argv[:2] == ["volume", "inspect"]:
            if not facts["volume"]:
                raise ResourceNotFound("SCRIPTED current volume absence")
            value = [{"Name": resource.names["volume"], "Driver": "local", "Mountpoint": "/SCRIPTED-ipc",
                      "CreatedAt": "SCRIPTED-original", "Scope": "local", "Labels": {LABEL: resource.identity},
                      "Options": {"type": "tmpfs", "device": "tmpfs", "o": "size=1048576,uid=10002,gid=10003,mode=0710"}}]
        elif argv[:2] == ["network", "create"]:
            facts["network"] = True
            return (network_id + "\n").encode()
        elif argv[:2] == ["network", "inspect"]:
            if not facts["network"]:
                raise ResourceNotFound("SCRIPTED current network absence")
            value = [{"Id": network_id, "Name": resource.names["network"], "Labels": {LABEL: resource.identity},
                      "Driver": "bridge", "Scope": "local", "EnableIPv4": True, "EnableIPv6": False,
                      "Internal": False, "Attachable": False, "Ingress": False, "ConfigOnly": False,
                      "ConfigFrom": {"Network": ""}, "Options": {"com.docker.network.enable_ipv4": "true"}, "Containers": {}}]
        elif argv[:2] == ["container", "inspect"]:
            raise ResourceNotFound("SCRIPTED current container absence")
        elif argv[:2] in (["network", "rm"], ["volume", "rm"]):
            facts[argv[0]] = False
            return b""
        else:
            raise AssertionError("unexpected scripted daemon operation: " + repr(argv[:2]))
        return json.dumps(value).encode()

    monkeypatch.setattr(_LocalWorkerResources, "create", create)
    monkeypatch.setattr(_LocalWorkerResources, "_retire", _RETIRE)
    monkeypatch.setattr(setup.provider.boundary, "invoke", invoke)
    async with setup.bound() as context:
        request = setup.request(context, request_id="original-create-" + kind + "-" + reply)
        error = None
        try:
            await setup.runtime.admit(context, request)
        except BaseException as caught:
            error = caught
            custody["known_debt"].extend(leaves(caught))
        resource = facts["resource"]
        saved = await setup.runtime.repository.get(request.scope, request_id=request.request_id)
        assert error is not None
        assert not resource._closed, "current absence fabricated original create retirement"
        assert saved.phase == "acquiring" and saved.lease is None
        assert setup.runtime.egress._calls.get(resource.original._key) is resource.original
        async with setup.runtime.operation(context, VisualAction.ADMIT) as check:
            new_request = setup.request(context, request_id="still-charged-" + kind + "-" + reply)
            with pytest.raises(ResourceConflict):
                await setup.runtime.repository.admit(new_request.scope,
                    CustodyRecord(request=new_request, session_id="not-admitted"), check)
        before = list(facts["commands"])
        for recover in (lambda: setup.runtime.release(context, saved.session_id),
                        lambda: setup.runtime.reconcile_request(context, request.request_id)):
            try:
                await recover()
            except BaseException as caught:
                custody["known_debt"].extend(leaves(caught))
            else:
                pytest.fail("unknown original create became a verified cleanup receipt")
        assert facts["commands"] == before, "recovery must retain the original failed closer, never recreate"
        retained = await setup.runtime.repository.get(request.scope, request_id=request.request_id)
        assert retained.phase == "acquiring"


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "observation"), [
    (kind, observation) for kind in ("volume", "network", "browser", "broker")
    for observation in ("owned", "wrong-identity", "missing")
] + [("network", "wrong-native-id")])
async def test_unacknowledged_create_requires_an_exact_observed_original_before_removal(kind, observation):
    """Only canonical retirement executes; all daemon observations are data."""
    profile = LocalWorkerProfile("browser@sha256:" + "a" * 64, "broker@sha256:" + "b" * 64)
    identity, native_id, calls, removed = "c" * 64, "d" * 64, [], set()
    native_kind = "container" if kind in {"browser", "broker"} else kind
    name = "SCRIPTED-original-" + kind

    class Daemon:
        async def inspect(self, target):
            calls.append(("container", "inspect", target))
            if observation == "missing" or target in removed:
                raise ResourceNotFound("SCRIPTED current container absence")
            return {"Id": "malformed" if observation == "wrong-identity" else native_id,
                    "Image": profile.image(kind), "Config": {"Labels": {LABEL: identity}},
                    "State": {"Running": False, "Status": "created", "Pid": 0, "ExitCode": 0}}

        async def invoke(self, argv, *, timeout=None):
            calls.append(tuple(argv))
            if argv[1] == "inspect":
                if observation == "missing" or argv[-1] in removed:
                    raise ResourceNotFound("SCRIPTED current named resource absence")
                return json.dumps([{"Id": "malformed" if observation == "wrong-native-id" else native_id,
                                    "Name": "wrong-name" if observation == "wrong-identity" else name,
                                    "Labels": {LABEL: identity}}]).encode()
            assert argv[1] == "rm" and "--force" not in argv
            assert argv[-1] == (name if kind == "volume" else native_id)
            removed.add(argv[-1])
            return b""

    resource = object.__new__(_LocalWorkerResources)
    resource.profile, resource.identity, resource.boundary = profile, identity, Daemon()
    resource.containers = {kind: name} if native_kind == "container" else {}
    resource._native_instances, resource._container_waits = {}, {}
    resource.volume = name if kind == "volume" else None
    resource.network = name if kind == "network" else None
    resource._volume_identity = resource._network_identity = None
    resource._pending_creates, resource._closed = {(native_kind, name)}, False
    if observation == "owned":
        await resource._retire()
        assert resource._closed and not resource._pending_creates
        assert removed == {name if kind == "volume" else native_id}
    else:
        with pytest.raises((ResourceConflict, ResourceOutcomeUnknown)):
            await resource._retire()
        assert not resource._closed and not removed


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["browser", "broker"])
async def test_discovered_unacknowledged_original_cannot_lose_its_native_wait(role):
    profile = LocalWorkerProfile("browser@sha256:" + "a" * 64, "broker@sha256:" + "b" * 64)
    identity, native_id, name, calls = "c" * 64, "d" * 64, "SCRIPTED-original-" + role, []

    class Daemon:
        async def inspect(self, target):
            return {"Id": native_id, "Image": profile.image(role), "Config": {"Labels": {LABEL: identity}},
                    "State": {"Running": True, "Status": "running", "Pid": 123}}

        async def invoke(self, argv, *, timeout=None):
            calls.append(tuple(argv))
            if argv[0] == "wait":
                raise ResourceNotFound("SCRIPTED original native wait lost after discovery")
            assert argv == ["kill", "--signal=SIGTERM", native_id]
            return b""

    resource = object.__new__(_LocalWorkerResources)
    resource.profile, resource.identity, resource.boundary = profile, identity, Daemon()
    resource.containers, resource._native_instances, resource._container_waits = {role: name}, {}, {}
    resource.network = resource.volume = resource._network_identity = resource._volume_identity = None
    resource._pending_creates, resource._closed = {("container", name)}, False
    with pytest.raises(ResourceOutcomeUnknown, match="disappeared_without_native_wait"):
        await resource._retire()
    assert not resource._closed and resource.containers == {role: native_id}
    assert calls == [("kill", "--signal=SIGTERM", native_id), ("wait", native_id)]
