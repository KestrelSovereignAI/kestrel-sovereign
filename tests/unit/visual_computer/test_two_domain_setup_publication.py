"""Original Host/SQLite flow with scripted native status/cleanup I/O only.

The models returned by ScriptedTLS are test data, not a genuine browser/native
result. The actual setup owner, HMAC connection, SDK validators, Runtime state
transitions, encrypted SQL and current grants are unmodified.
"""
from datetime import UTC, datetime, timedelta
import json

import pytest
import pytest_asyncio
from kestrel_sdk.computer_use import ActionRequest, ComputerSession
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.visual_computer.local_worker import _LocalWorkerResources
from kestrel_sovereign.visual_computer.worker_profile import DRIVER, OPERATIONS
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_local_two_domain_admit import local_setup as local_setup


@pytest_asyncio.fixture
async def scripted_native(local_setup, monkeypatch):
    setup = local_setup
    state = {"exchanges": [], "closed": [], "wrong_status": False}
    async def create(resource):
        resource.containers = {"browser": "a"*64, "broker": "b"*64}
        resource.incarnations = {"browser": "c"*64, "broker": "d"*64}
    async def live(resource, current):
        current()  # The SAME genuine current restriction is never replaced.
    class ScriptedTLS:
        def __init__(self, resource):
            self.resource = resource
        async def exchange(self, raw, *, admission, timeout_seconds):
            async with admission as current:
                current()
                message = json.loads(raw)
                state["exchanges"].append(message["operation"])
                request = self.resource.scripted_request
                session = ComputerSession(session_id=request.session_id, scope=request.scope,
                    lease_id=request.lease_id, driver_id=DRIVER, surface="browser", capabilities=OPERATIONS,
                    state="ready", controller="agent", control_epoch=0, expires_at=request.expires_at)
                if message["operation"] == "control":
                    session = session.model_copy(update={"state": "closed", "controller": "none", "control_epoch": 1})
                elif message["operation"] != "status":
                    raise AssertionError("no application RPC is allowed during setup")
                if state["wrong_status"]:
                    session = session.model_copy(update={"driver_id": "wrong-driver"})
                current()
                return json.dumps({"schema_version": 1, "id": message["id"],
                                   "result": session.model_dump(mode="json")}).encode()+b"\n"
    async def bootstrap(resource, request, credential):
        assert resource._connection.credential.get_secret_value() == credential
        resource.scripted_request = request
        resource._tls = {role: ScriptedTLS(resource) for role in ("normal", "priority")}
    async def retire(resource):
        state["closed"].append(resource)
        resource._closed = True
    monkeypatch.setattr(_LocalWorkerResources, "create", create)
    monkeypatch.setattr(_LocalWorkerResources, "verify_live", live)
    monkeypatch.setattr(_LocalWorkerResources, "bootstrap", bootstrap)
    monkeypatch.setattr(_LocalWorkerResources, "_retire", retire)
    yield setup, state


@pytest.mark.asyncio
async def test_original_status_and_active_ack_precede_foreground_borrows(scripted_native, monkeypatch):
    setup, state = scripted_native
    save, reached = setup.runtime.repository.save, []
    async def before_active(scope, record, guard=None):
        if record.phase == "active":
            reached.append(record.session)
            before = list(state["exchanges"])
            session = record.session
            request = ActionRequest(action_id="not-admitted", session_id=session.session_id,
                observation_id="no-active-frame", control_epoch=session.control_epoch,
                not_after=datetime.now(UTC)+timedelta(seconds=2), action={"kind": "wait", "milliseconds": 0})
            with pytest.raises(OperatorAuthorizationError):
                await setup.driver.status(scope, session.session_id)
            with pytest.raises(OperatorAuthorizationError):
                await setup.driver.observe(session)
            with pytest.raises(OperatorAuthorizationError):
                await setup.driver.act(session, request)
            with pytest.raises(ResourceConflict):
                await setup.runtime.observe(context, session.session_id)
            assert state["exchanges"] == before == ["status"]
        return await save(scope, record, guard)
    monkeypatch.setattr(setup.runtime.repository, "save", before_active)
    async with setup.bound() as context:
        session = await setup.runtime.admit(context, setup.request(context))
        assert reached == [session]
        assert setup.runtime.egress._calls == {}
        saved = await setup.runtime.repository.get(session.scope, session_id=session.session_id)
        assert saved.phase == "active" and saved.session == session
        try:
            # A fresh real current STATUS activation can now use the same
            # scripted controller and actual finite session borrow.
            async with setup.runtime.operation(context, VisualAction.STATUS) as current:
                actual = await setup.runtime._session_call(context, saved, current, "status")
            assert actual == session
        finally:
            await setup.runtime.release(context, session.session_id)
    assert len(state["closed"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["wrong-status", "lost-active-ack"])
async def test_invalid_status_or_missing_active_ack_settles_after_original_cleanup(scripted_native, monkeypatch, failure):
    setup, state = scripted_native
    if failure == "wrong-status":
        state["wrong_status"] = True
    save = setup.runtime.repository.save
    async def lost_ack(scope, record, guard=None):
        result = await save(scope, record, guard)
        if record.phase == "active":
            raise ResourceOutcomeUnknown("scripted lost original active COMMIT acknowledgement")
        return result
    if failure == "lost-active-ack":
        monkeypatch.setattr(setup.runtime.repository, "save", lost_ack)
    async with setup.bound() as context:
        request = setup.request(context)
        with pytest.raises((ResourceConflict, ResourceOutcomeUnknown)):
            await setup.runtime.admit(context, request)
        saved = await setup.runtime.repository.get(request.scope, request_id=request.request_id)
        assert saved.phase == "released" and saved.lease.state == "released"
        assert setup.runtime._key(request.scope, saved.session_id) not in setup.runtime._verified_sessions
        assert setup.runtime.egress._calls == {}
        assert setup.driver._connections == {}
        assert len(state["closed"]) == 1 and state["closed"][0]._closed
        before = list(state["exchanges"])
        with pytest.raises(ResourceConflict):
            await setup.runtime.observe(context, saved.session_id)
        assert state["exchanges"] == before == ["status"]


# Reuse independent measured HostConfig vectors; only Docker command results,
# cleanup IO and daemon identity fields are SCRIPTED. Host ADMIT/check/SQL and
# the canonical create/bootstrap/live methods execute normally. No native
# resources or ready ComputerSession are supplied by this fixture.
from tests.unit.visual_computer.test_two_domain_worker_profile import (
    measured_created_browser_profile as measured_created_browser_profile,
    measured_created_broker_profile as measured_created_broker_profile,
)


@pytest_asyncio.fixture
async def scripted_daemon_maps(local_setup, monkeypatch, measured_created_browser_profile, measured_created_broker_profile):
    import copy
    from kestrel_sdk.resources import ResourceNotFound, ResourceUnsupported
    from kestrel_sovereign.visual_computer import local_worker as worker
    from kestrel_sovereign.visual_computer.worker_profile import CONTRACT, LABEL

    setup = local_setup
    state = {"failure": None, "resources": None, "create_returned": False, "live_returned": False,
             "initialized": [], "transport_boundary": False, "closed": [], "commands": [],
             "started": set(), "inspects": {"browser": 0, "broker": 0}, "volume": False, "network": False}
    measured = {"browser": measured_created_browser_profile, "broker": measured_created_broker_profile}
    cids = {"browser": "a" * 64, "broker": "b" * 64}
    network_id, mountpoint = "d" * 64, "/SCRIPTED-owned-ipc"
    original_create = _LocalWorkerResources.create

    async def create(resource):
        state["resources"] = resource
        await original_create(resource)
        state["create_returned"] = True
        if state["failure"] == "changed-live-map":
            await resource.verify_live(resource.original._check_original)
            state["live_returned"] = True

    def image(role):
        return {"Id": setup.provider.profile.image(role), "Os": "linux", "Architecture": "arm64",
                "Config": {"Env": ["SCRIPTED_DAEMON_MAP=1"],
                           "Labels": {LABEL + ".contract": ("core-sdk046-two-domain-v1"
                               if state["failure"] == "old-v1-image" else CONTRACT)}}}

    def container(role):
        resource = state["resources"]
        record = copy.deepcopy(measured[role][0])
        record["Id"], record["Image"] = cids[role], image(role)["Id"]
        host = record["HostConfig"]
        host["Mounts"][0]["Source"] = resource.names["volume"]
        host["NetworkMode"] = "none" if role == "browser" else resource.names["network"]
        record["Mounts"][0].update(Name=resource.names["volume"], Source=mountpoint)
        record["Config"]["Env"] = image(role)["Config"]["Env"]
        record["Config"]["Labels"] = {LABEL: resource.identity, LABEL + ".profile": resource.profile.digest()}
        running = role in state["started"]
        record["State"] = {"Status": "running" if running else "created", "Running": running,
                           "Pid": (9101 if role == "browser" else 9102) if running else 0,
                           "StartedAt": "SCRIPTED-" + role + "-original-start"}
        ports = {"41401/tcp": [{"HostIp": "127.0.0.1", "HostPort": "64476"}],
                 "41402/tcp": [{"HostIp": "127.0.0.1", "HostPort": "64477"}]}
        if role == "broker" and state["failure"] == "empty-map":
            ports["41401/tcp"] = []
        if (role == "broker" and state["failure"] in {"changed-bootstrap-map", "changed-live-map"}
                and state["inspects"][role] > 2):
            ports["41401/tcp"][0]["HostPort"] = "64478"
        attachments = {resource.names["network"]: {"NetworkID": network_id}}
        if role == "broker" and state["failure"] == "foreign-attachment":
            attachments["foreign"] = {"NetworkID": "e" * 64}
        record["NetworkSettings"] = {"Ports": ports if role == "broker" else {},
                                     "Networks": attachments if role == "broker" else {}}
        return record

    def network():
        resource = state["resources"]
        return {"Id": network_id, "Name": resource.names["network"], "Labels": {LABEL: resource.identity},
                "Driver": "bridge", "Scope": "local", "EnableIPv4": True, "EnableIPv6": False,
                "Internal": state["failure"] == "internal-network", "Attachable": False, "Ingress": False,
                "ConfigOnly": False, "ConfigFrom": {"Network": ""},
                "Options": {"com.docker.network.enable_ipv4": "true"},
                "Containers": ({cids["broker"]: {}} if "broker" in state["started"] else {})}

    async def invoke(args, *, data=None, current):
        current()  # Original Host checker, never a replacement permission.
        state["commands"].append(tuple(args))  # No bootstrap stdin/key content.
        resource = state["resources"]
        result = None
        if args[0] == "info":
            result = {"ServerVersion": "29.8.2", "OSType": "linux", "Architecture": "aarch64", "DefaultRuntime": "runc"}
        elif args[:2] == ["image", "inspect"]:
            role = next(role for role in cids if args[2] == resource.profile.image(role))
            result = [image(role)]
        elif args[:2] == ["volume", "create"]:
            state["volume"] = True
            return (resource.names["volume"] + "\n").encode()
        elif args[:2] == ["volume", "inspect"]:
            if not state["volume"]:
                raise ResourceNotFound("scripted absent exact volume")
            result = [{"Name": resource.names["volume"], "Driver": "local", "Mountpoint": mountpoint,
                       "CreatedAt": "SCRIPTED-original-volume", "Scope": "local", "Labels": {LABEL: resource.identity},
                       "Options": {"type": "tmpfs", "device": "tmpfs", "o": "size=1048576,uid=10002,gid=10003,mode=0710"}}]
        elif args[:2] == ["network", "create"]:
            assert "--internal" not in args
            assert args == ["network", "create", "--driver", "bridge", "--ipv6=false", "--label",
                            LABEL + "=" + resource.identity, resource.names["network"]]
            state["network"] = True
            return (network_id + "\n").encode()
        elif args[:2] == ["network", "inspect"]:
            if not state["network"]:
                raise ResourceNotFound("scripted absent exact network")
            result = [network()]
        elif args[0] == "create":
            role = args[-1]
            return (cids[role] + "\n").encode()
        elif args[:2] == ["container", "inspect"]:
            if args[2] in resource.names.values():
                raise ResourceNotFound("scripted absent exact container")
            role = next(role for role in cids if args[2] == cids[role])
            state["inspects"][role] += 1
            result = [container(role)]
        elif args[0] == "start":
            state["started"].add(next(role for role in cids if args[1] == cids[role]))
            return b""
        elif args[0] == "exec":
            role = next(role for role in cids if cids[role] in args)
            if "--initialize" in args:
                assert data is not None
                state["initialized"].append(role)
                return b""
            if "admission.json" in args[-1]:
                result = {"uid": 10002, "gid": 10003, "groups": [10003],
                          "capabilities": {key: 0 for key in ("CapEff", "CapBnd", "CapPrm", "CapInh", "CapAmb")},
                          "no_new_privileges": True, "rtprio_limit": [0, 0],
                          "worker_incarnation": resource.incarnations["browser"],
                          "sandbox": "PID namespaces\tYes\nNetwork namespaces\tYes\nSeccomp-BPF sandbox\tYes"}
            else:
                assert "incarnation" in args[-1]
                return (("c" if role == "browser" else "d") * 64 + "\n").encode()
        else:
            raise AssertionError("unexpected scripted daemon operation: " + repr(args[:2]))
        current()
        return json.dumps(result).encode()

    async def retire(resource):
        # SCRIPTED no-resource retirement only. Canonical close Task and Host
        # cleanup still run; this is no physical container/native wait ACK.
        state["closed"].append(resource)
        resource._closed = True

    def transport_boundary():
        state["transport_boundary"] = True
        raise ResourceUnsupported("scripted_transport_boundary_after_metadata")

    monkeypatch.setattr(_LocalWorkerResources, "create", create)
    monkeypatch.setattr(_LocalWorkerResources, "_retire", retire)
    monkeypatch.setattr(setup.provider.boundary, "invoke", invoke)
    monkeypatch.setattr(worker, "_transport_types", transport_boundary)
    return setup, state


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["empty-map", "old-v1-image", "internal-network", "foreign-attachment",
                                    "changed-bootstrap-map", "changed-live-map", "valid-before-transport"])
async def test_v2_original_setup_rejects_bad_metadata_at_its_original_boundary(scripted_daemon_maps, failure):
    from kestrel_sdk.resources import ResourceUnsupported
    setup, state = scripted_daemon_maps
    state["failure"] = failure
    async with setup.bound() as context:
        request = setup.request(context)
        with pytest.raises(ResourceUnsupported):
            await setup.runtime.admit(context, request)
        saved = await setup.runtime.repository.get(request.scope, request_id=request.request_id)
        if failure == "old-v1-image":
            assert saved.phase == "retired" and saved.retired_from == "quoted"
        elif failure in {"changed-bootstrap-map", "valid-before-transport"}:
            assert saved.phase == "released" and saved.lease.state == "released"
        else:
            assert saved.phase == "retired" and saved.retired_from == "acquiring"
            assert saved.lease is None
        assert saved.session is None
    resource = state["resources"]
    assert state["closed"] == [resource] and resource._closed
    assert setup.runtime.egress._calls == {}
    assert setup.driver._connections == {}
    assert resource.original._finished.done() and resource.original._finished.exception() is None
    assert not resource.original._handed_off
    if failure == "old-v1-image":
        assert not state["volume"] and not state["network"] and state["started"] == set()
    elif failure == "internal-network":
        assert state["volume"] and state["network"] and state["started"] == set()
    elif failure in {"empty-map", "foreign-attachment"}:
        assert state["started"] == {"browser", "broker"}
    if failure in {"empty-map", "old-v1-image", "internal-network", "foreign-attachment"}:
        # Independent reached flag kills removal of the early guard even when
        # a later bootstrap/driver refusal would satisfy outer raises().
        assert state["create_returned"] is False
        assert state["initialized"] == [] and state["transport_boundary"] is False
    elif failure == "changed-live-map":
        assert state["create_returned"] is True and state["live_returned"] is False
        assert state["initialized"] == [] and state["transport_boundary"] is False
    else:
        assert state["create_returned"] is True
        assert state["initialized"] == ["browser", "broker"]
        assert state["transport_boundary"] is (failure == "valid-before-transport")
