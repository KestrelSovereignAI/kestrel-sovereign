"""Malformed native Human replies remain unknown after original cleanup.

Actual authenticated private HTTP, canonical driver/RPC/owned executable and
native encrypted SQLite. The finite daemon binding is controlled, as in the
original native_input helper; this is not Docker/Chromium conformance.
"""
import asyncio
from datetime import UTC, datetime, timedelta
import json
import sys

import pytest

from tests.unit.visual_computer.test_host import visual_host as visual_host, AUTH
from tests.unit.visual_computer.test_human_http import setup as setup_http


PRIVATE = "PRIVATE-MALFORMED-NATIVE-HUMAN-REPLY"


async def native_http(visual_host, harness, monkeypatch, tmp_path, mode):
    from kestrel_sovereign.visual_computer.docker import DockerComputerDriver, PROVIDER

    client, agents, _ = visual_host
    runtime = agents["kite"].features["VisualComputerFeature"].runtime
    # Configure the existing controlled adapter's native driver ID BEFORE its
    # genuine HTTP admission, as committed native_input does before setup().
    monkeypatch.setattr(harness.adapter, "driver_id", PROVIDER)
    runtime.drivers = {PROVIDER: harness.adapter}
    client, _, extension, sid, base, lease = await setup_http(visual_host, harness)
    hp = base + "/human/control"
    viewed = await client.get(hp + "/observation", headers=AUTH)
    assert viewed.status_code == 200, viewed.text
    good = viewed.json()
    session = harness.adapter.sessions[sid]
    driver = DockerComputerDriver(docker_socket=str(tmp_path / "absent-owned.sock"))
    driver.human_guard = runtime.egress.human_operation
    driver.human_status_guard = runtime.egress.human_status_operation
    driver._connections[sid] = harness.adapter.last_connection

    async def bound(*args, **kwargs):
        return {"container_id": "c" * 64}

    monkeypatch.setattr(driver, "_bound", bound)
    runtime.drivers[PROVIDER] = driver
    result = dict(good)
    if mode == "observe-valid":
        pass
    elif mode == "observe-bad-base64":
        result["image_base64"] = "!" + PRIVATE
    elif mode == "observe-bad-control-id":
        result["control_id"] = ""
    elif mode == "observe-bad-consumer-id":
        result["consumer_id"] = ""
    elif mode == "observe-missing-image":
        del result["image_base64"]
    elif mode in {"observe-null", "revoke-null"}:
        result = None
    elif mode == "revoke-integer":
        result = 7
    elif mode == "observe-invalid-frame":
        result["frame"] = {"unexpected": PRIVATE}
    elif mode == "revoke-invalid-map":
        result = {"unexpected": PRIVATE}
    elif mode == "shared-envelope-null":
        result = None
    else:
        raise AssertionError("undeclared finite peer response")
    received = tmp_path / "original-native-requests.jsonl"
    executable = tmp_path / "owned-native-worker"
    executable.write_text(
        "#!" + sys.executable + " -B\n"
        "import json,sys\nfrom pathlib import Path\n"
        "data=sys.stdin.buffer.read()\n"
        "with Path(" + repr(str(received)) + ").open('ab') as out:out.write(data)\n"
        "value=json.loads(data)\n"
        "result=json.loads(" + repr(json.dumps(result)) + ")\n"
        + ("print('null')\n" if mode == "shared-envelope-null" else
           "print(json.dumps({'id':value['id'],'result':result}))\n")
    )
    executable.chmod(0o700)
    driver.boundary.executable = str(executable)
    return client, runtime, extension, session, base, hp, lease, good, driver, received


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [
    "observe-valid",  # Genuine original native data must still reach its owner.
    "observe-bad-base64", "observe-bad-control-id", "observe-bad-consumer-id",
    "observe-missing-image", "observe-null",
    "observe-invalid-frame",  # Existing _record UNKNOWN inverse, not new defect.
    "revoke-null", "revoke-integer",
    "revoke-invalid-map",  # Existing _record UNKNOWN inverse, not new defect.
    "shared-envelope-null",  # Pre-existing shared _rpc issue; distinct scope.
])
async def test_native_malformed_human_reply_classification_after_cleanup(
    visual_host, harness, monkeypatch, tmp_path, mode,
):
    from kestrel_sovereign.visual_computer.human import HumanRecord

    client, runtime, extension, session, base, hp, lease, good, driver, received = (
        await native_http(visual_host, harness, monkeypatch, tmp_path, mode)
    )
    scope, sid = session.scope, session.session_id
    original = await runtime.repository.private_record(scope, sid, "human", "control", HumanRecord)
    native_code = asyncio.create_subprocess_exec.__code__
    process_type = asyncio.subprocess.Process
    previous = sys.getprofile()
    children = []

    def observe_return(frame, event, value):
        if previous is not None:
            previous(frame, event, value)
        if (event == "return" and frame.f_code is native_code
                and type(value) is process_type and all(value is not p for p in children)):
            children.append(value)

    response = released = None
    native_wait_results = None
    sys.setprofile(observe_return)
    try:
        if mode.startswith("observe-"):
            response = await client.get(hp + "/observation", headers=AUTH)
        elif mode.startswith("revoke-"):
            response = await client.post(hp + "/revoke", headers=AUTH,
                json={"request_id": "native-revoke-original", "control_epoch": session.control_epoch})
        else:
            # Real frame/consumer/epoch and real HTTP action admission. No
            # fabricated result or no-send fact: native stdin is fully accepted.
            payload = {"control_id": "control", "consumer_id": lease["request"]["consumer_id"],
                "action": {"action_id": "native-original-input", "session_id": sid,
                    "observation_id": good["frame"]["observation_id"],
                    "control_epoch": session.control_epoch,
                    "not_after": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
                    "action": {"kind": "type", "text": "PRIVATE-ORIGINAL-HUMAN-INPUT"}}}
            from kestrel_sdk.computer_use.human_control import HumanActionRequest
            original_request = HumanActionRequest.model_validate_json(json.dumps(payload))
            response = await client.post(hp + "/actions", headers=AUTH, json=payload)
            saved = await runtime.repository.action(scope, sid, "native-original-input")
            assert saved.channel == "human" and saved.control_id == "control"
            assert saved.request == original_request.action
            assert saved.dispatch_started and saved.receipt is None
            assert await runtime.repository.has_pending_actions(scope, sid)
            before_retry = received.read_bytes()
            replay = await client.post(hp + "/actions", headers=AUTH, json=payload)
            assert replay.status_code == 503 and received.read_bytes() == before_retry
        assert received.is_file() and received.stat().st_size > 17
        wires = [json.loads(line) for line in received.read_bytes().splitlines()]
        assert len(wires) == 1
        wire = wires[0]
        assert wire["operation"] == "human"
        expected = "observe" if mode.startswith("observe-") else "revoke" if mode.startswith("revoke-") else "act"
        assert wire["arguments"]["operation"] == expected
        from kestrel_sdk.computer_use import ComputerSession
        assert ComputerSession.model_validate_json(json.dumps(wire["arguments"]["session"])) == session
        if expected == "act":
            assert wire["arguments"]["arguments"]["request"] == original_request.model_dump(mode="json")
        if expected == "revoke":
            assert wire["arguments"]["arguments"]["request"]["request_id"] == "native-revoke-original"
        stored = await runtime.repository.private_record(scope, sid, "human", "control", HumanRecord)
        assert stored.request == original.request and stored.lease == original.lease
        if expected == "observe":
            assert stored.frame == original.frame and stored.revocation is None
        elif expected == "revoke":
            assert stored.revocation.request_id == "native-revoke-original"
            assert stored.revocation.control_epoch == session.control_epoch
            assert stored.frame is None and stored.tombstone is None
        assert extension.effects == 0
    finally:
        try:
            released = await client.post(base + "/release", headers=AUTH)
        finally:
            # Original production boundary has already joined these Process
            # handles. Retain and await their actual native ACKs even on failure.
            try:
                native_wait_results = await asyncio.gather(*(child.wait() for child in children))
            finally:
                sys.setprofile(previous)
    assert children and all(type(p) is process_type and p.returncode == 0 for p in children)
    assert released.status_code == 200 and released.json()["released"] is True
    assert not runtime.egress._calls and runtime._inflight == 0
    assert not driver._workspace_borrowers and not driver._borrowers and not driver._pumps
    assert not runtime.repository._original_results and not harness.db.backend._original_result_holds
    if mode == "shared-envelope-null":
        saved = await runtime.repository.action(scope, sid, "native-original-input")
        assert saved.dispatch_started and saved.receipt is None
        assert await runtime.repository.has_pending_actions(scope, sid)
    assert native_wait_results == [0] * len(children)
    (tmp_path / "native-peer-custody.json").write_text(json.dumps({
        "mode": mode,
        "original_processes": [{"pid": child.pid, "native_returncode": child.returncode,
            "original_wait_return": waited} for child, waited in zip(children, native_wait_results, strict=True)],
        "foreground_release": released.json(),
        "original_native_input_count": len(wires),
        "original_calls_and_result_holds_empty": True,
        "controlled_daemon_binding_not_Docker_conformance": True,
    }, indent=2) + "\n")
    # Classification LAST: unknown native reply is not malformed client input,
    # no replay or proven-no-input denial, and no private error text disclosure.
    if mode == "observe-valid":
        assert response.status_code == 200, response.text
        assert response.json() == good
        assert response.headers["cache-control"].startswith("private, no-store")
    else:
        assert PRIVATE not in response.text and "image_base64" not in response.text
        assert response.status_code == 503, (mode, response.status_code, response.text)
        assert response.json() == {"error": {"code": "visual_outcome_unknown"}}
