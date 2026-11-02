"""Actual HTTP/SQLite/native peer validation; no Docker daemon or browser effects."""
import base64
import hashlib
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from kestrel_sdk.computer_use import ActionReceipt, ScreenFrame, ScreenObservation
from kestrel_sdk.computer_use.human_control import (
    HumanActionReceipt,
    HumanActionRequest,
    HumanControlLease,
    HumanControlRequest,
    HumanScreenObservation,
)

from kestrel_sovereign.visual_computer.docker import PROVIDER, DockerComputerDriver
from kestrel_sovereign.visual_computer.repository import ActionRecord
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_host import AUTH
from tests.unit.visual_computer.test_host import (
    visual_host as visual_host,  # noqa: PLC0414
)
from tests.unit.visual_computer.test_human_http import setup

CASES = ["foreign-observe", "foreign-act", "foreign-action-status", "foreign-revoke", "unhashable-error", "unhashable-list-error", "withdrawn-view", "valid-observe", "valid-act", "valid-revoke"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES)
async def test_original_native_human_peer_reply_classification(visual_host, harness, monkeypatch, tmp_path, case):
    # Canonical C3 seam: set the bounded adapter's real driver identity before
    # genuine admission, then select the production native driver afterwards.
    harness.adapter.driver_id = PROVIDER
    client, feature, extension, sid, base, body = await setup(visual_host, harness)
    runtime = feature.runtime
    lease = HumanControlLease.model_validate_json(json.dumps(body))
    original_drivers = dict(runtime.drivers)
    driver = DockerComputerDriver(docker_socket=str(tmp_path / "owned-unused-daemon.sock"))
    driver.human_guard = runtime.egress.human_operation
    driver.human_status_guard = runtime.egress.human_status_operation
    driver._connections[sid] = harness.adapter.last_connection
    assert driver.human_guard.__self__ is runtime.egress and driver.human_status_guard.__self__ is runtime.egress
    async def bound(*args, **kwargs):
        return {"container_id": "c" * 64}
    monkeypatch.setattr(driver, "_bound", bound)
    executable, received = tmp_path / "owned-native-peer", tmp_path / "original-wire.jsonl"
    driver.boundary.executable = str(executable)
    hp = base + "/human/control"
    facts = {"case": case, "controlled_peer_not_browser": True, "only_daemon_binding_substituted": True,
        "original_driver_id": driver.driver_id, "lease_driver_id": lease.request.driver_id}
    response = cleanup = repeated = status = None
    payload = request = None

    def program(envelope):
        # Adversarial result DATA, not an asserted real SDK effect/outcome.
        script = "#!" + sys.executable + " -B\nimport json,sys\nfrom pathlib import Path\n"
        script += "raw=sys.stdin.buffer.read()\nvalue=json.loads(raw)\n"
        script += "with Path(" + repr(str(received)) + ").open('ab') as log: log.write(raw)\n"
        script += "reply=" + repr(envelope) + "\nreply['id']=value['id']\nprint(json.dumps(reply))\n"
        executable.write_text(script)
        executable.chmod(0o700)

    async def charged():
        # Native read only observer; no new capability/checker, row write,
        # result settlement or lease repair is manufactured for the assertion.
        owner = runtime.repository._owner(lease.request.scope)
        session_key = runtime.repository._address("session", sid)
        key = f"{session_key}:{request.action.action_id}"
        row = await harness.db.fetchone("SELECT payload, pending FROM visual_computer_actions WHERE owner = ? AND session_key = ? AND action_id = ?", (owner, session_key, request.action.action_id))
        assert row is not None
        model = runtime.repository._open(owner, key, bytes(row[0]), ActionRecord)
        return {"pending": row[1], "dispatch_started": model.dispatch_started, "receipt_is_none": model.receipt is None,
            "original_request_equal": model.request == request.action, "channel": model.channel, "control_id": model.control_id}

    try:
        assert lease.request.driver_id == PROVIDER
        view = await client.get(hp + "/observation", headers=AUTH)
        assert view.status_code == 200, view.text
        session = harness.adapter.sessions[sid]
        frame = view.json()["frame"]
        payload = {"control_id": lease.request.control_id, "consumer_id": lease.request.consumer_id, "action": {
            "action_id": "original-native-" + case, "session_id": sid, "observation_id": frame["observation_id"],
            "control_epoch": session.control_epoch, "not_after": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
            "action": {"kind": "type", "text": "PRIVATE-NATIVE-PEER-INPUT"}}}
        request = HumanActionRequest.model_validate_json(json.dumps(payload))
        if case in {"foreign-observe", "valid-observe"}:
            image = b"PRIVATE-NATIVE-PEER-PIXELS"
            peer_frame = ScreenFrame(observation_id="peer-frame", session_id=sid, control_epoch=session.control_epoch,
                observed_at=datetime.now(UTC), width=100, height=100, media_type="image/png", image_sha256=hashlib.sha256(image).hexdigest())
            foreign = HumanScreenObservation(control_id=lease.request.control_id, consumer_id=("foreign-consumer" if case == "foreign-observe" else lease.request.consumer_id),
                observation=ScreenObservation(frame=peer_frame, image=image))
            envelope = {"result": {"control_id": foreign.control_id, "consumer_id": foreign.consumer_id,
                "frame": foreign.observation.frame.model_dump(mode="json"), "image_base64": base64.b64encode(image).decode()}}
            facts["peer_model_schema_valid"] = True
        elif case in {"foreign-act", "foreign-action-status", "valid-act"}:
            foreign = HumanActionReceipt(control_id=lease.request.control_id, consumer_id=lease.request.consumer_id,
                receipt=ActionReceipt(action_id=(request.action.action_id if case == "valid-act" else "foreign-action"), session_id=sid, observation_id=request.action.observation_id,
                    control_epoch=session.control_epoch, outcome="completed", recorded_at=datetime.now(UTC)))
            envelope = {"result": foreign.model_dump(mode="json")}
            facts["peer_model_schema_valid"] = True
        elif case in {"foreign-revoke", "valid-revoke"}:
            foreign_request = HumanControlRequest.model_validate_json(json.dumps({**lease.request.model_dump(mode="json"), "consumer_id": (lease.request.consumer_id if case == "valid-revoke" else "foreign-consumer")}))
            foreign = HumanControlLease(request=foreign_request, request_sha256=foreign_request.digest(), state="revoked",
                accepted_at=lease.accepted_at, revoked_at=datetime.now(UTC))
            envelope = {"result": foreign.model_dump(mode="json")}
            facts["peer_model_schema_valid"] = True
        elif case in {"unhashable-error", "unhashable-list-error"}:
            envelope = {"error": ({"PRIVATE-NATIVE-PEER-ERROR": []} if case == "unhashable-error" else [])}
        else:
            envelope = {"result": {}}
        program(envelope)
        runtime.drivers = {PROVIDER: driver}
        if case == "withdrawn-view":
            feature._cfg["granted_actions"].remove(VisualAction.HUMAN_VIEW.value)
            response = await client.get(hp + "/observation", headers=AUTH)
        elif case in {"foreign-act", "valid-act"}:
            response = await client.post(hp + "/actions", headers=AUTH, json=payload)
        elif case == "foreign-action-status":
            program({"error": "human_outcome_unknown"})
            initial = await client.post(hp + "/actions", headers=AUTH, json=payload)
            assert initial.status_code == 503, initial.text
            facts["initial_original_act_status"] = initial.status_code
            program(envelope)
            response = await client.get(hp + "/actions/" + request.action.action_id, headers=AUTH)
        elif case in {"foreign-revoke", "valid-revoke"}:
            response = await client.post(hp + "/revoke", headers=AUTH, json={"request_id": "original-native-revoke", "control_epoch": session.control_epoch})
        else:
            response = await client.get(hp + "/observation", headers=AUTH)
        facts.update(actual_http_status=response.status_code, actual_http_body=response.json())
        if case in {"foreign-act", "foreign-action-status"}:
            facts["charged_before_release"] = await charged()
            before = received.read_bytes()
            repeated = await client.post(hp + "/actions", headers=AUTH, json=payload)
            facts["same_id_replay_status"] = repeated.status_code
            facts["no_native_replay"] = received.read_bytes() == before
            status = await client.get(base, headers=AUTH)
            facts["ordinary_status_before_release"] = status.status_code
    finally:
        runtime.drivers = original_drivers
        # Restore the fixture selection only after original native operations
        # returned. Release is genuine foreground HTTP; no custody erasure or
        # fake native human acknowledgement is used for cleanup.
        cleanup = await client.post(base + "/release", headers=AUTH)
        wire = [json.loads(line) for line in received.read_bytes().splitlines()] if received.exists() else []
        facts["original_native_operations"] = [value["arguments"]["operation"] for value in wire]
        facts["native_full_requests_match"] = all(HumanActionRequest.model_validate_json(json.dumps(value["arguments"]["arguments"]["request"])) == request
            for value in wire if value["arguments"]["operation"] == "act")
        facts.update(foreground_release_status=cleanup.status_code, foreground_release_body=cleanup.json(),
            inflight=runtime._inflight, egress_empty=not runtime.egress._calls,
            workspace_borrowers=len(driver._workspace_borrowers), ordinary_borrowers=len(driver._borrowers), native_pumps=len(driver._pumps),
            original_results=len(runtime.repository._original_results), original_holds=len(harness.db.backend._original_result_holds),
            native_wire_lines=len(received.read_bytes().splitlines()) if received.exists() else 0,
            fixture_human_effects=extension.effects, fixture_agent_effects=harness.adapter.effects)
        if case in {"foreign-act", "foreign-action-status"} and request is not None:
            facts["charged_after_release"] = await charged()
        (Path(os.environ.get("CURRENT_ROW_PROOF_DIR", tmp_path)) / (case + "-facts.json")).write_text(json.dumps(facts, indent=2) + "\n")
    # Actual original cleanup/custody oracles run BEFORE classification.
    assert cleanup.status_code == 200 and cleanup.json()["released"] is True, facts
    assert facts["inflight"] == 0 and facts["egress_empty"] and facts["workspace_borrowers"] == 0 and facts["ordinary_borrowers"] == 0 and facts["native_pumps"] == 0, facts
    assert facts["original_results"] == 0 and facts["original_holds"] == 0, facts
    assert extension.effects == 0 and harness.adapter.effects == 0
    assert facts["native_full_requests_match"], facts
    if case == "withdrawn-view":
        assert facts["native_wire_lines"] == 0 and response.status_code == 403, facts
    else:
        assert facts["native_wire_lines"] == (2 if case == "foreign-action-status" else 1), facts
        if case in {"foreign-act", "foreign-action-status"}:
            for key in ("charged_before_release", "charged_after_release"):
                assert facts[key] == {"pending": 1, "dispatch_started": True, "receipt_is_none": True,
                    "original_request_equal": True, "channel": "human", "control_id": "control"}, facts
            assert repeated.status_code == 503 and facts["no_native_replay"], facts
        if case.startswith("valid-"):
            assert response.status_code == 200, facts
        else:
            assert response.status_code == 503, facts
            assert response.json() == {"error": {"code": "visual_outcome_unknown"}}, facts
