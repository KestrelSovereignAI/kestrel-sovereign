"""Synthetic offline IPC acceptance; invoked by the owned substrate runner."""
import base64
import hashlib
import json
import secrets
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4


def exercise(checked, cid, output, *, monotonic=time.monotonic):
    secret = secrets.token_hex(32)
    scope = {"schema_version": 1, "tenant_id": "synthetic", "principal_id": "owner", "agent_id": "fixture"}
    request = {"schema_version": 1, "session_id": uuid4().hex, "lease_id": uuid4().hex, "scope": scope,
               "surface": "browser", "required_capabilities": ["screenshot", "click", "type"],
               "expires_at": (datetime.now(UTC) + timedelta(seconds=45)).isoformat()}
    program = ["exec", "-i", cid, "python", "/opt/kestrel-browser/rpc_client.py"]
    # Bootstrap is one-shot. Wait for its directory before submitting; do not
    # retry an uncertain initialize operation with a second secret or session.
    checked(["exec", cid, "python", "-c",
             ("import pathlib,time; p=pathlib.Path('/tmp/kestrel-browser'); "
             "\nfor _ in range(100):\n if p.is_dir(): break\n time.sleep(.05)\nelse: raise SystemExit(1)")])
    html = '<!doctype html><style>body{font:24px sans-serif}input{position:absolute;left:20px;top:20px;width:500px;height:40px;font-size:24px}</style><input aria-label="Persistent draft">'
    incarnation = checked(["exec", cid, "python", "-c",
        "from pathlib import Path; print(Path('/tmp/kestrel-browser/incarnation').read_text())"]).decode().strip()
    initial = {"schema_version": 1, "request": request, "secret": secret, "document_html": html,
               "incarnation": incarnation}
    checked([*program, "initialize"], input=json.dumps(initial).encode())
    checked(["exec", cid, "python", "-c",
             ("import pathlib,time; p=pathlib.Path('/tmp/kestrel-browser/worker.sock'); "
             "\nfor _ in range(200):\n if p.exists(): break\n time.sleep(.05)\nelse: raise SystemExit(1)")])
    admission = json.loads(checked(["exec", cid, "python", "-c",
                                   "from pathlib import Path; import sys; sys.stdout.buffer.write(Path('/tmp/kestrel-browser/admission.json').read_bytes())"]))
    if not isinstance(admission, dict) or not admission.get("sandbox") or not admission.get("browser_version"):
        raise RuntimeError("worker sandbox admission evidence is missing")

    def rpc(operation, arguments, *, credential=secret):
        message = {"schema_version": 1, "secret": credential, "id": uuid4().hex,
                   "operation": operation, "arguments": arguments}
        response = json.loads(checked(program, input=json.dumps(message).encode() + b"\n"))
        if response.get("id") != message["id"]:
            raise RuntimeError("IPC response identity differs")
        return response

    session = rpc("status", {"scope": scope, "session_id": request["session_id"]})["result"]
    if rpc("status", {"scope": scope, "session_id": request["session_id"]}, credential="0" * 64).get("error") != "unauthenticated":
        raise RuntimeError("worker accepted a wrong credential")

    def observe():
        result = rpc("observe", {"session": session})["result"]
        image = base64.b64decode(result["image_base64"], validate=True)
        if hashlib.sha256(image).hexdigest() != result["frame"]["image_sha256"] or not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("worker observation integrity failed")
        return result["frame"], image

    def action(frame, effect):
        return {"schema_version": 1, "action_id": uuid4().hex, "session_id": session["session_id"],
                "observation_id": frame["observation_id"], "control_epoch": session["control_epoch"],
                "not_after": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(), "action": effect}

    frame, blank = observe()
    click = action(frame, {"schema_version": 1, "kind": "click", "x": 50, "y": 40, "button": "left"})
    if rpc("act", {"session": session, "request": click})["result"]["outcome"] != "completed":
        raise RuntimeError("worker did not complete focus action")
    frame, _ = observe()
    typed = action(frame, {"schema_version": 1, "kind": "type", "text": "Persistent local worker draft"})
    receipt = rpc("act", {"session": session, "request": typed})["result"]
    if receipt["outcome"] != "completed":
        raise RuntimeError("worker did not complete typing")
    frame, saved = observe()
    if blank == saved:
        raise RuntimeError("browser form did not change")
    if rpc("act", {"session": session, "request": typed})["result"] != receipt:
        raise RuntimeError("accepted action did not retain its receipt")
    if rpc("action_status", {"session": session, "action_id": typed["action_id"]})["result"] != receipt:
        raise RuntimeError("reconnect did not retain the receipt")
    _, repeated = observe()
    if repeated != saved:
        raise RuntimeError("same accepted input changed the page a second time")
    stale = action(frame, {"schema_version": 1, "kind": "type", "text": "STALE"})
    if rpc("act", {"session": session, "request": stale}).get("error") != "stale_observation":
        raise RuntimeError("worker accepted a superseded frame")
    old_session = session
    pause = {"schema_version": 1, "request_id": uuid4().hex, "session_id": session["session_id"],
             "expected_control_epoch": session["control_epoch"], "operation": "pause"}
    session = rpc("control", {"session": session, "request": pause})["result"]
    if session["state"] != "paused" or session["control_epoch"] != 1:
        raise RuntimeError("pause did not fence input")
    if rpc("observe", {"session": old_session}).get("error") != "stale_control":
        raise RuntimeError("old controller retained observation access")
    if rpc("observe", {"session": session}).get("error") != "input_fenced":
        raise RuntimeError("paused observations were not suspended")
    resume = {**pause, "request_id": uuid4().hex, "expected_control_epoch": 1, "operation": "resume"}
    session = rpc("control", {"session": session, "request": resume})["result"]
    frame, resumed = observe()
    if resumed != saved:
        raise RuntimeError("control handoff lost browser state")
    for effect in ({"schema_version": 1, "kind": "key", "key": "ArrowDown"},
                   {"schema_version": 1, "kind": "scroll", "delta_x": 0, "delta_y": 10},
                   {"schema_version": 1, "kind": "wait", "milliseconds": 10000}):
        request_action = action(frame, effect)
        request_action["not_after"] = (datetime.now(UTC) + timedelta(seconds=20)).isoformat()
        started = monotonic()
        if rpc("act", {"session": session, "request": request_action})["result"]["outcome"] != "completed":
            raise RuntimeError("supported key, scroll or maximum wait did not complete")
        if effect["kind"] == "wait":
            wait_seconds = monotonic() - started
            if wait_seconds < 9.9:
                raise RuntimeError("maximum wait returned before its requested duration")
        frame, _ = observe()
    (output / "worker.png").write_bytes(saved)
    # No close request: prove expiry physically retires the process/container.
    deadline = monotonic() + 45
    while monotonic() < deadline:
        record = json.loads(checked(["container", "inspect", cid]))[0]
        if not record["State"]["Running"]:
            state = record["State"]
            try:
                finished = datetime.fromisoformat(state["FinishedAt"])
                expiry = datetime.fromisoformat(request["expires_at"])
            except (KeyError, TypeError, ValueError):
                raise RuntimeError("worker expiry evidence lacks a valid physical retirement time") from None
            if (state["ExitCode"] != 0 or state.get("OOMKilled") is not False
                    or finished.tzinfo is None or finished < expiry):
                raise RuntimeError("worker did not retire normally at expiry")
            break
        time.sleep(0.25)
    else:
        raise RuntimeError("worker did not physically retire on expiry")
    return {"passed": True, "session_id": session["session_id"], "retired_exit_code": record["State"]["ExitCode"],
            "expires_at": request["expires_at"], "finished_at": record["State"]["FinishedAt"],
            "sandbox_admission": admission, "maximum_wait_preserved_session": True, "key_scroll_completed": True,
            "maximum_wait_seconds": wait_seconds,
            "png_sha256": hashlib.sha256(saved).hexdigest(), "same_action_not_replayed": True,
            "transport_reconnect_preserved_state": True, "stale_frame_refused": True,
            "pause_fenced_input_and_observation": True, "expiry_physically_retired": True}
