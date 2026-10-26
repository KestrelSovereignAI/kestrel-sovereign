#!/usr/bin/env python3
"""Opt-in real local SDK adapter/browser acceptance, using only owned resources."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import secrets
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from kestrel_sdk.computer_use import ActionRequest, ControlRequest, EnvironmentRequest, SessionRequest, TypeText
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceScope
from kestrel_sovereign.visual_computer.docker import (
    DockerComputerDriver, DockerEnvironmentProvider, LABEL, PROFILE, PROVIDER,
)
from scripts.verify_visual_browser_sandbox import CONTEXT, source_identity


async def run(output, socket):
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    identity = uuid4().hex
    tag = "kestrel-visual-adapter-proof:" + identity
    custody = {"identity": identity, "image_tag": tag, "image_id": None, "lease": None,
               "cleanup_confirmed": False, "source": source_identity(), "scope": "local SDK Docker adapter"}
    def record():
        (output / "custody.json").write_text(json.dumps(custody, indent=2) + "\n")
    record()
    from kestrel_sovereign.visual_computer.docker import DockerBoundary
    boundary = DockerBoundary(socket)
    provider, lease, request = None, None, None
    uncertain = False
    try:
        build = await boundary.invoke(["build", "--label", f"{LABEL}.proof={identity}", "--tag", tag, str(CONTEXT)], timeout=600)
        (output / "build.log").write_bytes(build)
        image = json.loads(await boundary.invoke(["image", "inspect", tag]))[0]
        custody["image_id"] = image["Id"]
        if image["Config"].get("Labels", {}).get(LABEL + ".proof") != identity:
            raise RuntimeError("image ownership mismatch")
        record()
        provider = DockerEnvironmentProvider(docker_socket=socket, image="local-worker@" + image["Id"],
                                               credential_key=secrets.token_bytes(32))
        scope = ResourceScope(tenant_id="synthetic", principal_id="adapter-proof")
        now = datetime.now(UTC)
        request = EnvironmentRequest(request_id=identity, scope=scope, provider_id=PROVIDER,
            profile_id=PROFILE, image=provider.image, surface="browser",
            required_capabilities=frozenset({"screenshot", "type", "key", "click", "takeover"}), location="local",
            requested_at=now, expires_at=now + timedelta(seconds=65), idle_ttl_seconds=60,
            ready_timeout_seconds=25, max_hourly_cost_usd=Decimal(0), max_total_cost_usd=Decimal(0))
        quote = await provider.quote(request)
        lease = await provider.acquire(request, quote)
        custody["lease"] = lease.model_dump(mode="json"); record()
        repeated = await provider.acquire(request, quote)
        assert repeated.provider_resource_id == lease.provider_resource_id
        connection = await provider.connect(scope, lease)
        assert connection == await provider.connect(scope, lease)
        driver = DockerComputerDriver(docker_socket=socket)
        session_request = SessionRequest(session_id="session-" + identity, scope=scope, lease_id=lease.lease_id,
            surface="browser", required_capabilities=request.required_capabilities, expires_at=lease.expires_at)
        session = await driver.open(session_request, connection)
        assert (await provider.inspect(scope, lease)).state == "ready"
        admission = await driver._admission(connection)
        observation = await driver.observe(session)
        before = observation.image.get_secret_value()
        async def act(effect):
            observation = await driver.observe(session)
            action = ActionRequest(action_id=uuid4().hex, session_id=session.session_id,
                observation_id=observation.frame.observation_id, control_epoch=session.control_epoch,
                not_after=datetime.now(UTC)+timedelta(seconds=5), action=effect)
            receipt = await driver.act(session, action)
            assert receipt.outcome == "completed"
            return action, receipt
        from kestrel_sdk.computer_use import Click
        await act(Click(x=20, y=20))
        typed, receipt = await act(TypeText(text="Persistent SDK browser draft"))
        assert await driver.act(session, typed) == receipt
        assert await driver.action_status(session, typed.action_id) == receipt
        after = (await driver.observe(session)).image.get_secret_value()
        assert before != after
        (output / "browser.png").write_bytes(after)
        # A separately constructed driver binds the same connection/page. It
        # cannot initialize a second browser or discard deduplication state.
        reconnect = DockerComputerDriver(docker_socket=socket)
        assert await reconnect.open(session_request, connection) == session
        assert await reconnect.action_status(session, typed.action_id) == receipt
        assert (await reconnect.observe(session)).image.get_secret_value() == after
        # Occupy all normal slots at the real in-container Unix socket.
        cid = lease.provider_resource_id
        holding = asyncio.create_task(boundary.invoke(["exec", cid, "python", "-c",
            "import socket,time; from pathlib import Path; "
            "s=[socket.socket(socket.AF_UNIX) for _ in range(4)]; "
            "[c.connect('/tmp/kestrel-browser/worker.sock') for c in s]; "
            "Path('/tmp/adapter-proof-slots').touch(); time.sleep(8)"]))
        await boundary.invoke(["exec", cid, "python", "-c",
            "import time; from pathlib import Path; p=Path('/tmp/adapter-proof-slots'); "
            "\nfor _ in range(100):\n if p.exists(): break\n time.sleep(.02)\nelse: raise SystemExit(1)"])
        try:
            await driver.observe(session)
        except ResourceOutcomeUnknown:
            pass
        else:
            raise AssertionError("normal slots were not occupied")
        started = asyncio.get_running_loop().time()
        taken = await driver.control(session, ControlRequest(request_id="takeover", session_id=session.session_id,
            expected_control_epoch=session.control_epoch, operation="takeover"))
        control_seconds = asyncio.get_running_loop().time() - started
        assert taken.controller == "human" and taken.state == "paused" and taken.control_epoch == 1
        assert control_seconds < 5
        assert await driver.status(scope, session.session_id) == taken
        await holding
        try:
            await driver.act(session, typed)
        except ResourceConflict:
            pass
        else:
            raise AssertionError("stale control accepted")
        # Independently measure network=none across families and destinations.
        refused = json.loads(await boundary.invoke(["exec", cid, "python", "-c",
            "import json,socket; result={}; "
            "\nfor family,host in [(socket.AF_INET,'1.1.1.1'),(socket.AF_INET,'169.254.169.254'),"
            "(socket.AF_INET,'192.168.1.1'),(socket.AF_INET,'172.17.0.1'),(socket.AF_INET6,'2606:4700:4700::1111')]:"
            "\n s=socket.socket(family); s.settimeout(.2)"
            "\n try: s.connect((host,80)); result[host]=False"
            "\n except OSError: result[host]=True"
            "\n finally: s.close()"
            "\nprint(json.dumps(result))"]))
        assert all(refused.values())
        # No close: fixed lease expiry must terminate the actual PID tree.
        deadline = asyncio.get_running_loop().time() + 75
        while asyncio.get_running_loop().time() < deadline:
            state = (await boundary.inspect(cid))["State"]
            if not state["Running"]:
                break
            await asyncio.sleep(.5)
        else:
            raise AssertionError("worker failed autonomous expiry")
        assert state["ExitCode"] == 0 and not state["OOMKilled"]
        assert datetime.fromisoformat(state["FinishedAt"]) >= lease.expires_at
        after_source = source_identity()
        if after_source != custody["source"]:
            raise RuntimeError("source changed during real Docker acceptance")
        custody["source_unchanged"] = True
        custody["result"] = {
            "persistent_pixels": True, "deduplicated_action": receipt.model_dump(mode="json"),
            "priority_takeover_seconds": control_seconds, "stale_control_refused": True,
            "refused_destinations": refused, "autonomous_expiry": state,
            "image_id": image["Id"], "image_architecture": image["Architecture"],
            "docker_version": json.loads(await boundary.invoke(["version", "--format", "{{json .}}"])),
            "sandbox_admission": admission, "png_sha256": hashlib.sha256(after).hexdigest()}
        record()
    except ResourceOutcomeUnknown:
        uncertain = True
        custody["outcome_unknown"] = True
        record()
        raise
    finally:
        if provider is not None and lease is not None:
            released = await provider.release(lease.scope, lease)
            custody["lease"] = released.model_dump(mode="json"); record()
        elif provider is not None and request is not None:
            # An unknown acquire can still have created the deterministic
            # resource. Retain uncertainty even after this current cleanup.
            from kestrel_sovereign.visual_computer.docker import _identity
            owner = _identity(request.scope, request.request_id)
            try:
                owned = await boundary.inspect("kestrel-visual-" + owner)
            except ResourceNotFound:
                owned = None
            if owned is not None:
                if owned["Config"].get("Labels", {}).get(LABEL) != owner:
                    raise RuntimeError("refusing cleanup of unowned container")
                custody["unknown_container_id"] = owned["Id"]; record()
                await boundary.invoke(["container", "rm", "--force", owned["Id"]])
                try:
                    await boundary.inspect(owned["Id"])
                except ResourceNotFound:
                    pass
                else:
                    raise RuntimeError("owned container cleanup unconfirmed")
        try:
            image = json.loads(await boundary.invoke(["image", "inspect", tag]))[0]
        except ResourceNotFound:
            image = None
        if image is not None:
            if ((custody["image_id"] is not None and image["Id"] != custody["image_id"])
                    or image["Config"].get("Labels", {}).get(LABEL + ".proof") != identity):
                raise RuntimeError("refusing cleanup of replaced image")
            custody["image_id"] = image["Id"]; record()
            await boundary.invoke(["image", "rm", tag])
            try:
                await boundary.invoke(["image", "inspect", image["Id"]])
            except ResourceNotFound:
                pass
            else:
                raise RuntimeError("owned image cleanup unconfirmed")
        custody["cleanup_confirmed"] = not uncertain
        record()
    return custody


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-socket", required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args.output.resolve(), args.docker_socket))
    print(json.dumps({"passed": "result" in result, "cleanup_confirmed": result["cleanup_confirmed"], "evidence": str(args.output)}))


if __name__ == "__main__":
    main()
