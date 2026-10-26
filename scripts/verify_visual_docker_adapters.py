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
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceScope, ResourceUnsupported
from kestrel_sovereign.visual_computer.docker import (
    DockerCommandError, DockerComputerDriver, DockerEnvironmentProvider, LABEL, PROFILE, PROVIDER,
)
from scripts.verify_visual_browser_sandbox import CONTEXT, source_identity
from scripts.visual_owned_peers import OwnedNetworkPeers


async def run(output, socket, *, scenario=None):
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
        try:
            build = await boundary.invoke(["build", "--label", f"{LABEL}.proof={identity}", "--tag", tag, str(CONTEXT)], timeout=600)
        except DockerCommandError as error:
            (output / "build.log").write_bytes(error.diagnostics.get_secret_value())
            raise
        (output / "build.log").write_bytes(build)
        image = json.loads(await boundary.invoke(["image", "inspect", tag]))[0]
        custody["image_id"] = image["Id"]
        if image["Config"].get("Labels", {}).get(LABEL + ".proof") != identity:
            raise RuntimeError("image ownership mismatch")
        record()
        if scenario is not None:
            custody["result"] = await scenario(boundary, image, identity, output)
            if source_identity() != custody["source"]:
                raise RuntimeError("source changed during real Docker acceptance")
            custody["source_unchanged"] = True
            record()
            return custody
        provider = DockerEnvironmentProvider(docker_socket=socket, image="local-worker@" + image["Id"],
                                               credential_key=secrets.token_bytes(32))
        scope = ResourceScope(tenant_id="synthetic", principal_id="adapter-proof")
        now = datetime.now(UTC)
        request = EnvironmentRequest(request_id=identity, scope=scope, provider_id=PROVIDER,
            profile_id=PROFILE, image=provider.image, surface="browser",
            required_capabilities=frozenset({"screenshot", "type", "key", "click", "takeover", "files"}), location="local",
            requested_at=now, expires_at=now + timedelta(seconds=90), idle_ttl_seconds=80,
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
        from kestrel_sdk.artifacts import ArtifactRef, ArtifactFile, ArtifactManifest
        from kestrel_sdk.workspaces import TransferLimits, WorkspaceExportRequest, WorkspaceImportRequest
        from pydantic import SecretBytes
        limits = TransferLimits(max_files=4, max_file_bytes=128*1024, max_total_bytes=256*1024)
        session = await driver.control(session, ControlRequest(request_id="workspace-pause", session_id=session.session_id,
            expected_control_epoch=session.control_epoch, operation="pause"))
        transfer = driver.workspace_transfer()
        def export_request(transfer_id, workspace_id="browser", paths=("document.html",)):
            return WorkspaceExportRequest(transfer_id=transfer_id, session_id=session.session_id, lease_id=session.lease_id,
                control_epoch=session.control_epoch, workspace_id=workspace_id, expires_at=session.expires_at,
                limits=limits, paths=paths, entrypoint=paths[0])
        frozen_request = export_request("browser-export")
        snapshot = await transfer.prepare_export(session, frozen_request)
        document = (await transfer.read_export(session, snapshot.snapshot_id, "document.html")).get_secret_value()
        assert b"Persistent SDK browser draft" in document
        assert await transfer.prepare_export(session, frozen_request) == snapshot
        (output / "document.html").write_bytes(document)
        changed = document.replace(b"Persistent SDK browser draft", b"Pinned revision materialized in Docker")
        file = ArtifactFile(path="nested/document.html", sha256=hashlib.sha256(changed).hexdigest(),
                            size_bytes=len(changed), media_type="text/html")
        incoming = WorkspaceImportRequest(transfer_id="materialize", session_id=session.session_id, lease_id=session.lease_id,
            control_epoch=session.control_epoch, workspace_id="fresh-revision", expires_at=session.expires_at, limits=limits,
            source=ArtifactRef(artifact_id="synthetic-artifact", revision_id="synthetic-revision"),
            manifest=ArtifactManifest(files=(file,), entrypoint=file.path))
        assert (await transfer.prepare_import(session, incoming)).state == "ready"
        await transfer.write_import_file(session, incoming.transfer_id, file, SecretBytes(changed))
        materialized = await transfer.commit_import(session, incoming.transfer_id)
        assert await transfer.commit_import(session, incoming.transfer_id) == materialized
        assert (await transfer.prepare_import(session, incoming)).materialization == materialized
        exported = await transfer.prepare_export(session, export_request("revision-export", "fresh-revision", (file.path,)))
        assert (await transfer.read_export(session, exported.snapshot_id, file.path)).get_secret_value() == changed
        assert (await transfer.release_export(session, "revision-export")).state == "released"
        # Exercise real symlink, hardlink and FIFO boundaries in this exact
        # uniquely owned container; no host filesystem is mounted or addressed.
        cid = lease.provider_resource_id
        adversarial = hashlib.sha256(b"adversarial").hexdigest()
        rejected = []
        for kind in ("symlink", "hardlink", "fifo"):
            await boundary.invoke(["exec", cid, "python", "-c",
                "from pathlib import Path; import os; "
                "root=Path('/tmp/kestrel-browser/workspace-transfer/workspaces/" + adversarial + "'); "
                "root.mkdir(exist_ok=True); p=root/'bad.txt'; p.unlink(missing_ok=True); "
                "outside=Path('/tmp/adapter-proof-outside'); outside.write_bytes(b'PRIVATE FIXTURE'); "
                + ("p.symlink_to(outside)" if kind == "symlink" else "os.link(outside,p)" if kind == "hardlink" else "os.mkfifo(p)")])
            test_request = export_request("reject-" + kind, "adversarial", ("bad.txt",))
            try:
                await transfer.prepare_export(session, test_request)
            except (ResourceUnsupported, ResourceOutcomeUnknown):
                rejected.append(kind)
            else:
                raise AssertionError("unsupported filesystem object exported")
            assert (await transfer.release_export(session, test_request.transfer_id)).state == "released"
        assert rejected == ["symlink", "hardlink", "fifo"]
        # Cleanup a partial import; no completion receipt can exist for it.
        pending = incoming.model_copy(update={"transfer_id": "partial", "workspace_id": "partial-target"})
        await transfer.prepare_import(session, pending)
        assert (await transfer.abort_import(session, pending.transfer_id)).state == "aborted"
        try:
            await transfer.prepare_import(session, pending)
        except ResourceNotFound:
            pass
        else:
            raise AssertionError("aborted transfer was allocated again")
        # A racing empty destination cannot be overwritten by atomic commit.
        racing = incoming.model_copy(update={"transfer_id": "racing", "workspace_id": "racing-target"})
        await transfer.prepare_import(session, racing)
        await transfer.write_import_file(session, racing.transfer_id, file, SecretBytes(changed))
        racing_key = hashlib.sha256(b"racing-target").hexdigest()
        await boundary.invoke(["exec", cid, "python", "-c",
            "from pathlib import Path; Path('/tmp/kestrel-browser/workspace-transfer/workspaces/" + racing_key + "').mkdir()"])
        try:
            await transfer.commit_import(session, racing.transfer_id)
        except ResourceConflict:
            pass
        else:
            raise AssertionError("atomic import replaced a racing destination")
        assert (await transfer.abort_import(session, racing.transfer_id)).state == "aborted"
        session = await driver.control(session, ControlRequest(request_id="workspace-resume", session_id=session.session_id,
            expected_control_epoch=session.control_epoch, operation="resume"))
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
        assert taken.controller == "human" and taken.state == "paused" and taken.control_epoch == session.control_epoch + 1
        assert control_seconds < 5
        assert await driver.status(scope, session.session_id) == taken
        await holding
        try:
            await transfer.read_export(taken, snapshot.snapshot_id, "document.html")
        except ResourceConflict:
            pass
        else:
            raise AssertionError("takeover did not fence workspace reads")
        assert (await transfer.release_export(session, snapshot.snapshot_id)).state == "released"
        try:
            await driver.act(session, typed)
        except ResourceConflict:
            pass
        else:
            raise AssertionError("stale control accepted")
        # Both destinations are inspected listeners owned by this proof. A
        # separate contained peer proves actual IPv4/IPv6 reachability first.
        async with OwnedNetworkPeers(boundary.invoke, image["Id"], identity,
                output/"owned-network", image_label=LABEL+".proof") as peers:
            refused = await peers.refuse_from(cid)
        # Exercise Linux-only atomic rename and lost receipt reconciliation
        # directly against the same image's production filesystem transport.
        linux_filesystem = json.loads(await boundary.invoke(["exec", cid, "python", "-c", r'''
import hashlib,json,secrets,sys
sys.path.insert(0,'/opt/kestrel-browser')
from pathlib import Path
from datetime import UTC,datetime,timedelta
from controller import BrowserController
from workspace import WorkspaceTransport
from kestrel_sdk.artifacts import ArtifactFile,ArtifactManifest,ArtifactRef
from kestrel_sdk.workspaces import TransferLimits,WorkspaceImportRequest
from kestrel_sdk.resources import ResourceConflict
request={"schema_version":1,"session_id":"filesystem-proof","scope":{"schema_version":1,"tenant_id":"synthetic","principal_id":"filesystem-proof","agent_id":None},"lease_id":"filesystem-proof","surface":"browser","required_capabilities":["screenshot"],"expires_at":(datetime.now(UTC)+timedelta(seconds=30)).isoformat()}
controller=BrowserController(object(),request)
controller._session.update(state="paused",controller="none",control_epoch=1)
session=dict(controller._session)
transport=WorkspaceTransport(controller,Path('/tmp/filesystem-proof-'+secrets.token_hex(8)))
content=b'VERIFIED REVISION'
file=ArtifactFile(path="draft.txt",sha256=hashlib.sha256(content).hexdigest(),size_bytes=len(content),media_type="text/plain")
def incoming(name):
 return WorkspaceImportRequest(transfer_id=name,session_id=session['session_id'],lease_id=session['lease_id'],control_epoch=1,workspace_id=name,expires_at=datetime.now(UTC)+timedelta(seconds=20),limits=TransferLimits(max_files=2,max_file_bytes=1024,max_total_bytes=2048),source=ArtifactRef(artifact_id="artifact",revision_id="revision"),manifest=ArtifactManifest(files=(file,),entrypoint=file.path))
racing=incoming('racing')
transport.prepare_import(session,racing)
transport.write_import_file(session,'racing',file,content)
destination=transport._area('workspaces','racing');destination.mkdir();inode=destination.stat().st_ino
try:transport.commit_import(session,'racing')
except ResourceConflict:pass
else:raise AssertionError('atomic install replaced a racing target')
assert destination.stat().st_ino==inode and list(destination.iterdir())==[]
assert transport.abort_import(session,'racing').state=='aborted'
lost=incoming('lost-receipt')
transport.prepare_import(session,lost);transport.write_import_file(session,lost.transfer_id,file,content)
original=transport._persist
failed=[False]
def persist(record):
 if record['status'].state=='complete' and not failed[0]:
  failed[0]=True
  raise OSError('injected receipt fsync failure')
 return original(record)
transport._persist=persist
try:transport.commit_import(session,lost.transfer_id)
except OSError:pass
else:raise AssertionError('receipt persistence failure was hidden')
assert transport.records[lost.transfer_id]['status'].state=='unknown'
installed=transport._area('workspaces',lost.workspace_id);inode=installed.stat().st_ino
status=transport.status(session,lost.transfer_id)
assert status.state=='complete'
assert transport.commit_import(session,lost.transfer_id)==status.materialization
assert installed.stat().st_ino==inode
assert transport._file(installed,file.path,1024)==content
print(json.dumps({'racing_empty_target_not_replaced':True,'lost_receipt_reconciled_without_second_install':True}))
''']))
        # No close: fixed lease expiry must terminate the actual PID tree.
        deadline = asyncio.get_running_loop().time() + 100
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
            "workspace_snapshot": snapshot.model_dump(mode="json"),
            "materialization": materialized.model_dump(mode="json"), "filesystem_refusals": rejected,
            "linux_filesystem": linux_filesystem,
            "partial_import_aborted": True, "racing_target_not_replaced": True,
            "takeover_fenced_workspace_data": True,
            "persistent_pixels": True, "deduplicated_action": receipt.model_dump(mode="json"),
            "priority_takeover_seconds": control_seconds, "stale_control_refused": True,
            "refused_destinations": refused, "autonomous_expiry": state,
            "image_id": image["Id"], "image_architecture": image["Architecture"],
            "docker_version": json.loads(await boundary.invoke(["version", "--format", "{{json .}}"])),
            "sandbox_admission": admission, "png_sha256": hashlib.sha256(after).hexdigest()}
        record()
    except ResourceOutcomeUnknown as error:
        if isinstance(error, DockerCommandError):
            (output / "operation.private.log").write_bytes(error.diagnostics.get_secret_value())
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
