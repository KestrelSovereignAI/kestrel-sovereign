"""Workspace Linux filesystem guards; real SDK/Chromium proof is opt-in."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.artifacts import ArtifactFile, ArtifactManifest, ArtifactRef
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceUnsupported
from kestrel_sdk.workspaces import TransferLimits, WorkspaceExportRequest, WorkspaceImportRequest
from tests.unit.test_visual_browser_controller import controller

root = Path(__file__).parents[2] / "docker/visual-browser"
_previous = sys.modules.get("controller")
_spec = importlib.util.spec_from_file_location("controller", root / "controller.py")
_controller = importlib.util.module_from_spec(_spec)
sys.modules["controller"] = _controller
_spec.loader.exec_module(_controller)
try:
    _spec = importlib.util.spec_from_file_location("browser_workspace", root / "workspace.py")
    workspace = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(workspace)
finally:
    if _previous is None:
        sys.modules.pop("controller", None)
    else:
        sys.modules["controller"] = _previous


@pytest.fixture
def transport(tmp_path):
    driver, page, session = controller()
    session.update(state="paused", controller="none", control_epoch=1)
    driver._session = dict(session)
    page.document = "<html><body>frozen browser draft</body></html>"
    async def content():
        return page.document
    page.content = content
    value = workspace.WorkspaceTransport(driver, tmp_path / "transfer")
    return value, session, page


def export(session, **changes):
    return WorkspaceExportRequest(transfer_id="export", session_id=session["session_id"], lease_id=session["lease_id"],
        control_epoch=session["control_epoch"], workspace_id="browser", expires_at=datetime.now(UTC)+timedelta(seconds=30),
        limits=TransferLimits(max_files=4, max_file_bytes=4096, max_total_bytes=8192), paths=("document.html",),
        entrypoint="document.html", **changes)


def incoming(session, *, transfer_id="import", workspace_id="fresh", path="nested/draft.txt", content=b"source"):
    file = ArtifactFile(path=path, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content), media_type="text/plain")
    request = WorkspaceImportRequest(transfer_id=transfer_id, session_id=session["session_id"], lease_id=session["lease_id"],
        control_epoch=session["control_epoch"], workspace_id=workspace_id, expires_at=datetime.now(UTC)+timedelta(seconds=30),
        limits=TransferLimits(max_files=4, max_file_bytes=4096, max_total_bytes=8192),
        source=ArtifactRef(artifact_id="artifact", revision_id="revision"), manifest=ArtifactManifest(files=(file,), entrypoint=path))
    return request, file, content


@pytest.mark.asyncio
async def test_export_is_frozen_idempotent_and_released_ids_do_not_reexport(transport):
    value, session, page = transport
    request = export(session)
    frozen = await value.prepare_export(session, request)
    page.document = "CHANGED"
    assert await value.prepare_export(session, request) == frozen
    import base64
    content = base64.b64decode(value.read_export(session, frozen.snapshot_id, "document.html")["content_base64"])
    assert b"frozen browser draft" in content
    assert hashlib.sha256(content).hexdigest() == frozen.manifest.files[0].sha256
    assert value.release_export(session, request.transfer_id).state == "released"
    assert not value._area("exports", request.transfer_id).exists()
    with pytest.raises(ResourceNotFound):
        await value.prepare_export(session, request)


@pytest.mark.asyncio
async def test_takeover_fences_data_while_cleanup_remains_available(transport):
    value, session, page = transport
    request = export(session)
    await value.prepare_export(session, request)
    value.controller._session.update(control_epoch=2, controller="human")
    current = dict(value.controller._session)
    with pytest.raises(ResourceConflict):
        value.read_export(current, request.transfer_id, "document.html")
    assert value.release_export(session, request.transfer_id).state == "released"


@pytest.mark.parametrize("kind", ["symlink", "directory_symlink", "hardlink", "fifo"])
def test_actual_filesystem_refuses_link_and_special_file_escapes(transport, tmp_path, kind):
    value, _, _ = transport
    source = value._area("workspaces", "browser")
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "private.txt").write_bytes(b"PRIVATE")
    if kind == "directory_symlink":
        (source / "nested").symlink_to(outside, target_is_directory=True)
        path = "nested/private.txt"
    else:
        path = "private.txt"
        if kind == "symlink":
            (source / path).symlink_to(outside / path)
        elif kind == "hardlink":
            os.link(outside / path, source / path)
        else:
            os.mkfifo(source / path)
    with pytest.raises((OSError, ResourceUnsupported)):
        value._file(source, path, 4096)
    assert (outside / "private.txt").read_bytes() == b"PRIVATE"


def test_import_checks_each_payload_retry_and_partial_abort(transport):
    value, session, _ = transport
    request, file, content = incoming(session)
    assert value.prepare_import(session, request).state == "ready"
    with pytest.raises(ResourceConflict, match="integrity"):
        value.write_import_file(session, request.transfer_id, file, b"wrong")
    assert value.write_import_file(session, request.transfer_id, file, content).state == "transferring"
    assert value.write_import_file(session, request.transfer_id, file, content).state == "transferring"
    assert not value._area("workspaces", request.workspace_id).exists()
    assert value.abort_import(session, request.transfer_id).state == "aborted"
    assert not value._area("imports", request.transfer_id).exists()
    with pytest.raises(ResourceNotFound):
        value.prepare_import(session, request)


@pytest.mark.skipif(sys.platform != "linux", reason="atomic RENAME_NOREPLACE is a Linux worker capability")
def test_atomic_import_and_reconciliation_keep_exact_original_receipt(transport):
    value, session, _ = transport
    request, file, content = incoming(session)
    value.prepare_import(session, request)
    value.write_import_file(session, request.transfer_id, file, content)
    receipt = value.commit_import(session, request.transfer_id)
    assert value.commit_import(session, request.transfer_id) == receipt
    assert value.prepare_import(session, request).materialization == receipt
    assert value._file(value._area("workspaces", request.workspace_id), file.path, 4096) == content
    request2, _, _ = incoming(session, transfer_id="another")
    with pytest.raises(ResourceConflict, match="target_exists"):
        value.prepare_import(session, request2)


def test_extra_staged_files_never_become_materialized_content(transport):
    value, session, _ = transport
    request, file, content = incoming(session)
    value.prepare_import(session, request)
    value.write_import_file(session, request.transfer_id, file, content)
    (value._area("imports", request.transfer_id) / "extra").write_bytes(b"UNADMITTED")
    with pytest.raises(ResourceConflict, match="differs"):
        value.commit_import(session, request.transfer_id)
    assert not value._area("workspaces", request.workspace_id).exists()


def test_oversized_transport_request_and_active_session_are_refused(transport):
    value, session, _ = transport
    request, file, content = incoming(session)
    oversized = request.model_copy(update={"limits": TransferLimits(max_files=33, max_file_bytes=4096, max_total_bytes=8192)})
    with pytest.raises(ResourceUnsupported, match="limits"):
        value.prepare_import(session, oversized)
    value.controller._session.update(state="ready", controller="agent")
    with pytest.raises(ResourceConflict, match="paused"):
        value.prepare_import(dict(value.controller._session), request)


def test_worker_image_includes_workspace_contract_and_implementation():
    assert "workspace.py" in (root / "Dockerfile").read_text()
    assert "!workspace.py" in (root / ".dockerignore").read_text().splitlines()
    assert "kestrel-sovereign-sdk==0.43.0" in (root / "requirements.in").read_text()


@pytest.mark.skipif(sys.platform != "linux", reason="atomic RENAME_NOREPLACE is a Linux worker capability")
def test_atomic_install_never_replaces_a_target_created_after_prepare(transport):
    value, session, _ = transport
    request, file, content = incoming(session)
    value.prepare_import(session, request)
    value.write_import_file(session, request.transfer_id, file, content)
    destination = value._area("workspaces", request.workspace_id)
    destination.mkdir()
    inode = destination.stat().st_ino
    with pytest.raises(ResourceConflict, match="target_exists"):
        value.commit_import(session, request.transfer_id)
    assert destination.stat().st_ino == inode
    assert list(destination.iterdir()) == []
    assert value.abort_import(session, request.transfer_id).state == "aborted"


@pytest.mark.skipif(sys.platform != "linux", reason="atomic RENAME_NOREPLACE is a Linux worker capability")
def test_lost_install_receipt_reconciles_exact_files_without_second_install(transport, monkeypatch):
    value, session, _ = transport
    request, file, content = incoming(session)
    value.prepare_import(session, request)
    value.write_import_file(session, request.transfer_id, file, content)
    original = value._persist
    fail = [True]
    def persist(record):
        if record["status"].state == "complete" and fail[0]:
            fail[0] = False
            raise OSError("injected receipt persistence failure")
        return original(record)
    monkeypatch.setattr(value, "_persist", persist)
    with pytest.raises(OSError):
        value.commit_import(session, request.transfer_id)
    assert value.records[request.transfer_id]["status"].state == "unknown"
    inode = value._area("workspaces", request.workspace_id).stat().st_ino
    status = value.status(session, request.transfer_id)
    assert status.state == "complete"
    assert value.commit_import(session, request.transfer_id) == status.materialization
    assert value._area("workspaces", request.workspace_id).stat().st_ino == inode
