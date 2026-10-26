"""Bounded SDK workspace transport at the actual private Linux filesystem.

The host artifact broker alone authorizes publication/materialization. This
worker freezes explicit regular files and installs verified files into fresh
contained workspaces. It never accepts a host path or extracts an archive.
"""
from __future__ import annotations

import asyncio
import base64
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError
from kestrel_sdk.artifacts import ArtifactFile, ArtifactManifest
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceUnsupported
from kestrel_sdk.workspaces import (
    WorkspaceExportRequest, WorkspaceImportRequest, WorkspaceMaterialization,
    WorkspaceSnapshot, WorkspaceTransferStatus,
)
from controller import Refused

MAX_FILES = 32
MAX_FILE_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MAX_RESERVED_BYTES = 16 * 1024 * 1024
MAX_TRANSFERS = 64
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def model(kind, value):
    return kind.model_validate_json(json.dumps(value, allow_nan=False))


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def rename_fresh(source, destination):
    """Linux atomic install that can never replace a concurrently created target."""
    libc = ctypes.CDLL(None, use_errno=True)
    operation = getattr(libc, "renameat2", None)
    if operation is None:
        raise ResourceUnsupported("atomic_fresh_install_unavailable")
    operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    operation.restype = ctypes.c_int
    if operation(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise ResourceConflict("workspace_target_exists")
        raise OSError(code, "contained_atomic_install_failed")


class WorkspaceTransport:
    def __init__(self, controller, root):
        self.controller, self.root = controller, Path(root)
        self.root.mkdir(mode=0o700, exist_ok=False)
        self.records = {}
        for name in ("workspaces", "exports", "imports", "records"):
            (self.root / name).mkdir(mode=0o700)
        (self.root / "workspaces" / key("browser")).mkdir(mode=0o700)

    def _data(self, session, request=None):
        self.controller._bound(session)
        current = self.controller._session
        if (current["state"] != "paused" or current["controller"] != "none"
                or self.controller._control_pending):
            raise ResourceConflict("workspace_requires_paused_uncontrolled_session")
        if request is not None:
            if ((request.session_id, request.lease_id, request.control_epoch)
                    != (current["session_id"], current["lease_id"], current["control_epoch"])
                    or request.expires_at > model_deadline(current["expires_at"])
                    or request.expires_at <= datetime.now(UTC)):
                raise ResourceConflict("workspace_stale_or_expired")

    def _cleanup(self, session):
        # Retirement is available to the authenticated cleanup owner even after
        # epoch/expiry revocation; it never grants new reads or writes.
        self.controller._owner(session.get("scope"), session.get("session_id"))
        if any(session.get(field) != self.controller._session[field] for field in ("lease_id", "driver_id")):
            raise ResourceNotFound("workspace_unavailable")

    def _limits(self, request):
        limits = request.limits
        if (limits.max_files > MAX_FILES or limits.max_file_bytes > MAX_FILE_BYTES
                or limits.max_total_bytes > MAX_TOTAL_BYTES):
            raise ResourceUnsupported("workspace_transport_limits_exceeded")
        paths = request.paths if isinstance(request, WorkspaceExportRequest) else tuple(file.path for file in request.manifest.files)
        if any(len(path.split("/")) > 16 or len(path.encode()) > 1024 for path in paths):
            raise ResourceUnsupported("workspace_transport_path_bounds_exceeded")

    def _existing(self, request, direction):
        previous = self.records.get(request.transfer_id)
        if previous is not None:
            if previous["status"].request_sha256 != request.digest() or previous["status"].direction != direction:
                raise ResourceConflict("workspace_transfer_conflict")
        return previous

    def _reserve(self, request, direction):
        self._limits(request)
        if len(self.records) >= MAX_TRANSFERS:
            raise ResourceUnsupported("workspace_transfer_budget_exhausted")
        reserved = sum(record["reserved"] for record in self.records.values())
        if reserved + request.limits.max_total_bytes > MAX_RESERVED_BYTES:
            raise ResourceUnsupported("workspace_storage_budget_exhausted")
        record = {"request": request, "reserved": request.limits.max_total_bytes,
                  "status": WorkspaceTransferStatus(transfer_id=request.transfer_id,
                    session_id=request.session_id, request_sha256=request.digest(), direction=direction,
                    state="preparing")}
        self.records[request.transfer_id] = record
        self._persist(record)
        return record

    def _persist(self, record):
        # Receipt/state bytes are private tmpfs custody, acknowledged after fsync.
        name = key(record["request"].transfer_id)
        target = self.root / "records" / name
        temporary = target.with_suffix("." + uuid4().hex)
        payload = {"request": record["request"].model_dump(mode="json"),
                   "status": record["status"].model_dump(mode="json"), "reserved": record["reserved"],
                   "intent": record.get("intent").model_dump(mode="json") if record.get("intent") else None}
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(json.dumps(payload, allow_nan=False).encode())
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory = os.open(target.parent, DIRECTORY_FLAGS)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def _transition(self, record, state, **values):
        previous = record["status"]
        record["status"] = record["status"].model_copy(update={"state": state, **values})
        # Validate state/receipt wiring before making it observable.
        record["status"] = model(WorkspaceTransferStatus, record["status"].model_dump(mode="json"))
        try:
            self._persist(record)
        except OSError:
            record["status"] = previous
            raise
        return record["status"]

    def _file(self, root, path, limit):
        """Resolve every component relative to held directory descriptors."""
        ArtifactFile.contained_path(path)
        descriptor = os.open(root, DIRECTORY_FLAGS)
        root_stat = os.fstat(descriptor)
        try:
            parts = path.split("/")
            for component in parts[:-1]:
                following = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor); descriptor = following
                if os.fstat(descriptor).st_dev != root_stat.st_dev:
                    raise ResourceUnsupported("workspace_cross_device_path")
            source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
            try:
                before = os.fstat(source)
                if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid()
                        or before.st_dev != root_stat.st_dev or before.st_size > limit):
                    raise ResourceUnsupported("workspace_requires_bounded_regular_file")
                with os.fdopen(source, "rb", closefd=False) as stream:
                    content = stream.read(limit + 1)
                after = os.fstat(source)
                signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_nlink)
                if len(content) > limit or signature(before) != signature(after) or len(content) != after.st_size:
                    raise ResourceConflict("workspace_file_changed")
                return content
            finally:
                os.close(source)
        finally:
            os.close(descriptor)

    def _write(self, root, path, content):
        ArtifactFile.contained_path(path)
        descriptor = os.open(root, DIRECTORY_FLAGS)
        try:
            parts = path.split("/")
            for component in parts[:-1]:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                following = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor); descriptor = following
            target = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
            with os.fdopen(target, "wb") as stream:
                stream.write(content)
                stream.flush(); os.fsync(stream.fileno())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _area(self, name, transfer_id):
        return self.root / name / key(transfer_id)

    def _lookup(self, transfer_id, direction):
        record = self.records.get(transfer_id)
        if record is None or record["status"].direction != direction:
            raise ResourceNotFound("workspace_transfer_unavailable")
        return record

    async def prepare_export(self, session, request):
        self.controller._bound(session)
        previous = self._existing(request, "export")
        if previous is not None:
            state = previous["status"].state
            if state == "released":
                raise ResourceNotFound("workspace_export_released")
            if state in {"ready", "transferring"}:
                return previous["status"].snapshot
            raise ResourceConflict("workspace_export_requires_reconciliation")
        self._data(session, request)
        record = self._reserve(request, "export")
        target = self._area("exports", request.transfer_id)
        target.mkdir(mode=0o700)
        source = self._area("workspaces", request.workspace_id)
        if request.workspace_id == "browser":
            if request.paths != ("document.html",):
                raise ResourceUnsupported("browser_export_requires_document_html")
            async with asyncio.timeout(5):
                document = (await self.controller.page.content()).encode()
            self._data(session, request)
            if len(document) > request.limits.max_file_bytes:
                raise ResourceUnsupported("browser_document_exceeds_export_limit")
            # The requested browser source is a fresh, explicit HTML snapshot.
            (source / "document.html").unlink(missing_ok=True)
            self._write(source, "document.html", document)
        files, total = [], 0
        for path in request.paths:
            self._data(session, request)
            content = self._file(source, path, request.limits.max_file_bytes)
            total += len(content)
            if total > request.limits.max_total_bytes:
                raise ResourceUnsupported("workspace_total_bytes_exceeded")
            self._write(target, path, content)
            files.append(ArtifactFile(path=path, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content),
                media_type="text/html" if path.endswith(".html") else "application/octet-stream"))
        manifest = ArtifactManifest(files=tuple(files), entrypoint=request.entrypoint)
        snapshot = WorkspaceSnapshot(request=request, snapshot_id=request.transfer_id, manifest=manifest)
        self._data(session, request)
        self._transition(record, "ready", snapshot=snapshot)
        return snapshot

    def read_export(self, session, snapshot_id, path):
        record = self._lookup(snapshot_id, "export")
        self._data(session, record["request"])
        snapshot = record["status"].snapshot
        if record["status"].state not in {"ready", "transferring"} or snapshot is None:
            raise ResourceNotFound("workspace_export_unavailable")
        file = next((file for file in snapshot.manifest.files if file.path == path), None)
        if file is None:
            raise ResourceNotFound("workspace_file_not_selected")
        content = self._file(self._area("exports", snapshot_id), path, file.size_bytes)
        if len(content) != file.size_bytes or hashlib.sha256(content).hexdigest() != file.sha256:
            raise ResourceConflict("workspace_snapshot_integrity_failure")
        self._data(session, record["request"])
        self._transition(record, "transferring")
        return {"content_base64": base64.b64encode(content).decode()}

    def _remove(self, directory):
        if directory.is_symlink():
            directory.unlink()
            return
        if directory.exists():
            # Python's fd-relative implementation refuses symlink swaps.
            if not shutil.rmtree.avoids_symlink_attacks:
                raise ResourceUnsupported("contained_cleanup_unavailable")
            shutil.rmtree(directory)

    def release_export(self, session, transfer_id):
        self._cleanup(session)
        record = self._lookup(transfer_id, "export")
        if record["status"].state == "released":
            return record["status"]
        self._transition(record, "releasing", snapshot=None)
        self._remove(self._area("exports", transfer_id))
        record["reserved"] = 0
        return self._transition(record, "released")

    def prepare_import(self, session, request):
        self.controller._bound(session)
        previous = self._existing(request, "import")
        if previous is not None:
            state = previous["status"].state
            if state == "aborted":
                raise ResourceNotFound("workspace_import_aborted")
            if state == "complete":
                return previous["status"]
            if state in {"ready", "transferring"}:
                self._data(session, request)
                return previous["status"]
            raise ResourceConflict("workspace_import_requires_reconciliation")
        self._data(session, request)
        if request.workspace_id == "browser":
            raise ResourceConflict("workspace_target_exists")
        if self._area("workspaces", request.workspace_id).exists():
            raise ResourceConflict("workspace_target_exists")
        record = self._reserve(request, "import")
        self._area("imports", request.transfer_id).mkdir(mode=0o700)
        return self._transition(record, "ready")

    def write_import_file(self, session, transfer_id, file, content):
        record = self._lookup(transfer_id, "import")
        request = record["request"]
        self._data(session, request)
        if record["status"].state not in {"ready", "transferring"} or file not in request.manifest.files:
            raise ResourceConflict("workspace_file_not_admitted")
        if len(content) != file.size_bytes or hashlib.sha256(content).hexdigest() != file.sha256:
            raise ResourceConflict("workspace_file_integrity_failure")
        target = self._area("imports", transfer_id)
        try:
            existing = self._file(target, file.path, request.limits.max_file_bytes)
        except FileNotFoundError:
            self._write(target, file.path, content)
        else:
            if existing != content:
                raise ResourceConflict("workspace_file_retry_conflict")
        self._data(session, request)
        return self._transition(record, "transferring")

    def _exact_tree(self, root, manifest):
        paths, count = set(), [0]
        def walk(descriptor, prefix="", depth=0):
            if depth > 16:
                raise ResourceUnsupported("workspace_tree_depth_exceeded")
            for name in os.listdir(descriptor):
                count[0] += 1
                if count[0] > MAX_FILES * 16:
                    raise ResourceUnsupported("workspace_tree_entry_limit_exceeded")
                value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                path = prefix + name
                if stat.S_ISDIR(value.st_mode):
                    child = os.open(name, DIRECTORY_FLAGS, dir_fd=descriptor)
                    try:
                        walk(child, path + "/", depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(value.st_mode) and value.st_nlink == 1:
                    paths.add(path)
                else:
                    raise ResourceUnsupported("workspace_tree_contains_unsupported_file")
        descriptor = os.open(root, DIRECTORY_FLAGS)
        try:
            walk(descriptor)
        finally:
            os.close(descriptor)
        if paths != {file.path for file in manifest.files}:
            raise ResourceConflict("workspace_tree_differs_from_manifest")
        for file in manifest.files:
            content = self._file(root, file.path, MAX_FILE_BYTES)
            if len(content) != file.size_bytes or hashlib.sha256(content).hexdigest() != file.sha256:
                raise ResourceConflict("workspace_materialization_integrity_failure")

    def commit_import(self, session, transfer_id):
        record = self._lookup(transfer_id, "import")
        request = record["request"]
        self._data(session, request)
        if record["status"].state == "complete":
            return record["status"].materialization
        if record["status"].state not in {"ready", "transferring"}:
            raise ResourceConflict("workspace_import_not_ready")
        source = self._area("imports", transfer_id)
        self._exact_tree(source, request.manifest)
        self._data(session, request)
        receipt = WorkspaceMaterialization(transfer_id=transfer_id, session_id=request.session_id,
            lease_id=request.lease_id, control_epoch=request.control_epoch, workspace_id=request.workspace_id,
            source=request.source, manifest_sha256=request.manifest.digest(), request_sha256=request.digest(),
            completed_at=datetime.now(UTC))
        # Persist intent before the non-replacing atomic installation. A failed
        # acknowledgement stays unknown and must never cause a second install.
        previous = record["status"].state
        record["intent"] = receipt
        self._transition(record, "unknown")
        try:
            rename_fresh(source, self._area("workspaces", request.workspace_id))
        except (ResourceConflict, ResourceUnsupported, OSError):
            self._transition(record, previous)
            raise
        self._transition(record, "complete", materialization=receipt)
        return receipt

    def abort_import(self, session, transfer_id):
        self._cleanup(session)
        record = self._lookup(transfer_id, "import")
        if record["status"].state in {"complete", "unknown"}:
            raise ResourceConflict("workspace_install_requires_reconciliation")
        if record["status"].state == "aborted":
            return record["status"]
        self._transition(record, "aborting")
        self._remove(self._area("imports", transfer_id))
        record["reserved"] = 0
        return self._transition(record, "aborted")

    def status(self, session, transfer_id):
        self.controller._bound(session)
        record = self.records.get(transfer_id)
        if record is not None and record["status"].state == "unknown" and record.get("intent") is not None:
            target = self._area("workspaces", record["request"].workspace_id)
            if target.exists() and not self._area("imports", transfer_id).exists():
                self._exact_tree(target, record["request"].manifest)
                self._transition(record, "complete", materialization=record["intent"])
        return record["status"] if record is not None else None

    async def dispatch(self, session, operation, arguments):
        try:
            async with self.controller._lock:
                if operation == "prepare_export" and set(arguments) == {"request"}:
                    result = await self.prepare_export(session, model(WorkspaceExportRequest, arguments["request"]))
                elif operation == "prepare_import" and set(arguments) == {"request"}:
                    result = self.prepare_import(session, model(WorkspaceImportRequest, arguments["request"]))
                elif operation == "read_export" and set(arguments) == {"snapshot_id", "path"}:
                    return self.read_export(session, **arguments)
                elif operation == "write_import_file" and set(arguments) == {"transfer_id", "file", "content_base64"}:
                    content = base64.b64decode(arguments["content_base64"], validate=True)
                    result = self.write_import_file(session, arguments["transfer_id"], model(ArtifactFile, arguments["file"]), content)
                elif operation in {"release_export", "commit_import", "abort_import", "status"} and set(arguments) == {"transfer_id"}:
                    result = getattr(self, operation)(session, arguments["transfer_id"])
                else:
                    raise ResourceUnsupported("workspace_operation_unsupported")
                return result.model_dump(mode="json") if result is not None else None
        except ResourceConflict:
            raise Refused("workspace_conflict") from None
        except ResourceNotFound:
            raise Refused("workspace_unavailable") from None
        except (ResourceUnsupported, ValidationError, ValueError, TypeError):
            raise Refused("workspace_unsupported") from None
        except (OSError, TimeoutError, ResourceOutcomeUnknown):
            raise Refused("workspace_outcome_unknown") from None


def model_deadline(value):
    return datetime.fromisoformat(value)
