#!/usr/bin/env python3
"""Dry-run/apply cleanup for historical cloud backup backlog.

The sync targets prune newly-created agent-scoped backups during normal backup
cycles. This script handles the older shared-account backlog where Lighthouse
snapshots may be flat uploads and must be attributed through manifest files
before any deletion is considered.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from kestrel_sovereign.storage.sync.retention import (
    DataClass,
    RetentionItem,
    RetentionPolicy,
    classify,
    load_retention_policy,
    parse_timestamp,
)


DEFAULT_GCS_BUCKET = "kestrel-agent-backup"
DEFAULT_GCS_PREFIX = "kestrel/"
CONFIRMATION_PHRASE = "DELETE BACKUPS"
AUDIT_LOG = "backup_cleanup_deletions.log"
QUARANTINE_STATE = "backup_cleanup_quarantine.json"
RETENTION_POLICY_VERSION = "backup-retention-v1"
TOOL_VERSION = "backup-cleanup-mutation-v1"
DEAL_IMMUTABILITY_CAVEAT = (
    "Lighthouse/Filecoin deletion removes the object from the active backup "
    "namespace/quota and restore catalogs; immutable Filecoin deals may "
    "persist until expiry."
)
QUARANTINE_CLASSES = frozenset(
    {
        "legacy_private_candidate",
        "unattributed_bin",
        "unattributed_private_candidate",
    }
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackupRecord:
    store: str
    key: str
    agent_id: str | None
    name: str
    size: int
    timestamp: datetime | None
    data_class: DataClass
    metadata: Mapping[str, Any]
    attributed: bool = True
    test_artifact: bool = False
    protected: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PlannedRecord:
    record: BackupRecord
    keep: bool
    reason: str


@dataclass(frozen=True)
class CleanupPlan:
    records: tuple[PlannedRecord, ...]

    @property
    def deletions(self) -> tuple[PlannedRecord, ...]:
        return tuple(row for row in self.records if not row.keep)


@dataclass(frozen=True)
class ManifestAttribution:
    agent_id: str
    timestamp: datetime | None
    snapshot_format: str | None
    manifest_cid: str
    cid_kind: str


@dataclass(frozen=True)
class ClassifiedRecord:
    record: BackupRecord
    inventory_class: str
    confidence: str
    reason: str


class LighthouseClient(Protocol):
    async def get_uploads(self, last_key: str | None = None) -> Mapping[str, Any]: ...
    async def download(self, cid: str, timeout: float | None = None) -> bytes: ...
    async def delete_file(self, cid: str) -> Mapping[str, Any]: ...
    async def get_deal_status(self, cid: str) -> Mapping[str, Any]: ...
    async def close(self) -> None: ...


def _size(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _cid(upload: Mapping[str, Any]) -> str | None:
    for key in ("cid", "Hash", "hash", "fileHash", "CID"):
        value = upload.get(key)
        if value:
            return str(value)
    return None


def _filename(upload: Mapping[str, Any]) -> str:
    return str(upload.get("fileName") or upload.get("Name") or upload.get("name") or "")


def _upload_size(upload: Mapping[str, Any]) -> int:
    for key in ("fileSizeInBytes", "Size", "size", "fileSize"):
        if key in upload:
            return _size(upload.get(key))
    return 0


def _upload_timestamp(upload: Mapping[str, Any]) -> datetime | None:
    return (
        parse_timestamp(
            upload.get("createdAt")
            or upload.get("created_at")
            or upload.get("uploadedAt")
            or upload.get("timestamp")
        )
        or parse_timestamp(_filename(upload))
    )


def _looks_like_latest(name: str, key: str) -> bool:
    return Path(name).name == "latest.db" or key.endswith("/latest.db")


def _is_test_artifact(record: BackupRecord) -> bool:
    haystack = " ".join(
        part
        for part in (
            record.agent_id or "",
            record.key,
            record.name,
            str(record.metadata.get("tag") or ""),
        )
        if part
    ).lower()
    filename = Path(record.name).name.lower()
    return (
        (record.agent_id or "").startswith("did:test:")
        or bool(re.match(r"test[^/]*\.db$", filename))
        or "gcs-live-test" in haystack
        or "codex-live-check" in haystack
        or "fixture" in haystack
    )


def _gcs_agent_id(object_name: str, prefix: str) -> str | None:
    normalized_prefix = prefix.strip("/")
    path = object_name.strip("/")
    if normalized_prefix and path.startswith(normalized_prefix + "/"):
        path = path[len(normalized_prefix) + 1 :]
    parts = [part for part in path.split("/") if part]
    if not parts:
        return None
    for part in parts:
        if part.startswith("did:"):
            return part
    if len(parts) >= 2 and parts[1] in {"snapshots", "identity", "manifests"}:
        return parts[0]
    return parts[0] if len(parts) >= 2 else None


def parse_gsutil_ls(output: str, *, bucket: str, prefix: str) -> list[BackupRecord]:
    records: list[BackupRecord] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("TOTAL:") or line.endswith(":"):
            continue
        # `gsutil ls -l` emits: "<size>  <ISO8601-timestamp>  gs://...",
        # e.g. "299177514  2026-06-20T12:00:00Z  gs://bucket/obj". The
        # timestamp is a SINGLE whitespace-free token.
        match = re.match(
            r"(?P<size>\d+)\s+(?P<ts>\S+)\s+(?P<uri>gs://\S+)",
            line,
        )
        if not match:
            continue
        uri = match.group("uri")
        object_name = uri.removeprefix(f"gs://{bucket}/")
        name = Path(object_name).name
        timestamp = parse_timestamp(match.group("ts")) or parse_timestamp(object_name)
        protected = _looks_like_latest(name, object_name)
        record = BackupRecord(
            store="gcs",
            key=uri,
            agent_id=_gcs_agent_id(object_name, prefix),
            name=name,
            size=_size(match.group("size")),
            timestamp=timestamp,
            data_class=classify({"key": object_name, "filename": name}),
            metadata={"object_name": object_name},
            attributed=_gcs_agent_id(object_name, prefix) is not None,
            protected=protected,
            reason="latest.db" if protected else "",
        )
        records.append(_with_test_flag(record))
    return records


def _with_test_flag(record: BackupRecord) -> BackupRecord:
    test_artifact = _is_test_artifact(record)
    return BackupRecord(
        store=record.store,
        key=record.key,
        agent_id=record.agent_id,
        name=record.name,
        size=record.size,
        timestamp=record.timestamp,
        data_class=record.data_class,
        metadata=record.metadata,
        attributed=record.attributed,
        test_artifact=test_artifact,
        protected=record.protected,
        reason=record.reason,
    )


def list_gcs_records(bucket: str, prefix: str) -> list[BackupRecord]:
    target = f"gs://{bucket}/{prefix.rstrip('/')}/**"
    completed = subprocess.run(
        ["gsutil", "ls", "-l", "-r", target],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "gsutil ls failed")
    return parse_gsutil_ls(completed.stdout, bucket=bucket, prefix=prefix)


def _agent_from_manifest_filename(filename: str) -> str | None:
    if filename.startswith("manifest_") and filename.endswith(".json"):
        return filename[len("manifest_") : -len(".json")]
    match = re.match(
        r"kestrel_manifest__(?P<agent>.+)__20\d{6}[_-]\d{6}\.json$",
        filename,
    )
    if match:
        return match.group("agent")
    return None


def _coerce_cid(value: Any) -> str | None:
    """Extract a CID string from a scalar or a dict entry in a collection."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("cid", "Hash", "hash", "snapshot_cid", "payload_cid"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return None


def _manifest_cid_entries(manifest: Mapping[str, Any]) -> dict[str, str]:
    entries: dict[str, str] = {}
    # Scalar CID fields across current + historical manifest formats.
    for key, kind in (
        ("snapshot_cid", "snapshot"),
        ("snapshot_payload_cid", "snapshot_payload"),
        ("cid", "snapshot"),
        ("state_cid", "snapshot"),
        ("backup_cid", "snapshot"),
    ):
        cid = _coerce_cid(manifest.get(key))
        if cid:
            entries.setdefault(cid, kind)
    # Collection CID fields (lists of CIDs or of {cid: ...} dicts).
    for key in ("snapshots", "snapshot_cids", "files", "items"):
        value = manifest.get(key)
        if isinstance(value, (list, tuple)):
            for item in value:
                cid = _coerce_cid(item)
                if cid:
                    entries.setdefault(cid, "snapshot")
    return entries


def _manifest_schema_valid(manifest: Mapping[str, Any]) -> bool:
    # Valid if the manifest references at least one snapshot CID by any
    # supported field (current or historical). Don't hard-require
    # snapshot_cid — older manifests used cid/state_cid/backup_cid/etc.
    if not _manifest_cid_entries(manifest):
        return False
    optional_strings = (
        "snapshot_cid",
        "snapshot_payload_cid",
        "snapshot_format",
        "uploaded_at",
        "source_file",
        "content_hash",
    )
    if any(
        key in manifest and not isinstance(manifest.get(key), str)
        for key in optional_strings
    ):
        return False
    if (
        "uploaded_at" in manifest
        and parse_timestamp(manifest.get("uploaded_at")) is None
    ):
        return False
    if "raw_snapshot_size" in manifest:
        try:
            int(manifest.get("raw_snapshot_size") or 0)
        except (TypeError, ValueError):
            return False
    return True


def _valid_manifest_agent(
    manifest: Mapping[str, Any],
    *,
    filename: str,
) -> str | None:
    filename_agent = _agent_from_manifest_filename(filename)
    raw_body = manifest.get("agent_id")
    body_agent = raw_body.strip() if isinstance(raw_body, str) and raw_body.strip() else None
    # Reject when both are present and disagree (provenance mismatch).
    if body_agent and filename_agent and body_agent != filename_agent:
        return None
    # Fall back to filename attribution when the body lacks agent_id (legacy
    # manifest_<agent>.json / kestrel_manifest__<agent>__... formats).
    agent_id = body_agent or filename_agent
    if not agent_id:
        return None
    if not _manifest_schema_valid(manifest):
        return None
    return agent_id


async def build_manifest_index(
    uploads: Iterable[Mapping[str, Any]],
    client: LighthouseClient,
) -> tuple[dict[str, ManifestAttribution], set[str], dict[str, str]]:
    cid_index: dict[str, ManifestAttribution] = {}
    manifest_cids: set[str] = set()
    manifest_agents: dict[str, str] = {}

    for upload in uploads:
        cid = _cid(upload)
        if not cid:
            continue
        filename = _filename(upload)
        if _agent_from_manifest_filename(filename) is None:
            continue
        manifest_cids.add(cid)
        try:
            body = await client.download(cid)
            manifest = json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - malformed manifests are skipped.
            logger.warning("Skipping unreadable Lighthouse manifest %s: %s", cid, exc)
            continue
        if not isinstance(manifest, Mapping):
            logger.warning("Skipping non-object Lighthouse manifest %s", cid)
            continue
        agent_id = _valid_manifest_agent(manifest, filename=filename)
        if agent_id is None:
            logger.warning("Skipping malformed Lighthouse manifest %s", cid)
            continue
        manifest_agents[cid] = agent_id
        timestamp = parse_timestamp(manifest.get("uploaded_at")) or _upload_timestamp(
            upload
        )
        snapshot_format = manifest.get("snapshot_format")
        if snapshot_format is not None:
            snapshot_format = str(snapshot_format)
        for snapshot_cid, cid_kind in _manifest_cid_entries(manifest).items():
            cid_index[snapshot_cid] = ManifestAttribution(
                agent_id=agent_id,
                timestamp=timestamp,
                snapshot_format=snapshot_format,
                manifest_cid=cid,
                cid_kind=cid_kind,
            )
    return cid_index, manifest_cids, manifest_agents


async def _all_lighthouse_uploads(client: LighthouseClient) -> list[Mapping[str, Any]]:
    uploads: list[Mapping[str, Any]] = []
    last_key: str | None = None
    seen_cursors: set[str] = set()
    account_total: int | None = None
    while True:
        page = await client.get_uploads(last_key=last_key)
        file_list = page.get("fileList", [])
        if not isinstance(file_list, list):
            break
        uploads.extend(item for item in file_list if isinstance(item, Mapping))
        if account_total is None and page.get("totalFiles") is not None:
            account_total = _size(page.get("totalFiles"))
        next_key = page.get("nextLastKey") or page.get("lastKey")
        next_key = str(next_key) if next_key else None
        if not next_key:
            break
        if next_key in seen_cursors:
            logger.warning(
                "Stopping Lighthouse pagination on repeated cursor %s",
                next_key,
            )
            break
        seen_cursors.add(next_key)
        last_key = next_key
    logger.info("Lighthouse total files seen: %s", len(uploads))
    if account_total is not None and len(uploads) != account_total:
        logger.warning(
            "Lighthouse enumeration saw %s files, account reported %s",
            len(uploads),
            account_total,
        )
    return uploads


async def lighthouse_records(client: LighthouseClient) -> list[BackupRecord]:
    uploads = await _all_lighthouse_uploads(client)
    manifest_index, manifest_cids, manifest_agents = await build_manifest_index(
        uploads, client
    )

    records: list[BackupRecord] = []
    for upload in uploads:
        cid = _cid(upload)
        if not cid:
            continue
        filename = _filename(upload) or cid
        attribution = manifest_index.get(cid)
        agent_id = attribution.agent_id if attribution else manifest_agents.get(cid)
        metadata = dict(upload)
        if attribution:
            metadata.update(
                {
                    "manifest_cid": attribution.manifest_cid,
                    "snapshot_format": attribution.snapshot_format,
                    "manifest_cid_kind": attribution.cid_kind,
                }
            )
        if cid in manifest_cids and agent_id:
            metadata["manifest_cid"] = cid
        if cid in manifest_cids:
            role = "identity"
        elif attribution:
            role = "snapshot"
        else:
            role = upload.get("role")
        record = BackupRecord(
            store="lighthouse",
            key=cid,
            agent_id=agent_id,
            name=filename,
            size=_upload_size(upload),
            timestamp=_upload_timestamp(upload),
            data_class=classify(
                {
                    "role": role,
                    "tag": upload.get("tag"),
                    "fileName": filename,
                    "cid": cid,
                }
            ),
            metadata=metadata,
            attributed=agent_id is not None,
        )
        records.append(_with_test_flag(record))
    return records


def classify_inventory_record(record: BackupRecord) -> ClassifiedRecord:
    _ = classify(
        {
            "role": record.data_class.value,
            "fileName": record.name,
            "key": record.key,
            "tag": record.metadata.get("tag"),
            "cid": record.key,
        }
    )
    filename = Path(record.name).name.lower()
    if record.test_artifact or _is_test_artifact(record):
        return ClassifiedRecord(
            record,
            "test_proven_orphan",
            "high",
            "matches did:test/test file/live-check/fixture marker",
        )
    if record.metadata.get("manifest_cid_kind"):
        return ClassifiedRecord(
            record,
            "attributed_snapshot",
            "high",
            f"CID found in manifest index as {record.metadata['manifest_cid_kind']}",
        )
    # GCS is the EXPEDIENT tier: snapshot objects are attributed to an agent by
    # bucket path (kestrel/<agent>/snapshots/<ts>.db), not by a Lighthouse
    # manifest. Only the .../<agent>/snapshots/... shape is retention-managed
    # (eligible past retention; newest-per-agent + live pointers protected).
    # Other GCS objects (e.g. raw .../<agent>/kestrel_prime.db) fall through to
    # the legacy/quarantine branches below — they are NOT auto-deletable.
    if record.store == "gcs" and record.agent_id and "/snapshots/" in record.key:
        return ClassifiedRecord(
            record,
            "attributed_snapshot",
            "high",
            "GCS snapshot attributed to agent by bucket path",
        )
    if filename in {"kestrel_prime.db", "kestrel_prime.db-wal"} or filename.endswith(
        "-wal"
    ):
        return ClassifiedRecord(
            record,
            "legacy_private_candidate",
            "medium",
            "raw legacy database/WAL upload without manifest attribution",
        )
    if filename.endswith(".bin"):
        return ClassifiedRecord(
            record,
            "unattributed_bin",
            "medium",
            "binary blob without manifest attribution",
        )
    return ClassifiedRecord(
        record,
        "unattributed_private_candidate",
        "low" if record.attributed else "medium",
        "no manifest index attribution for this object",
    )


def classify_records(records: Iterable[BackupRecord]) -> list[ClassifiedRecord]:
    return [classify_inventory_record(record) for record in records]


def _object_id(record: BackupRecord) -> str:
    return f"{record.store}:{record.key}"


def _state_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_quarantine_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "objects": {}}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"quarantine state must be a JSON object: {path}")
    objects = data.setdefault("objects", {})
    if not isinstance(objects, dict):
        raise ValueError(f"quarantine state objects must be a JSON object: {path}")
    data.setdefault("version", 1)
    return data


def save_quarantine_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")


def quarantine_records(
    classified: Iterable[ClassifiedRecord],
    *,
    state_path: Path,
) -> dict[str, Any]:
    state = load_quarantine_state(state_path)
    objects: dict[str, Any] = state.setdefault("objects", {})
    now = _state_now()
    added = 0
    kept = 0
    for row in classified:
        if row.inventory_class not in QUARANTINE_CLASSES:
            continue
        object_id = _object_id(row.record)
        existing = objects.get(object_id)
        status = (
            existing.get("status")
            if isinstance(existing, Mapping) and existing.get("status")
            else "quarantined"
        )
        if existing is None:
            added += 1
            first_seen_at = now
        else:
            kept += 1
            first_seen_at = str(existing.get("first_seen_at") or now)
        objects[object_id] = {
            "status": status,
            "store": row.record.store,
            "key": row.record.key,
            "agent_id": row.record.agent_id,
            "name": row.record.name,
            "size": row.record.size,
            "inventory_class": row.inventory_class,
            "confidence": row.confidence,
            "reason": row.reason,
            "first_seen_at": first_seen_at,
            "last_seen_at": now,
        }
    state["updated_at"] = now
    save_quarantine_state(state_path, state)
    return {"added": added, "kept": kept, "path": str(state_path)}


def promote_quarantine_object(state_path: Path, object_ref: str) -> dict[str, Any]:
    state = load_quarantine_state(state_path)
    objects: dict[str, Any] = state.setdefault("objects", {})
    matches = [
        object_id
        for object_id, entry in objects.items()
        if object_ref == object_id
        or (
            isinstance(entry, Mapping)
            and object_ref in {str(entry.get("key")), str(entry.get("name"))}
        )
    ]
    if not matches:
        raise ValueError(f"quarantine object not found: {object_ref}")
    if len(matches) > 1:
        raise ValueError(
            f"quarantine object reference is ambiguous: {object_ref} "
            f"matched {', '.join(sorted(matches))}"
        )
    object_id = matches[0]
    entry = objects[object_id]
    if not isinstance(entry, dict):
        raise ValueError(f"quarantine entry is malformed: {object_id}")
    entry["status"] = "promoted"
    entry["promoted_at"] = _state_now()
    state["updated_at"] = entry["promoted_at"]
    save_quarantine_state(state_path, state)
    return {"promoted": object_id, "path": str(state_path)}


def _promoted_quarantine_ids(state: Mapping[str, Any] | None) -> set[str]:
    if not state:
        return set()
    objects = state.get("objects")
    if not isinstance(objects, Mapping):
        return set()
    return {
        str(object_id)
        for object_id, entry in objects.items()
        if isinstance(entry, Mapping) and entry.get("status") == "promoted"
    }


def _newest_keys_by_agent_class(records: Iterable[BackupRecord]) -> set[str]:
    newest: dict[tuple[str, DataClass], BackupRecord] = {}
    for record in records:
        if not record.agent_id or record.timestamp is None:
            continue
        key = (record.agent_id, record.data_class)
        current = newest.get(key)
        if current is None or record.timestamp > (current.timestamp or record.timestamp):
            newest[key] = record
    return {record.key for record in newest.values()}


def _live_manifest_protected_keys(records: Iterable[BackupRecord]) -> set[str]:
    newest_manifest: dict[str, BackupRecord] = {}
    record_list = list(records)
    for record in record_list:
        if (
            record.store != "lighthouse"
            or not record.agent_id
            or record.metadata.get("manifest_cid") != record.key
            or record.timestamp is None
        ):
            continue
        current = newest_manifest.get(record.agent_id)
        if current is None or record.timestamp > (current.timestamp or record.timestamp):
            newest_manifest[record.agent_id] = record
    live_manifest_cids = {record.key for record in newest_manifest.values()}
    protected = set(live_manifest_cids)
    for record in record_list:
        manifest_cid = record.metadata.get("manifest_cid")
        if isinstance(manifest_cid, str) and manifest_cid in live_manifest_cids:
            protected.add(record.key)
    return protected


def _manifest_index_hash(records: Iterable[BackupRecord]) -> str:
    rows: list[dict[str, Any]] = []
    for record in records:
        manifest_cid = record.metadata.get("manifest_cid")
        manifest_kind = record.metadata.get("manifest_cid_kind")
        if manifest_cid or manifest_kind:
            rows.append(
                {
                    "store": record.store,
                    "key": record.key,
                    "agent_id": record.agent_id,
                    "manifest_cid": manifest_cid,
                    "manifest_cid_kind": manifest_kind,
                }
            )
    payload = json.dumps(
        sorted(rows, key=lambda row: (row["store"], row["key"])),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_delete_plan(
    records: Iterable[BackupRecord],
    policy: RetentionPolicy,
    *,
    quarantine_state: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> CleanupPlan:
    record_list = list(records)
    retention_plan = build_plan(record_list, policy, now=now)
    classified = {row.record.key: row for row in classify_records(record_list)}
    promoted_ids = _promoted_quarantine_ids(quarantine_state)
    newest_keys = _newest_keys_by_agent_class(record_list)
    live_manifest_keys = _live_manifest_protected_keys(record_list)

    rows: list[PlannedRecord] = []
    for row in retention_plan.records:
        record = row.record
        object_id = _object_id(record)
        inventory_class = classified[record.key].inventory_class
        if record.protected:
            rows.append(PlannedRecord(record, True, record.reason or "protected"))
            continue
        if record.key in newest_keys:
            rows.append(PlannedRecord(record, True, "newest"))
            continue
        if record.key in live_manifest_keys:
            rows.append(PlannedRecord(record, True, "live_manifest_referenced"))
            continue
        if inventory_class == "test_proven_orphan":
            rows.append(PlannedRecord(record, False, "test_proven_orphan"))
            continue
        if inventory_class == "attributed_snapshot":
            rows.append(
                PlannedRecord(
                    record,
                    row.keep,
                    row.reason if row.keep else "attributed_snapshot_past_retention",
                )
            )
            continue
        if inventory_class in QUARANTINE_CLASSES:
            if object_id in promoted_ids:
                rows.append(PlannedRecord(record, False, f"promoted_{inventory_class}"))
            else:
                rows.append(
                    PlannedRecord(
                        record,
                        True,
                        f"quarantine_required:{inventory_class}",
                    )
                )
            continue
        rows.append(
            PlannedRecord(
                record,
                True,
                f"class_not_delete_eligible:{inventory_class}",
            )
        )

    rows.sort(
        key=lambda planned: (
            planned.record.store,
            planned.record.agent_id or "",
            planned.record.key,
        )
    )
    return CleanupPlan(tuple(rows))


def build_plan(
    records: Iterable[BackupRecord],
    policy: RetentionPolicy,
    *,
    now: datetime | None = None,
) -> CleanupPlan:
    rows: list[PlannedRecord] = []
    grouped_items: dict[tuple[str, str], list[RetentionItem]] = {}
    by_key: dict[str, BackupRecord] = {}

    for record in records:
        if record.protected:
            rows.append(PlannedRecord(record, True, record.reason or "protected"))
            continue
        if record.test_artifact:
            rows.append(PlannedRecord(record, False, "test_artifact"))
            continue
        if not record.attributed or not record.agent_id:
            rows.append(PlannedRecord(record, True, "unattributed"))
            continue
        if record.timestamp is None:
            rows.append(PlannedRecord(record, True, "missing_timestamp"))
            continue
        item = RetentionItem(
            key=record.key,
            name=record.name,
            timestamp=record.timestamp,
            data_class=record.data_class,
            metadata=record.metadata,
        )
        grouped_items.setdefault((record.store, record.agent_id), []).append(item)
        by_key[record.key] = record

    for items in grouped_items.values():
        decisions = policy.decide(items, now=now)
        rows.extend(
            PlannedRecord(by_key[decision.item.key], decision.keep, decision.reason)
            for decision in decisions
        )
    rows.sort(
        key=lambda row: (row.record.store, row.record.agent_id or "", row.record.key)
    )
    return CleanupPlan(tuple(rows))


def _fmt_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def render_report(plan: CleanupPlan) -> str:
    summary: dict[tuple[str, str], dict[str, int]] = {}
    unattributed: list[PlannedRecord] = []
    for row in plan.records:
        agent = row.record.agent_id or "(unattributed)"
        bucket = summary.setdefault(
            (row.record.store, agent),
            {"keep": 0, "delete": 0, "keep_bytes": 0, "delete_bytes": 0},
        )
        if row.keep:
            bucket["keep"] += 1
            bucket["keep_bytes"] += row.record.size
        else:
            bucket["delete"] += 1
            bucket["delete_bytes"] += row.record.size
        if row.reason == "unattributed":
            unattributed.append(row)

    lines = ["Backup cleanup dry-run report", ""]
    for (store, agent), counts in sorted(summary.items()):
        lines.append(
            f"{store} {agent}: keep {counts['keep']} ({_fmt_bytes(counts['keep_bytes'])}), "
            f"delete {counts['delete']} ({_fmt_bytes(counts['delete_bytes'])})"
        )
    if unattributed:
        lines.append("")
        lines.append("Unattributed Lighthouse/GCS items kept:")
        for row in unattributed:
            lines.append(f"  {row.record.store} {row.record.key} {row.record.name}")
    return "\n".join(lines)


def render_inventory(classified: Iterable[ClassifiedRecord]) -> str:
    summary: dict[tuple[str, str], dict[str, int]] = {}
    for row in classified:
        agent = row.record.agent_id or "(unattributed)"
        bucket = summary.setdefault(
            (row.inventory_class, agent),
            {"count": 0, "bytes": 0},
        )
        bucket["count"] += 1
        bucket["bytes"] += row.record.size

    lines = ["Backup inventory report", ""]
    for (inventory_class, agent), counts in sorted(summary.items()):
        lines.append(
            f"{inventory_class} {agent}: "
            f"{counts['count']} files ({_fmt_bytes(counts['bytes'])})"
        )
    return "\n".join(lines)


def render_classification(classified: Iterable[ClassifiedRecord]) -> str:
    lines = ["Backup classification report", ""]
    for row in sorted(
        classified,
        key=lambda item: (
            item.record.store,
            item.record.agent_id or "",
            item.record.key,
        ),
    ):
        lines.append(
            f"{row.record.store} {row.record.key} {row.record.name}: "
            f"{row.inventory_class} confidence={row.confidence} "
            f"agent={row.record.agent_id or '(unattributed)'} reason={row.reason}"
        )
    return "\n".join(lines)


def render_quarantine_report(result: Mapping[str, Any]) -> str:
    return (
        "Backup quarantine report\n\n"
        f"state: {result['path']}\n"
        f"added: {result['added']}\n"
        f"already_tracked: {result['kept']}\n"
        "provider_deletes: 0"
    )


def render_delete_preflight(
    plan: CleanupPlan,
    *,
    manifest_index_hash: str,
    policy_version: str = RETENTION_POLICY_VERSION,
) -> str:
    summary: dict[str, dict[str, int]] = {}
    for row in plan.deletions:
        inventory_class = classify_inventory_record(row.record).inventory_class
        bucket = summary.setdefault(inventory_class, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += row.record.size
    lines = [
        "Backup delete preflight",
        "",
        f"manifest-index hash: {manifest_index_hash}",
        f"policy version: {policy_version}",
    ]
    if not summary:
        lines.append("eligible deletes: 0")
        return "\n".join(lines)
    for inventory_class, counts in sorted(summary.items()):
        lines.append(
            f"{inventory_class}: {counts['count']} objects "
            f"({_fmt_bytes(counts['bytes'])})"
        )
    return "\n".join(lines)


def _deal_expiry_from_status(status: Mapping[str, Any] | None) -> Any:
    if not status:
        return None
    for key in ("dealExpiry", "deal_expiry", "endEpoch", "expiration", "expiry"):
        if key in status:
            return status.get(key)
    data = status.get("data")
    if isinstance(data, Mapping):
        return _deal_expiry_from_status(data)
    deals = status.get("deals")
    if isinstance(deals, list):
        for deal in deals:
            if isinstance(deal, Mapping):
                expiry = _deal_expiry_from_status(deal)
                if expiry is not None:
                    return expiry
    return None


def _append_audit(
    audit_log: Path,
    row: PlannedRecord,
    result: str,
    *,
    delete_call_result: Any = None,
    actor: str = "backup_cleanup",
    tool_version: str = TOOL_VERSION,
    manifest_index_hash: str | None = None,
    policy_version: str = RETENTION_POLICY_VERSION,
    deal_status: Mapping[str, Any] | None = None,
) -> None:
    entry = {
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "store": row.record.store,
        "key": row.record.key,
        "agent_id": row.record.agent_id,
        "name": row.record.name,
        "size": row.record.size,
        "reason": row.reason,
        "result": result,
        "delete_call_result": delete_call_result,
        "actor": actor,
        "tool_version": tool_version,
        "manifest_index_hash": manifest_index_hash,
        "policy_version": policy_version,
        "inventory_class": classify_inventory_record(row.record).inventory_class,
        "deal_immutability_caveat": DEAL_IMMUTABILITY_CAVEAT,
    }
    if row.record.store == "lighthouse":
        entry["filecoin_status"] = (
            "deleted_from_account_but_deal_may_persist_until_expiry"
        )
        deal_expiry = _deal_expiry_from_status(deal_status)
        if deal_expiry is not None:
            entry["deal_expiry"] = deal_expiry
    with audit_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _confirm_apply(confirmation: str | None) -> bool:
    if confirmation is not None:
        return confirmation == CONFIRMATION_PHRASE
    try:
        answer = input(f'Type "{CONFIRMATION_PHRASE}" to delete planned backups: ')
    except EOFError:
        return False
    return answer == CONFIRMATION_PHRASE


def _gcs_delete_batches(
    rows: list[PlannedRecord],
    audit_log: Path,
    batch_size: int,
    *,
    manifest_index_hash: str | None,
    policy_version: str,
) -> None:
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        completed = subprocess.run(
            ["gsutil", "-m", "rm", *[row.record.key for row in batch]],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "gsutil rm failed")
        for row in batch:
            _append_audit(
                audit_log,
                row,
                "deleted",
                delete_call_result={
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                },
                manifest_index_hash=manifest_index_hash,
                policy_version=policy_version,
            )


def _deletion_allowed(row: PlannedRecord) -> bool:
    inventory_class = classify_inventory_record(row.record).inventory_class
    if inventory_class in {"attributed_snapshot", "test_proven_orphan"}:
        return True
    return (
        inventory_class in QUARANTINE_CLASSES
        and row.reason == f"promoted_{inventory_class}"
    )


async def apply_plan(
    plan: CleanupPlan,
    *,
    lighthouse_client: LighthouseClient | None,
    confirmation: str | None,
    audit_log: Path,
    gcs_batch_size: int = 100,
    manifest_index_hash: str | None = None,
    policy_version: str = RETENTION_POLICY_VERSION,
) -> int:
    deletions = list(plan.deletions)
    if not deletions:
        return 0
    unsafe = [row for row in deletions if not _deletion_allowed(row)]
    if unsafe:
        names = ", ".join(f"{row.record.store}:{row.record.key}" for row in unsafe)
        print(
            "Refusing to delete: plan contains records without mutation proof: "
            f"{names}",
            file=sys.stderr,
        )
        return 2
    if not _confirm_apply(confirmation):
        print("Refusing to delete: typed confirmation did not match.", file=sys.stderr)
        return 2

    gcs_rows = [row for row in deletions if row.record.store == "gcs"]
    lighthouse_rows = [row for row in deletions if row.record.store == "lighthouse"]
    if lighthouse_rows and lighthouse_client is None:
        raise RuntimeError("Lighthouse deletion requested without a client")
    if gcs_rows:
        _gcs_delete_batches(
            gcs_rows,
            audit_log,
            gcs_batch_size,
            manifest_index_hash=manifest_index_hash,
            policy_version=policy_version,
        )
    if lighthouse_rows:
        for row in lighthouse_rows:
            deal_status = None
            get_deal_status = getattr(lighthouse_client, "get_deal_status", None)
            if get_deal_status is not None:
                try:
                    deal_status = await get_deal_status(row.record.key)
                except Exception as exc:  # noqa: BLE001 - audit deletion still proceeds.
                    deal_status = {"error": str(exc)}
            delete_result = await lighthouse_client.delete_file(row.record.key)
            _append_audit(
                audit_log,
                row,
                "deleted",
                delete_call_result=delete_result,
                manifest_index_hash=manifest_index_hash,
                policy_version=policy_version,
                deal_status=deal_status,
            )
    return 0


async def _main_async(args: argparse.Namespace) -> int:
    policy = load_retention_policy()
    if args.mode == "promote":
        if not args.promote_object:
            raise RuntimeError("--promote-object is required in promote mode")
        result = promote_quarantine_object(
            Path(args.quarantine_state),
            args.promote_object,
        )
        print(
            "Backup quarantine promotion\n\n"
            f"state: {result['path']}\n"
            f"promoted: {result['promoted']}"
        )
        return 0

    records: list[BackupRecord] = []
    if not args.skip_gcs:
        records.extend(list_gcs_records(args.gcs_bucket, args.gcs_prefix))

    lighthouse_client: LighthouseClient | None = None
    if not args.skip_lighthouse:
        api_key = os.environ.get("LIGHTHOUSE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LIGHTHOUSE_API_KEY is required unless --skip-lighthouse is set"
            )
        from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient

        lighthouse_client = LighthouseRestClient(api_key=api_key)
        try:
            records.extend(await lighthouse_records(lighthouse_client))
        except Exception:
            await lighthouse_client.close()
            raise

    if args.mode == "inventory":
        print(render_inventory(classify_records(records)))
        if lighthouse_client is not None:
            await lighthouse_client.close()
        return 0
    if args.mode == "classify":
        print(render_classification(classify_records(records)))
        if lighthouse_client is not None:
            await lighthouse_client.close()
        return 0
    if args.mode == "quarantine":
        result = quarantine_records(
            classify_records(records),
            state_path=Path(args.quarantine_state),
        )
        print(render_quarantine_report(result))
        if lighthouse_client is not None:
            await lighthouse_client.close()
        return 0
    quarantine_state = load_quarantine_state(Path(args.quarantine_state))
    if args.mode == "delete":
        plan = build_delete_plan(
            records,
            policy,
            quarantine_state=quarantine_state,
        )
        manifest_hash = _manifest_index_hash(records)
        print(
            render_delete_preflight(
                plan,
                manifest_index_hash=manifest_hash,
                policy_version=RETENTION_POLICY_VERSION,
            )
        )
        try:
            return await apply_plan(
                plan,
                lighthouse_client=lighthouse_client,
                confirmation=args.confirm,
                audit_log=Path(args.audit_log),
                gcs_batch_size=args.gcs_batch_size,
                manifest_index_hash=manifest_hash,
                policy_version=RETENTION_POLICY_VERSION,
            )
        finally:
            if lighthouse_client is not None:
                await lighthouse_client.close()

    # Build the SAME quarantine-aware plan whether previewing or applying, so
    # the dry-run preview is a faithful preview of what --apply will delete
    # (incl. promoted quarantine entries and mutation-proof protections).
    plan = build_delete_plan(
        records,
        policy,
        quarantine_state=quarantine_state,
    )
    manifest_hash = _manifest_index_hash(records)
    print(
        render_delete_preflight(
            plan,
            manifest_index_hash=manifest_hash,
            policy_version=RETENTION_POLICY_VERSION,
        )
    )
    if not args.apply:
        if lighthouse_client is not None:
            await lighthouse_client.close()
        return 0

    try:
        return await apply_plan(
            plan,
            lighthouse_client=lighthouse_client,
            confirmation=args.confirm,
            audit_log=Path(args.audit_log),
            gcs_batch_size=args.gcs_batch_size,
            manifest_index_hash=manifest_hash,
            policy_version=RETENTION_POLICY_VERSION,
        )
    finally:
        if lighthouse_client is not None:
            await lighthouse_client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gcs-bucket",
        default=os.environ.get("GCS_BACKUP_BUCKET", DEFAULT_GCS_BUCKET),
    )
    parser.add_argument(
        "--gcs-prefix",
        default=os.environ.get("GCS_BACKUP_PREFIX", DEFAULT_GCS_PREFIX),
    )
    parser.add_argument("--skip-gcs", action="store_true")
    parser.add_argument("--skip-lighthouse", action="store_true")
    parser.add_argument(
        "--mode",
        choices=(
            "dry-run",
            "inventory",
            "classify",
            "quarantine",
            "delete",
            "promote",
        ),
        default="dry-run",
        help=(
            "inventory/classify are read-only; quarantine writes metadata only; "
            "delete requires typed confirmation; dry-run preserves the historical report."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete planned records after typed confirmation",
    )
    parser.add_argument(
        "--confirm",
        help=f'Non-interactive confirmation; must equal "{CONFIRMATION_PHRASE}"',
    )
    parser.add_argument("--audit-log", default=AUDIT_LOG)
    parser.add_argument("--quarantine-state", default=QUARANTINE_STATE)
    parser.add_argument(
        "--promote-object",
        help="Object id (store:key) or unique key/name to promote from quarantine",
    )
    parser.add_argument("--gcs-batch-size", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return asyncio.run(_main_async(args))
    except Exception as exc:  # noqa: BLE001 - CLI should surface exact failure.
        print(f"backup cleanup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
