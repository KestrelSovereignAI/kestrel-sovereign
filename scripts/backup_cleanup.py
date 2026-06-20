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
import json
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


class LighthouseClient(Protocol):
    async def get_uploads(self, last_key: str | None = None) -> Mapping[str, Any]: ...
    async def download(self, cid: str, timeout: float | None = None) -> bytes: ...
    async def delete_file(self, cid: str) -> Mapping[str, Any]: ...
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


def _manifest_snapshot_cids(manifest: Mapping[str, Any]) -> set[str]:
    cids: set[str] = set()
    for key in ("snapshot_cid", "cid", "state_cid", "backup_cid"):
        value = manifest.get(key)
        if value:
            cids.add(str(value))
    for key in ("snapshots", "snapshot_cids", "files", "items"):
        value = manifest.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    for cid_key in ("snapshot_cid", "cid", "Hash", "hash", "fileHash"):
                        cid_value = item.get(cid_key)
                        if cid_value:
                            cids.add(str(cid_value))
                elif item:
                    cids.add(str(item))
        elif isinstance(value, Mapping):
            for cid_value in value.values():
                if isinstance(cid_value, Mapping):
                    cid_value = cid_value.get("snapshot_cid") or cid_value.get("cid")
                if cid_value:
                    cids.add(str(cid_value))
    return cids


async def _all_lighthouse_uploads(client: LighthouseClient) -> list[Mapping[str, Any]]:
    uploads: list[Mapping[str, Any]] = []
    last_key: str | None = None
    while True:
        page = await client.get_uploads(last_key=last_key)
        file_list = page.get("fileList", [])
        if not isinstance(file_list, list):
            break
        uploads.extend(item for item in file_list if isinstance(item, Mapping))
        last_key = page.get("lastKey") or page.get("nextLastKey")
        if not last_key or not file_list:
            break
    return uploads


async def lighthouse_records(client: LighthouseClient) -> list[BackupRecord]:
    uploads = await _all_lighthouse_uploads(client)
    cid_to_agent: dict[str, str] = {}
    manifest_cids: set[str] = set()

    for upload in uploads:
        cid = _cid(upload)
        if not cid:
            continue
        filename = _filename(upload)
        agent_id = _agent_from_manifest_filename(filename)
        if agent_id is None and filename.startswith("kestrel_manifest__"):
            agent_id = _agent_from_manifest_filename(filename)
        if agent_id is None:
            continue
        manifest_cids.add(cid)
        cid_to_agent[cid] = agent_id
        try:
            body = await client.download(cid)
            manifest = json.loads(body.decode("utf-8"))
        except Exception:
            continue
        if isinstance(manifest, Mapping):
            body_agent = manifest.get("agent_id")
            if body_agent:
                agent_id = str(body_agent)
                cid_to_agent[cid] = agent_id
            for snapshot_cid in _manifest_snapshot_cids(manifest):
                cid_to_agent[snapshot_cid] = agent_id

    records: list[BackupRecord] = []
    for upload in uploads:
        cid = _cid(upload)
        if not cid:
            continue
        filename = _filename(upload) or cid
        agent_id = cid_to_agent.get(cid)
        metadata = dict(upload)
        role = "identity" if cid in manifest_cids else upload.get("role")
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


def _append_audit(audit_log: Path, row: PlannedRecord, result: str) -> None:
    entry = {
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "store": row.record.store,
        "key": row.record.key,
        "agent_id": row.record.agent_id,
        "name": row.record.name,
        "size": row.record.size,
        "reason": row.reason,
        "result": result,
    }
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
    rows: list[PlannedRecord], audit_log: Path, batch_size: int
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
            _append_audit(audit_log, row, "deleted")


async def apply_plan(
    plan: CleanupPlan,
    *,
    lighthouse_client: LighthouseClient | None,
    confirmation: str | None,
    audit_log: Path,
    gcs_batch_size: int = 100,
) -> int:
    deletions = list(plan.deletions)
    if not deletions:
        return 0
    if not _confirm_apply(confirmation):
        print("Refusing to delete: typed confirmation did not match.", file=sys.stderr)
        return 2

    gcs_rows = [row for row in deletions if row.record.store == "gcs"]
    lighthouse_rows = [row for row in deletions if row.record.store == "lighthouse"]
    if lighthouse_rows and lighthouse_client is None:
        raise RuntimeError("Lighthouse deletion requested without a client")
    if gcs_rows:
        _gcs_delete_batches(gcs_rows, audit_log, gcs_batch_size)
    if lighthouse_rows:
        for row in lighthouse_rows:
            await lighthouse_client.delete_file(row.record.key)
            _append_audit(audit_log, row, "deleted")
    return 0


async def _main_async(args: argparse.Namespace) -> int:
    policy = load_retention_policy()
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

    plan = build_plan(records, policy)
    print(render_report(plan))
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
        "--apply",
        action="store_true",
        help="Delete planned records after typed confirmation",
    )
    parser.add_argument(
        "--confirm",
        help=f'Non-interactive confirmation; must equal "{CONFIRMATION_PHRASE}"',
    )
    parser.add_argument("--audit-log", default=AUDIT_LOG)
    parser.add_argument("--gcs-batch-size", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_main_async(args))
    except Exception as exc:  # noqa: BLE001 - CLI should surface exact failure.
        print(f"backup cleanup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
