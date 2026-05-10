"""Storage health checks for operator-facing backup status."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


DEFAULT_LIGHTHOUSE_DEAL_GRACE = timedelta(hours=24)


@dataclass
class TargetHealth:
    """Health result for one storage target."""

    name: str
    configured: bool
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StorageHealthReport:
    """Combined storage health report."""

    status: str
    lighthouse: TargetHealth
    gcs: TargetHealth

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "lighthouse": self.lighthouse.to_dict(),
            "gcs": self.gcs.to_dict(),
        }


def _parse_upload_time(value: Any) -> Optional[datetime]:
    """Parse Lighthouse upload timestamps from milliseconds, seconds, or ISO."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1000
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            return _parse_upload_time(int(raw))
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except ValueError:
            return None
    return None


def _upload_cid(upload: Mapping[str, Any]) -> Optional[str]:
    for key in ("cid", "Hash", "hash", "CID"):
        value = upload.get(key)
        if value:
            return str(value)
    return None


def _latest_lighthouse_snapshot(
    uploads: list[Mapping[str, Any]],
    agent_id: str,
) -> Optional[Mapping[str, Any]]:
    tag = f"kestrel-state-{agent_id}"
    candidates = [
        upload
        for upload in uploads
        if upload.get("tag") == tag
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda upload: _parse_upload_time(upload.get("createdAt"))
        or datetime.min.replace(tzinfo=timezone.utc),
    )


def _latest_lighthouse_manifest(
    uploads: list[Mapping[str, Any]],
    agent_id: str,
) -> Optional[Mapping[str, Any]]:
    tag = f"kestrel-manifest-{agent_id}"
    filename = f"manifest_{agent_id}.json"
    candidates = [
        upload
        for upload in uploads
        if upload.get("tag") == tag or upload.get("fileName") == filename
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda upload: _parse_upload_time(upload.get("createdAt"))
        or datetime.min.replace(tzinfo=timezone.utc),
    )


async def check_lighthouse_health(
    *,
    api_key: Optional[str],
    agent_id: str,
    grace_period: timedelta = DEFAULT_LIGHTHOUSE_DEAL_GRACE,
    now: Optional[datetime] = None,
    client_factory: Optional[Callable[..., Any]] = None,
) -> TargetHealth:
    """Check latest Lighthouse snapshot deal status."""
    if not api_key:
        return TargetHealth(
            name="lighthouse",
            configured=False,
            status="not_configured",
            message="LIGHTHOUSE_API_KEY is not set",
        )

    if client_factory is None:
        from kestrel_sovereign.storage.providers.lighthouse_rest import (
            LighthouseRestClient,
        )

        client_factory = LighthouseRestClient

    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    client = client_factory(api_key=api_key)
    try:
        uploads_response = await client.get_uploads()
        uploads = uploads_response.get("fileList", [])
        if not isinstance(uploads, list):
            uploads = []

        latest = _latest_lighthouse_snapshot(uploads, agent_id)
        manifest = None
        if latest is None:
            latest_manifest = _latest_lighthouse_manifest(uploads, agent_id)
            manifest_cid = _upload_cid(latest_manifest) if latest_manifest else None
            if manifest_cid:
                manifest_bytes = await client.download(manifest_cid)
                manifest = json.loads(manifest_bytes)
                latest = latest_manifest

        if latest is None:
            return TargetHealth(
                name="lighthouse",
                configured=True,
                status="unavailable",
                message="No Lighthouse snapshot upload found",
            )

        cid = str(manifest.get("snapshot_cid")) if manifest else _upload_cid(latest)
        uploaded_at = (
            _parse_upload_time(manifest.get("uploaded_at")) if manifest else None
        ) or _parse_upload_time(latest.get("createdAt"))
        age = current_time - uploaded_at if uploaded_at else None
        deals = await client.get_deal_status(cid) if cid else []
        deal_count = len(deals) if isinstance(deals, list) else 0

        details = {
            "cid": cid,
            "file_name": latest.get("fileName"),
            "tag": latest.get("tag"),
            "resolved_via": "manifest" if manifest else "snapshot_upload",
            "uploaded_at": uploaded_at.isoformat() if uploaded_at else None,
            "age_seconds": int(age.total_seconds()) if age else None,
            "deal_count": deal_count,
            "deals": deals if isinstance(deals, list) else deals,
            "grace_seconds": int(grace_period.total_seconds()),
        }

        if deal_count > 0:
            return TargetHealth(
                name="lighthouse",
                configured=True,
                status="ok",
                message=f"Latest Lighthouse snapshot has {deal_count} Filecoin deal(s)",
                details=details,
            )
        if age is not None and age <= grace_period:
            return TargetHealth(
                name="lighthouse",
                configured=True,
                status="pending",
                message="Latest Lighthouse snapshot has no deals yet but is within grace",
                details=details,
            )
        return TargetHealth(
            name="lighthouse",
            configured=True,
            status="warning",
            message="Latest Lighthouse snapshot has no Filecoin deals beyond grace",
            details=details,
        )
    except Exception as exc:
        return TargetHealth(
            name="lighthouse",
            configured=True,
            status="error",
            message=f"Lighthouse health check failed: {exc}",
        )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()


async def check_gcs_health(
    *,
    bucket: Optional[str],
    agent_id: str,
    prefix: str = "kestrel/",
    project: Optional[str] = None,
    credentials_path: Optional[str] = None,
    target_factory: Optional[Callable[..., Any]] = None,
) -> TargetHealth:
    """Check whether GCS fallback is configured and has a latest snapshot."""
    if not bucket:
        return TargetHealth(
            name="gcs",
            configured=False,
            status="not_configured",
            message="GCS_BACKUP_BUCKET is not set",
        )

    if target_factory is None:
        from kestrel_sovereign.storage.sync.gcs_target import GCSTarget

        target_factory = GCSTarget

    target = target_factory(
        bucket=bucket,
        prefix=prefix,
        agent_id=agent_id,
        project=project,
        credentials_path=credentials_path,
    )
    try:
        healthy = await target.health_check()
        latest_exists = False
        if healthy:
            latest_blob = f"{target.prefix}{agent_id}/latest.db"

            def _exists() -> bool:
                return bool(target._get_bucket().blob(latest_blob).exists())

            latest_exists = await asyncio.to_thread(_exists)

        details = {
            "bucket": bucket,
            "prefix": target.prefix,
            "agent_id": agent_id,
            "latest_blob": f"{target.prefix}{agent_id}/latest.db",
            "latest_exists": latest_exists,
        }
        if healthy and latest_exists:
            return TargetHealth(
                name="gcs",
                configured=True,
                status="ok",
                message="GCS fallback is configured and latest snapshot exists",
                details=details,
            )
        if healthy:
            return TargetHealth(
                name="gcs",
                configured=True,
                status="warning",
                message="GCS bucket is reachable but latest snapshot is missing",
                details=details,
            )
        return TargetHealth(
            name="gcs",
            configured=True,
            status="unavailable",
            message="GCS bucket is not reachable",
            details=details,
        )
    except Exception as exc:
        return TargetHealth(
            name="gcs",
            configured=True,
            status="error",
            message=f"GCS health check failed: {exc}",
            details={"bucket": bucket, "agent_id": agent_id},
        )


async def build_storage_health_report(
    *,
    agent_id: str,
    env: Mapping[str, str] = os.environ,
    lighthouse_grace: timedelta = DEFAULT_LIGHTHOUSE_DEAL_GRACE,
    gcs_prefix: str = "kestrel/",
    now: Optional[datetime] = None,
) -> StorageHealthReport:
    """Build a combined Lighthouse/GCS storage health report from environment."""
    lighthouse, gcs = await asyncio.gather(
        check_lighthouse_health(
            api_key=env.get("LIGHTHOUSE_API_KEY"),
            agent_id=agent_id,
            grace_period=lighthouse_grace,
            now=now,
        ),
        check_gcs_health(
            bucket=env.get("GCS_BACKUP_BUCKET"),
            agent_id=agent_id,
            prefix=gcs_prefix,
            project=env.get("GCP_PROJECT"),
            credentials_path=env.get("GOOGLE_APPLICATION_CREDENTIALS"),
        ),
    )
    statuses = {lighthouse.status, gcs.status}
    if "error" in statuses or "warning" in statuses:
        status = "warning"
    elif "ok" in statuses or "pending" in statuses:
        status = "ok"
    else:
        status = "unavailable"
    return StorageHealthReport(status=status, lighthouse=lighthouse, gcs=gcs)


def load_env_file(path: Path, env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Load simple KEY=VALUE lines from an env file without overwriting env."""
    values = dict(os.environ if env is None else env)
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return values
