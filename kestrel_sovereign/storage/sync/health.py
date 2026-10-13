"""Storage health checks for operator-facing backup status."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from kestrel_sovereign.storage.sync.retention import parse_timestamp
from kestrel_sovereign.storage.sync.sovereignty_exports import sovereignty_export_uploads
from kestrel_sovereign.storage.sync.targets import TierState, TrustTier, TIER_LABELS


DEFAULT_LIGHTHOUSE_DEAL_GRACE = timedelta(hours=24)


@dataclass
class TargetHealth:
    """Health result for one storage target."""

    name: str
    label: str
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
    sovereign_ipfs: TargetHealth
    lighthouse: TargetHealth
    gcs: TargetHealth

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sovereign_ipfs": self.sovereign_ipfs.to_dict(),
            "lighthouse": self.lighthouse.to_dict(),
            "gcs": self.gcs.to_dict(),
        }


async def check_sovereign_ipfs_health(
    *,
    api_url: Optional[str],
    client_factory: Optional[Callable[..., Any]] = None,
) -> TargetHealth:
    """Check whether the sovereign-operated IPFS backup tier is active."""
    label = TIER_LABELS[TrustTier.SOVEREIGN]
    if not api_url:
        return TargetHealth(
            name="sovereign_ipfs",
            label=label,
            configured=False,
            status=TierState.DECOMMISSIONED.value,
            message=(
                "No sovereign-operated IPFS node is configured; this tier is "
                "decommissioned"
            ),
        )

    if client_factory is None:
        import httpx

        client_factory = httpx.AsyncClient

    normalized_url = api_url.rstrip("/")
    client = client_factory(timeout=10.0)
    try:
        response = await client.post(f"{normalized_url}/api/v0/id")
        response.raise_for_status()
        peer_info = response.json()
        return TargetHealth(
            name="sovereign_ipfs",
            label=label,
            configured=True,
            status=TierState.ACTIVE.value,
            message="Sovereign-operated IPFS node is reachable",
            details={
                "api_url": normalized_url,
                "peer_id": peer_info.get("ID"),
                "agent_version": peer_info.get("AgentVersion"),
            },
        )
    except Exception as exc:
        return TargetHealth(
            name="sovereign_ipfs",
            label=label,
            configured=True,
            status=TierState.DECOMMISSIONED.value,
            message=(
                "Configured sovereign-operated IPFS node is unreachable; "
                "this tier is decommissioned"
            ),
            details={"api_url": normalized_url, "error": str(exc)},
        )
    finally:
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


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


def _lighthouse_filename_agent_id(agent_id: str) -> str:
    return agent_id.replace("/", "_")


def _upload_filename(upload: Mapping[str, Any]) -> str:
    return str(upload.get("fileName") or upload.get("Name") or "")


def _upload_order_key(upload: Mapping[str, Any]) -> datetime:
    return (
        parse_timestamp(upload.get("createdAt"))
        or parse_timestamp(_upload_filename(upload))
        or datetime.min.replace(tzinfo=timezone.utc)
    )


def _upload_time(upload: Mapping[str, Any]) -> Optional[datetime]:
    return _parse_upload_time(upload.get("createdAt")) or parse_timestamp(
        _upload_filename(upload)
    )


def _latest_lighthouse_snapshot(
    uploads: list[Mapping[str, Any]],
    agent_id: str,
) -> Optional[Mapping[str, Any]]:
    tag = f"kestrel-state-{agent_id}"
    structured_prefix = (
        f"kestrel_state__{_lighthouse_filename_agent_id(agent_id)}__"
    )
    candidates = [
        upload
        for upload in uploads
        if upload.get("tag") == tag
        or (
            _upload_filename(upload).startswith(structured_prefix)
            and _upload_filename(upload).endswith(".car")
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=_upload_order_key)


def _latest_sovereignty_export(
    uploads: list[Mapping[str, Any]],
    agent_id: str,
) -> Optional[Mapping[str, Any]]:
    """This agent's latest sovereignty export, attributed by its record (#2874)."""
    candidates = sovereignty_export_uploads(uploads, agent_id)
    if not candidates:
        return None
    return max(candidates, key=_upload_order_key)


_DEAL_STATUS_RANK = {"ok": 0, "pending": 1, "warning": 2}


async def _deal_details(
    client: Any,
    *,
    cid: Optional[str],
    uploaded_at: Optional[datetime],
    current_time: datetime,
    grace_period: timedelta,
) -> dict[str, Any]:
    """An upload's Filecoin deals, its age, and the status they amount to."""
    age = current_time - uploaded_at if uploaded_at else None
    deals = await client.get_deal_status(cid) if cid else []
    deal_count = len(deals) if isinstance(deals, list) else 0
    if deal_count > 0:
        status = "ok"
    elif age is not None and age <= grace_period:
        status = "pending"
    else:
        status = "warning"
    return {
        "cid": cid,
        "uploaded_at": uploaded_at.isoformat() if uploaded_at else None,
        "age_seconds": int(age.total_seconds()) if age else None,
        "deal_count": deal_count,
        "deals": deals,
        "status": status,
    }


def _deal_phrase(details: Mapping[str, Any]) -> str:
    if details["status"] == "ok":
        return f"has {details['deal_count']} Filecoin deal(s)"
    if details["status"] == "pending":
        return "has no deals yet but is within grace"
    return "has no Filecoin deals beyond grace"


def _latest_lighthouse_manifest(
    uploads: list[Mapping[str, Any]],
    agent_id: str,
) -> Optional[Mapping[str, Any]]:
    tag = f"kestrel-manifest-{agent_id}"
    filename = f"manifest_{agent_id}.json"
    structured_prefix = (
        f"kestrel_manifest__{_lighthouse_filename_agent_id(agent_id)}__"
    )
    candidates = [
        upload
        for upload in uploads
        if upload.get("tag") == tag
        or _upload_filename(upload) == filename
        or (
            _upload_filename(upload).startswith(structured_prefix)
            and _upload_filename(upload).endswith(".json")
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=_upload_order_key)


async def check_lighthouse_health(
    *,
    api_key: Optional[str],
    agent_id: str,
    grace_period: timedelta = DEFAULT_LIGHTHOUSE_DEAL_GRACE,
    now: Optional[datetime] = None,
    client_factory: Optional[Callable[..., Any]] = None,
) -> TargetHealth:
    """Check the deal status of the agent's latest Lighthouse snapshot and
    sovereignty export.

    A sovereignty export is reported under ``details["sovereignty_export"]``
    and attributed to ``agent_id`` by its agent-scoped upload record (#2874).
    With no snapshot it is the upload the check reports; with both, the
    status is the worse of the two.
    """
    if not api_key:
        return TargetHealth(
            name="lighthouse",
            label=TIER_LABELS[TrustTier.DELEGATED],
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
        export = _latest_sovereignty_export(uploads, agent_id)

        if latest is None and export is None:
            return TargetHealth(
                name="lighthouse",
                label=TIER_LABELS[TrustTier.DELEGATED],
                configured=True,
                status="unavailable",
                message="No Lighthouse snapshot or sovereignty export upload found",
            )

        export_details = None
        if export is not None:
            export_details = await _deal_details(
                client,
                cid=_upload_cid(export),
                uploaded_at=_upload_time(export),
                current_time=current_time,
                grace_period=grace_period,
            )
            export_details.update(
                {
                    # The one agent the export's record names (#2874).
                    "agent_id": agent_id,
                    "file_name": export.get("fileName"),
                    "tag": export.get("tag"),
                }
            )

        if latest is None:
            # No state snapshot: the sovereignty export is this agent's
            # restore point on Lighthouse, so it is what the check reports.
            details = {
                "cid": export_details["cid"],
                "file_name": export_details["file_name"],
                "tag": export_details["tag"],
                "resolved_via": "sovereignty_export",
                "uploaded_at": export_details["uploaded_at"],
                "age_seconds": export_details["age_seconds"],
                "deal_count": export_details["deal_count"],
                "deals": export_details["deals"],
                "grace_seconds": int(grace_period.total_seconds()),
                "sovereignty_export": export_details,
            }
            return TargetHealth(
                name="lighthouse",
                label=TIER_LABELS[TrustTier.DELEGATED],
                configured=True,
                status=export_details["status"],
                message=(
                    "Latest Lighthouse sovereignty export "
                    + _deal_phrase(export_details)
                ),
                details=details,
            )

        cid = str(manifest.get("snapshot_cid")) if manifest else _upload_cid(latest)
        uploaded_at = (
            _parse_upload_time(manifest.get("uploaded_at")) if manifest else None
        ) or _parse_upload_time(latest.get("createdAt"))
        snapshot = await _deal_details(
            client,
            cid=cid,
            uploaded_at=uploaded_at,
            current_time=current_time,
            grace_period=grace_period,
        )

        details = {
            "cid": cid,
            "file_name": latest.get("fileName"),
            "tag": latest.get("tag"),
            "resolved_via": "manifest" if manifest else "snapshot_upload",
            "uploaded_at": snapshot["uploaded_at"],
            "age_seconds": snapshot["age_seconds"],
            "deal_count": snapshot["deal_count"],
            "deals": snapshot["deals"],
            "grace_seconds": int(grace_period.total_seconds()),
            "sovereignty_export": export_details,
        }
        status = snapshot["status"]
        message = "Latest Lighthouse snapshot " + _deal_phrase(snapshot)
        if export_details is not None:
            # The snapshot does not stand in for the export: the check
            # reports the worse of the two.
            if _DEAL_STATUS_RANK[export_details["status"]] > _DEAL_STATUS_RANK[status]:
                status = export_details["status"]
            message += "; latest sovereignty export " + _deal_phrase(export_details)
        return TargetHealth(
            name="lighthouse",
            label=TIER_LABELS[TrustTier.DELEGATED],
            configured=True,
            status=status,
            message=message,
            details=details,
        )
    except Exception as exc:
        return TargetHealth(
            name="lighthouse",
            label=TIER_LABELS[TrustTier.DELEGATED],
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
            label=TIER_LABELS[TrustTier.EXPEDIENT],
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
                label=TIER_LABELS[TrustTier.EXPEDIENT],
                configured=True,
                status="ok",
                message="GCS fallback is configured and latest snapshot exists",
                details=details,
            )
        if healthy:
            return TargetHealth(
                name="gcs",
                label=TIER_LABELS[TrustTier.EXPEDIENT],
                configured=True,
                status="warning",
                message="GCS bucket is reachable but latest snapshot is missing",
                details=details,
            )
        return TargetHealth(
            name="gcs",
            label=TIER_LABELS[TrustTier.EXPEDIENT],
            configured=True,
            status="unavailable",
            message="GCS bucket is not reachable",
            details=details,
        )
    except Exception as exc:
        return TargetHealth(
            name="gcs",
            label=TIER_LABELS[TrustTier.EXPEDIENT],
            configured=True,
            status="error",
            message=f"GCS health check failed: {exc}",
            details={"bucket": bucket, "agent_id": agent_id},
        )


def _aggregate_overall_status(statuses: set[str]) -> str:
    """Reduce per-tier statuses to an overall backup-storage status.

    "active" (a reachable sovereign-operated node) counts as healthy even when
    it is the only configured tier; "decommissioned"/"not_configured" alone do
    not make storage healthy.
    """
    if "error" in statuses or "warning" in statuses:
        return "warning"
    if statuses & {"ok", "pending", "active"}:
        return "ok"
    return "unavailable"


async def build_storage_health_report(
    *,
    agent_id: str,
    env: Mapping[str, str] = os.environ,
    lighthouse_grace: timedelta = DEFAULT_LIGHTHOUSE_DEAL_GRACE,
    gcs_prefix: str = "kestrel/",
    now: Optional[datetime] = None,
) -> StorageHealthReport:
    """Build a combined backup storage health report from environment.

    The Lighthouse check reads the host ``LIGHTHOUSE_API_KEY`` account. An
    agent publishes its sovereignty exports with its own PayerPolicy
    credential, which may name another account; the agent checks those
    through its provider (``SovereigntyFeature.check_sovereignty_status``,
    #2874).
    """
    sovereign_ipfs, lighthouse, gcs = await asyncio.gather(
        check_sovereign_ipfs_health(
            api_url=env.get("SOVEREIGN_IPFS_URL"),
        ),
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
    status = _aggregate_overall_status(
        {sovereign_ipfs.status, lighthouse.status, gcs.status}
    )
    return StorageHealthReport(
        status=status,
        sovereign_ipfs=sovereign_ipfs,
        lighthouse=lighthouse,
        gcs=gcs,
    )


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
