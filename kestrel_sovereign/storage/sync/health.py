"""Storage health checks for operator-facing backup status."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional

from kestrel_sovereign.storage.providers.lighthouse_rest import (
    deal_records,
    list_all_uploads,
)
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
    deal_count = len(deal_records(deals))
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


async def _snapshot_health(
    client: Any,
    uploads: list[Mapping[str, Any]],
    *,
    agent_id: str,
    current_time: datetime,
    grace_period: timedelta,
) -> tuple[
    Optional[Mapping[str, Any]], Optional[Mapping[str, Any]], Optional[dict[str, Any]]
]:
    """The agent's latest state snapshot on ``client``'s account: its upload
    record, the manifest it was resolved through (if any), and its deals.
    All three are ``None`` without one."""
    manifest = None
    latest = _latest_lighthouse_snapshot(uploads, agent_id)
    if latest is None:
        latest_manifest = _latest_lighthouse_manifest(uploads, agent_id)
        manifest_cid = _upload_cid(latest_manifest) if latest_manifest else None
        if manifest_cid:
            manifest_bytes = await client.download(manifest_cid)
            manifest = json.loads(manifest_bytes)
            latest = latest_manifest
    if latest is None:
        return None, None, None
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
    return latest, manifest, snapshot


async def _sovereignty_export_health(
    client: Any,
    uploads: list[Mapping[str, Any]],
    *,
    agent_id: str,
    current_time: datetime,
    grace_period: timedelta,
) -> Optional[dict[str, Any]]:
    """The agent's latest sovereignty export on ``client``'s account and its
    deals, or ``None`` without one."""
    export = _latest_sovereignty_export(uploads, agent_id)
    if export is None:
        return None
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
    return export_details


async def check_lighthouse_health(
    *,
    snapshot_api_key: Optional[str],
    export_api_key: Optional[str],
    agent_id: str,
    export_key_error: Optional[str] = None,
    grace_period: timedelta = DEFAULT_LIGHTHOUSE_DEAL_GRACE,
    now: Optional[datetime] = None,
    client_factory: Optional[Callable[..., Any]] = None,
) -> TargetHealth:
    """Check the deal status of the agent's latest Lighthouse snapshot and
    sovereignty export, each on the account it is uploaded to.

    State snapshots are read with ``snapshot_api_key``, the host's
    ``LIGHTHOUSE_API_KEY``: the key the snapshot target uploads with and
    cold-start restore reads. Sovereignty exports are read with
    ``export_api_key``, the agent's key as its PayerPolicy resolves it,
    which need not name the same account (#2874). An account both keys name
    is listed once. Either key may be ``None``, and its uploads are then not
    looked for.

    A sovereignty export is reported under ``details["sovereignty_export"]``
    and attributed to ``agent_id`` by its agent-scoped upload record. With no
    snapshot it is the upload the check reports; with both, the status is the
    worse of the two. ``export_key_error`` is why the export key could not be
    resolved: the status is then ``error``, whatever the snapshot shows.

    Each account is read on its own. A failure reading one, listing its
    uploads, a manifest or a deal status, is that account's error, under
    ``details["snapshot_error"]`` or ``details["sovereignty_export_error"]``,
    and the status is ``error``; what the other account shows is still
    reported beside it (#3602).
    """
    if not snapshot_api_key and not export_api_key and export_key_error is None:
        return TargetHealth(
            name="lighthouse",
            label=TIER_LABELS[TrustTier.DELEGATED],
            configured=False,
            status="not_configured",
            message="No Lighthouse credential is configured for this agent",
        )

    if client_factory is None:
        from kestrel_sovereign.storage.providers.lighthouse_rest import (
            LighthouseRestClient,
        )

        client_factory = LighthouseRestClient

    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    clients: dict[str, Any] = {}
    listings: dict[str, list[Mapping[str, Any]]] = {}
    listing_errors: dict[str, Exception] = {}

    async def account(api_key: str) -> tuple[Any, list[Mapping[str, Any]]]:
        """The client for ``api_key``'s account and every upload in it.

        An account is listed once, whether the listing succeeds or fails.
        """
        if api_key not in clients:
            clients[api_key] = client_factory(api_key=api_key)
        if api_key in listing_errors:
            raise listing_errors[api_key]
        if api_key not in listings:
            try:
                # Every page: the latest upload need not be on the first (#2874).
                listings[api_key] = await list_all_uploads(clients[api_key])
            except Exception as exc:
                listing_errors[api_key] = exc
                raise
        return clients[api_key], listings[api_key]

    latest = manifest = snapshot = None
    export_details = None
    # Each account's failure is its own, never the other account's: an export
    # account that answers is reported whatever the snapshot account does,
    # and the reverse (#3602).
    snapshot_error: Optional[str] = None
    export_errors = [export_key_error] if export_key_error is not None else []
    try:
        if snapshot_api_key:
            try:
                client, uploads = await account(snapshot_api_key)
                latest, manifest, snapshot = await _snapshot_health(
                    client,
                    uploads,
                    agent_id=agent_id,
                    current_time=current_time,
                    grace_period=grace_period,
                )
            except Exception as exc:
                snapshot_error = f"Lighthouse snapshot check failed: {exc}"

        if export_api_key:
            try:
                client, uploads = await account(export_api_key)
                export_details = await _sovereignty_export_health(
                    client,
                    uploads,
                    agent_id=agent_id,
                    current_time=current_time,
                    grace_period=grace_period,
                )
            except Exception as exc:
                export_errors.append(
                    f"Lighthouse sovereignty export check failed: {exc}"
                )
    finally:
        for client in clients.values():
            close = getattr(client, "close", None)
            if close is not None:
                await close()

    grace_seconds = int(grace_period.total_seconds())
    if snapshot is not None:
        details = {
            "cid": snapshot["cid"],
            "file_name": latest.get("fileName"),
            "tag": latest.get("tag"),
            "resolved_via": "manifest" if manifest else "snapshot_upload",
            "uploaded_at": snapshot["uploaded_at"],
            "age_seconds": snapshot["age_seconds"],
            "deal_count": snapshot["deal_count"],
            "deals": snapshot["deals"],
            "grace_seconds": grace_seconds,
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
    elif export_details is not None:
        # No state snapshot: the sovereignty export is this agent's restore
        # point on Lighthouse, so it is what the check reports.
        details = {
            "cid": export_details["cid"],
            "file_name": export_details["file_name"],
            "tag": export_details["tag"],
            "resolved_via": "sovereignty_export",
            "uploaded_at": export_details["uploaded_at"],
            "age_seconds": export_details["age_seconds"],
            "deal_count": export_details["deal_count"],
            "deals": export_details["deals"],
            "grace_seconds": grace_seconds,
            "sovereignty_export": export_details,
        }
        status = export_details["status"]
        message = "Latest Lighthouse sovereignty export " + _deal_phrase(export_details)
    else:
        details = {}
        status = "unavailable"
        message = "No Lighthouse snapshot or sovereignty export upload found"

    export_error = "; ".join(export_errors) or None
    details["snapshot_error"] = snapshot_error
    details["sovereignty_export_error"] = export_error
    errors = [error for error in (snapshot_error, export_error) if error is not None]
    if errors:
        # An account that cannot be read is an error, whatever the other
        # shows; what the other shows is still reported beside it.
        status = "error"
        if snapshot is not None or export_details is not None:
            errors.append(message)
        message = "; ".join(errors)
    return TargetHealth(
        name="lighthouse",
        label=TIER_LABELS[TrustTier.DELEGATED],
        configured=True,
        status=status,
        message=message,
        details=details,
    )


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
    export_api_key: Optional[str],
    export_key_error: Optional[str] = None,
    env: Mapping[str, str] = os.environ,
    lighthouse_grace: timedelta = DEFAULT_LIGHTHOUSE_DEAL_GRACE,
    gcs_prefix: str = "kestrel/",
    now: Optional[datetime] = None,
) -> StorageHealthReport:
    """Build a combined backup storage health report for one agent.

    The Lighthouse tier reads each of the agent's uploads on the account it
    went to (``check_lighthouse_health``). Its state snapshots are read with
    ``env``'s ``LIGHTHOUSE_API_KEY``, the host key the snapshot target uploads
    with and cold-start restore reads. Its sovereignty exports are read with
    ``export_api_key``, the agent's Lighthouse credential as its PayerPolicy
    resolves it (``resolve_lighthouse_storage`` and ``lighthouse_api_key_for``
    in ``lighthouse_provider``), the key they are published with (#2874).
    ``export_key_error`` is why that credential could not be resolved, which
    the Lighthouse tier reports as its error while the other tiers are still
    checked.
    """
    sovereign_ipfs, lighthouse, gcs = await asyncio.gather(
        check_sovereign_ipfs_health(
            api_url=env.get("SOVEREIGN_IPFS_URL"),
        ),
        check_lighthouse_health(
            snapshot_api_key=env.get("LIGHTHOUSE_API_KEY"),
            export_api_key=export_api_key,
            export_key_error=export_key_error,
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
