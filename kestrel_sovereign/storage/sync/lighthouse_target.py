"""
Lighthouse Sync Target

Lighthouse (Filecoin) decentralized storage target with CID tracking
and cold-start restore support.
"""

import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from kestrel_sovereign.execution_custody import (
    close_native_execution_stream,
    is_execution_control_error,
)
from kestrel_sovereign.storage.car_builder import (
    CARReader,
    single_raw_block_car_prefix,
)
from kestrel_sovereign.storage.sync.manifest_manager import ManifestManagerMixin
from kestrel_sovereign.storage.sync.quota import (
    DEFAULT_QUOTA_THRESHOLD,
    StorageQuota,
    StorageQuotaExceeded,
    format_bytes,
    validate_quota_threshold,
)
from kestrel_sovereign.storage.sync.retention import (
    RetentionItem,
    RetentionPolicy,
    classify,
    load_retention_policy,
    parse_timestamp,
)
from kestrel_sovereign.storage.sync.targets import (
    SyncResult,
    SyncTarget,
    TrustTier,
    _offloaded_snapshot_file,
)

if TYPE_CHECKING:
    from kestrel_sovereign.storage.providers.lighthouse_rest import UploadBody

logger = logging.getLogger(__name__)


def _snapshot_size(manifest: Optional[Dict[str, Any]]) -> Optional[int]:
    """The positive ``snapshot_size`` a manifest records, else ``None``."""
    if not isinstance(manifest, dict):
        return None
    try:
        size = int(manifest.get("snapshot_size") or 0)
    except (TypeError, ValueError):
        return None
    return size or None


def _byte_count(value: Any) -> Optional[int]:
    """A non-negative byte count from a usage field (number or numeric
    string), else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        value = int(value)
    if not isinstance(value, int) or value < 0:
        return None
    return value


class LighthouseTarget(ManifestManagerMixin, SyncTarget):
    """
    Lighthouse (Filecoin) storage target.

    Stores database snapshots and WAL on decentralized storage
    for maximum data sovereignty. Supports cold-start restore for
    ephemeral environments (Cloud Run scale-to-zero).

    CID tracking: After each snapshot upload, a manifest file is uploaded
    containing the snapshot CID. The manifest CID is stored locally and
    can also be found via the Lighthouse uploads API.

    Quota: before each upload the target reads the account's
    ``user_data_usage``. When the upload would take ``dataUsed`` past
    ``quota_threshold`` of ``dataLimit`` it logs a WARNING and prunes under
    ``retention_policy`` first; an upload that then fails for lack of room
    reports ``dataUsed`` against ``dataLimit``. The last reading is
    ``storage_quota`` (#3894).
    """

    # Tag prefix for filtering uploads via Lighthouse API
    SNAPSHOT_TAG = "kestrel-state"
    MANIFEST_TAG = "kestrel-manifest"
    SNAPSHOT_FORMAT_CAR_V1 = "car-v1/raw-sqlite"
    SNAPSHOT_NAME_PREFIX = "kestrel_state"
    MANIFEST_NAME_PREFIX = "kestrel_manifest"

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        state_dir: Optional[Path] = None,
        *,
        retention_policy: Optional[RetentionPolicy] = None,
        quota_threshold: float = DEFAULT_QUOTA_THRESHOLD,
    ):
        """
        Initialize Lighthouse target.

        Args:
            api_key: Lighthouse API key
            agent_id: Unique agent identifier for namespacing
            state_dir: Directory to store local state (latest CID tracking).
                       Defaults to parent of the DB path.
            retention_policy: The policy a prune before an upload applies:
                the sync service's, which also prunes after each success.
                Loaded from the host config when not given.
            quota_threshold: The fraction of ``dataLimit`` an upload may
                take ``dataUsed`` to before retention prunes first.
        """
        self.api_key = api_key
        self.agent_id = agent_id
        self._state_dir = state_dir
        self._manifest_filename = f".lighthouse_manifest_{self.agent_id}.json"
        self._latest_cid: Optional[str] = None
        self._retention_policy = retention_policy
        self._quota_threshold = validate_quota_threshold(quota_threshold)
        self.storage_quota: Optional[StorageQuota] = None

    kind = "lighthouse"

    @property
    def name(self) -> str:
        return f"lighthouse://{self.agent_id}"

    @property
    def trust_tier(self) -> TrustTier:
        return TrustTier.DELEGATED

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """Upload snapshot to Lighthouse and update manifest.

        Uses sqlite3.backup() for a consistent snapshot and skips upload
        if the content hash matches the previous upload. The snapshot stays
        on disk and its CAR is streamed from the file in bounded chunks:
        neither is ever held in memory whole (#3662). The backup and the
        hash run off the event loop (#3916).
        """
        timestamp = datetime.now(timezone.utc)

        try:
            from kestrel_sovereign.storage.providers.lighthouse_rest import (
                LighthouseRestClient,
            )

            # Use sqlite3.backup() for consistent snapshot (safe with active WAL)
            async with _offloaded_snapshot_file(db_path) as snapshot:
                snapshot_path = snapshot.path
                digest = snapshot.sha256
                raw_size = snapshot.size

                # Skip upload if content unchanged
                content_hash = digest.hex()
                manifest = self._load_local_manifest()
                if manifest and manifest.get("content_hash") == content_hash:
                    logger.debug(f"Snapshot unchanged (hash={content_hash[:12]}), skipping upload")
                    return SyncResult(
                        success=True,
                        target_name=self.name,
                        bytes_synced=0,
                        frames_synced=0,
                        timestamp=timestamp,
                        metadata={"cid": manifest.get("snapshot_cid"), "skipped": True},
                    )

                client = LighthouseRestClient(api_key=self.api_key)
                try:
                    tag = f"{self.SNAPSHOT_TAG}-{self.agent_id}"
                    timestamp_token = self._timestamp_token(timestamp)
                    snapshot_filename = self._structured_snapshot_filename(timestamp_token)
                    body, payload_cid = self._snapshot_car_body(digest, snapshot_path, raw_size)
                    with body:
                        pruned_first = await self._make_room(client, body.length)
                        try:
                            result = await client.upload_car(
                                body=body,
                                tag=tag,
                                filename=snapshot_filename,
                            )
                        except Exception as upload_error:
                            if is_execution_control_error(upload_error):
                                raise
                            raise await self._named_upload_failure(
                                client, upload_error, body.length
                            )

                    cid = result.get("Hash") or result.get("cid")
                    size = int(result.get("Size", body.length))

                    if not cid:
                        raise ValueError(f"No CID returned from Lighthouse upload: {result}")

                    logger.info(f"Uploaded snapshot to Lighthouse: {cid} ({size} bytes)")
                    self._latest_cid = cid
                    self._latest_size = size

                    # Upload manifest pointing to this snapshot
                    new_manifest = {
                        "agent_id": self.agent_id,
                        "snapshot_cid": cid,
                        "snapshot_size": size,
                        "snapshot_format": self.SNAPSHOT_FORMAT_CAR_V1,
                        "snapshot_payload_cid": payload_cid,
                        "raw_snapshot_size": raw_size,
                        "uploaded_at": timestamp.isoformat(),
                        "upload_name": snapshot_filename,
                        "source_file": db_path.name,
                        "content_hash": content_hash,
                    }
                    await self._upload_manifest(client, new_manifest)
                finally:
                    await close_native_execution_stream(client, retain_originals=True)

            metadata: Dict[str, Any] = {"cid": cid, "format": self.SNAPSHOT_FORMAT_CAR_V1}
            if pruned_first is not None:
                metadata["prune_before_upload"] = pruned_first
            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=size,
                frames_synced=0,
                timestamp=timestamp,
                metadata=metadata,
            )

        except Exception as e:
            if is_execution_control_error(e):
                raise
            # ``str(e)`` is empty for an httpx timeout, and 880 empty
            # "Failed to sync" lines hid an 18-day outage (#3189): the type
            # is the diagnosis, so it travels with the message.
            logger.error(
                "Failed to sync to Lighthouse: %s: %s", type(e).__name__, e
            )
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=f"{type(e).__name__}: {e}",
            )

    def _retention(self) -> RetentionPolicy:
        if self._retention_policy is None:
            self._retention_policy = load_retention_policy()
        return self._retention_policy

    async def _read_quota(self, client: Any) -> Optional[StorageQuota]:
        """The account's ``user_data_usage`` as a quota, recorded as
        ``storage_quota``; ``None`` when it cannot be read or names no
        ``dataUsed`` and positive ``dataLimit``."""
        try:
            response = await client.get_balance()
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.warning(
                "Could not read Lighthouse storage usage for %s: %s: %s",
                self.name,
                type(e).__name__,
                e,
            )
            return None
        data = response.get("data") if isinstance(response, Mapping) else None
        used = _byte_count(data.get("dataUsed")) if isinstance(data, Mapping) else None
        limit = _byte_count(data.get("dataLimit")) if isinstance(data, Mapping) else None
        if used is None or not limit:
            logger.warning(
                "Lighthouse user_data_usage for %s named no dataUsed and positive "
                "dataLimit; its quota is unknown",
                self.name,
            )
            return None
        self.storage_quota = StorageQuota(
            used_bytes=used,
            limit_bytes=limit,
            observed_at=datetime.now(timezone.utc),
            threshold=self._quota_threshold,
        )
        return self.storage_quota

    async def _make_room(self, client: Any, upload_bytes: int) -> Optional[Dict[str, Any]]:
        """Before an upload: when ``upload_bytes`` would take the account past
        the quota threshold, warn and prune under retention first.

        Returns the prune's result when one ran. Never fails the upload: an
        unreadable usage or a failed prune is logged, and the upload goes
        ahead.
        """
        quota = await self._read_quota(client)
        if quota is None or not quota.would_cross_threshold(upload_bytes):
            return None
        logger.warning(
            "Lighthouse account near quota before uploading %s's snapshot: dataUsed "
            "%s of dataLimit %s (%.1f%%); a %s upload would cross %.0f%% of "
            "dataLimit, so retention prunes first",
            self.agent_id,
            format_bytes(quota.used_bytes),
            format_bytes(quota.limit_bytes),
            quota.used_fraction * 100,
            format_bytes(upload_bytes),
            quota.threshold * 100,
        )
        try:
            pruned = await self._prune_uploads(client, self._retention())
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.warning(
                "Retention prune before upload failed for %s: %s: %s",
                self.name,
                type(e).__name__,
                e,
            )
            return {"error": f"{type(e).__name__}: {e}"}
        if pruned["deleted"]:
            quota = await self._read_quota(client) or quota
        if quota.would_cross_threshold(upload_bytes):
            logger.warning(
                "Lighthouse account still near quota after pruning %d of %s's "
                "upload(s): dataUsed %s of dataLimit %s (%.1f%%); uploading %s anyway",
                pruned["deleted"],
                self.agent_id,
                format_bytes(quota.used_bytes),
                format_bytes(quota.limit_bytes),
                quota.used_fraction * 100,
                format_bytes(upload_bytes),
            )
        else:
            logger.info(
                "Pruned %d of %s's Lighthouse upload(s) before uploading: dataUsed "
                "now %s of dataLimit %s",
                pruned["deleted"],
                self.agent_id,
                format_bytes(quota.used_bytes),
                format_bytes(quota.limit_bytes),
            )
        return pruned

    async def _named_upload_failure(
        self, client: Any, error: Exception, upload_bytes: int
    ) -> Exception:
        """``error``, or, when the account has no room for the upload, a
        ``StorageQuotaExceeded`` naming dataUsed against dataLimit and
        chained from it. The usage is read again: the precheck's reading may
        predate other agents' uploads to the same account."""
        quota = await self._read_quota(client)
        if quota is None or quota.fits(upload_bytes):
            return error
        failure = StorageQuotaExceeded(
            quota,
            upload_bytes,
            account="Lighthouse",
            used_label="dataUsed",
            limit_label="dataLimit",
            cause=error,
        )
        failure.__cause__ = error
        return failure

    async def _upload_manifest(self, client: Any, manifest: Dict[str, Any]) -> None:
        """Upload manifest file to Lighthouse and save locally."""
        try:
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            tag = f"{self.MANIFEST_TAG}-{self.agent_id}"
            manifest_time = parse_timestamp(manifest.get("uploaded_at")) or datetime.now(
                timezone.utc
            )
            timestamp_token = self._timestamp_token(manifest_time)
            manifest_filename = self._structured_manifest_filename(timestamp_token)

            result = await client.upload(
                content=manifest_bytes,
                filename=manifest_filename,
                tag=tag,
            )

            manifest_cid = result.get("Hash") or result.get("cid")
            manifest["manifest_cid"] = manifest_cid
            manifest["manifest_upload_name"] = manifest_filename
            self._save_local_manifest(manifest)
            logger.debug(f"Uploaded manifest to Lighthouse: {manifest_cid}")

        except Exception as e:
            # Manifest upload failure is non-fatal -- snapshot is already safe
            logger.warning(
                "Failed to upload manifest (snapshot is safe): %s: %s", type(e).__name__, e
            )
            self._save_local_manifest(manifest)

    async def restore_snapshot(self, dest_path: Path) -> Optional[SyncResult]:
        """
        Download the latest snapshot from Lighthouse and write to dest_path.

        Resolution order for finding the latest snapshot CID:
        1. LIGHTHOUSE_STATE_CID env var (explicit pointer)
        2. Local manifest file (from previous sync on this host)
        3. Lighthouse uploads API (search by tag)

        Args:
            dest_path: Where to write the restored database file

        Returns:
            SyncResult if restore succeeded, None if no snapshot found
        """
        timestamp = datetime.now(timezone.utc)

        try:
            snapshot_cid, expected_bytes = await self._resolve_latest_snapshot()
            if not snapshot_cid:
                logger.info("No Lighthouse snapshot found to restore")
                return None

            logger.info(f"Restoring snapshot from Lighthouse: {snapshot_cid}")

            from kestrel_sovereign.storage.providers.lighthouse_rest import (
                LighthouseRestClient,
            )

            client = LighthouseRestClient(api_key=self.api_key)
            # Whichever path named the snapshot also named its size when it
            # could (the local manifest, the uploads API record, a remote
            # manifest); a known size buys the gateway the patience a 1.2 GB
            # CAR needs before its first byte (#3189).
            content = await client.download(snapshot_cid, expected_bytes=expected_bytes)
            await client.close()

            if not content:
                logger.warning("Empty snapshot downloaded from Lighthouse")
                return None

            content = self._extract_snapshot_content(content)

            # Write to destination
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(content)

            logger.info(
                f"Restored {len(content)} bytes from Lighthouse to {dest_path}"
            )

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(content),
                frames_synced=0,
                timestamp=timestamp,
                metadata={"cid": snapshot_cid, "action": "restore"},
            )

        except Exception as e:
            # The same empty-message timeout as the upload door (#3189).
            logger.error(
                "Failed to restore from Lighthouse: %s: %s", type(e).__name__, e
            )
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=f"{type(e).__name__}: {e}",
            )

    def _snapshot_car_body(
        self, sha256_digest: bytes, snapshot_path: Path, raw_size: int
    ) -> tuple["UploadBody", str]:
        """The snapshot file as a single-block CAR, streamed from disk.

        The archive is a few hundred bytes of header and block prefix
        followed by the file itself, so the body reads the file in bounded
        chunks as it uploads instead of building the archive in memory.
        """
        from kestrel_sovereign.storage.providers.lighthouse_rest import UploadBody

        prefix, payload_cid = single_raw_block_car_prefix(sha256_digest, raw_size)
        return UploadBody([prefix, snapshot_path]), payload_cid

    def _build_snapshot_car(self, content: bytes) -> tuple[bytes, str]:
        """Pack a SQLite snapshot held in memory as a single-block CAR archive:
        the same bytes ``_snapshot_car_body`` streams for a file holding it."""
        prefix, payload_cid = single_raw_block_car_prefix(
            hashlib.sha256(content).digest(), len(content)
        )
        return prefix + content, payload_cid

    def _filename_agent_id(self) -> str:
        return self.agent_id.replace("/", "_")

    def _timestamp_token(self, timestamp: datetime) -> str:
        return timestamp.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _structured_snapshot_filename(self, timestamp_token: str) -> str:
        return (
            f"{self.SNAPSHOT_NAME_PREFIX}__{self._filename_agent_id()}__"
            f"{timestamp_token}.car"
        )

    def _structured_manifest_filename(self, timestamp_token: str) -> str:
        return (
            f"{self.MANIFEST_NAME_PREFIX}__{self._filename_agent_id()}__"
            f"{timestamp_token}.json"
        )

    def _upload_filename(self, upload: Dict[str, Any]) -> str:
        return str(upload.get("fileName") or upload.get("Name") or "")

    def _upload_cid(self, upload: Dict[str, Any]) -> Optional[str]:
        for key in ("cid", "Hash", "hash", "fileHash", "CID"):
            value = upload.get(key)
            if value:
                return str(value)
        return None

    def _upload_timestamp(self, upload: Dict[str, Any]) -> datetime:
        filename = self._upload_filename(upload)
        return (
            parse_timestamp(
                upload.get("createdAt")
                or upload.get("created_at")
                or upload.get("uploadedAt")
                or upload.get("timestamp")
            )
            or parse_timestamp(filename)
            or datetime.min.replace(tzinfo=timezone.utc)
        )

    def _is_structured_snapshot_upload(self, upload: Dict[str, Any]) -> bool:
        return self._upload_filename(upload).startswith(
            f"{self.SNAPSHOT_NAME_PREFIX}__{self._filename_agent_id()}__"
        ) and self._upload_filename(upload).endswith(".car")

    def _is_structured_manifest_upload(self, upload: Dict[str, Any]) -> bool:
        return self._upload_filename(upload).startswith(
            f"{self.MANIFEST_NAME_PREFIX}__{self._filename_agent_id()}__"
        ) and self._upload_filename(upload).endswith(".json")

    def _is_legacy_agent_manifest_upload(self, upload: Dict[str, Any]) -> bool:
        return self._upload_filename(upload) == f"manifest_{self.agent_id}.json"

    def _extract_snapshot_content(self, content: bytes) -> bytes:
        """Return raw SQLite bytes from CAR snapshots, preserving legacy raw files."""
        try:
            reader = CARReader(content)
            if not reader.verify():
                raise ValueError("CAR verification failed")

            root_block = reader.get_block(reader.root_cid)
            if root_block is None:
                raise ValueError(f"CAR root block missing: {reader.root_cid}")
            return root_block
        except Exception as e:
            logger.debug(
                "Downloaded snapshot is not a CAR archive; treating as raw SQLite: %s",
                e,
            )
            return content

    async def _resolve_latest_snapshot(self) -> tuple[Optional[str], Optional[int]]:
        """The latest snapshot's CID and, when whichever path named it also
        recorded a size (the local manifest, this process's own upload, an
        uploads API record, a remote manifest), that size."""
        cid = await self._resolve_latest_cid()
        return cid, (self._known_snapshot_size(cid) if cid else None)

    def _known_snapshot_size(self, cid: str) -> Optional[int]:
        manifest = self._load_local_manifest()
        if manifest and manifest.get("snapshot_cid") == cid:
            return _snapshot_size(manifest)
        if getattr(self, "_latest_cid", None) == cid:
            return getattr(self, "_latest_size", None)
        remote = getattr(self, "_last_remote_manifest", None)
        if isinstance(remote, dict) and remote.get("snapshot_cid") == cid:
            return _snapshot_size(remote)
        return getattr(self, "_upload_sizes_by_cid", {}).get(cid)

    async def _resolve_latest_cid(self) -> Optional[str]:
        """
        Find the latest snapshot CID using multiple resolution strategies.

        Returns:
            CID string or None
        """
        # 1. Explicit env var
        explicit_cid = os.environ.get("LIGHTHOUSE_STATE_CID")
        if explicit_cid:
            logger.debug(f"Using LIGHTHOUSE_STATE_CID env var: {explicit_cid}")
            return explicit_cid

        # 2. Local manifest
        manifest = self._load_local_manifest()
        if manifest and manifest.get("snapshot_cid"):
            logger.debug(f"Using local manifest CID: {manifest['snapshot_cid']}")
            return manifest["snapshot_cid"]

        # 3. In-memory cache from last sync
        if self._latest_cid:
            return self._latest_cid

        # 4. Lighthouse uploads API
        return await self._query_uploads_api()

    async def _query_uploads_api(self) -> Optional[str]:
        """Query Lighthouse uploads API to find the latest snapshot.

        Records each snapshot upload's size by CID so a restore can size its
        download budget (#3189).
        """
        try:
            from kestrel_sovereign.storage.providers.lighthouse_rest import (
                LighthouseRestClient,
            )

            client = LighthouseRestClient(api_key=self.api_key)
            try:
                result = await client.get_uploads()
            finally:
                await client.close()

            uploads = result.get("fileList", [])
            if not isinstance(uploads, list):
                return None

            agent_snapshots = [
                u for u in uploads if self._is_structured_snapshot_upload(u)
            ]
            self._upload_sizes_by_cid = {
                cid: size
                for u in agent_snapshots
                if (cid := self._upload_cid(u)) and (size := self._upload_size(u))
            }
            if agent_snapshots:
                latest = max(agent_snapshots, key=self._upload_timestamp)
                cid = self._upload_cid(latest)
                logger.info(f"Found latest snapshot via uploads API: {cid}")
                return cid

            manifests = [
                u
                for u in uploads
                if self._is_structured_manifest_upload(u)
                or self._is_legacy_agent_manifest_upload(u)
            ]
            if not manifests:
                return None

            latest_manifest = max(manifests, key=self._upload_timestamp)
            manifest_cid = self._upload_cid(latest_manifest)
            if manifest_cid:
                return await self._read_manifest_cid(manifest_cid)
            return None

        except Exception as e:
            logger.warning("Failed to query Lighthouse uploads API: %s: %s", type(e).__name__, e)
            return None

    @staticmethod
    def _upload_size(upload: Dict[str, Any]) -> Optional[int]:
        for key in ("fileSizeInBytes", "Size", "size"):
            value = upload.get(key)
            try:
                if value is not None and int(value) > 0:
                    return int(value)
            except (TypeError, ValueError):
                continue
        return None

    async def _read_manifest_cid(self, manifest_cid: str) -> Optional[str]:
        """Download a manifest file and extract the snapshot CID."""
        try:
            from kestrel_sovereign.storage.providers.lighthouse_rest import (
                LighthouseRestClient,
            )

            client = LighthouseRestClient(api_key=self.api_key)
            content = await client.download(manifest_cid)
            await client.close()
            manifest = json.loads(content)
            self._last_remote_manifest = manifest if isinstance(manifest, dict) else None
            manifest_agent = manifest.get("agent_id")
            if manifest_agent and manifest_agent != self.agent_id:
                logger.warning(
                    "Ignoring Lighthouse manifest %s for different agent_id: %s",
                    manifest_cid,
                    manifest_agent,
                )
                return None
            return manifest.get("snapshot_cid")
        except Exception as e:
            logger.warning(f"Failed to read manifest {manifest_cid}: {e}")
            return None

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        """No-op for Lighthouse. WAL sync is not appropriate for content-addressed storage.

        Lighthouse uses full consistent snapshots via sync_snapshot(). Uploading
        the raw WAL file is wasteful (new CID on every write, not recoverable
        without the matching DB, and burns storage quota for no benefit).
        """
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=0,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_latest_position(self) -> Optional[int]:
        """Return max int to signal no WAL sync needed.

        Returning None caused SyncService to always sync from position 0.
        Returning a large value ensures the wal_position > last_pos check
        in SyncService._sync_pending() is never true.
        """
        return 2**63

    async def prune(self, policy: RetentionPolicy) -> Dict[str, Any]:
        """Prune Lighthouse upload records when the API can enumerate/delete."""
        from kestrel_sovereign.storage.providers.lighthouse_rest import (
            LighthouseRestClient,
        )

        client = LighthouseRestClient(api_key=self.api_key)
        try:
            return await self._prune_uploads(client, policy)
        finally:
            await close_native_execution_stream(client, retain_originals=True)

    async def _prune_uploads(self, client: Any, policy: RetentionPolicy) -> Dict[str, Any]:
        """Delete this agent's upload records that ``policy`` does not keep."""
        from kestrel_sovereign.storage.providers.lighthouse_rest import (
            list_all_uploads,
        )

        uploads = await list_all_uploads(client)

        items: list[RetentionItem] = []
        for upload in uploads:
            # Only prune uploads attributable to THIS agent by name:
            # structured snapshots/manifests, and the legacy agent-scoped
            # manifest_<did>.json. Legacy flat / `*-wal` uploads carry no
            # agent id, so on a shared Lighthouse key (multi-tenant host)
            # pruning them here could delete another agent's data. That
            # unattributed legacy backlog is handled by the cleanup tool
            # (#1841) via manifest attribution under dry-run + confirm.
            #
            # Sovereignty exports are not this target's: the agent's
            # Lighthouse provider publishes and thins them, on the account
            # its PayerPolicy resolves, which need not be this target's
            # (#2874).
            if self._is_structured_snapshot_upload(upload):
                role = "snapshot"
            elif self._is_structured_manifest_upload(
                upload
            ) or self._is_legacy_agent_manifest_upload(upload):
                role = "identity"
            else:
                continue
            cid = self._upload_cid(upload)
            if not cid:
                continue
            ts = self._upload_timestamp(upload)
            if ts is None:
                logger.debug(
                    "Lighthouse retention skipped upload without timestamp: %s",
                    cid,
                )
                continue
            item_data = {
                "role": role,
                "tag": upload.get("tag"),
                "fileName": self._upload_filename(upload),
                "cid": cid,
            }
            items.append(
                RetentionItem(
                    key=str(cid),
                    name=str(item_data.get("fileName") or cid),
                    timestamp=ts,
                    data_class=classify(item_data),
                    metadata=upload,
                )
            )

        deletions = policy.deletions(items)
        deleted = []
        skipped_missing_id = 0
        for item in deletions:
            file_id = item.metadata.get("id")
            if not file_id:
                skipped_missing_id += 1
                logger.warning(
                    "Skipping Lighthouse prune delete for %s without file id",
                    item.key,
                )
                continue
            await client.delete_file(str(file_id))
            deleted.append(item.key)
        return {
            "deleted": len(deleted),
            "cids": deleted,
            "scanned": len(items),
            "skipped_missing_id": skipped_missing_id,
        }

    async def health_check(self) -> bool:
        """Check Lighthouse API connectivity."""
        try:
            from kestrel_sovereign.storage.providers.lighthouse_rest import (
                LighthouseRestClient,
            )

            client = LighthouseRestClient(api_key=self.api_key)
            await client.get_balance()
            await client.close()
            return True
        except Exception as e:
            logger.warning(f"Lighthouse health check failed: {e}")
            return False
