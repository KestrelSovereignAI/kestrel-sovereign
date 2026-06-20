"""
Sovereign IPFS Sync Target

Self-hosted IPFS (Kubo) sync target -- the sovereign default.
Stores SQLite snapshots on infrastructure we own and operate.
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from kestrel_sovereign.storage.sync.manifest_manager import ManifestManagerMixin
from kestrel_sovereign.storage.sync.retention import RetentionPolicy
from kestrel_sovereign.storage.sync.targets import (
    SyncTarget,
    SyncResult,
    TrustTier,
    _create_consistent_snapshot,
)

logger = logging.getLogger(__name__)


class SovereignIPFSTarget(ManifestManagerMixin, SyncTarget):
    """
    Self-hosted IPFS (Kubo) sync target.

    The sovereign default. Stores SQLite snapshots on infrastructure
    we own and operate. Requires only a reachable Kubo API endpoint --
    no third-party credentials.

    Trust tier: SOVEREIGN (highest)
    """

    def __init__(
        self,
        api_url: str,
        agent_id: str,
        state_dir: Optional[Path] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.agent_id = agent_id
        self._state_dir = state_dir
        self._manifest_filename = f".sovereign_ipfs_manifest_{self.agent_id}.json"
        self._client = None

    @property
    def name(self) -> str:
        return f"ipfs://{self.agent_id}"

    @property
    def trust_tier(self) -> TrustTier:
        return TrustTier.SOVEREIGN

    async def _get_client(self):
        """Get or create httpx async client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """Upload consistent snapshot via Kubo /api/v0/add?pin=true."""
        timestamp = datetime.now(timezone.utc)

        try:
            content = _create_consistent_snapshot(db_path)
            content_hash = hashlib.sha256(content).hexdigest()

            # Content-hash dedup
            manifest = self._load_local_manifest()
            if manifest and manifest.get("content_hash") == content_hash:
                logger.debug(f"Sovereign IPFS snapshot unchanged (hash={content_hash[:12]}), skipping")
                return SyncResult(
                    success=True,
                    target_name=self.name,
                    bytes_synced=0,
                    frames_synced=0,
                    timestamp=timestamp,
                    metadata={"skipped": True, "cid": manifest.get("cid")},
                )

            client = await self._get_client()

            # Kubo API: POST /api/v0/add?pin=true
            filename = f"{self.agent_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.db"
            response = await client.post(
                f"{self.api_url}/api/v0/add",
                params={"pin": "true", "quieter": "true"},
                files={"file": (filename, content, "application/x-sqlite3")},
            )
            response.raise_for_status()
            result_data = response.json()
            cid = result_data["Hash"]

            logger.info(
                f"Uploaded snapshot to sovereign IPFS: {cid} ({len(content)} bytes)"
            )

            new_manifest = {
                "agent_id": self.agent_id,
                "cid": cid,
                "content_hash": content_hash,
                "size": len(content),
                "uploaded_at": timestamp.isoformat(),
            }
            self._save_local_manifest(new_manifest)

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(content),
                frames_synced=0,
                timestamp=timestamp,
                metadata={"cid": cid},
            )

        except Exception as e:
            logger.error(f"Failed to sync snapshot to sovereign IPFS: {e}")
            return SyncResult(
                success=False,
                target_name=self.name,
                bytes_synced=0,
                frames_synced=0,
                timestamp=timestamp,
                error=str(e),
            )

    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        """No-op. Sovereign IPFS uses full snapshots only."""
        return SyncResult(
            success=True,
            target_name=self.name,
            bytes_synced=0,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
        )

    async def get_latest_position(self) -> Optional[int]:
        """Signal that WAL sync is not needed."""
        return 2**63

    async def prune(self, policy: RetentionPolicy) -> Dict[str, Any]:
        """No-op: decommissioned IPFS tier cannot enumerate safe upload records."""
        logger.info(
            "Sovereign IPFS retention prune skipped for %s: enumeration/delete "
            "is not supported by this sync target",
            self.agent_id,
        )
        return {"deleted": 0, "skipped": True, "reason": "enumeration_not_supported"}

    async def restore_snapshot(self, dest_path: Path) -> Optional[SyncResult]:
        """Download latest snapshot from sovereign IPFS via /api/v0/cat."""
        timestamp = datetime.now(timezone.utc)

        # Resolve CID: env var -> local manifest
        cid = os.environ.get("SOVEREIGN_IPFS_CID")
        if not cid:
            manifest = self._load_local_manifest()
            if manifest:
                cid = manifest.get("cid")

        if not cid:
            logger.debug("No sovereign IPFS CID available for restore")
            return None

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_url}/api/v0/cat",
                params={"arg": cid},
                timeout=120.0,
            )
            response.raise_for_status()
            content = response.content

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(content)

            logger.info(
                f"Restored snapshot from sovereign IPFS: {cid} ({len(content)} bytes)"
            )

            return SyncResult(
                success=True,
                target_name=self.name,
                bytes_synced=len(content),
                frames_synced=0,
                timestamp=timestamp,
                metadata={"cid": cid},
            )

        except Exception as e:
            logger.warning(f"Sovereign IPFS restore failed: {e}")
            return None

    async def health_check(self) -> bool:
        """Check Kubo connectivity via /api/v0/id."""
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_url}/api/v0/id",
                timeout=10.0,
            )
            response.raise_for_status()
            peer_info = response.json()
            logger.debug(f"Sovereign IPFS node: {peer_info.get('ID', 'unknown')}")
            return True
        except Exception as e:
            logger.warning(f"Sovereign IPFS health check failed: {e}")
            return False
