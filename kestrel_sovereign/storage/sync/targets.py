"""
Sync Targets

Abstractions for sync destinations. Each target declares a TrustTier
reflecting how much sovereignty the agent retains over its data:

    SOVEREIGN   Sovereign-operated target; currently decommissioned by default
    DELEGATED   Delegated-decentralized service with API key (Lighthouse)
    EXPEDIENT   Expedient-cloud target (GCS, S3)

Write to all configured targets. Restore from most trusted.
"""

import logging
import os
import sqlite3
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

from kestrel_sovereign.storage.sync.retention import RetentionPolicy

logger = logging.getLogger(__name__)


class TrustTier(Enum):
    """Trust hierarchy for persistence targets.

    Lower value = higher trust. Restore walks tiers in this order.
    """
    SOVEREIGN = 1   # Sovereign-operated — self-hosted IPFS/Kubo
    FEDERATED = 2   # Agent-controlled auth (DID/UCAN); reserved
    DELEGATED = 3   # Delegated-decentralized — Lighthouse
    EXPEDIENT = 4   # Expedient-cloud — GCS, S3


class TierState(Enum):
    """Operational state for a backup tier."""

    ACTIVE = "active"
    DISABLED = "disabled"
    DECOMMISSIONED = "decommissioned"


TIER_LABELS = {
    TrustTier.SOVEREIGN: "sovereign-operated",
    TrustTier.FEDERATED: "federated",
    TrustTier.DELEGATED: "delegated-decentralized",
    TrustTier.EXPEDIENT: "expedient-cloud",
}


def _create_consistent_snapshot(db_path: Path) -> bytes:
    """Create a consistent snapshot using sqlite3.backup().

    Safe to call while the database is in use with an active WAL.
    Falls back to raw file read if backup fails (e.g. not a SQLite DB).
    """
    tmp_path = None
    try:
        src = sqlite3.connect(str(db_path))
        try:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_path = tmp.name
            dst = sqlite3.connect(tmp_path)
            try:
                src.backup(dst)
            finally:
                dst.close()
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            src.close()
    except sqlite3.DatabaseError:
        logger.debug(f"sqlite3.backup() failed for {db_path}, using raw read")
        with open(db_path, "rb") as f:
            return f.read()
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    target_name: str
    bytes_synced: int
    frames_synced: int
    timestamp: datetime
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SyncTarget(ABC):
    """Abstract base class for sync targets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Target name for logging and identification."""
        ...

    @property
    def trust_tier(self) -> TrustTier:
        """Trust level of this target. Override in subclasses."""
        return TrustTier.EXPEDIENT

    @property
    def tier_state(self) -> TierState:
        """Operational state of this target."""
        return TierState.ACTIVE

    @property
    def tier_label(self) -> str:
        """Honest operator-facing label for this target's trust tier."""
        return TIER_LABELS.get(self.trust_tier, self.trust_tier.name.lower())

    @abstractmethod
    async def sync_snapshot(self, db_path: Path) -> SyncResult:
        """
        Sync a full database snapshot.

        Args:
            db_path: Path to SQLite database file

        Returns:
            SyncResult with status
        """
        ...

    @abstractmethod
    async def sync_wal(self, wal_path: Path, position: int) -> SyncResult:
        """
        Sync WAL changes from a specific position.

        Args:
            wal_path: Path to WAL file
            position: Byte offset to start from

        Returns:
            SyncResult with status
        """
        ...

    @abstractmethod
    async def get_latest_position(self) -> Optional[int]:
        """
        Get the latest synced WAL position.

        Returns:
            Byte offset of last sync, or None if no previous sync
        """
        ...

    async def health_check(self) -> bool:
        """Check if target is reachable and healthy."""
        return True

    async def prune(self, policy: RetentionPolicy) -> Dict[str, Any]:
        """Apply backup retention after a successful snapshot.

        Targets that cannot enumerate their remote objects should override only
        when they can safely delete. The default no-op keeps older targets
        compatible and makes pruning best-effort.
        """
        return {"deleted": 0, "skipped": True, "reason": "not_supported"}


# ---------------------------------------------------------------------------
# Re-export provider classes for backward compatibility.
#
# All imports below are lazy — the individual provider modules import
# from *this* file (SyncTarget, SyncResult, TrustTier, _create_consistent_snapshot)
# so there is no circular dependency.
# ---------------------------------------------------------------------------

from kestrel_sovereign.storage.sync.s3_target import S3Target  # noqa: E402, F401
from kestrel_sovereign.storage.sync.gcs_target import GCSTarget  # noqa: E402, F401
from kestrel_sovereign.storage.sync.lighthouse_target import LighthouseTarget  # noqa: E402, F401
from kestrel_sovereign.storage.sync.sovereign_ipfs_target import SovereignIPFSTarget  # noqa: E402, F401
from kestrel_sovereign.storage.sync.retention import (  # noqa: E402, F401
    DataClass,
    RetentionItem,
    classify,
    load_retention_policy,
)

__all__ = [
    "TrustTier",
    "TierState",
    "TIER_LABELS",
    "SyncResult",
    "SyncTarget",
    "_create_consistent_snapshot",
    "DataClass",
    "RetentionItem",
    "RetentionPolicy",
    "classify",
    "load_retention_policy",
    "S3Target",
    "GCSTarget",
    "LighthouseTarget",
    "SovereignIPFSTarget",
]
