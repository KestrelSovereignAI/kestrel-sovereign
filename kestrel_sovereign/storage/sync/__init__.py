"""
SQLite-First Sync Layer

Event-driven snapshot service for sovereign persistence.
Snapshots are triggered by lifecycle events (shutdown, scheduled backup)
and fan out to all configured targets.

Architecture:
    SQLite (primary) ---> SyncService ---> Targets (ordered by trust)

    SOVEREIGN   Self-hosted IPFS (Kubo)
    DELEGATED   Lighthouse (API key)
    EXPEDIENT   GCS / S3

Write to all. Restore from most trusted.
"""

from kestrel_sovereign.storage.sync.service import SyncService
from kestrel_sovereign.storage.sync.targets import (
    SyncTarget,
    SyncResult,
    TrustTier,
    DataClass,
    RetentionItem,
    RetentionPolicy,
    classify,
    load_retention_policy,
    GCSTarget,
    LighthouseTarget,
    SovereignIPFSTarget,
    S3Target,
)

__all__ = [
    "SyncService",
    "SyncTarget",
    "SyncResult",
    "TrustTier",
    "DataClass",
    "RetentionItem",
    "RetentionPolicy",
    "classify",
    "load_retention_policy",
    "GCSTarget",
    "LighthouseTarget",
    "SovereignIPFSTarget",
    "S3Target",
]
