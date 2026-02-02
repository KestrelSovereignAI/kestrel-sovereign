"""
Storage Providers Package

Provides a unified interface for multi-tier sovereign storage:
- Tier 1 (Browser): IndexedDB + cloud backup
- Tier 2 (Local): Self-hosted IPFS via Docker
- Tier 3 (Cloud): Lighthouse IPFS + Filecoin
"""

from kestrel_sovereign.storage.providers.base import (
    StorageProvider,
    StorageResult,
    StorageTier,
    SyncStatus,
)

__all__ = [
    "StorageProvider",
    "StorageResult",
    "StorageTier",
    "SyncStatus",
]
