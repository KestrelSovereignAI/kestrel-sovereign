"""
Storage Providers Package

Provides a unified interface for multi-tier sovereign storage:
- Tier 1 (Browser):    IndexedDB + cloud backup
- Tier 2 (Local):      Self-hosted IPFS via Docker (Kubo)
- Tier 3 (Cloud Hot):  Filebase (S3-compatible IPFS) or Lighthouse IPFS pinning
- Tier 4 (Cloud Cold): Lighthouse Filecoin perpetual storage

Provider selection order (CLOUD_HOT):
  1. Filebase   — when FILEBASE_API_KEY + FILEBASE_API_KEY_SECRET are set
  2. Lighthouse — when LIGHTHOUSE_API_KEY is set

Provider implementations are NOT re-exported here to avoid importing optional
dependencies at package import time.  Import directly when needed:

    from kestrel_sovereign.storage.providers.filebase_provider import FilebaseProvider
    from kestrel_sovereign.storage.providers.lighthouse_provider import LighthouseProvider

External packages can register storage providers via entry_points::

    [project.entry-points."kestrel_sovereign.storage_providers"]
    MyStorageProvider = "my_storage_package:MyStorageProvider"
"""

import logging
from typing import Dict, Type

from kestrel_sovereign.storage.providers.base import (
    CryostasisCapable,
    StorageProvider,
    StorageResult,
    StorageTier,
    SyncStatus,
)

logger = logging.getLogger(__name__)

STORAGE_PROVIDER_ENTRY_POINT_GROUP = "kestrel_sovereign.storage_providers"


def discover_storage_providers() -> Dict[str, Type[StorageProvider]]:
    """Discover storage provider classes from entry_points.

    Returns:
        Dict mapping entry point name to StorageProvider subclass.
    """
    from kestrel_sovereign.entrypoints import discover_entry_point_classes
    return discover_entry_point_classes(STORAGE_PROVIDER_ENTRY_POINT_GROUP, StorageProvider)


__all__ = [
    "CryostasisCapable",
    "StorageProvider",
    "StorageResult",
    "StorageTier",
    "SyncStatus",
    "STORAGE_PROVIDER_ENTRY_POINT_GROUP",
    "discover_storage_providers",
]
