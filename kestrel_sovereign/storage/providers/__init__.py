"""
Storage Providers Package

Provides a unified interface for multi-tier sovereign storage:
- Tier 1 (Browser):    IndexedDB + cloud backup
- Tier 2 (Local):      Self-hosted IPFS via Docker (Kubo)
- Tier 3 (Cloud Hot):  Storacha (web3.storage, preferred) or Lighthouse IPFS pinning
- Tier 4 (Cloud Cold): Lighthouse Filecoin perpetual storage

Provider selection order (CLOUD_HOT):
  1. Storacha   — when STORACHA_SPACE_DID + STORACHA_AGENT_KEY + STORACHA_PROOF are set
  2. Filebase   — when FILEBASE_API_KEY + FILEBASE_API_KEY_SECRET are set
  3. Lighthouse — when LIGHTHOUSE_API_KEY is set

Provider implementations are NOT re-exported here to avoid importing optional
dependencies at package import time.  Import directly when needed:

    from kestrel_sovereign.storage.providers.storacha_provider import StorachaProvider
    from kestrel_sovereign.storage.providers.filebase_provider import FilebaseProvider
    from kestrel_sovereign.storage.providers.lighthouse_provider import LighthouseProvider
"""

from kestrel_sovereign.storage.providers.base import (
    CryostasisCapable,
    StorageProvider,
    StorageResult,
    StorageTier,
    SyncStatus,
)

__all__ = [
    "CryostasisCapable",
    "StorageProvider",
    "StorageResult",
    "StorageTier",
    "SyncStatus",
]
