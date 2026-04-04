"""
Storage Provider Protocol.

Re-exports from kestrel_sdk.storage.providers.base for backward compatibility.
Feature packages should import from kestrel_sdk.storage.providers.base directly.
"""

# Re-export everything from kestrel_sdk
from kestrel_sdk.storage.providers.base import (  # noqa: F401
    StorageTier,
    SyncStatus,
    StorageResult,
    SyncItem,
    SyncManifest,
    StorageProvider,
    CryostasisCapable,
    MultiCurrencyPayment,
    _utc_now,
)
