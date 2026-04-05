"""
Storage Provider Protocol

Defines the unified interface for all storage providers in the Kestrel
multi-tier sovereign storage architecture.

Tiers:
- BROWSER: IndexedDB with encrypted cloud backup
- LOCAL: Self-hosted IPFS via Docker (Kubo)
- CLOUD_HOT: Lighthouse IPFS pinning (fast access)
- CLOUD_COLD: Lighthouse Filecoin deals (permanent archive)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class StorageTier(Enum):
    """Storage tiers for the multi-tier architecture."""

    # Tier 1: Browser/Phone
    BROWSER = "browser"  # IndexedDB, user's device only

    # Tier 2: Local Docker (self-hosted)
    LOCAL = "local"  # Local IPFS node, SQLite, user's hardware

    # Tier 3: Cloud Hosted (Lighthouse)
    CLOUD_HOT = "cloud_hot"  # Lighthouse IPFS pinning (fast retrieval)
    CLOUD_COLD = "cloud_cold"  # Lighthouse Filecoin (permanent archive)

    # Legacy (backward compatibility with FilecoinAdapter)
    LOCAL_ONLY = "local_only"
    IPFS = "ipfs"
    FILECOIN = "filecoin"
    ENCRYPTED_FILECOIN = "encrypted_filecoin"


class SyncStatus(Enum):
    """Status of sync operations between tiers."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


@dataclass
class StorageResult:
    """Result of a storage operation.

    Note: Includes backward-compatible aliases for filecoin_adapter.py fields:
    - storage_tier -> tier
    - ipfs_cid -> cid
    - filecoin_deal_id -> deal_id
    """

    # Content identification
    content_hash: str  # SHA256 of original content
    cid: Optional[str] = None  # IPFS/IPLD Content ID

    # Storage location
    tier: StorageTier = StorageTier.LOCAL
    provider: str = "local"  # 'local', 'lighthouse'

    # Filecoin deal info (for CLOUD_COLD)
    deal_id: Optional[str] = None
    deal_status: Optional[str] = None

    # Encryption
    encrypted: bool = False
    encryption_key_hash: Optional[str] = None

    # Metadata
    size_bytes: int = 0
    content_type: Optional[str] = None
    filename: Optional[str] = None

    # Cost tracking
    storage_cost_usd: Optional[Decimal] = None

    # Timestamps
    created_at: datetime = field(default_factory=_utc_now)
    last_verified_at: Optional[datetime] = None

    # Backward compatibility aliases (legacy filecoin_adapter.py field names)
    @property
    def storage_tier(self) -> StorageTier:
        """Alias for tier (backward compatibility)."""
        return self.tier

    @property
    def ipfs_cid(self) -> Optional[str]:
        """Alias for cid (backward compatibility)."""
        return self.cid

    @property
    def filecoin_deal_id(self) -> Optional[str]:
        """Alias for deal_id (backward compatibility)."""
        return self.deal_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content_hash": self.content_hash,
            "cid": self.cid,
            "tier": self.tier.value,
            "provider": self.provider,
            "deal_id": self.deal_id,
            "deal_status": self.deal_status,
            "encrypted": self.encrypted,
            "encryption_key_hash": self.encryption_key_hash,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "filename": self.filename,
            "storage_cost_usd": str(self.storage_cost_usd) if self.storage_cost_usd else None,
            "created_at": self.created_at.isoformat(),
            "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageResult":
        """Create from dictionary."""
        return cls(
            content_hash=data["content_hash"],
            cid=data.get("cid"),
            tier=StorageTier(data.get("tier", "local")),
            provider=data.get("provider", "local"),
            deal_id=data.get("deal_id"),
            deal_status=data.get("deal_status"),
            encrypted=data.get("encrypted", False),
            encryption_key_hash=data.get("encryption_key_hash"),
            size_bytes=data.get("size_bytes", 0),
            content_type=data.get("content_type"),
            filename=data.get("filename"),
            storage_cost_usd=Decimal(data["storage_cost_usd"]) if data.get("storage_cost_usd") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else _utc_now(),
            last_verified_at=datetime.fromisoformat(data["last_verified_at"]) if data.get("last_verified_at") else None,
        )


@dataclass
class SyncItem:
    """Item to be synced between tiers."""
    content_hash: str
    source_tier: StorageTier
    target_tier: StorageTier
    size_bytes: int
    status: SyncStatus = SyncStatus.PENDING
    error_message: Optional[str] = None


@dataclass
class SyncManifest:
    """Manifest describing a sync operation between tiers."""
    source_tier: StorageTier
    target_tier: StorageTier
    items: List[SyncItem] = field(default_factory=list)
    total_bytes: int = 0
    estimated_cost_usd: Optional[Decimal] = None
    created_at: datetime = field(default_factory=_utc_now)


class StorageProvider(ABC):
    """
    Abstract base class for storage providers.

    All storage providers (local IPFS, Lighthouse, browser) must implement
    this interface to work with the TieredStorageManager.
    """

    @property
    @abstractmethod
    def tier(self) -> StorageTier:
        """Return the storage tier this provider handles."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'lighthouse', 'local')."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is currently available."""
        ...

    @abstractmethod
    async def store(
        self,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        encrypt: bool = True,
    ) -> StorageResult:
        """
        Store content in this provider.

        Args:
            content: Raw content bytes to store
            metadata: Optional metadata (filename, content_type, etc.)
            encrypt: Whether to encrypt before storing (default True)

        Returns:
            StorageResult with storage details
        """
        ...

    @abstractmethod
    async def retrieve(self, cid: str, encryption_key_hash: Optional[str] = None) -> bytes:
        """
        Retrieve content by CID.

        Args:
            cid: Content identifier (IPFS CID or content hash)
            encryption_key_hash: Hash of encryption key if content is encrypted

        Returns:
            Original content bytes
        """
        ...

    @abstractmethod
    async def list_content(self, limit: int = 100, offset: int = 0) -> List[StorageResult]:
        """
        List stored content.

        Args:
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of StorageResult objects
        """
        ...

    @abstractmethod
    async def delete(self, cid: str) -> bool:
        """
        Delete content by CID.

        Args:
            cid: Content identifier to delete

        Returns:
            True if deleted successfully
        """
        ...

    @abstractmethod
    async def verify(self, cid: str) -> bool:
        """
        Verify that content is still accessible.

        Args:
            cid: Content identifier to verify

        Returns:
            True if content is verified accessible
        """
        ...

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get provider statistics.

        Returns:
            Dictionary with provider-specific stats
        """
        return {
            "tier": self.tier.value,
            "provider": self.provider_name,
            "available": self.is_available(),
        }

    async def estimate_cost(self, size_bytes: int) -> Decimal:
        """
        Estimate storage cost for given size.

        Args:
            size_bytes: Size of content in bytes

        Returns:
            Estimated cost in USD
        """
        return Decimal("0")  # Override in paid providers


class CryostasisCapable(ABC):
    """
    Mixin for providers that support agent cryostasis (dormancy).

    Cryostasis allows agents to enter a low-cost dormant state when
    their wallet balance is low, preserving state on Filecoin.
    """

    @abstractmethod
    async def archive_for_cryostasis(
        self,
        agent_id: str,
        state_snapshot: bytes,
        metadata: Dict[str, Any],
    ) -> StorageResult:
        """
        Archive agent state for cryostasis (permanent Filecoin storage).

        Args:
            agent_id: Unique agent identifier
            state_snapshot: Serialized agent state
            metadata: Agent metadata (identity, last active, etc.)

        Returns:
            StorageResult with Filecoin deal info
        """
        ...

    @abstractmethod
    async def restore_from_cryostasis(self, cid: str, encryption_key_hash: str) -> bytes:
        """
        Restore agent state from cryostasis.

        Args:
            cid: Content identifier of archived state
            encryption_key_hash: Encryption key hash for decryption

        Returns:
            Decrypted agent state bytes
        """
        ...

    @abstractmethod
    async def calculate_cryostasis_cost(self, size_bytes: int) -> Decimal:
        """
        Calculate the one-time cost for cryostasis storage.

        Args:
            size_bytes: Size of agent state in bytes

        Returns:
            One-time cost in USD for permanent Filecoin storage
        """
        ...


class MultiCurrencyPayment(ABC):
    """
    Mixin for providers that support on-chain payments.

    Allows agents to pay for storage directly using FIL, USDC, or USDT.
    """

    SUPPORTED_CURRENCIES = ["FIL", "USDC", "USDT"]

    @abstractmethod
    async def pay_for_storage(
        self,
        amount_usd: Decimal,
        currency: str,
        wallet_address: str,
    ) -> Dict[str, Any]:
        """
        Pay for storage using cryptocurrency.

        Args:
            amount_usd: Amount to pay in USD equivalent
            currency: Currency to pay with (FIL, USDC, USDT)
            wallet_address: Agent's wallet address

        Returns:
            Transaction details (tx_hash, status, etc.)
        """
        ...

    @abstractmethod
    async def get_balance(self, wallet_address: str, currency: str) -> Decimal:
        """
        Get wallet balance for a specific currency.

        Args:
            wallet_address: Wallet address to check
            currency: Currency to check (FIL, USDC, USDT)

        Returns:
            Balance in the specified currency
        """
        ...
