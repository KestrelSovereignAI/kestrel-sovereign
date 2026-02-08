"""
Lighthouse Storage Provider

Integrates with Lighthouse.storage for IPFS pinning and Filecoin permanent storage.
Supports direct on-chain payments (FIL, USDC, USDT) for true agent sovereignty.

Lighthouse provides:
- IPFS pinning (hot access, dedicated gateways)
- Filecoin deals via endowment pool (perpetual archive, ~$2-5/GB one-time)
- Pay-per-use (x402 protocol) OR lifetime plans
- Kavach threshold encryption (BLS key sharding, NFT-gated access)
- No Lotus node required

Pricing (as of Feb 2026):
- Free tier: 5 GB
- Lifetime plans: $20/5GB, $100/25GB, $500/150GB (~$4/GB perpetual)
- Raw Filecoin deal cost is ~$0.00005/GB but Lighthouse charges $2-5/GB
  to fund the endowment pool that auto-renews deals in perpetuity.

Note on Python SDK:
- lighthouseweb3 v0.1.1 (May 2023) only supports upload(). Unmaintained.
- For encryption/token-gating/deal status, use REST API directly.
- Migration to Kavach threshold encryption is recommended over local Fernet
  keys to eliminate the single-point-of-failure in cryostasis key recovery.

References:
- Docs: https://docs.lighthouse.storage/
- Python SDK: https://pypi.org/project/lighthouseweb3/
- Kavach encryption: https://github.com/lighthouse-web3/encryption-sdk
- x402 protocol: https://github.com/coinbase/x402
- Endowment pool: https://www.lighthouse.storage/blogs/Discover%20How%20the%20Endowment%20Pool%20Makes%20Your%20Data%20Immortal
"""

import hashlib
import io
import logging
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.storage.providers.base import (
    CryostasisCapable,
    MultiCurrencyPayment,
    StorageProvider,
    StorageResult,
    StorageTier,
)
from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_MEDIUM
from kestrel_sovereign.kestrel_config.defaults import get_lighthouse_gateway_url

logger = logging.getLogger(__name__)

# Try to import lighthouseweb3, gracefully degrade if not installed
try:
    from lighthouseweb3 import Lighthouse
    LIGHTHOUSE_AVAILABLE = True
except ImportError:
    Lighthouse = None
    LIGHTHOUSE_AVAILABLE = False
    logger.warning("lighthouseweb3 not installed. Run: pip install lighthouseweb3")


# Pricing constants (USD)
# Lighthouse perpetual storage costs $2-5/GB one-time (funds endowment pool).
# Raw Filecoin deal cost is ~$0.00005/GB but that doesn't include the
# endowment buffer that keeps deals renewed forever.
LIGHTHOUSE_COST_PER_GB_MONTHLY = Decimal("0.05")  # Hot IPFS pinning (monthly)
LIGHTHOUSE_PERPETUAL_COST_PER_GB = Decimal("4.00")  # One-time perpetual via endowment pool
LIGHTHOUSE_RAW_FILECOIN_COST_PER_GB = Decimal("0.00005")  # Raw deal cost (no endowment)
CRYOSTASIS_BUFFER_USD = Decimal("0.50")  # Safety buffer for cryostasis archival


class LighthouseProvider(StorageProvider, CryostasisCapable, MultiCurrencyPayment):
    """
    Lighthouse storage provider for Tier 3 (Cloud Hosted) storage.

    Provides both IPFS pinning (CLOUD_HOT) and Filecoin deals (CLOUD_COLD).
    Agents can pay directly using FIL, USDC, or USDT.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        default_tier: StorageTier = StorageTier.CLOUD_HOT,
        key_resolver: Optional["KeyResolutionService"] = None,
    ):
        """
        Initialize Lighthouse provider.

        Args:
            api_key: Lighthouse API key (or from LIGHTHOUSE_API_KEY env var)
            cache_dir: Local cache directory for performance
            default_tier: Default storage tier (CLOUD_HOT or CLOUD_COLD)
            key_resolver: Optional KeyResolutionService for dynamic key resolution
        """
        self._key_resolver = key_resolver
        self.api_key = api_key or os.environ.get("LIGHTHOUSE_API_KEY")
        self._default_tier = default_tier

        # Initialize cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.environ.get("KESTREL_CACHE_DIR", "./storage_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Lighthouse client
        self._client: Optional["Lighthouse"] = None
        self._available = False

        if LIGHTHOUSE_AVAILABLE and self.api_key:
            try:
                self._client = Lighthouse(token=self.api_key)
                self._available = True
                logger.info("✅ Lighthouse provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Lighthouse: {e}")
        elif not LIGHTHOUSE_AVAILABLE:
            logger.warning("⚠️ lighthouseweb3 not installed")
        else:
            logger.warning("⚠️ No LIGHTHOUSE_API_KEY configured")

        # Track stored content locally for list operations
        self._index_file = self.cache_dir / "lighthouse_index.json"

    async def _get_api_key(self) -> Optional[str]:
        """Get API key, using resolver if available."""
        if self._key_resolver:
            try:
                key = await self._key_resolver.resolve_key("lighthouse", require=False)
                if key:
                    return key
            except Exception as e:
                logger.warning(f"Key resolver failed: {e}")
        return self.api_key

    async def _ensure_client(self) -> bool:
        """Ensure client is initialized with current API key."""
        api_key = await self._get_api_key()
        if api_key and LIGHTHOUSE_AVAILABLE:
            if not self._client or self.api_key != api_key:
                try:
                    self._client = Lighthouse(token=api_key)
                    self.api_key = api_key
                    self._available = True
                    return True
                except Exception as e:
                    logger.error(f"Failed to initialize Lighthouse: {e}")
                    return False
        return self._available

    @property
    def tier(self) -> StorageTier:
        """Return the storage tier this provider handles."""
        return self._default_tier

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "lighthouse"

    def is_available(self) -> bool:
        """Check if Lighthouse is available."""
        return self._available and self._client is not None

    async def store(
        self,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        encrypt: bool = True,
    ) -> StorageResult:
        """
        Store content in Lighthouse (IPFS + optional Filecoin).

        Args:
            content: Raw content bytes
            metadata: Optional metadata (filename, content_type, etc.)
            encrypt: Whether to encrypt before storing

        Returns:
            StorageResult with CID and storage details
        """
        # Ensure client is initialized with latest key
        await self._ensure_client()
        if not self.is_available():
            raise ConnectionError("Lighthouse provider not available")

        metadata = metadata or {}
        content_hash = hashlib.sha256(content).hexdigest()
        filename = metadata.get("filename", f"{content_hash[:16]}.bin")

        # Handle encryption
        final_content = content
        encryption_key_hash = None
        if encrypt:
            final_content, encryption_key_hash = await self._encrypt_content(content)

        # Write to temp file for upload (Lighthouse SDK requires file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(final_content)
            tmp_path = tmp.name

        try:
            # Upload to Lighthouse
            tag = metadata.get("tag", "kestrel-storage")
            upload_response = self._client.upload(source=tmp_path, tag=tag)

            # Parse response - format: {'data': {'Name': '...', 'Hash': '...', 'Size': '...'}}
            if isinstance(upload_response, dict) and "data" in upload_response:
                cid = upload_response["data"].get("Hash")
                size = int(upload_response["data"].get("Size", len(final_content)))
            else:
                # Handle alternative response format
                cid = getattr(upload_response, "Hash", None) or str(upload_response)
                size = len(final_content)

            logger.info(f"📤 Uploaded to Lighthouse: {cid}")

            # Cache locally for fast retrieval
            await self._cache_content(content_hash, final_content)

            # Calculate cost
            size_gb = Decimal(size) / Decimal(1024 * 1024 * 1024)
            storage_cost = size_gb * LIGHTHOUSE_COST_PER_GB_MONTHLY

            result = StorageResult(
                content_hash=content_hash,
                cid=cid,
                tier=self._default_tier,
                provider=self.provider_name,
                encrypted=encrypt,
                encryption_key_hash=encryption_key_hash,
                size_bytes=size,
                content_type=metadata.get("content_type"),
                filename=filename,
                storage_cost_usd=storage_cost,
            )

            # Update local index
            await self._update_index(result)

            return result

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    async def retrieve(self, cid: str, encryption_key_hash: Optional[str] = None) -> bytes:
        """
        Retrieve content by CID.

        Args:
            cid: IPFS Content ID
            encryption_key_hash: Key hash if content is encrypted

        Returns:
            Original content bytes
        """
        await self._ensure_client()
        if not self.is_available():
            raise ConnectionError("Lighthouse provider not available")

        # Try local cache first
        cached = await self._get_from_cache(cid)
        if cached:
            logger.info(f"📂 Retrieved from cache: {cid[:16]}...")
            content = cached
        else:
            # Download from Lighthouse IPFS gateway
            logger.info(f"📡 Downloading from Lighthouse: {cid}")
            import requests
            gateway_url = f"{get_lighthouse_gateway_url()}/{cid}"
            response = requests.get(gateway_url, timeout=HTTP_TIMEOUT_MEDIUM)
            response.raise_for_status()
            content = response.content

            # Cache for next time
            await self._cache_content(cid, content)

        # Decrypt if needed
        if encryption_key_hash:
            content = await self._decrypt_content(content, encryption_key_hash)

        return content

    async def list_content(self, limit: int = 100, offset: int = 0) -> List[StorageResult]:
        """
        List stored content from local index.

        Args:
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of StorageResult objects
        """
        index = await self._load_index()
        items = list(index.values())[offset : offset + limit]
        return [StorageResult.from_dict(item) for item in items]

    async def delete(self, cid: str) -> bool:
        """
        Delete content (removes from local index only).

        Note: IPFS content cannot be truly deleted from the network,
        but we can unpin and remove from our index.

        Args:
            cid: Content ID to delete

        Returns:
            True if removed from index
        """
        index = await self._load_index()
        if cid in index:
            del index[cid]
            await self._save_index(index)
            # Remove from cache
            cache_file = self.cache_dir / f"{cid}.cache"
            if cache_file.exists():
                cache_file.unlink()
            logger.info(f"🗑️ Removed from index: {cid}")
            return True
        return False

    async def verify(self, cid: str) -> bool:
        """
        Verify content is accessible via Lighthouse.

        Args:
            cid: Content ID to verify

        Returns:
            True if content is accessible
        """
        await self._ensure_client()
        if not self.is_available():
            return False

        try:
            # Check deal status
            status = self._client.getDealStatus(cid)
            logger.info(f"✅ Verified: {cid} - {status}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Verification failed for {cid}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get Lighthouse provider statistics."""
        index = await self._load_index()
        total_size = sum(item.get("size_bytes", 0) for item in index.values())

        return {
            "tier": self.tier.value,
            "provider": self.provider_name,
            "available": self.is_available(),
            "total_items": len(index),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "cost_per_gb_monthly": str(LIGHTHOUSE_COST_PER_GB_MONTHLY),
            "perpetual_cost_per_gb": str(LIGHTHOUSE_PERPETUAL_COST_PER_GB),
            "raw_filecoin_cost_per_gb": str(LIGHTHOUSE_RAW_FILECOIN_COST_PER_GB),
        }

    async def estimate_cost(self, size_bytes: int) -> Decimal:
        """
        Estimate monthly storage cost.

        Args:
            size_bytes: Size of content

        Returns:
            Estimated monthly cost in USD
        """
        size_gb = Decimal(size_bytes) / Decimal(1024 * 1024 * 1024)
        return size_gb * LIGHTHOUSE_COST_PER_GB_MONTHLY

    # =========================================================================
    # CryostasisCapable Implementation
    # =========================================================================

    async def archive_for_cryostasis(
        self,
        agent_id: str,
        state_snapshot: bytes,
        metadata: Dict[str, Any],
    ) -> StorageResult:
        """
        Archive agent state for cryostasis (permanent Filecoin storage).

        This is called when an agent's wallet balance falls below the
        cryostasis trigger threshold.

        Args:
            agent_id: Agent's unique identifier
            state_snapshot: Serialized agent state (memory, identity, graph)
            metadata: Agent metadata (DID, last active timestamp, etc.)

        Returns:
            StorageResult with Filecoin deal info
        """
        logger.info(f"🧊 Archiving agent {agent_id} for cryostasis...")

        # Store with cryostasis-specific metadata
        cryostasis_metadata = {
            **metadata,
            "agent_id": agent_id,
            "cryostasis": True,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "tag": f"cryostasis-{agent_id}",
        }

        result = await self.store(
            content=state_snapshot,
            metadata=cryostasis_metadata,
            encrypt=True,  # Always encrypt agent state
        )

        # Mark as cryostasis in result
        result.deal_status = "cryostasis"

        logger.info(f"🧊 Agent {agent_id} archived to CID: {result.cid}")
        return result

    async def restore_from_cryostasis(self, cid: str, encryption_key_hash: str) -> bytes:
        """
        Restore agent state from cryostasis.

        Called when an agent is "woken up" after being funded.

        Args:
            cid: CID of archived state
            encryption_key_hash: Encryption key hash

        Returns:
            Decrypted agent state bytes
        """
        logger.info(f"🌡️ Restoring agent from cryostasis: {cid}")
        return await self.retrieve(cid, encryption_key_hash)

    async def calculate_cryostasis_cost(self, size_bytes: int) -> Decimal:
        """
        Calculate the cryostasis trigger threshold.

        This is the minimum balance needed to archive to Filecoin plus a buffer.

        Args:
            size_bytes: Size of agent state

        Returns:
            Cryostasis trigger threshold in USD
        """
        size_gb = Decimal(size_bytes) / Decimal(1024 * 1024 * 1024)
        filecoin_cost = size_gb * LIGHTHOUSE_PERPETUAL_COST_PER_GB
        return filecoin_cost + CRYOSTASIS_BUFFER_USD

    # =========================================================================
    # MultiCurrencyPayment Implementation
    # =========================================================================

    async def pay_for_storage(
        self,
        amount_usd: Decimal,
        currency: str,
        wallet_address: str,
    ) -> Dict[str, Any]:
        """
        Pay for storage using cryptocurrency.

        Note: Lighthouse handles payments through their platform.
        This method would integrate with their payment API.

        Args:
            amount_usd: Amount in USD
            currency: FIL, USDC, or USDT
            wallet_address: Agent's wallet address

        Returns:
            Transaction details
        """
        if currency not in self.SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {currency}. Use: {self.SUPPORTED_CURRENCIES}")

        # Payment API integration not yet implemented
        # Lighthouse payment requires web3.py integration for on-chain transactions
        raise NotImplementedError(
            "Lighthouse payment API integration is planned for a future release. "
            "Currently using API key quota instead of per-transaction payments. "
            f"Requested: ${amount_usd} in {currency} from {wallet_address}"
        )

    async def get_balance(self, wallet_address: str, currency: str) -> Decimal:
        """
        Get wallet balance.

        Note: This would integrate with a blockchain provider.

        Args:
            wallet_address: Wallet to check
            currency: Currency to check

        Returns:
            Balance in specified currency
        """
        # Web3 balance check not yet implemented
        # Requires integration with Ethereum/Polygon RPC provider
        raise NotImplementedError(
            "Web3 balance check is planned for a future release. "
            f"Cannot check balance for wallet {wallet_address}"
        )

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _encrypt_content(self, content: bytes) -> tuple[bytes, str]:
        """Encrypt content using Fernet with a derived key."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise ImportError("cryptography package required for encryption")

        # Generate content-specific key
        content_key = Fernet.generate_key()
        f = Fernet(content_key)
        encrypted = f.encrypt(content)

        # Get master key from environment
        master_key = self._get_master_key()
        f_master = Fernet(master_key)

        # Encrypt the content key with master key
        encrypted_key = f_master.encrypt(content_key)
        key_hash = hashlib.sha256(encrypted_key).hexdigest()

        # Store encrypted key locally
        key_file = self.cache_dir / f"key_{key_hash}.key"
        with open(key_file, "wb") as f:
            f.write(encrypted_key)

        return encrypted, key_hash

    async def _decrypt_content(self, encrypted: bytes, key_hash: str) -> bytes:
        """Decrypt content using stored key."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise ImportError("cryptography package required for decryption")

        # Load encrypted key
        key_file = self.cache_dir / f"key_{key_hash}.key"
        if not key_file.exists():
            raise FileNotFoundError(f"Encryption key not found: {key_hash}")

        with open(key_file, "rb") as f:
            encrypted_key = f.read()

        # Decrypt key with master
        master_key = self._get_master_key()
        f_master = Fernet(master_key)
        content_key = f_master.decrypt(encrypted_key)

        # Decrypt content
        f_content = Fernet(content_key)
        return f_content.decrypt(encrypted)

    def _get_master_key(self) -> bytes:
        """Get master encryption key from centralized encryption module.

        Uses the same key derivation as the rest of the system, supporting
        both raw Fernet keys and passphrases (via SHA-256 derivation).
        """
        from kestrel_sovereign.storage.encryption import get_master_key_bytes

        key = get_master_key_bytes()
        if key:
            return key

        # No key configured - fail explicitly (no hardcoded fallback)
        raise ValueError(
            "KESTREL_DATA_KEY environment variable required for Lighthouse encryption. "
            "Set it to a passphrase or a valid Fernet key."
        )

    async def _cache_content(self, key: str, content: bytes) -> None:
        """Cache content locally."""
        cache_file = self.cache_dir / f"{key}.cache"
        with open(cache_file, "wb") as f:
            f.write(content)

    async def _get_from_cache(self, key: str) -> Optional[bytes]:
        """Get content from local cache."""
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return f.read()
        return None

    async def _load_index(self) -> Dict[str, Any]:
        """Load the local content index."""
        import json

        if self._index_file.exists():
            with open(self._index_file, "r") as f:
                return json.load(f)
        return {}

    async def _save_index(self, index: Dict[str, Any]) -> None:
        """Save the local content index."""
        import json

        with open(self._index_file, "w") as f:
            json.dump(index, f, indent=2)

    async def _update_index(self, result: StorageResult) -> None:
        """Update the local index with a new result."""
        index = await self._load_index()
        index[result.cid] = result.to_dict()
        await self._save_index(index)


# Factory function for easy instantiation
def create_lighthouse_provider(
    api_key: Optional[str] = None,
    tier: StorageTier = StorageTier.CLOUD_HOT,
    key_resolver: Optional["KeyResolutionService"] = None,
) -> LighthouseProvider:
    """
    Create a Lighthouse provider instance.

    Args:
        api_key: Lighthouse API key (or uses LIGHTHOUSE_API_KEY env var)
        tier: Default storage tier (CLOUD_HOT or CLOUD_COLD)
        key_resolver: Optional KeyResolutionService for dynamic key resolution

    Returns:
        Configured LighthouseProvider
    """
    return LighthouseProvider(api_key=api_key, default_tier=tier, key_resolver=key_resolver)
