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

References:
- Docs: https://docs.lighthouse.storage/
- REST API: used directly via LighthouseRestClient (replaces unmaintained SDK)
- Kavach encryption: https://github.com/lighthouse-web3/encryption-sdk
- x402 protocol: https://github.com/coinbase/x402
- Endowment pool: https://www.lighthouse.storage/blogs/Discover%20How%20the%20Endowment%20Pool%20Makes%20Your%20Data%20Immortal
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from kestrel_sovereign.storage.providers.base import (
    CryostasisCapable,
    MultiCurrencyPayment,
    StorageProvider,
    StorageResult,
    StorageTier,
)
from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseRestClient
from kestrel_sovereign.kestrel_config.defaults import get_lighthouse_gateway_url

logger = logging.getLogger(__name__)


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
            api_key: Lighthouse API key. When both ``api_key`` and
                ``key_resolver`` are None, ``LIGHTHOUSE_API_KEY`` is read
                from the environment as a fallback (today's standalone
                behavior). When ``key_resolver`` is supplied, the env-var
                fallback is NOT consulted at construction time — the
                resolver is the single source of truth, so a PayerPolicy
                slot of ``NONE`` actually means NONE even on a host where
                ``LIGHTHOUSE_API_KEY`` happens to be set.
            cache_dir: Local cache directory for performance
            default_tier: Default storage tier (CLOUD_HOT or CLOUD_COLD)
            key_resolver: Optional KeyResolutionService for dynamic key resolution
        """
        self._key_resolver = key_resolver
        # Env-var fallback only when no resolver is in charge (otherwise the
        # resolver would be silently overridden at construction time).
        if api_key is None and key_resolver is None:
            api_key = os.environ.get("LIGHTHOUSE_API_KEY")
        self.api_key = api_key
        self._default_tier = default_tier

        # Initialize cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.environ.get("KESTREL_CACHE_DIR", "./storage_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize async REST client
        self._client: Optional[LighthouseRestClient] = None
        self._available = False

        if self.api_key:
            self._client = LighthouseRestClient(
                api_key=self.api_key,
                gateway_url=get_lighthouse_gateway_url(),
            )
            self._available = True
            logger.info("Lighthouse provider initialized (REST API)")
        else:
            logger.warning("No LIGHTHOUSE_API_KEY configured")

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
        if api_key:
            if not self._client or self.api_key != api_key:
                # Close old client if key changed
                if self._client:
                    await self._client.close()
                self._client = LighthouseRestClient(
                    api_key=api_key,
                    gateway_url=get_lighthouse_gateway_url(),
                )
                self.api_key = api_key
                self._available = True
                return True
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
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> StorageResult:
        """
        Store content in Lighthouse (IPFS + optional Filecoin).

        Args:
            content: Raw content bytes
            metadata: Optional metadata (filename, content_type, etc.)
            encrypt: Whether to encrypt before storing
            on_progress: Optional callback(bytes_sent, total_bytes)

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

        # Upload to Lighthouse via REST API
        tag = metadata.get("tag", "kestrel-storage")
        upload_response = await self._client.upload(
            content=final_content,
            filename=filename,
            tag=tag,
            on_progress=on_progress,
        )

        cid = upload_response.get("Hash")
        size = int(upload_response.get("Size", len(final_content)))

        logger.info(f"Uploaded to Lighthouse: {cid}")

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
            logger.info(f"Retrieved from cache: {cid[:16]}...")
            content = cached
        else:
            # Download from Lighthouse IPFS gateway
            logger.info(f"Downloading from Lighthouse: {cid}")
            content = await self._client.download(cid)

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
            status = await self._client.get_deal_status(cid)
            logger.info(f"Verified: {cid} - {status}")
            return True
        except Exception as e:
            logger.warning(f"Verification failed for {cid}: {e}")
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

        Uses Lighthouse payment API to create a storage deal payment.
        Supports FIL, USDC, and USDT on Filecoin network.

        Args:
            amount_usd: Amount in USD
            currency: FIL, USDC, or USDT
            wallet_address: Agent's wallet address

        Returns:
            Transaction details including payment status and deal info

        Raises:
            ValueError: If currency is not supported
            ConnectionError: If Lighthouse API is not available
        """
        if currency not in self.SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {currency}. Use: {self.SUPPORTED_CURRENCIES}")

        # Ensure client is available
        await self._ensure_client()
        if not self.is_available():
            raise ConnectionError("Lighthouse provider not available")

        try:
            import httpx

            # Convert USD to the specified currency
            conversion_rates = {
                "FIL": Decimal("5.50"),   # $5.50 per FIL (approximate)
                "USDC": Decimal("1.00"),  # 1:1 with USD
                "USDT": Decimal("1.00"),  # 1:1 with USD
            }

            amount_in_currency = amount_usd / conversion_rates[currency]

            # Lighthouse REST API for payment operations
            client = await self._client._get_client()
            response = await client.post(
                "https://api.lighthouse.storage/api/payment/deal",
                headers=self._client._auth_headers,
                json={
                    "amount": str(amount_in_currency),
                    "currency": currency,
                    "wallet_address": wallet_address,
                    "duration_days": 365,
                },
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Payment successful: {amount_in_currency} {currency}")

                return {
                    "status": "success",
                    "payment_id": result.get("paymentId"),
                    "amount": str(amount_in_currency),
                    "currency": currency,
                    "amount_usd": str(amount_usd),
                    "wallet_address": wallet_address,
                    "deal_id": result.get("dealId"),
                    "expires_at": result.get("expiresAt"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                error_msg = response.json().get("error", "Unknown error")
                logger.error(f"Payment failed: {error_msg}")

                return {
                    "status": "failed",
                    "error": error_msg,
                    "amount": str(amount_in_currency),
                    "currency": currency,
                    "amount_usd": str(amount_usd),
                    "wallet_address": wallet_address,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        except httpx.HTTPError as e:
            logger.error(f"Payment request failed: {e}")
            raise ConnectionError(f"Failed to connect to Lighthouse payment API: {e}")
        except Exception as e:
            logger.error(f"Payment error: {e}")
            raise

    async def get_balance(self, wallet_address: str, currency: str) -> Decimal:
        """
        Get wallet balance and Lighthouse storage quota.

        Returns both the on-chain balance and Lighthouse storage usage.

        Args:
            wallet_address: Wallet to check (optional, for on-chain balance)
            currency: Currency to check (FIL, USDC, or USDT)

        Returns:
            Balance in specified currency (Decimal)

        Raises:
            ValueError: If currency is not supported
            ConnectionError: If Lighthouse API is not available
        """
        if currency not in self.SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {currency}. Use: {self.SUPPORTED_CURRENCIES}")

        # Ensure client is available
        await self._ensure_client()
        if not self.is_available():
            raise ConnectionError("Lighthouse provider not available")

        try:
            # Use REST API to get balance/quota information
            balance_data = await self._client.get_balance()

            # Balance data format from SDK:
            # {
            #   "data": {
            #     "dataUsed": "1234567890",  # bytes used
            #     "dataLimit": "5368709120"  # bytes limit (e.g., 5GB)
            #   }
            # }

            if isinstance(balance_data, dict) and "data" in balance_data:
                data = balance_data["data"]
                data_used = int(data.get("dataUsed", 0))
                data_limit = int(data.get("dataLimit", 0))
                available_bytes = max(0, data_limit - data_used)

                # Convert available storage to USD value
                # Using perpetual storage cost: $4/GB one-time
                available_gb = Decimal(available_bytes) / Decimal(1024 * 1024 * 1024)
                available_usd = available_gb * LIGHTHOUSE_PERPETUAL_COST_PER_GB

                # Convert to requested currency
                conversion_rates = {
                    "FIL": Decimal("5.50"),   # $5.50 per FIL (approximate)
                    "USDC": Decimal("1.00"),  # 1:1 with USD
                    "USDT": Decimal("1.00"),  # 1:1 with USD
                }

                balance_in_currency = available_usd / conversion_rates[currency]

                logger.info(
                    f"📊 Lighthouse balance: {available_bytes / (1024**3):.2f} GB available "
                    f"(~{balance_in_currency:.2f} {currency})"
                )

                return balance_in_currency
            else:
                logger.warning("⚠️ Unexpected balance data format")
                return Decimal("0")

        except Exception as e:
            logger.error(f"❌ Failed to get balance: {e}")
            # Graceful fallback - return 0 instead of raising
            return Decimal("0")

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _encrypt_content(self, content: bytes) -> tuple[bytes, str]:
        """Encrypt content using Fernet with a derived key."""
        try:
            from kestrel_sdk.security.aead import AEADCipher
        except ImportError:
            raise ImportError("cryptography package required for encryption")

        # Generate content-specific key
        content_key = AEADCipher.generate_key()
        f = AEADCipher(content_key)
        encrypted = f.encrypt(content)

        # Get master key from environment
        master_key = self._get_master_key()
        f_master = AEADCipher(master_key)

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
            from kestrel_sdk.security.aead import AEADCipher
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
        f_master = AEADCipher(master_key)
        content_key = f_master.decrypt(encrypted_key)

        # Decrypt content
        f_content = AEADCipher(content_key)
        return f_content.decrypt(encrypted)

    def _get_master_key(self) -> bytes:
        """Get master encryption key from centralized encryption module.

        Uses the same key derivation as the rest of the system, supporting
        both raw Fernet keys and passphrases (via SHA-256 derivation).
        """
        from kestrel_sovereign.security.encryption import get_master_key_bytes

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
            with open(self._index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    async def _save_index(self, index: Dict[str, Any]) -> None:
        """Save the local content index."""
        import json

        with open(self._index_file, "w", encoding="utf-8") as f:
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
