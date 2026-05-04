"""
Storacha (web3.storage) Storage Provider

Integrates with Storacha's w3up protocol for IPFS-pinned storage with
Filecoin-backed persistence. Uses UCAN/DID authentication — the agent's
Ed25519 identity is used directly as the UCAN signing key, aligning
with Kestrel's existing DID identity model.

Advantages over Lighthouse:
- UCAN/DID native auth (no opaque API key, agent controls access)
- Open-source client stack + self-hostable via local.storage
- Content-addressed, verifiable storage (IPFS CIDs)
- Free tier: 5 GB

Pricing (as of March 2026):
- Free tier: 5 GB
- Paid: ~$0.05/GB/month (similar to Lighthouse hot tier)

One-time setup (run once per deployment):
    npm install -g @web3-storage/w3cli
    w3 key create                                            # → STORACHA_AGENT_KEY
    w3 space create kestrel                                  # → STORACHA_SPACE_DID
    w3 delegation create --can '*' <agent-did> | base64     # → STORACHA_PROOF

References:
- Storacha docs:   https://docs.storacha.network/
- w3up GitHub:     https://github.com/storacha/w3up
- local.storage:   https://github.com/storacha/local.storage
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.storage.providers.base import (
    CryostasisCapable,
    StorageProvider,
    StorageResult,
    StorageTier,
)
from kestrel_sovereign.storage.providers.storacha_ucan import StorachaUCAN
from kestrel_sovereign.storage.providers.storacha_rest import StorachaRestClient
from kestrel_sovereign.kestrel_config.defaults import get_storacha_gateway_url

logger = logging.getLogger(__name__)

# Pricing constants (USD)
STORACHA_COST_PER_GB_MONTHLY = Decimal("0.05")   # Hot IPFS pinning
STORACHA_FREE_TIER_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB free tier
CRYOSTASIS_BUFFER_USD = Decimal("0.50")


class StorachaProvider(StorageProvider, CryostasisCapable):
    """
    Storacha storage provider for Tier 3 (CLOUD_HOT) storage.

    Stores content on IPFS via the Storacha w3up bridge, using UCAN/DID
    auth. Content is retrievable from any IPFS gateway using the CID.

    Required environment variables:
        STORACHA_SPACE_DID  - Space DID (from `w3 space create`)
        STORACHA_AGENT_KEY  - Ed25519 agent key (from `w3 key create`)
        STORACHA_PROOF      - Base64 delegation proof CAR (from `w3 delegation create`)

    Optional:
        STORACHA_GATEWAY_URL - IPFS gateway base URL (default: https://w3s.link/ipfs)
        KESTREL_CACHE_DIR    - Local cache directory (default: ./storage_cache)
    """

    def __init__(
        self,
        space_did: Optional[str] = None,
        agent_key: Optional[str] = None,
        proof: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialise Storacha provider.

        Args:
            space_did: Space DID (falls back to STORACHA_SPACE_DID env var)
            agent_key: Ed25519 agent key string (falls back to STORACHA_AGENT_KEY)
            proof:     Base64 delegation proof (falls back to STORACHA_PROOF)
            cache_dir: Local cache directory path (falls back to KESTREL_CACHE_DIR)
        """
        self._space_did = space_did or os.environ.get("STORACHA_SPACE_DID", "")
        self._agent_key = agent_key or os.environ.get("STORACHA_AGENT_KEY", "")
        self._proof = proof or os.environ.get("STORACHA_PROOF", "")

        # Local cache for performance (same pattern as LighthouseProvider)
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.environ.get("KESTREL_CACHE_DIR", "./storage_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "storacha_index.json"

        # Lazy-initialise the UCAN signer and REST client
        self._ucan: Optional[StorachaUCAN] = None
        self._client: Optional[StorachaRestClient] = None
        self._available = False

        if self._space_did and self._agent_key and self._proof:
            try:
                self._ucan = StorachaUCAN(
                    agent_key=self._agent_key,
                    space_did=self._space_did,
                    proof=self._proof,
                )
                self._client = StorachaRestClient(
                    ucan=self._ucan,
                    gateway_url=get_storacha_gateway_url(),
                )
                self._available = True
                logger.info(
                    "StorachaProvider initialized — space: %s  agent: %s",
                    self._space_did[:30],
                    self._ucan.agent_did[:30],
                )
            except Exception as e:
                logger.warning("StorachaProvider: failed to initialize UCAN client: %s", e)
        else:
            missing = [
                v for v, val in [
                    ("STORACHA_SPACE_DID", self._space_did),
                    ("STORACHA_AGENT_KEY", self._agent_key),
                    ("STORACHA_PROOF", self._proof),
                ] if not val
            ]
            logger.warning("StorachaProvider: missing env vars: %s", ", ".join(missing))

    # ------------------------------------------------------------------
    # StorageProvider interface
    # ------------------------------------------------------------------

    @property
    def tier(self) -> StorageTier:
        return StorageTier.CLOUD_HOT

    @property
    def provider_name(self) -> str:
        return "storacha"

    def is_available(self) -> bool:
        return self._available and self._client is not None

    async def store(
        self,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        encrypt: bool = True,
    ) -> StorageResult:
        """
        Store content in Storacha (IPFS + w3up).

        Args:
            content:  Raw content bytes
            metadata: Optional metadata dict (filename, content_type, tag, …)
            encrypt:  Whether to encrypt before storing (Fernet, same as LighthouseProvider)

        Returns:
            StorageResult with CID and storage details
        """
        if not self.is_available():
            raise ConnectionError("StorachaProvider is not available")

        metadata = metadata or {}
        content_hash = hashlib.sha256(content).hexdigest()
        filename = metadata.get("filename", f"{content_hash[:16]}.bin")

        # Encrypt if requested
        final_content = content
        encryption_key_hash = None
        if encrypt:
            final_content, encryption_key_hash = await self._encrypt_content(content)

        # Upload via w3up bridge
        result_data = await self._client.upload(
            content=final_content,
            filename=filename,
            content_type=metadata.get("content_type", "application/octet-stream"),
        )
        cid = result_data["cid"]
        size = len(final_content)

        # Cache locally for fast re-retrieval
        await self._cache_content(cid, final_content)

        size_gb = Decimal(size) / Decimal(1024 * 1024 * 1024)
        storage_cost = size_gb * STORACHA_COST_PER_GB_MONTHLY

        storage_result = StorageResult(
            content_hash=content_hash,
            cid=cid,
            tier=self.tier,
            provider=self.provider_name,
            encrypted=encrypt,
            encryption_key_hash=encryption_key_hash,
            size_bytes=size,
            content_type=metadata.get("content_type"),
            filename=filename,
            storage_cost_usd=storage_cost,
        )

        await self._update_index(storage_result)
        return storage_result

    async def retrieve(self, cid: str, encryption_key_hash: Optional[str] = None) -> bytes:
        """
        Retrieve content by CID string.

        Args:
            cid:                  CID string (base32lower "b..." or base58 "Qm...")
            encryption_key_hash:  Fernet key hash if content was encrypted

        Returns:
            Original (decrypted) content bytes
        """
        if not self.is_available():
            raise ConnectionError("StorachaProvider is not available")

        # Try local cache first
        cached = await self._get_from_cache(cid)
        if cached:
            logger.debug("Retrieved from local cache: %s", cid[:20])
            content = cached
        else:
            logger.info("Fetching from Storacha gateway: %s", cid[:20])
            content = await self._client.get_by_cid(cid)
            await self._cache_content(cid, content)

        if encryption_key_hash:
            content = await self._decrypt_content(content, encryption_key_hash)

        return content

    async def list_content(self, limit: int = 100, offset: int = 0) -> List[StorageResult]:
        """List stored content from the local index."""
        index = await self._load_index()
        items = list(index.values())[offset: offset + limit]
        return [StorageResult.from_dict(item) for item in items]

    async def delete(self, cid: str) -> bool:
        """
        Remove CID from the space index and local cache.

        Note: Content is content-addressed on IPFS and cannot be deleted
        from the network; this removes it from the Storacha space index.
        """
        if self.is_available():
            await self._client.delete(cid)

        index = await self._load_index()
        if cid in index:
            del index[cid]
            await self._save_index(index)

        cache_file = self.cache_dir / f"{cid}.cache"
        if cache_file.exists():
            cache_file.unlink()

        logger.info("Removed from Storacha index: %s", cid[:20])
        return True

    async def verify(self, cid: str) -> bool:
        """Verify content is accessible via the IPFS gateway."""
        if not self.is_available():
            return False
        try:
            client = await self._client._get_client()
            url = f"{self._client.gateway_url}/{cid}"
            response = await client.head(url, timeout=15.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning("Verification failed for %s: %s", cid[:20], e)
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics including space usage."""
        index = await self._load_index()
        total_size = sum(item.get("size_bytes", 0) for item in index.values())

        stats: Dict[str, Any] = {
            "tier": self.tier.value,
            "provider": self.provider_name,
            "available": self.is_available(),
            "total_items": len(index),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "cost_per_gb_monthly": str(STORACHA_COST_PER_GB_MONTHLY),
            "free_tier_bytes": STORACHA_FREE_TIER_BYTES,
        }

        if self.is_available() and self._ucan:
            stats["space_did"] = self._space_did
            stats["agent_did"] = self._ucan.agent_did

        return stats

    async def estimate_cost(self, size_bytes: int) -> Decimal:
        """Estimate monthly storage cost. Returns $0 within the 5 GB free tier."""
        if size_bytes <= STORACHA_FREE_TIER_BYTES:
            return Decimal("0")
        billable = size_bytes - STORACHA_FREE_TIER_BYTES
        size_gb = Decimal(billable) / Decimal(1024 * 1024 * 1024)
        return size_gb * STORACHA_COST_PER_GB_MONTHLY

    # ------------------------------------------------------------------
    # CryostasisCapable implementation
    # ------------------------------------------------------------------

    async def archive_for_cryostasis(
        self,
        agent_id: str,
        state_snapshot: bytes,
        metadata: Dict[str, Any],
    ) -> StorageResult:
        """
        Archive agent state for cryostasis.

        Stores the state snapshot on IPFS/Storacha with cryostasis metadata.
        Unlike Lighthouse (which uses Filecoin deals), Storacha archives on
        IPFS with Filecoin-backed persistence via the w3up protocol.

        Args:
            agent_id:       Unique agent identifier
            state_snapshot: Serialised agent state
            metadata:       Agent metadata dict

        Returns:
            StorageResult with cryostasis CID
        """
        logger.info("Archiving agent %s for cryostasis (Storacha)...", agent_id)

        cryo_metadata = {
            **metadata,
            "agent_id": agent_id,
            "cryostasis": True,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "tag": f"cryostasis-{agent_id}",
        }

        result = await self.store(
            content=state_snapshot,
            metadata=cryo_metadata,
            encrypt=True,
        )
        result.deal_status = "cryostasis"
        logger.info("Agent %s archived to Storacha CID: %s", agent_id, result.cid)
        return result

    async def restore_from_cryostasis(self, cid: str, encryption_key_hash: str) -> bytes:
        """Restore agent state from cryostasis storage."""
        logger.info("Restoring from Storacha cryostasis: %s", cid[:20])
        return await self.retrieve(cid, encryption_key_hash)

    async def calculate_cryostasis_cost(self, size_bytes: int) -> Decimal:
        """
        Estimate cryostasis storage cost (one-time, includes IPFS + Filecoin backup).
        """
        monthly_cost = await self.estimate_cost(size_bytes)
        # Estimate 12 months of storage as a proxy for long-term archival cost
        return monthly_cost * 12 + CRYOSTASIS_BUFFER_USD

    # ------------------------------------------------------------------
    # Encryption helpers (same pattern as LighthouseProvider)
    # ------------------------------------------------------------------

    async def _encrypt_content(self, content: bytes) -> tuple:
        """Encrypt content using Fernet with a master-key-wrapped content key."""
        try:
            from kestrel_sdk.security.aead import AEADCipher
        except ImportError:
            raise ImportError("cryptography package required for encryption")

        content_key = AEADCipher.generate_key()
        f = AEADCipher(content_key)
        encrypted = f.encrypt(content)

        master_key = self._get_master_key()
        f_master = AEADCipher(master_key)
        encrypted_key = f_master.encrypt(content_key)
        key_hash = hashlib.sha256(encrypted_key).hexdigest()

        key_file = self.cache_dir / f"key_{key_hash}.key"
        with open(key_file, "wb") as fh:
            fh.write(encrypted_key)

        return encrypted, key_hash

    async def _decrypt_content(self, encrypted: bytes, key_hash: str) -> bytes:
        """Decrypt content using stored Fernet key."""
        try:
            from kestrel_sdk.security.aead import AEADCipher
        except ImportError:
            raise ImportError("cryptography package required for decryption")

        key_file = self.cache_dir / f"key_{key_hash}.key"
        if not key_file.exists():
            raise FileNotFoundError(f"Encryption key not found: {key_hash}")

        with open(key_file, "rb") as fh:
            encrypted_key = fh.read()

        master_key = self._get_master_key()
        f_master = AEADCipher(master_key)
        content_key = f_master.decrypt(encrypted_key)

        f_content = AEADCipher(content_key)
        return f_content.decrypt(encrypted)

    def _get_master_key(self) -> bytes:
        """Get master encryption key from centralised encryption module."""
        from kestrel_sovereign.security.encryption import get_master_key_bytes

        key = get_master_key_bytes()
        if key:
            return key
        raise ValueError(
            "KESTREL_DATA_KEY environment variable is required for StorachaProvider encryption."
        )

    # ------------------------------------------------------------------
    # Local cache / index helpers
    # ------------------------------------------------------------------

    async def _cache_content(self, key: str, content: bytes) -> None:
        cache_file = self.cache_dir / f"{key}.cache"
        with open(cache_file, "wb") as fh:
            fh.write(content)

    async def _get_from_cache(self, key: str) -> Optional[bytes]:
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            with open(cache_file, "rb") as fh:
                return fh.read()
        return None

    async def _load_index(self) -> Dict[str, Any]:
        import json
        if self._index_file.exists():
            with open(self._index_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    async def _save_index(self, index: Dict[str, Any]) -> None:
        import json
        with open(self._index_file, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)

    async def _update_index(self, result: StorageResult) -> None:
        index = await self._load_index()
        index[result.cid] = result.to_dict()
        await self._save_index(index)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_storacha_provider(
    space_did: Optional[str] = None,
    agent_key: Optional[str] = None,
    proof: Optional[str] = None,
) -> StorachaProvider:
    """
    Create a StorachaProvider from environment or explicit parameters.

    Args:
        space_did: Space DID (falls back to STORACHA_SPACE_DID env var)
        agent_key: Agent key string (falls back to STORACHA_AGENT_KEY)
        proof:     Delegation proof (falls back to STORACHA_PROOF)

    Returns:
        Configured StorachaProvider
    """
    return StorachaProvider(space_did=space_did, agent_key=agent_key, proof=proof)
