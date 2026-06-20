"""
Tiered Storage Manager

Orchestrates multiple storage providers across tiers, handles fallback logic,
and respects privacy modes. This is the main entry point for storage operations.

Architecture:
    Browser (Tier 1) → Local IPFS (Tier 2) → Lighthouse Cloud (Tier 3)

Features:
- Automatic tier selection based on privacy mode
- Fallback to lower tiers when preferred tier unavailable
- Sync between tiers for redundancy
- Cryostasis integration for agent dormancy
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Type

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.providers.base import (
    CryostasisCapable,
    StorageProvider,
    StorageResult,
    StorageTier,
    SyncItem,
    SyncManifest,
    SyncStatus,
)

logger = logging.getLogger(__name__)


# `PrivacyMode` used to be a local bare-class shadow with string class
# attributes (predating the SDK extraction in #1094). It now re-exports
# the canonical enum from `kestrel_sovereign.privacy` so this module
# shares one identity with the rest of the agent. The str-Enum mixin
# means every existing comparison (`mode == "normal"`, dict lookups
# keyed by either form, the `privacy_mode: str = PrivacyMode.NORMAL`
# default below) keeps working unchanged.

# Map privacy modes to allowed storage tiers
PRIVACY_TIER_POLICY = {
    PrivacyMode.EPHEMERAL: [],  # No storage
    PrivacyMode.ISOLATED: [StorageTier.LOCAL, StorageTier.BROWSER],
    PrivacyMode.ANONYMOUS: [StorageTier.LOCAL, StorageTier.BROWSER, StorageTier.CLOUD_HOT],
    PrivacyMode.NORMAL: [StorageTier.LOCAL, StorageTier.BROWSER, StorageTier.CLOUD_HOT, StorageTier.CLOUD_COLD],
    PrivacyMode.PUBLIC: [StorageTier.LOCAL, StorageTier.BROWSER, StorageTier.CLOUD_HOT, StorageTier.CLOUD_COLD],
    PrivacyMode.DEIDENTIFIED: [StorageTier.LOCAL, StorageTier.BROWSER],
}

REMOTE_STORAGE_TIERS = {
    StorageTier.CLOUD_HOT,
    StorageTier.CLOUD_COLD,
    StorageTier.IPFS,
    StorageTier.FILECOIN,
    StorageTier.ENCRYPTED_FILECOIN,
}


def allowed_tiers_for_privacy_mode(privacy_mode: str) -> List[StorageTier]:
    """Return the storage tiers allowed by the canonical privacy tier map."""
    mode = privacy_mode
    if isinstance(privacy_mode, str):
        try:
            mode = PrivacyMode(privacy_mode)
        except ValueError:
            return []
    return PRIVACY_TIER_POLICY.get(mode, [])


def privacy_allows_remote_tiers(privacy_mode: str) -> bool:
    """Whether this privacy mode permits any non-local storage tier."""
    return any(
        tier in REMOTE_STORAGE_TIERS
        for tier in allowed_tiers_for_privacy_mode(privacy_mode)
    )


class TieredStorageManager:
    """
    Orchestrates storage across multiple tiers.

    Responsibilities:
    - Register and manage providers for each tier
    - Route storage requests based on privacy mode
    - Handle fallback when preferred provider unavailable
    - Coordinate sync between tiers
    - Manage cryostasis (agent dormancy) workflow
    """

    def __init__(self, privacy_mode: str = PrivacyMode.NORMAL):
        """
        Initialize the tiered storage manager.

        Args:
            privacy_mode: Current privacy mode (affects tier availability)
        """
        self._providers: Dict[StorageTier, StorageProvider] = {}
        self._privacy_mode = privacy_mode
        self._index: Dict[str, StorageResult] = {}  # In-memory index

    def register_provider(self, tier: StorageTier, provider: StorageProvider) -> None:
        """
        Register a storage provider for a tier.

        Args:
            tier: Storage tier to register for
            provider: Provider instance implementing StorageProvider
        """
        if not isinstance(provider, StorageProvider):
            raise TypeError(f"Provider must implement StorageProvider, got {type(provider)}")

        self._providers[tier] = provider
        logger.info(f"📁 Registered {provider.provider_name} for tier {tier.value}")

    def set_privacy_mode(self, mode: str) -> None:
        """
        Set the current privacy mode.

        Args:
            mode: Privacy mode (ephemeral, isolated, anonymous, normal, public)
        """
        if isinstance(mode, str):
            try:
                PrivacyMode(mode)
            except ValueError:
                raise ValueError(f"Unknown privacy mode: {mode}") from None
        elif mode not in PRIVACY_TIER_POLICY:
            raise ValueError(f"Unknown privacy mode: {mode}")
        self._privacy_mode = mode
        logger.info(f"🔒 Privacy mode set to: {mode}")

    def get_available_tiers(self) -> List[StorageTier]:
        """
        Get available tiers based on privacy mode and provider status.

        Returns:
            List of available StorageTier values
        """
        allowed = allowed_tiers_for_privacy_mode(self._privacy_mode)
        return [
            tier for tier in allowed
            if tier in self._providers and self._providers[tier].is_available()
        ]

    def _select_provider(
        self,
        preferred_tier: Optional[StorageTier] = None,
    ) -> tuple[StorageTier, StorageProvider]:
        """
        Select the best available provider.

        Args:
            preferred_tier: Preferred tier (falls back if unavailable)

        Returns:
            Tuple of (tier, provider)

        Raises:
            RuntimeError: If no providers available
        """
        available = self.get_available_tiers()

        if not available:
            if self._privacy_mode == PrivacyMode.EPHEMERAL:
                raise RuntimeError("Storage disabled in EPHEMERAL mode")
            raise RuntimeError(f"No storage providers available (privacy: {self._privacy_mode})")

        # Try preferred tier first
        if preferred_tier and preferred_tier in available:
            return preferred_tier, self._providers[preferred_tier]

        # Fallback priority: LOCAL > BROWSER > CLOUD_HOT > CLOUD_COLD
        fallback_order = [
            StorageTier.LOCAL,
            StorageTier.BROWSER,
            StorageTier.CLOUD_HOT,
            StorageTier.CLOUD_COLD,
        ]

        for tier in fallback_order:
            if tier in available:
                logger.info(f"⚠️ Using fallback tier: {tier.value}")
                return tier, self._providers[tier]

        raise RuntimeError("No providers available after fallback")

    async def store(
        self,
        content: bytes,
        preferred_tier: Optional[StorageTier] = None,
        metadata: Optional[Dict[str, Any]] = None,
        encrypt: Optional[bool] = None,
    ) -> StorageResult:
        """
        Store content using the best available provider.

        Args:
            content: Content bytes to store
            preferred_tier: Preferred storage tier
            metadata: Optional metadata
            encrypt: Whether to encrypt (default based on privacy mode)

        Returns:
            StorageResult with storage details
        """
        # Determine encryption based on privacy mode
        if encrypt is None:
            encrypt = self._privacy_mode in [PrivacyMode.ANONYMOUS, PrivacyMode.NORMAL]

        # Force encryption in ANONYMOUS mode
        if self._privacy_mode == PrivacyMode.ANONYMOUS:
            encrypt = True

        tier, provider = self._select_provider(preferred_tier)

        result = await provider.store(content, metadata, encrypt)

        # Update in-memory index
        self._index[result.content_hash] = result

        logger.info(f"📤 Stored {result.size_bytes} bytes -> {tier.value} ({provider.provider_name})")
        return result

    async def retrieve(
        self,
        content_hash: str,
        cid: Optional[str] = None,
        encryption_key_hash: Optional[str] = None,
    ) -> bytes:
        """
        Retrieve content from any available tier.

        Tries local cache first, then IPFS, then cloud.

        Args:
            content_hash: SHA256 hash of content
            cid: Optional IPFS CID (for direct retrieval)
            encryption_key_hash: Key hash if encrypted

        Returns:
            Content bytes
        """
        # Check in-memory index for storage location
        if content_hash in self._index:
            result = self._index[content_hash]
            cid = cid or result.cid
            encryption_key_hash = encryption_key_hash or result.encryption_key_hash

        # Try providers in order: LOCAL → CLOUD_HOT → CLOUD_COLD
        retrieval_order = [
            StorageTier.LOCAL,
            StorageTier.CLOUD_HOT,
            StorageTier.CLOUD_COLD,
        ]

        errors = []
        for tier in retrieval_order:
            if tier not in self._providers:
                continue

            provider = self._providers[tier]
            if not provider.is_available():
                continue

            try:
                # Use CID if available, otherwise content hash
                lookup_key = cid or content_hash
                content = await provider.retrieve(lookup_key, encryption_key_hash)
                logger.info(f"📥 Retrieved from {tier.value}")
                return content
            except Exception as e:
                errors.append(f"{tier.value}: {e}")
                continue

        raise ValueError(f"Content not found: {content_hash}. Errors: {errors}")

    async def sync(
        self,
        source_tier: StorageTier,
        target_tier: StorageTier,
    ) -> SyncManifest:
        """
        Sync content between tiers.

        Args:
            source_tier: Source storage tier
            target_tier: Target storage tier

        Returns:
            SyncManifest with results
        """
        if source_tier not in self._providers or target_tier not in self._providers:
            raise ValueError("Both source and target tiers must have registered providers")

        source = self._providers[source_tier]
        target = self._providers[target_tier]

        if not source.is_available() or not target.is_available():
            raise ConnectionError("Both source and target providers must be available")

        # Get content from source
        source_content = await source.list_content()

        # Create sync manifest
        manifest = SyncManifest(
            source_tier=source_tier,
            target_tier=target_tier,
            items=[],
            total_bytes=0,
        )

        # Check what needs to be synced
        for result in source_content:
            manifest.items.append(SyncItem(
                content_hash=result.content_hash,
                source_tier=source_tier,
                target_tier=target_tier,
                size_bytes=result.size_bytes,
                status=SyncStatus.PENDING,
            ))
            manifest.total_bytes += result.size_bytes

        # Estimate cost
        if hasattr(target, "estimate_cost"):
            manifest.estimated_cost_usd = await target.estimate_cost(manifest.total_bytes)

        # Execute sync
        for item in manifest.items:
            try:
                item.status = SyncStatus.IN_PROGRESS

                # Retrieve from source
                content = await source.retrieve(item.content_hash)

                # Store to target
                await target.store(content)

                item.status = SyncStatus.COMPLETED
                logger.info(f"✅ Synced {item.content_hash[:16]}... to {target_tier.value}")

            except Exception as e:
                item.status = SyncStatus.FAILED
                item.error_message = str(e)
                logger.error(f"❌ Sync failed for {item.content_hash[:16]}...: {e}")

        return manifest

    async def list_all_content(self) -> List[StorageResult]:
        """
        List all content across all available tiers.

        Returns:
            Merged list of StorageResult from all tiers
        """
        all_content: Dict[str, StorageResult] = {}

        for tier, provider in self._providers.items():
            if not provider.is_available():
                continue

            try:
                content = await provider.list_content()
                for result in content:
                    # Deduplicate by content hash, prefer higher tier
                    if result.content_hash not in all_content:
                        all_content[result.content_hash] = result
            except Exception as e:
                logger.warning(f"Failed to list content from {tier.value}: {e}")

        return list(all_content.values())

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics across all providers.

        Returns:
            Aggregated statistics
        """
        stats = {
            "privacy_mode": self._privacy_mode,
            "available_tiers": [t.value for t in self.get_available_tiers()],
            "providers": {},
        }

        for tier, provider in self._providers.items():
            try:
                provider_stats = await provider.get_stats()
                stats["providers"][tier.value] = provider_stats
            except Exception as e:
                stats["providers"][tier.value] = {"error": str(e)}

        return stats

    # =========================================================================
    # Cryostasis (Agent Dormancy) Support
    # =========================================================================

    async def initiate_cryostasis(
        self,
        agent_id: str,
        state_snapshot: bytes,
        metadata: Dict[str, Any],
    ) -> StorageResult:
        """
        Archive agent to permanent storage for cryostasis.

        Called when agent's wallet balance falls below threshold.

        Args:
            agent_id: Agent's unique identifier
            state_snapshot: Serialized agent state
            metadata: Agent metadata

        Returns:
            StorageResult with archive CID
        """
        # Find a cryostasis-capable provider (Lighthouse)
        cryostasis_provider = None
        for tier in [StorageTier.CLOUD_COLD, StorageTier.CLOUD_HOT]:
            if tier in self._providers:
                provider = self._providers[tier]
                if isinstance(provider, CryostasisCapable):
                    cryostasis_provider = provider
                    break

        if not cryostasis_provider:
            raise RuntimeError("No cryostasis-capable provider available")

        logger.info(f"🧊 Initiating cryostasis for agent {agent_id}")
        return await cryostasis_provider.archive_for_cryostasis(
            agent_id, state_snapshot, metadata
        )

    async def restore_from_cryostasis(
        self,
        cid: str,
        encryption_key_hash: str,
    ) -> bytes:
        """
        Restore agent from cryostasis.

        Called when agent is funded to wake up.

        Args:
            cid: Archive CID
            encryption_key_hash: Encryption key hash

        Returns:
            Decrypted agent state
        """
        # Find cryostasis-capable provider
        for tier in [StorageTier.CLOUD_HOT, StorageTier.CLOUD_COLD]:
            if tier in self._providers:
                provider = self._providers[tier]
                if isinstance(provider, CryostasisCapable):
                    logger.info(f"🌡️ Restoring agent from cryostasis: {cid}")
                    return await provider.restore_from_cryostasis(cid, encryption_key_hash)

        raise RuntimeError("No cryostasis-capable provider available")

    async def calculate_cryostasis_trigger(self, state_size_bytes: int) -> Decimal:
        """
        Calculate the wallet balance trigger for cryostasis.

        Args:
            state_size_bytes: Size of agent state

        Returns:
            Minimum balance in USD before cryostasis triggers
        """
        for tier in [StorageTier.CLOUD_COLD, StorageTier.CLOUD_HOT]:
            if tier in self._providers:
                provider = self._providers[tier]
                if isinstance(provider, CryostasisCapable):
                    return await provider.calculate_cryostasis_cost(state_size_bytes)

        # Default fallback (enough for ~1GB perpetual via Lighthouse endowment pool)
        return Decimal("5.00")


def create_default_manager(
    privacy_mode: str = PrivacyMode.NORMAL,
    lighthouse_api_key: Optional[str] = None,
) -> TieredStorageManager:
    """
    Create a TieredStorageManager with default cloud storage providers.

    Registers available providers in priority order:
    - Filebase (CLOUD_HOT): S3-compatible IPFS storage when FILEBASE_* env vars are set
    - Lighthouse (CLOUD_HOT + CLOUD_COLD): fallback / legacy

    Args:
        privacy_mode:       Initial privacy mode
        lighthouse_api_key: Lighthouse API key (or from LIGHTHOUSE_API_KEY env var)

    Returns:
        Configured TieredStorageManager
    """
    import os

    manager = TieredStorageManager(privacy_mode=privacy_mode)

    # Filebase (CLOUD_HOT) — S3-compatible IPFS storage
    if os.environ.get("FILEBASE_API_KEY") and os.environ.get("FILEBASE_API_KEY_SECRET"):
        try:
            from kestrel_sovereign.storage.providers.filebase_provider import FilebaseProvider
            filebase = FilebaseProvider()
            if filebase.is_available():
                manager.register_provider(StorageTier.CLOUD_HOT, filebase)
                logger.info("TieredStorageManager: registered FilebaseProvider for CLOUD_HOT")
        except Exception as e:
            logger.warning(f"TieredStorageManager: FilebaseProvider init failed: {e}")

    # Lighthouse — CLOUD_HOT fallback (if Filebase not available) + CLOUD_COLD
    try:
        from kestrel_sovereign.storage.providers.lighthouse_provider import LighthouseProvider
        lighthouse = LighthouseProvider(api_key=lighthouse_api_key)
        if lighthouse.is_available():
            if StorageTier.CLOUD_HOT not in manager._providers:
                manager.register_provider(StorageTier.CLOUD_HOT, lighthouse)
            manager.register_provider(StorageTier.CLOUD_COLD, lighthouse)
            logger.info("TieredStorageManager: registered LighthouseProvider")
    except Exception as e:
        logger.warning(f"TieredStorageManager: LighthouseProvider init failed: {e}")

    # LOCAL runtime storage is owned by AsyncStorage/SQLite (or Postgres in
    # advanced deployments), not this content-provider registry. BROWSER storage
    # is client-side IndexedDB/SQLite-WASM territory and is intentionally not
    # registered by the Python server.

    return manager
