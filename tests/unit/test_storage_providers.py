"""
Unit tests for Storage Provider Protocol and Implementations.

Tests:
1. StorageResult dataclass serialization
2. StorageProvider ABC interface compliance
3. LighthouseProvider (with mocked SDK)
4. TieredStorageManager fallback logic
5. SyncProtocol manifest generation

Note: These are unit tests with mocked external dependencies.
For real integration tests, see tests/integration/test_sovereignty_*.py
"""

import hashlib
import json
import pytest
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import test subjects
from kestrel_sovereign.storage.providers.base import (
    CryostasisCapable,
    MultiCurrencyPayment,
    StorageProvider,
    StorageResult,
    StorageTier,
    SyncItem,
    SyncManifest,
    SyncStatus,
)
from kestrel_sovereign.storage.tiered_manager import (
    PrivacyMode,
    PRIVACY_TIER_POLICY,
    TieredStorageManager,
)
from kestrel_sovereign.storage.sync_protocol import (
    ConflictResolution,
    SyncConflict,
    SyncProgress,
    SyncProtocol,
    BrowserSyncState,
)


# =============================================================================
# StorageResult Tests
# =============================================================================

class TestStorageResult:
    """Tests for StorageResult dataclass."""

    def test_storage_result_creation(self):
        """Test creating a basic StorageResult."""
        result = StorageResult(
            content_hash="abc123",
            cid="QmTest123",
            tier=StorageTier.CLOUD_HOT,
            provider="lighthouse",
            size_bytes=1024,
        )

        assert result.content_hash == "abc123"
        assert result.cid == "QmTest123"
        assert result.tier == StorageTier.CLOUD_HOT
        assert result.provider == "lighthouse"
        assert result.size_bytes == 1024
        assert result.encrypted is False

    def test_storage_result_to_dict(self):
        """Test serializing StorageResult to dict."""
        now = datetime.now(timezone.utc)
        result = StorageResult(
            content_hash="abc123",
            cid="QmTest123",
            tier=StorageTier.LOCAL,
            provider="local",
            size_bytes=2048,
            encrypted=True,
            encryption_key_hash="key_hash_xyz",
            filename="test.txt",
            content_type="text/plain",
            storage_cost_usd=Decimal("0.01"),
            created_at=now,
        )

        data = result.to_dict()

        assert data["content_hash"] == "abc123"
        assert data["cid"] == "QmTest123"
        assert data["tier"] == "local"
        assert data["provider"] == "local"
        assert data["size_bytes"] == 2048
        assert data["encrypted"] is True
        assert data["encryption_key_hash"] == "key_hash_xyz"
        assert data["filename"] == "test.txt"
        assert data["content_type"] == "text/plain"
        assert data["storage_cost_usd"] == "0.01"

    def test_storage_result_from_dict(self):
        """Test deserializing StorageResult from dict."""
        data = {
            "content_hash": "abc123",
            "cid": "QmTest123",
            "tier": "cloud_hot",
            "provider": "lighthouse",
            "size_bytes": 4096,
            "encrypted": True,
            "encryption_key_hash": "key123",
            "filename": "data.bin",
            "content_type": "application/octet-stream",
            "storage_cost_usd": "0.05",
            "created_at": "2025-01-01T00:00:00",
        }

        result = StorageResult.from_dict(data)

        assert result.content_hash == "abc123"
        assert result.tier == StorageTier.CLOUD_HOT
        assert result.size_bytes == 4096
        assert result.storage_cost_usd == Decimal("0.05")

    def test_storage_result_roundtrip(self):
        """Test to_dict/from_dict roundtrip preserves data."""
        original = StorageResult(
            content_hash="xyz789",
            cid="QmRoundTrip",
            tier=StorageTier.CLOUD_COLD,
            provider="lighthouse",
            deal_id="deal123",
            deal_status="active",
            encrypted=True,
            encryption_key_hash="keyhash",
            size_bytes=8192,
            content_type="application/json",
            filename="state.json",
            storage_cost_usd=Decimal("0.001"),
        )

        # Roundtrip
        data = original.to_dict()
        restored = StorageResult.from_dict(data)

        assert restored.content_hash == original.content_hash
        assert restored.cid == original.cid
        assert restored.tier == original.tier
        assert restored.deal_id == original.deal_id
        assert restored.encrypted == original.encrypted
        assert restored.size_bytes == original.size_bytes
        assert restored.storage_cost_usd == original.storage_cost_usd


# =============================================================================
# StorageTier Tests
# =============================================================================

class TestStorageTier:
    """Tests for StorageTier enum."""

    def test_tier_values(self):
        """Test tier enum has expected values."""
        assert StorageTier.BROWSER.value == "browser"
        assert StorageTier.LOCAL.value == "local"
        assert StorageTier.CLOUD_HOT.value == "cloud_hot"
        assert StorageTier.CLOUD_COLD.value == "cloud_cold"

    def test_legacy_tier_values(self):
        """Test legacy tier values for backward compatibility."""
        assert StorageTier.LOCAL_ONLY.value == "local_only"
        assert StorageTier.IPFS.value == "ipfs"
        assert StorageTier.FILECOIN.value == "filecoin"


# =============================================================================
# SyncStatus Tests
# =============================================================================

class TestSyncStatus:
    """Tests for SyncStatus enum."""

    def test_sync_status_values(self):
        """Test all sync statuses exist."""
        assert SyncStatus.PENDING.value == "pending"
        assert SyncStatus.IN_PROGRESS.value == "in_progress"
        assert SyncStatus.COMPLETED.value == "completed"
        assert SyncStatus.FAILED.value == "failed"
        assert SyncStatus.CONFLICT.value == "conflict"


# =============================================================================
# Mock Storage Provider for Testing
# =============================================================================

class MockStorageProvider(StorageProvider):
    """Mock storage provider for testing."""

    def __init__(self, tier: StorageTier = StorageTier.LOCAL, available: bool = True):
        self._tier = tier
        self._available = available
        self._storage: Dict[str, bytes] = {}
        self._results: Dict[str, StorageResult] = {}

    @property
    def tier(self) -> StorageTier:
        return self._tier

    @property
    def provider_name(self) -> str:
        return f"mock_{self._tier.value}"

    def is_available(self) -> bool:
        return self._available

    async def store(
        self,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        encrypt: bool = True,
    ) -> StorageResult:
        content_hash = hashlib.sha256(content).hexdigest()
        self._storage[content_hash] = content

        result = StorageResult(
            content_hash=content_hash,
            cid=f"Qm{content_hash[:32]}",
            tier=self._tier,
            provider=self.provider_name,
            size_bytes=len(content),
            encrypted=encrypt,
        )
        self._results[content_hash] = result
        return result

    async def retrieve(self, cid: str, encryption_key_hash: Optional[str] = None) -> bytes:
        # Try by CID first
        for hash_id, content in self._storage.items():
            if f"Qm{hash_id[:32]}" == cid or hash_id == cid:
                return content
        raise FileNotFoundError(f"Content not found: {cid}")

    async def list_content(self, limit: int = 100, offset: int = 0) -> List[StorageResult]:
        return list(self._results.values())[offset:offset + limit]

    async def delete(self, cid: str) -> bool:
        for hash_id in list(self._storage.keys()):
            if f"Qm{hash_id[:32]}" == cid:
                del self._storage[hash_id]
                del self._results[hash_id]
                return True
        return False

    async def verify(self, cid: str) -> bool:
        for hash_id in self._storage.keys():
            if f"Qm{hash_id[:32]}" == cid:
                return True
        return False


class MockCryostasisProvider(MockStorageProvider, CryostasisCapable):
    """Mock provider that supports cryostasis."""

    async def archive_for_cryostasis(
        self,
        agent_id: str,
        state_snapshot: bytes,
        metadata: Dict[str, Any],
    ) -> StorageResult:
        result = await self.store(state_snapshot, metadata)
        result.deal_status = "cryostasis"
        return result

    async def restore_from_cryostasis(self, cid: str, encryption_key_hash: str) -> bytes:
        return await self.retrieve(cid)

    async def calculate_cryostasis_cost(self, size_bytes: int) -> Decimal:
        size_gb = Decimal(size_bytes) / Decimal(1024 * 1024 * 1024)
        return size_gb * Decimal("0.00005") + Decimal("0.01")


# =============================================================================
# StorageProvider ABC Tests
# =============================================================================

class TestStorageProviderABC:
    """Tests for StorageProvider abstract base class."""

    @pytest.mark.asyncio
    async def test_mock_provider_implements_interface(self):
        """Test that MockStorageProvider correctly implements the interface."""
        provider = MockStorageProvider(StorageTier.LOCAL)

        # Check properties
        assert provider.tier == StorageTier.LOCAL
        assert provider.provider_name == "mock_local"
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        """Test basic store and retrieve operations."""
        provider = MockStorageProvider(StorageTier.LOCAL)
        content = b"Hello, Kestrel!"

        # Store
        result = await provider.store(content)

        assert result.content_hash == hashlib.sha256(content).hexdigest()
        assert result.size_bytes == len(content)
        assert result.tier == StorageTier.LOCAL

        # Retrieve by CID
        retrieved = await provider.retrieve(result.cid)
        assert retrieved == content

    @pytest.mark.asyncio
    async def test_list_content(self):
        """Test listing stored content."""
        provider = MockStorageProvider()

        # Store multiple items
        await provider.store(b"item 1")
        await provider.store(b"item 2")
        await provider.store(b"item 3")

        # List
        items = await provider.list_content()
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_delete_content(self):
        """Test deleting content."""
        provider = MockStorageProvider()

        result = await provider.store(b"delete me")
        assert await provider.verify(result.cid) is True

        # Delete
        deleted = await provider.delete(result.cid)
        assert deleted is True

        # Verify deleted
        assert await provider.verify(result.cid) is False

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting provider stats."""
        provider = MockStorageProvider(StorageTier.CLOUD_HOT)
        stats = await provider.get_stats()

        assert stats["tier"] == "cloud_hot"
        assert stats["provider"] == "mock_cloud_hot"
        assert stats["available"] is True


# =============================================================================
# CryostasisCapable Tests
# =============================================================================

class TestCryostasisCapable:
    """Tests for CryostasisCapable mixin."""

    @pytest.mark.asyncio
    async def test_archive_for_cryostasis(self):
        """Test archiving agent for cryostasis."""
        provider = MockCryostasisProvider(StorageTier.CLOUD_COLD)

        state = b'{"memory": [], "graph": {}}'
        metadata = {"agent_did": "did:test:agent123"}

        result = await provider.archive_for_cryostasis(
            agent_id="agent123",
            state_snapshot=state,
            metadata=metadata,
        )

        assert result.deal_status == "cryostasis"
        assert result.size_bytes == len(state)

    @pytest.mark.asyncio
    async def test_restore_from_cryostasis(self):
        """Test restoring agent from cryostasis."""
        provider = MockCryostasisProvider(StorageTier.CLOUD_COLD)

        state = b'{"memory": [], "graph": {}}'
        result = await provider.archive_for_cryostasis(
            agent_id="agent123",
            state_snapshot=state,
            metadata={},
        )

        # Restore
        restored = await provider.restore_from_cryostasis(
            cid=result.cid,
            encryption_key_hash="key123",
        )

        assert restored == state

    @pytest.mark.asyncio
    async def test_calculate_cryostasis_cost(self):
        """Test cryostasis cost calculation."""
        provider = MockCryostasisProvider()

        # 1 GB
        cost = await provider.calculate_cryostasis_cost(1024 * 1024 * 1024)

        # Should be ~$0.00005/GB + $0.01 buffer
        assert cost > Decimal("0.01")
        assert cost < Decimal("0.02")


# =============================================================================
# Privacy Mode Policy Tests
# =============================================================================

class TestPrivacyModePolicy:
    """Tests for privacy mode to tier mapping."""

    def test_ephemeral_mode_no_storage(self):
        """Test EPHEMERAL mode disables all storage."""
        assert PRIVACY_TIER_POLICY[PrivacyMode.EPHEMERAL] == []

    def test_isolated_mode_local_only(self):
        """Test ISOLATED mode restricts to local storage."""
        allowed = PRIVACY_TIER_POLICY[PrivacyMode.ISOLATED]
        assert StorageTier.LOCAL in allowed
        assert StorageTier.BROWSER in allowed
        assert StorageTier.CLOUD_HOT not in allowed
        assert StorageTier.CLOUD_COLD not in allowed

    def test_anonymous_mode_allows_cloud(self):
        """Test ANONYMOUS mode allows cloud hot storage."""
        allowed = PRIVACY_TIER_POLICY[PrivacyMode.ANONYMOUS]
        assert StorageTier.CLOUD_HOT in allowed
        assert StorageTier.CLOUD_COLD not in allowed

    def test_normal_mode_full_access(self):
        """Test NORMAL mode allows all tiers."""
        allowed = PRIVACY_TIER_POLICY[PrivacyMode.NORMAL]
        assert StorageTier.LOCAL in allowed
        assert StorageTier.BROWSER in allowed
        assert StorageTier.CLOUD_HOT in allowed
        assert StorageTier.CLOUD_COLD in allowed


# =============================================================================
# TieredStorageManager Tests
# =============================================================================

class TestTieredStorageManager:
    """Tests for TieredStorageManager."""

    def test_manager_creation(self):
        """Test creating a manager."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)
        assert manager._privacy_mode == PrivacyMode.NORMAL

    def test_register_provider(self):
        """Test registering a provider."""
        manager = TieredStorageManager()
        provider = MockStorageProvider(StorageTier.LOCAL)

        manager.register_provider(StorageTier.LOCAL, provider)

        assert StorageTier.LOCAL in manager._providers

    def test_register_invalid_provider_raises(self):
        """Test registering non-provider raises TypeError."""
        manager = TieredStorageManager()

        with pytest.raises(TypeError):
            manager.register_provider(StorageTier.LOCAL, "not a provider")

    def test_set_privacy_mode(self):
        """Test changing privacy mode."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        manager.set_privacy_mode(PrivacyMode.ISOLATED)

        assert manager._privacy_mode == PrivacyMode.ISOLATED

    def test_set_invalid_privacy_mode_raises(self):
        """Test setting invalid privacy mode raises ValueError."""
        manager = TieredStorageManager()

        with pytest.raises(ValueError):
            manager.set_privacy_mode("invalid_mode")

    def test_get_available_tiers(self):
        """Test getting available tiers based on privacy and providers."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        # No providers registered
        assert manager.get_available_tiers() == []

        # Register LOCAL provider
        manager.register_provider(StorageTier.LOCAL, MockStorageProvider(StorageTier.LOCAL))

        available = manager.get_available_tiers()
        assert StorageTier.LOCAL in available

    def test_ephemeral_mode_disables_all_tiers(self):
        """Test EPHEMERAL mode returns no available tiers."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.EPHEMERAL)
        manager.register_provider(StorageTier.LOCAL, MockStorageProvider(StorageTier.LOCAL))

        assert manager.get_available_tiers() == []

    @pytest.mark.asyncio
    async def test_store_with_fallback(self):
        """Test storing with fallback to available tier."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        local = MockStorageProvider(StorageTier.LOCAL)
        manager.register_provider(StorageTier.LOCAL, local)

        content = b"test data"
        result = await manager.store(content)

        assert result.tier == StorageTier.LOCAL
        assert result.size_bytes == len(content)

    @pytest.mark.asyncio
    async def test_store_respects_preferred_tier(self):
        """Test store uses preferred tier when available."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        local = MockStorageProvider(StorageTier.LOCAL)
        cloud = MockStorageProvider(StorageTier.CLOUD_HOT)

        manager.register_provider(StorageTier.LOCAL, local)
        manager.register_provider(StorageTier.CLOUD_HOT, cloud)

        result = await manager.store(b"data", preferred_tier=StorageTier.CLOUD_HOT)

        assert result.tier == StorageTier.CLOUD_HOT

    @pytest.mark.asyncio
    async def test_store_fallback_when_preferred_unavailable(self):
        """Test fallback when preferred tier unavailable."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        local = MockStorageProvider(StorageTier.LOCAL)
        cloud = MockStorageProvider(StorageTier.CLOUD_HOT, available=False)

        manager.register_provider(StorageTier.LOCAL, local)
        manager.register_provider(StorageTier.CLOUD_HOT, cloud)

        # Prefer unavailable cloud, should fallback to local
        result = await manager.store(b"data", preferred_tier=StorageTier.CLOUD_HOT)

        assert result.tier == StorageTier.LOCAL

    @pytest.mark.asyncio
    async def test_store_ephemeral_raises(self):
        """Test storage fails in EPHEMERAL mode."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.EPHEMERAL)
        manager.register_provider(StorageTier.LOCAL, MockStorageProvider(StorageTier.LOCAL))

        with pytest.raises(RuntimeError, match="EPHEMERAL"):
            await manager.store(b"data")

    @pytest.mark.asyncio
    async def test_anonymous_mode_forces_encryption(self):
        """Test ANONYMOUS mode forces encryption."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.ANONYMOUS)
        manager.register_provider(StorageTier.LOCAL, MockStorageProvider(StorageTier.LOCAL))

        # Even with encrypt=False, ANONYMOUS should force encryption
        result = await manager.store(b"sensitive", encrypt=False)

        # The storage operation should have encrypt=True
        assert result.encrypted is True

    @pytest.mark.asyncio
    async def test_retrieve_from_local(self):
        """Test retrieving content."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        local = MockStorageProvider(StorageTier.LOCAL)
        manager.register_provider(StorageTier.LOCAL, local)

        content = b"retrieve me"
        result = await manager.store(content)

        retrieved = await manager.retrieve(result.content_hash, cid=result.cid)

        assert retrieved == content

    @pytest.mark.asyncio
    async def test_sync_between_tiers(self):
        """Test syncing content between tiers."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        local = MockStorageProvider(StorageTier.LOCAL)
        cloud = MockStorageProvider(StorageTier.CLOUD_HOT)

        manager.register_provider(StorageTier.LOCAL, local)
        manager.register_provider(StorageTier.CLOUD_HOT, cloud)

        # Store in local
        await manager.store(b"sync me", preferred_tier=StorageTier.LOCAL)

        # Sync to cloud
        manifest = await manager.sync(StorageTier.LOCAL, StorageTier.CLOUD_HOT)

        assert len(manifest.items) == 1
        assert manifest.items[0].status == SyncStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_list_all_content(self):
        """Test listing content across all tiers."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        local = MockStorageProvider(StorageTier.LOCAL)
        cloud = MockStorageProvider(StorageTier.CLOUD_HOT)

        manager.register_provider(StorageTier.LOCAL, local)
        manager.register_provider(StorageTier.CLOUD_HOT, cloud)

        await manager.store(b"local item", preferred_tier=StorageTier.LOCAL)
        await manager.store(b"cloud item", preferred_tier=StorageTier.CLOUD_HOT)

        all_content = await manager.list_all_content()

        # Should have 2 unique items
        assert len(all_content) == 2

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting aggregated stats."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        local = MockStorageProvider(StorageTier.LOCAL)
        manager.register_provider(StorageTier.LOCAL, local)

        stats = await manager.get_stats()

        assert stats["privacy_mode"] == PrivacyMode.NORMAL
        assert "local" in stats["providers"]


# =============================================================================
# TieredStorageManager Cryostasis Tests
# =============================================================================

class TestTieredStorageManagerCryostasis:
    """Tests for TieredStorageManager cryostasis functionality."""

    @pytest.mark.asyncio
    async def test_initiate_cryostasis(self):
        """Test initiating agent cryostasis."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        cryo_provider = MockCryostasisProvider(StorageTier.CLOUD_COLD)
        manager.register_provider(StorageTier.CLOUD_COLD, cryo_provider)

        state = b'{"agent_state": "test"}'
        result = await manager.initiate_cryostasis(
            agent_id="agent123",
            state_snapshot=state,
            metadata={"did": "did:test:123"},
        )

        assert result.deal_status == "cryostasis"

    @pytest.mark.asyncio
    async def test_restore_from_cryostasis(self):
        """Test restoring from cryostasis."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        cryo_provider = MockCryostasisProvider(StorageTier.CLOUD_COLD)
        manager.register_provider(StorageTier.CLOUD_COLD, cryo_provider)

        state = b'{"agent_state": "frozen"}'
        archive_result = await manager.initiate_cryostasis(
            agent_id="agent456",
            state_snapshot=state,
            metadata={},
        )

        restored = await manager.restore_from_cryostasis(
            cid=archive_result.cid,
            encryption_key_hash="key123",
        )

        assert restored == state

    @pytest.mark.asyncio
    async def test_cryostasis_requires_capable_provider(self):
        """Test cryostasis fails without capable provider."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        # Only register a non-cryostasis provider
        local = MockStorageProvider(StorageTier.LOCAL)
        manager.register_provider(StorageTier.LOCAL, local)

        with pytest.raises(RuntimeError, match="No cryostasis-capable provider"):
            await manager.initiate_cryostasis(
                agent_id="agent789",
                state_snapshot=b"state",
                metadata={},
            )

    @pytest.mark.asyncio
    async def test_calculate_cryostasis_trigger(self):
        """Test calculating cryostasis trigger balance."""
        manager = TieredStorageManager(privacy_mode=PrivacyMode.NORMAL)

        cryo_provider = MockCryostasisProvider(StorageTier.CLOUD_COLD)
        manager.register_provider(StorageTier.CLOUD_COLD, cryo_provider)

        trigger = await manager.calculate_cryostasis_trigger(1024 * 1024)  # 1MB

        assert trigger > Decimal("0")
        assert trigger < Decimal("1")  # Should be very small


# =============================================================================
# SyncProtocol Tests
# =============================================================================

class TestSyncProtocol:
    """Tests for SyncProtocol."""

    @pytest.mark.asyncio
    async def test_generate_manifest(self):
        """Test generating sync manifest."""
        protocol = SyncProtocol()

        source = MockStorageProvider(StorageTier.LOCAL)
        target = MockStorageProvider(StorageTier.CLOUD_HOT)

        # Add items to source
        await source.store(b"item 1")
        await source.store(b"item 2")

        manifest = await protocol.generate_manifest(source, target)

        assert len(manifest.items) == 2
        assert manifest.source_tier == StorageTier.LOCAL
        assert manifest.target_tier == StorageTier.CLOUD_HOT
        assert all(item.status == SyncStatus.PENDING for item in manifest.items)

    @pytest.mark.asyncio
    async def test_generate_manifest_with_existing_content(self):
        """Test manifest skips content that already exists in target."""
        protocol = SyncProtocol()

        source = MockStorageProvider(StorageTier.LOCAL)
        target = MockStorageProvider(StorageTier.CLOUD_HOT)

        # Store same content in both
        content = b"already synced"
        await source.store(content)
        await target.store(content)

        # Store unique content in source
        await source.store(b"needs sync")

        manifest = await protocol.generate_manifest(source, target)

        # Only the unique item should be in manifest
        assert len(manifest.items) == 1

    @pytest.mark.asyncio
    async def test_execute_sync(self):
        """Test executing sync operation."""
        protocol = SyncProtocol()

        source = MockStorageProvider(StorageTier.LOCAL)
        target = MockStorageProvider(StorageTier.CLOUD_HOT)

        await source.store(b"sync item 1")
        await source.store(b"sync item 2")

        manifest = await protocol.generate_manifest(source, target)
        progress = await protocol.execute_sync(manifest, source, target)

        assert progress.completed_items == 2
        assert progress.failed_items == 0
        assert progress.progress_percent == 100.0

    @pytest.mark.asyncio
    async def test_sync_progress_tracking(self):
        """Test progress tracking during sync."""
        progress_updates = []

        def progress_callback(progress: SyncProgress):
            progress_updates.append(progress.progress_percent)

        protocol = SyncProtocol(progress_callback=progress_callback)

        source = MockStorageProvider(StorageTier.LOCAL)
        target = MockStorageProvider(StorageTier.CLOUD_HOT)

        await source.store(b"item")

        manifest = await protocol.generate_manifest(source, target)
        await protocol.execute_sync(manifest, source, target)

        # Should have received at least one progress update
        assert len(progress_updates) >= 1

    def test_conflict_resolution_enum(self):
        """Test conflict resolution strategies exist."""
        assert ConflictResolution.KEEP_LOCAL.value == "keep_local"
        assert ConflictResolution.KEEP_REMOTE.value == "keep_remote"
        assert ConflictResolution.KEEP_NEWER.value == "keep_newer"
        assert ConflictResolution.KEEP_BOTH.value == "keep_both"
        assert ConflictResolution.MANUAL.value == "manual"


# =============================================================================
# SyncProgress Tests
# =============================================================================

class TestSyncProgress:
    """Tests for SyncProgress dataclass."""

    def test_progress_percent_empty(self):
        """Test progress percent with no items."""
        progress = SyncProgress()
        assert progress.progress_percent == 100.0

    def test_progress_percent_partial(self):
        """Test progress percent calculation."""
        progress = SyncProgress(
            total_items=10,
            completed_items=5,
        )
        assert progress.progress_percent == 50.0

    def test_progress_percent_complete(self):
        """Test progress percent at 100%."""
        progress = SyncProgress(
            total_items=10,
            completed_items=10,
        )
        assert progress.progress_percent == 100.0

    def test_elapsed_seconds(self):
        """Test elapsed time calculation."""
        import time
        progress = SyncProgress()
        time.sleep(0.1)
        assert progress.elapsed_seconds >= 0.1

    def test_transfer_rate_zero_elapsed(self):
        """Test transfer rate with zero elapsed time."""
        progress = SyncProgress()
        # Immediately after creation, might have very small elapsed time
        # Rate should not error
        rate = progress.transfer_rate_bps
        assert rate >= 0


# =============================================================================
# BrowserSyncState Tests
# =============================================================================

class TestBrowserSyncState:
    """Tests for BrowserSyncState."""

    def test_state_creation(self):
        """Test creating sync state."""
        state = BrowserSyncState()
        assert state.last_sync is None
        assert len(state.local_content_hashes) == 0

    def test_state_to_dict(self):
        """Test serializing state."""
        state = BrowserSyncState(
            last_sync=datetime(2025, 1, 1),
            local_content_hashes={"hash1", "hash2"},
            pending_uploads=["upload1"],
            pending_downloads=["download1"],
        )

        data = state.to_dict()

        assert data["last_sync"] == "2025-01-01T00:00:00"
        assert set(data["local_content_hashes"]) == {"hash1", "hash2"}
        assert data["pending_uploads"] == ["upload1"]

    def test_state_from_dict(self):
        """Test deserializing state."""
        data = {
            "last_sync": "2025-01-15T12:00:00",
            "local_content_hashes": ["hash1", "hash2"],
            "remote_content_hashes": ["hash3"],
            "pending_uploads": [],
            "pending_downloads": [],
        }

        state = BrowserSyncState.from_dict(data)

        assert state.last_sync == datetime(2025, 1, 15, 12, 0, 0)
        assert "hash1" in state.local_content_hashes
        assert "hash3" in state.remote_content_hashes

    def test_state_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = BrowserSyncState(
            last_sync=datetime(2025, 6, 15),
            local_content_hashes={"a", "b", "c"},
            remote_content_hashes={"x", "y"},
            pending_uploads=["a"],
            pending_downloads=["y"],
        )

        data = original.to_dict()
        restored = BrowserSyncState.from_dict(data)

        assert restored.last_sync == original.last_sync
        assert restored.local_content_hashes == original.local_content_hashes
        assert restored.pending_uploads == original.pending_uploads


# =============================================================================
# SyncItem and SyncManifest Tests
# =============================================================================

class TestSyncItem:
    """Tests for SyncItem dataclass."""

    def test_sync_item_creation(self):
        """Test creating a sync item."""
        item = SyncItem(
            content_hash="abc123",
            source_tier=StorageTier.LOCAL,
            target_tier=StorageTier.CLOUD_HOT,
            size_bytes=1024,
        )

        assert item.status == SyncStatus.PENDING
        assert item.error_message is None


class TestSyncManifest:
    """Tests for SyncManifest dataclass."""

    def test_manifest_creation(self):
        """Test creating a manifest."""
        manifest = SyncManifest(
            source_tier=StorageTier.LOCAL,
            target_tier=StorageTier.CLOUD_HOT,
        )

        assert manifest.total_bytes == 0
        assert len(manifest.items) == 0
        assert manifest.estimated_cost_usd is None

    def test_manifest_with_items(self):
        """Test manifest with items."""
        items = [
            SyncItem("hash1", StorageTier.LOCAL, StorageTier.CLOUD_HOT, 1024),
            SyncItem("hash2", StorageTier.LOCAL, StorageTier.CLOUD_HOT, 2048),
        ]

        manifest = SyncManifest(
            source_tier=StorageTier.LOCAL,
            target_tier=StorageTier.CLOUD_HOT,
            items=items,
            total_bytes=3072,
            estimated_cost_usd=Decimal("0.001"),
        )

        assert len(manifest.items) == 2
        assert manifest.total_bytes == 3072
        assert manifest.estimated_cost_usd == Decimal("0.001")
