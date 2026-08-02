"""Sovereignty export receipt seam contracts."""

import asyncio
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.sovereignty.feature import SovereigntyFeature
from kestrel_sovereign.storage.providers.base import StorageResult, StorageTier


class AuditAnchorFeature:
    def __init__(self):
        # AuditAnchorFeature.anchor_status now returns a ToolResult
        # envelope (#1061 wave 17); the dict the sovereignty receipt
        # quotes lives under .data.
        self.anchor_status = AsyncMock(
            return_value=ToolResult.ok(
                confirmation="audit anchor status",
                data={"cid": "bafyaudit"},
            ),
        )


@pytest.mark.asyncio
async def test_export_persists_receipt_with_import_and_audit_provenance():
    stored_nodes = []
    backup_blob = b"backup-bytes"

    storage = MagicMock()
    storage.create_backup_blob = AsyncMock(return_value=backup_blob)
    storage.record_backup_artifact = AsyncMock(return_value="hash123")
    storage.add_node = AsyncMock(side_effect=lambda node: stored_nodes.append(deepcopy(node)))

    wallet = MagicMock()
    wallet.can_afford.return_value = True
    wallet.transfer = AsyncMock()

    agent = SimpleNamespace(
        agent_id="did:kestrel:test",
        features={"audit": AuditAnchorFeature()},
        storage=storage,
        wallet=wallet,
    )

    storage_result = StorageResult(
        content_hash="hash123",
        cid="bafybackup",
        tier=StorageTier.IPFS,
        provider="ipfs",
        encrypted=True,
        encryption_key_hash="keyhash123",
        size_bytes=len(backup_blob),
    )
    adapter = MagicMock()
    adapter.store_content.return_value = storage_result

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter",
        return_value=adapter,
    ):
        envelope = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs",
            encrypt=True,
        )

    assert "CID: bafybackup" in envelope.confirmation
    assert envelope.data == {
        "cid": "bafybackup",
        "content_hash": "hash123",
        "tier": "ipfs",
        "tier_requested": "ipfs",
        "encrypted": True,
        "size_bytes": len(backup_blob),
        "node_id": "hash123",
    }
    assert len(stored_nodes) == 1
    receipt = stored_nodes[0]
    assert receipt.node_type == "sovereignty_receipt"
    assert receipt.properties == {
        "cid": "bafybackup",
        "ipfs_cid": "bafybackup",
        "content_hash": "hash123",
        "storage_tier": "ipfs",
        "provider": "ipfs",
        "encrypted": True,
        "encryption_key_hash": "keyhash123",
        "size_bytes": len(backup_blob),
        "created_at": receipt.properties["created_at"],
        "node_id": "hash123",
        "audit_anchors": {"cid": "bafyaudit"},
    }
    storage.record_backup_artifact.assert_awaited_once_with("did:kestrel:test", storage_result)
    wallet.transfer.assert_awaited_once()


@pytest.mark.asyncio
async def test_export_partial_when_tier_falls_back_to_local():
    """When the storage provider downgrades to LOCAL_ONLY (e.g. Lotus
    is unreachable), the envelope must surface the downgrade as
    PARTIAL and the LLM-facing fields must report the *actual* tier,
    not the requested one. Without this, the agent would tell the
    sovereign "your data is on IPFS" when it's actually only on local
    disk.
    """
    storage = MagicMock()
    storage.create_backup_blob = AsyncMock(return_value=b"backup-bytes")
    storage.record_backup_artifact = AsyncMock(return_value="hash123")
    storage.add_node = AsyncMock()

    wallet = MagicMock()
    wallet.can_afford.return_value = True
    wallet.transfer = AsyncMock()

    agent = SimpleNamespace(
        agent_id="did:kestrel:test",
        features={},
        storage=storage,
        wallet=wallet,
    )

    # The real FilecoinAdapter encrypts BEFORE it branches on tier, so a
    # tier fallback to LOCAL_ONLY still carries the requested encryption
    # (encrypted=True). This must remain a tier-downgrade PARTIAL, not an
    # encryption failure (#2872): a mocked ``encrypted=False`` here would be
    # unrealistic and would instead trip the encryption honesty gate.
    storage_result = StorageResult(
        content_hash="hash123",
        cid=None,
        tier=StorageTier.LOCAL_ONLY,
        provider="local",
        encrypted=True,
        encryption_key_hash="keyhash123",
        size_bytes=len(b"backup-bytes"),
    )
    adapter = MagicMock()
    adapter.store_content.return_value = storage_result

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter",
        return_value=adapter,
    ):
        envelope = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs",
            encrypt=True,
        )

    from kestrel_sdk.tools.result import ToolResultStatus
    assert envelope.status is ToolResultStatus.PARTIAL
    # LLM-facing tier must be the actual tier (local), not the requested one (ipfs).
    assert "Tier: local_only" in envelope.confirmation
    assert envelope.data["tier"] == "local_only"
    assert envelope.data["tier_requested"] == "ipfs"
    # Actual encryption is reported, and it was honoured on the local fallback.
    assert envelope.data["encrypted"] is True
    # Caveat names both tiers so the LLM can speak the downgrade.
    assert "requested tier 'ipfs'" in envelope.error
    assert "actual tier is 'local_only'" in envelope.error


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ephemeral", "isolated"])
async def test_export_refused_before_blob_by_real_privacy_storage(mode):
    """A privacy config that forbids cloud backups refuses the export loudly
    and NEVER reads the agent's private state into a blob (#2872 in-scope #3).

    Uses the real ``PrivacyEnforcingStorage`` — a permissive MagicMock would
    hide that production ``create_backup_blob`` raises ``PrivacyViolationError``
    in these modes.
    """
    from kestrel_sdk.tools.result import ToolResultStatus
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    underlying = MagicMock()
    underlying.create_backup_blob = AsyncMock(return_value=b"agent-private-state")
    storage = PrivacyEnforcingStorage(underlying, PrivacyMode(mode))

    agent = SimpleNamespace(
        agent_id="did:kestrel:test",
        features={},
        storage=storage,
        wallet=None,
    )

    adapter = MagicMock()
    adapter.store_content = MagicMock()

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter",
        return_value=adapter,
    ):
        # storage_tier='local' keeps the wallet out of it — the refusal is
        # purely the privacy policy, and it applies to every tier.
        envelope = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="local",
            encrypt=False,
        )

    assert envelope.status is ToolResultStatus.ERROR
    assert "backup" in envelope.error.lower()
    # The private state was never read into a blob, and nothing was stored.
    underlying.create_backup_blob.assert_not_awaited()
    adapter.store_content.assert_not_called()


@pytest.mark.asyncio
async def test_export_fails_loudly_when_encryption_requested_but_not_applied():
    """``encrypt=True`` that yields an unencrypted blob is a hard failure, not
    a silent ``encrypted: False`` under an OK envelope (#2872 Defect 1)."""
    from kestrel_sdk.tools.result import ToolResultStatus

    storage = MagicMock()
    storage.create_backup_blob = AsyncMock(return_value=b"backup-bytes")
    storage.record_backup_artifact = AsyncMock(return_value="node1")
    storage.add_node = AsyncMock()

    wallet = MagicMock()
    wallet.can_afford.return_value = True
    wallet.transfer = AsyncMock()

    agent = SimpleNamespace(
        agent_id="did:kestrel:test",
        features={},
        storage=storage,
        wallet=wallet,
    )

    # A durable off-host provider (so the tier check would pass) that
    # nonetheless returned an UNENCRYPTED blob — the encryption gate must fire.
    storage_result = StorageResult(
        content_hash="hash123",
        cid="bafybackup",
        tier=StorageTier.IPFS,
        provider="lighthouse",
        encrypted=False,
        encryption_key_hash=None,
        size_bytes=len(b"backup-bytes"),
    )
    adapter = MagicMock()
    adapter.store_content.return_value = storage_result

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter",
        return_value=adapter,
    ):
        envelope = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs",
            encrypt=True,
        )

    assert envelope.status is ToolResultStatus.ERROR
    assert "encrypt" in envelope.error.lower()
    # A refused export records no success receipt.
    storage.record_backup_artifact.assert_not_awaited()
    storage.add_node.assert_not_awaited()


@pytest.mark.asyncio
async def test_export_partial_for_local_kubo_ipfs_with_concrete_adapter(tmp_path):
    """A concrete ``FilecoinAdapter`` that pinned to a local Kubo node returns
    ``provider='local'`` with a CID. That is NOT a durable off-host replica, so
    the export must be PARTIAL — durability is validated by provenance, not by
    CID presence alone (#2872 in-scope #2)."""
    from kestrel_sdk.tools.result import ToolResultStatus
    from kestrel_sovereign.filecoin_adapter import FilecoinAdapter

    storage = MagicMock()
    storage.create_backup_blob = AsyncMock(return_value=b"backup-bytes")
    storage.record_backup_artifact = AsyncMock(return_value="node1")
    storage.add_node = AsyncMock()

    wallet = MagicMock()
    wallet.can_afford.return_value = True
    wallet.transfer = AsyncMock()

    agent = SimpleNamespace(
        agent_id="did:kestrel:test",
        features={},
        storage=storage,
        wallet=wallet,
    )

    # Build a real adapter, then stub its network so no Kubo/Lotus is needed.
    # It still returns provider="local" (its default) — a local pin.
    with patch.object(FilecoinAdapter, "_test_ipfs_connection", return_value=False):
        real_adapter = FilecoinAdapter(cache_dir=str(tmp_path))
    real_adapter.ipfs_is_available = lambda: True
    real_adapter._store_ipfs = lambda *a, **k: "bafyfakecid"
    real_adapter._store_local_cache = lambda *a, **k: None

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter",
        return_value=real_adapter,
    ):
        envelope = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs",
            encrypt=False,
        )

    assert envelope.status is ToolResultStatus.PARTIAL
    # The adapter kept the IPFS tier and produced a CID …
    assert envelope.data["tier"] == "ipfs"
    assert envelope.data["cid"] == "bafyfakecid"
    # … but a local-Kubo pin is not durable off-host.
    assert "not a durable off-host replica" in envelope.error.lower()


@pytest.mark.asyncio
async def test_export_does_not_charge_when_paid_tier_downgrades_to_local():
    """A paid-tier request (filecoin) that downgrades to local-only did NOT
    deliver the paid service, so the wallet must NOT be charged (#2872 review).

    The old code debited the wallet before durability was evaluated, so a
    downgraded export was billed for off-host storage it never got.
    """
    from kestrel_sdk.tools.result import ToolResultStatus

    storage = MagicMock()
    storage.create_backup_blob = AsyncMock(return_value=b"backup-bytes")
    storage.record_backup_artifact = AsyncMock(return_value="node1")
    storage.add_node = AsyncMock()

    wallet = MagicMock()
    wallet.can_afford.return_value = True
    wallet.transfer = AsyncMock(return_value=True)

    agent = SimpleNamespace(
        agent_id="did:kestrel:test",
        features={},
        storage=storage,
        wallet=wallet,
    )

    # Filecoin requested, but the provider stack was unreachable and the write
    # fell back to a local-only copy (no deal, no CID, provider="local").
    storage_result = StorageResult(
        content_hash="hash123",
        cid=None,
        tier=StorageTier.LOCAL_ONLY,
        provider="local",
        encrypted=False,
        encryption_key_hash=None,
        size_bytes=len(b"backup-bytes"),
    )
    adapter = MagicMock()
    adapter.store_content.return_value = storage_result

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter",
        return_value=adapter,
    ):
        envelope = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="filecoin",
            encrypt=False,
        )

    # PARTIAL because the durable tier was not achieved …
    assert envelope.status is ToolResultStatus.PARTIAL
    assert envelope.data["tier"] == "local_only"
    assert envelope.data["tier_requested"] == "filecoin"
    # … and CRUCIALLY the paid-tier fee was never charged for the downgrade.
    wallet.transfer.assert_not_awaited()


@pytest.mark.asyncio
async def test_export_fails_when_wallet_transfer_reports_failure():
    """``wallet.transfer`` returns ``False`` on failure; an ignored ``False``
    silently gives away paid storage. A failed charge must fail the export
    loudly rather than return OK (#2872 review)."""
    from kestrel_sdk.tools.result import ToolResultStatus

    storage = MagicMock()
    storage.create_backup_blob = AsyncMock(return_value=b"backup-bytes")
    storage.record_backup_artifact = AsyncMock(return_value="node1")
    storage.add_node = AsyncMock()

    wallet = MagicMock()
    wallet.can_afford.return_value = True
    # The real wallet contract returns a bool; False means the transfer failed.
    wallet.transfer = AsyncMock(return_value=False)

    agent = SimpleNamespace(
        agent_id="did:kestrel:test",
        features={},
        storage=storage,
        wallet=wallet,
    )

    # A genuinely durable Filecoin result (a deal exists), so the tier check
    # passes and the charge is actually attempted.
    storage_result = StorageResult(
        content_hash="hash123",
        cid="bafyfilecoin",
        tier=StorageTier.FILECOIN,
        provider="lighthouse",
        deal_id="deal-123",
        encrypted=False,
        encryption_key_hash=None,
        size_bytes=len(b"backup-bytes"),
    )
    adapter = MagicMock()
    adapter.store_content.return_value = storage_result

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter",
        return_value=adapter,
    ):
        envelope = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="filecoin",
            encrypt=False,
        )

    assert envelope.status is ToolResultStatus.ERROR
    assert "charged" in envelope.error.lower() or "transfer" in envelope.error.lower()
    # The charge was attempted (and reported failure) exactly once …
    wallet.transfer.assert_awaited_once()
    # … and storage truth was recorded (the blob is physically stored).
    storage.record_backup_artifact.assert_awaited_once()


@pytest.mark.asyncio
async def test_export_holds_privacy_lock_until_adapter_thread_drains_on_cancel():
    """Cancelling the export mid-write must NOT release the privacy-transition
    lock while the un-interruptible adapter thread is still writing (#2872
    review). The lock stays held until the drained thread completes, so a
    concurrent flip to EPHEMERAL/ISOLATED cannot land during the write.

    Uses the REAL ``asyncio.Lock`` the agent exposes as its transition lock and
    a genuinely blocking adapter, not a SimpleNamespace stand-in.
    """
    lock = asyncio.Lock()

    storage = MagicMock()
    storage.create_backup_blob = AsyncMock(return_value=b"backup-bytes")
    storage.record_backup_artifact = AsyncMock(return_value="node1")
    storage.add_node = AsyncMock()

    agent = SimpleNamespace(
        agent_id="did:kestrel:test",
        features={},
        storage=storage,
        wallet=None,
        _get_privacy_transition_lock=lambda: lock,
    )

    started = threading.Event()
    release = threading.Event()

    def blocking_store(*args, **kwargs):
        # Runs on the executor thread that asyncio.to_thread cannot interrupt.
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("release event was never set")
        return StorageResult(
            content_hash="hash123",
            tier=StorageTier.LOCAL_ONLY,
            provider="local",
            encrypted=False,
            size_bytes=len(b"backup-bytes"),
        )

    adapter = MagicMock()
    adapter.store_content = blocking_store

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter",
        return_value=adapter,
    ):
        task = asyncio.create_task(
            SovereigntyFeature(agent).export_sovereignty(
                storage_tier="local",
                encrypt=False,
            )
        )

        # Wait until the adapter has entered the blocking write.
        for _ in range(500):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "adapter store never started"
        assert lock.locked(), "export should hold the transition lock during the write"

        # Cancel mid-write. The synchronous thread keeps running; the feature
        # must drain it before releasing the lock.
        task.cancel()
        await asyncio.sleep(0.05)
        assert lock.locked(), "lock was released while the adapter write was still in flight"

        # Let the write finish: the drain completes, cancellation re-raises,
        # and only THEN is the lock released.
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lock.locked(), "lock must be released once the drained write completes"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FileNotFoundError("missing key material"), "missing key material"),
        (ValueError("rotated key cannot decrypt backup"), "rotated key cannot decrypt backup"),
    ],
)
async def test_import_fails_closed_when_backup_key_material_is_missing_or_rotated(
    error,
    expected,
):
    backup_node = SimpleNamespace(
        node_id="hash123",
        properties={
            "ipfs_cid": "bafybackup",
            "encrypted": True,
            "encryption_key_hash": "keyhash123",
        },
    )
    storage = MagicMock()
    storage.get_nodes_by_type = AsyncMock(return_value=[backup_node])
    storage.restore_from_backup_blob = AsyncMock()

    agent = SimpleNamespace(storage=storage)
    adapter = MagicMock()
    adapter.retrieve_content.side_effect = error

    with patch(
        "kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter",
        return_value=adapter,
    ):
        envelope = await SovereigntyFeature(agent).import_sovereignty("bafybackup")

    from kestrel_sdk.tools.result import ToolResultStatus
    assert envelope.status is ToolResultStatus.ERROR
    assert envelope.error.startswith("❌ Error during import:")
    assert expected in envelope.error
    adapter.retrieve_content.assert_called_once_with(
        "bafybackup",
        ipfs_cid="bafybackup",
        key_hash="keyhash123",
    )
    storage.restore_from_backup_blob.assert_not_awaited()
