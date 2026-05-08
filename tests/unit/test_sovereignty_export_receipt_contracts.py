"""Sovereignty export receipt seam contracts."""

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
            storage_tier="ipfs",
            encrypt=True,
        )

    from kestrel_sdk.tools.result import ToolResultStatus
    assert envelope.status is ToolResultStatus.PARTIAL
    # LLM-facing tier must be the actual tier (local), not the requested one (ipfs).
    assert "Tier: local_only" in envelope.confirmation
    assert envelope.data["tier"] == "local_only"
    assert envelope.data["tier_requested"] == "ipfs"
    # Caveat names both tiers so the LLM can speak the downgrade.
    assert "requested tier 'ipfs'" in envelope.error
    assert "actual tier is 'local_only'" in envelope.error


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
