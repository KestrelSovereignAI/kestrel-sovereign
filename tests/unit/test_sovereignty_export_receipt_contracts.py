"""Sovereignty export receipt seam contracts."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.features.sovereignty.feature import SovereigntyFeature
from kestrel_sovereign.storage.providers.base import StorageResult, StorageTier


class AuditAnchorFeature:
    def __init__(self):
        self.anchor_status = AsyncMock(return_value={"result": {"cid": "bafyaudit"}})


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
        result = await SovereigntyFeature(agent).export_sovereignty(
            storage_tier="ipfs",
            encrypt=True,
        )

    assert "CID: bafybackup" in result
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
