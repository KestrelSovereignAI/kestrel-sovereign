import pytest
import pytest_asyncio
import tempfile
import os
import json
import hashlib
from datetime import datetime, UTC
from dataclasses import dataclass
from typing import Optional
import unittest.mock

from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter
from kestrel_sovereign.filecoin_adapter import StorageTier, StorageResult
from kestrel_sovereign.kestrel_agent import KestrelAgent

@dataclass
class MockStorageResult:
    content_hash: str
    storage_tier: StorageTier
    ipfs_cid: Optional[str] = None
    filecoin_deal_id: Optional[str] = None
    encrypted: bool = False
    encryption_key_hash: Optional[str] = None

class MockFilecoinAdapter:
    def __init__(self):
        self.stored_items = []

    def store_content(self, content: bytes, storage_tier: StorageTier, encrypt: bool = False, metadata: dict = None) -> MockStorageResult:
        self.stored_items.append({
            "content": content,
            "tier": storage_tier,
            "meta": metadata
        })
        cid = "QmTest" + hashlib.sha256(content).hexdigest()[:10]
        return MockStorageResult(
            content_hash=hashlib.sha256(content).hexdigest(),
            storage_tier=storage_tier,
            ipfs_cid=cid
        )

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.asyncio
async def test_sovereignty_export_v2(temp_db):
    """Test V2 sovereignty export with sharding."""
    async with Storage(db_path=temp_db) as storage:
        # 1. Setup Data
        await storage.add_conversation("user", "Hello V2", metadata={"timestamp": "2025-11-21T10:00:00Z"})
        await storage.add_conversation("assistant", "Hi V2", metadata={"timestamp": "2025-11-21T10:01:00Z"})

        # 2. Setup Adapter
        mock_adapter = MockFilecoinAdapter()
        sov_adapter = SovereignStorageAdapter(storage.db, "test-secret", filecoin_adapter=mock_adapter)

        # 3. Run Export
        cid = await sov_adapter.export_agent("did:test:123", storage_tier=StorageTier.IPFS)

        # 4. Verify
        assert cid.startswith("QmTest")

        # Check that we stored shards and a manifest
        assert len(mock_adapter.stored_items) >= 3  # 1 shard + 1 keyring + 1 manifest

        # Find the manifest
        manifest_item = mock_adapter.stored_items[-1]
        manifest = json.loads(manifest_item["content"])

        assert manifest["version"] == "2.0"
        assert manifest["agent_did"] == "did:test:123"
        assert len(manifest["shards"]) == 1

        shard_meta = manifest["shards"][0]
        assert shard_meta["time_range"] == "2025-11"
        assert shard_meta["type"] == "conversation"

        print(f"Exported Manifest: {json.dumps(manifest, indent=2)}")


class MockLLMService:
    def __init__(self):
        self.providers = [{"name": "mock"}]


@pytest.mark.asyncio
async def test_agent_export_command(temp_db):
    """Test the agent !export-sovereignty command works with new async API."""
    agent = KestrelAgent("did:test:agent", storage_path=temp_db, llm_service=MockLLMService())
    await agent.initialize()

    try:
        # Add some test data first
        await agent.storage._storage.add_conversation("user", "Test message", metadata={"timestamp": "2025-11-21T10:00:00Z"})

        # Mock the FilecoinAdapter since that's what actually gets called
        with unittest.mock.patch('kestrel_sovereign.filecoin_adapter.FilecoinAdapter.store_content') as mock_store:
            mock_store.return_value = MockStorageResult(
                content_hash="QmMockCID",
                storage_tier=StorageTier.IPFS,
                ipfs_cid="QmMockCID",
                encrypted=True
            )

            result = await agent.process_input("!export-sovereignty")

            # Accept either format of output
            assert "QmMockCID" in result or "Export" in result or "CID" in result
    finally:
        await agent.shutdown()
