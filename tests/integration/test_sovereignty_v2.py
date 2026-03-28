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
from kestrel_sovereign.storage.car_builder import CARReader
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
async def test_sovereignty_export_v3_car(temp_db):
    """Test V3 sovereignty export produces a single CAR archive."""
    async with Storage(db_path=temp_db) as storage:
        # 1. Setup Data
        await storage.add_conversation("user", "Hello V3", metadata={"timestamp": "2025-11-21T10:00:00Z"})
        await storage.add_conversation("assistant", "Hi V3", metadata={"timestamp": "2025-11-21T10:01:00Z"})

        # 2. Setup Adapter
        mock_adapter = MockFilecoinAdapter()
        sov_adapter = SovereignStorageAdapter(storage.db, "test-secret", filecoin_adapter=mock_adapter)

        # 3. Run Export
        cid = await sov_adapter.export_agent("did:test:123", storage_tier=StorageTier.IPFS)

        # 4. Verify single upload (CAR archive)
        assert cid.startswith("QmTest")
        assert len(mock_adapter.stored_items) == 1  # Single CAR blob

        # 5. Parse the CAR archive
        car_bytes = mock_adapter.stored_items[0]["content"]
        reader = CARReader(car_bytes)
        assert reader.verify()

        # 6. Extract manifest from root
        manifest = reader.get_dag_cbor_block(reader.root_cid)
        assert manifest["version"] == "3.0"
        assert manifest["agent_did"] == "did:test:123"
        assert len(manifest["shards"]) == 1
        assert manifest["assets"] == []

        shard_meta = manifest["shards"][0]
        assert shard_meta["time_range"] == "2025-11"
        assert shard_meta["type"] == "conversation"

        # Verify shard block exists in CAR
        shard_block = reader.get_block(shard_meta["cid"])
        assert shard_block is not None

        print(f"Exported CAR: {reader.block_count} blocks, {len(car_bytes)} bytes")


class MockLLMService:
    def __init__(self):
        self.providers = [{"name": "mock"}]

    def get_active_model_id(self):
        return "mock-model"


@pytest.mark.asyncio
async def test_agent_export_command(temp_db, skip_bootstrap):
    """Test the agent !export-sovereignty command works with new async API."""
    agent = KestrelAgent("did:test:agent", storage_path=temp_db, llm_service=MockLLMService())
    await agent.initialize()

    # Skip bootstrap to test commands directly
    await skip_bootstrap(agent)

    try:
        # Add some test data first
        await agent.storage._storage.add_conversation("user", "Test message", metadata={"timestamp": "2025-11-21T10:00:00Z"})

        # Mock the FilecoinAdapter — now receives a single CAR blob
        with unittest.mock.patch('kestrel_sovereign.filecoin_adapter.FilecoinAdapter.store_content') as mock_store:
            mock_store.return_value = MockStorageResult(
                content_hash="QmMockCID",
                storage_tier=StorageTier.IPFS,
                ipfs_cid="QmMockCID",
                encrypted=True
            )

            result = await agent.process_input("!export-sovereignty")

            # Single upload for the whole CAR archive
            assert mock_store.call_count == 1

            # Accept either format of output
            assert "QmMockCID" in result or "Export" in result or "CID" in result
    finally:
        await agent.shutdown()
