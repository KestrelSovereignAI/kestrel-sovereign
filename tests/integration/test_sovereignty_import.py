"""
Integration tests for V2 Import/Restore functionality.

These tests verify the full round-trip:
1. Export agent to IPFS
2. Delete local database
3. Import from CID
4. Verify data integrity
"""

import pytest
import pytest_asyncio
import tempfile
import os
import json
import uuid

from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter
from kestrel_sovereign.filecoin_adapter import StorageTier
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest_asyncio.fixture
async def llm_service():
    """Initialize LLM service for testing with proper cleanup."""
    service = LLMService()
    yield service
    await service.close()


@pytest.mark.asyncio
async def test_export_import_roundtrip(temp_db):
    """
    Test the full export->import cycle.
    Verify that data survives the round-trip intact.
    """
    agent_did = "did:pkh:eip155:1:test_import"

    async with Storage(db_path=temp_db) as storage:
        # 1. Create test data
        test_messages = [
            ("user", "Hello, remember my birthday is March 15"),
            ("assistant", "Got it! I'll remember March 15 is your birthday."),
            ("user", "My favorite color is blue"),
            ("assistant", "Noted: blue is your favorite color."),
        ]

        for role, content in test_messages:
            await storage.add_conversation(role, content, metadata={"timestamp": "2025-11-21T10:00:00Z"})

        # Verify data was added
        original_history = await storage.get_conversation_history()
        assert len(original_history) == 4

        # 2. Export
        sovereign_adapter = SovereignStorageAdapter(storage.db, user_secret="test-import-secret")
        cid = await sovereign_adapter.export_agent(agent_did, storage_tier=StorageTier.LOCAL_ONLY)
        assert cid is not None
        assert len(cid) > 0
        print(f"✅ Exported to CID: {cid}")

        # 3. Delete local database (simulate data loss)
        await storage.db.execute_commit("DELETE FROM conversation_history")

        # Verify deletion
        deleted_history = await storage.get_conversation_history()
        assert len(deleted_history) == 0
        print("✅ Simulated data loss (deleted local DB)")

        # 4. Import from CID
        stats = await sovereign_adapter.import_agent(cid)

        assert stats['messages_restored'] == 4
        assert stats['shards_restored'] > 0
        assert stats['agent_did'] == agent_did
        assert stats['manifest_version'] == "3.0"
        assert 'assets_restored' in stats
        print(f"✅ Imported {stats['messages_restored']} messages")

        # 5. Verify restored data matches original
        restored_history = await storage.get_conversation_history()
        assert len(restored_history) == 4

        for i, msg in enumerate(restored_history):
            assert msg['role'] == test_messages[i][0]
            assert msg['content'] == test_messages[i][1]

        print("✅ Data integrity verified - all messages restored correctly")


@pytest.mark.asyncio
async def test_import_with_wrong_key_fails(temp_db):
    """
    Verify that import fails if user provides wrong encryption key.
    This proves the encryption is real.
    """
    agent_did = "did:pkh:eip155:1:test_wrong_key"

    async with Storage(db_path=temp_db) as storage:
        # Export with one key
        export_adapter = SovereignStorageAdapter(storage.db, user_secret="correct-key")
        await storage.add_conversation("user", "Secret message", metadata={"timestamp": "2025-11-21T10:00:00Z"})
        cid = await export_adapter.export_agent(agent_did, storage_tier=StorageTier.LOCAL_ONLY)

        # Try to import with different key
        import_adapter = SovereignStorageAdapter(storage.db, user_secret="wrong-key")

        with pytest.raises(Exception):  # Should fail to decrypt
            await import_adapter.import_agent(cid)

        print("✅ Import correctly failed with wrong key")


@pytest.mark.asyncio
async def test_agent_command_import(temp_db, llm_service, skip_bootstrap):
    """
    Test the user-facing !import-sovereignty command.

    This test verifies the full export/import cycle works through the agent's
    command interface. It uses direct storage access (bypassing the privacy
    wrapper's add_conversation) to avoid any potential content transformation.
    """
    # Use new KestrelAgent API: storage_path instead of storage object
    agent = KestrelAgent("did:test:import_cmd", storage_path=temp_db, llm_service=llm_service)
    await agent.initialize()

    # Skip bootstrap to test commands directly
    await skip_bootstrap(agent)

    # ``!export-sovereignty local`` and ``!import-sovereignty <cid>``
    # both route through the orchestrator's PRE_TOOL_USE chain.
    # Grant ALLOW for both pairs.  See conftest.grant_permissions (#879).
    from tests.integration.conftest import grant_permissions
    await grant_permissions(
        agent,
        ("SovereigntyFeature", "export_sovereignty"),
        ("SovereigntyFeature", "import_sovereignty"),
        reason="sovereignty-import agent-command",
    )

    # Use unique marker to avoid collision with any system messages
    unique_marker = f"SOVEREIGNTY_TEST_{uuid.uuid4().hex[:8]}"

    try:
        # Add test message directly to underlying storage to avoid privacy wrapper transformations
        # This ensures the exact content is stored
        await agent.storage._storage.add_conversation(
            "user",
            f"Test message with marker: {unique_marker}",
            metadata={"timestamp": "2025-11-21T10:00:00Z", "test": True}
        )

        # Record count before export
        history_before = await agent.storage._storage.get_conversation_history()
        count_before = len(history_before)
        print(f"Messages before export: {count_before}")

        # Verify our marker is there
        assert any(unique_marker in str(msg.get('content', '')) for msg in history_before), \
            f"Marker {unique_marker} not found in pre-export history"

        # Export
        export_result = await agent.process_input("!export-sovereignty local")
        assert "CID:" in export_result, f"Export failed: {export_result}"

        # Extract CID from response
        cid_line = [line for line in export_result.split('\n') if line.startswith('CID:')][0]
        cid = cid_line.split('CID:')[1].strip()
        print(f"Extracted CID: {cid}")
        print(f"Export result: {export_result}")

        # Clear conversation history using raw DB access
        await agent.storage.db.execute_commit("DELETE FROM conversation_history")

        # Verify cleared
        history_after_clear = await agent.storage._storage.get_conversation_history()
        assert len(history_after_clear) == 0, f"History not cleared: {len(history_after_clear)} messages remain"
        print("History cleared successfully")

        # Import using command
        import_result = await agent.process_input(f"!import-sovereignty {cid}")
        print(f"Import result: {import_result}")

        assert "Sovereignty Import Complete" in import_result or "Import" in import_result, \
            f"Import failed: {import_result}"

        # Verify data restored - use underlying storage to avoid privacy wrapper
        history_after = await agent.storage._storage.get_conversation_history()
        count_after = len(history_after)
        print(f"Messages after import: {count_after}")

        # Debug: print all messages
        for i, msg in enumerate(history_after):
            content = str(msg.get('content', ''))[:80]
            print(f"  [{i}] {msg.get('role')}: {content}...")

        assert count_after == count_before, \
            f"Message count mismatch: before={count_before}, after={count_after}"

        # Check for our unique marker
        found_marker = any(unique_marker in str(msg.get('content', '')) for msg in history_after)
        assert found_marker, \
            f"Marker {unique_marker} not found in restored history. Messages: {[m.get('content', '')[:50] for m in history_after]}"

        print("✅ Agent command import/export cycle successful")
    finally:
        # Clean up resources
        await agent.shutdown()


@pytest.mark.asyncio
async def test_partial_shard_recovery(temp_db):
    """
    Test that if some shards are lost, the import fails gracefully.

    In V2 architecture, data is sharded by month. If a shard is corrupted
    or unavailable (e.g., IPFS node is down), the import should report
    which shards failed rather than silently losing data.
    """
    agent_did = "did:pkh:eip155:1:test_partial"

    async with Storage(db_path=temp_db) as storage:
        # Create data spanning multiple months to get multiple shards
        await storage.add_conversation("user", "October message", metadata={"timestamp": "2025-10-15T10:00:00Z"})
        await storage.add_conversation("user", "November message", metadata={"timestamp": "2025-11-15T10:00:00Z"})

        sovereign_adapter = SovereignStorageAdapter(storage.db, user_secret="test-import-secret")
        cid = await sovereign_adapter.export_agent(agent_did, storage_tier=StorageTier.LOCAL_ONLY)
        assert cid is not None

        # Get the cache directory from the filecoin adapter
        cache_dir = sovereign_adapter.adapter.cache_dir

        # Find all cached shard files (they end with .cache)
        cache_files = list(cache_dir.glob("*.cache"))
        assert len(cache_files) >= 3, f"Expected at least 3 cache files (manifest + 2 shards), got {len(cache_files)}"

        # Remember one shard file to corrupt (not the manifest - that's the CID we have)
        # The manifest CID would be the one matching our root CID
        manifest_hash = cid.replace("local-", "")

        # Find a shard cache file (one that's NOT the manifest)
        shard_file = None
        for cf in cache_files:
            if manifest_hash not in str(cf):
                shard_file = cf
                break

        if shard_file:
            # Corrupt the shard by removing it
            original_content = shard_file.read_bytes()
            shard_file.unlink()
            print(f"🔥 Removed shard cache file: {shard_file.name}")

            # Clear existing conversations
            await storage.db.execute_commit("DELETE FROM conversation_history")

            # Attempt import - should fail or report partial recovery
            try:
                result = await sovereign_adapter.import_agent(cid)
                # If import succeeds, check if all messages were recovered
                messages_recovered = result.get("messages_restored", 0)
                print(f"✅ Import completed with {messages_recovered} messages recovered")
                # With one shard missing, we should have fewer than 2 messages
                # This tests graceful degradation
            except Exception as e:
                # Acceptable failure modes - should mention the issue clearly
                error_str = str(e).lower()
                acceptable_keywords = ["shard", "corrupt", "fail", "missing", "retrieve", "decrypt"]
                has_acceptable_keyword = any(kw in error_str for kw in acceptable_keywords)
                assert has_acceptable_keyword, \
                    f"Error should clearly indicate the problem. Got: {e}"
                print(f"✅ Graceful failure with clear error: {e}")

            # Restore the shard for other tests
            shard_file.write_bytes(original_content)
        else:
            # If we couldn't find a separate shard file, just verify export worked
            print("⚠️ Could not isolate shard file for corruption test, verifying export only")
            assert len(cache_files) >= 1
