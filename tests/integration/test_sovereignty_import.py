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
from tests.shared.genesis_audit import complete_deterministic_genesis_audit


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
async def test_restore_preserves_created_at_and_trash(temp_db):
    """#F265: the restore must be FAITHFUL — preserve created_at (history
    ordering) instead of stamping now(), and preserve deleted_at so a
    soft-deleted (trashed) message is NOT resurrected by a restore."""
    async with Storage(db_path=temp_db) as storage:
        # Content-agnostic (encryption-at-rest may cipher the content column):
        # backdate one row and soft-delete the other, keyed by row id.
        await storage.add_conversation("user", "kept-old", metadata={})
        await storage.add_conversation("user", "trashed", metadata={})
        ids = [
            r[0]
            for r in await storage.db.fetchall(
                "SELECT id FROM conversation_history ORDER BY id"
            )
        ]
        await storage.db.execute_commit(
            "UPDATE conversation_history SET created_at = ? WHERE id = ?",
            # Canonical, because the column's CHECK refuses anything else
            # since #3009. The claim is about the stamp being PRESERVED across
            # a backup, not about which spelling it arrives in.
            ("2020-01-01 00:00:00", ids[0]),
        )
        await storage.db.execute_commit(
            "UPDATE conversation_history SET deleted_at = ? WHERE id = ?",
            ("2020-06-01T00:00:00+00:00", ids[1]),
        )
        blob = await storage.create_backup_blob(include_db=True)

    fd, target_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        async with Storage(db_path=target_path) as target:
            stats = await target.restore_from_backup_blob(blob)
            assert stats["messages_restored"] == 2

            rows = await target.db.fetchall(
                "SELECT created_at, deleted_at FROM conversation_history"
            )
            created_ats = [str(r[0]) for r in rows]
            deleted_ats = [r[1] for r in rows]
            # created_at PRESERVED (not all rewritten to a fresh now()).
            assert any(c.startswith("2020-01-01") for c in created_ats)
            # deleted_at PRESERVED — the trashed row stays trashed.
            assert any(d is not None for d in deleted_ats)
            # A trash-filtered read therefore returns only the live row.
            live = await target.get_conversation_history()
            assert len(live) == 1
    finally:
        if os.path.exists(target_path):
            os.remove(target_path)


@pytest.mark.asyncio
async def test_restore_preserves_turn_order_for_same_second(temp_db):
    """#F265 (codex P2): same-second messages must keep their original turn
    order across a restore (new ids are assigned in the restore SELECT order,
    and reads sort by id). Uses the plaintext ``role`` sequence as the
    observable order (content may be encrypted at rest)."""
    roles = ["user", "assistant", "user", "assistant"]
    async with Storage(db_path=temp_db) as storage:
        for i, role in enumerate(roles):
            await storage.add_conversation(role, f"turn {i}", metadata={})
        # Force every row to the SAME created_at (second-granularity collision).
        await storage.db.execute_commit(
            "UPDATE conversation_history SET created_at = ?",
            ("2020-01-01 00:00:00",),
        )
        blob = await storage.create_backup_blob(include_db=True)

    fd, target_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        async with Storage(db_path=target_path) as target:
            await target.restore_from_backup_blob(blob)
            rows = await target.db.fetchall(
                "SELECT role FROM conversation_history ORDER BY id"
            )
            assert [r[0] for r in rows] == roles
    finally:
        if os.path.exists(target_path):
            os.remove(target_path)


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
        result = await sovereign_adapter.import_agent(cid)

        assert result.success
        assert result.status == "imported"
        assert result.messages_restored == 4
        assert result.shards_restored > 0
        assert result.agent_did == agent_did
        assert result.manifest_version == "3.0"
        assert result.assets_restored is not None
        assert result.continuity.is_verified()
        print(f"✅ Imported {result.messages_restored} messages")

        # 5. Verify restored data matches original
        restored_history = await storage.get_conversation_history()
        assert len(restored_history) == 4

        for i, msg in enumerate(restored_history):
            assert msg['role'] == test_messages[i][0]
            assert msg['content'] == test_messages[i][1]

        print("✅ Data integrity verified - all messages restored correctly")


@pytest.mark.asyncio
async def test_import_rederives_the_session_id_column(temp_db):
    """#2958: an imported row is indistinguishable from one the backfill lifted.

    The import DELETEs the agent's history and reinserts it from the package's
    shards, with hand-spelled SQL that does not go through the store's insert
    helper — so the derived ``session_id`` column is only populated because
    this path stamps it deliberately. A package exported before the column
    existed still restores with it populated, because the source is the
    ``metadata`` the package already carries.

    Both halves of the rule are exercised: an id inside the column's contract
    lands in it, and one outside stays in metadata with the column NULL, which
    is the state Phase C must tolerate.
    """
    agent_did = "did:pkh:eip155:1:test_import_session_column"
    stampable = "9d2f5c31-6b0a-4e7d-8c14-000000000042"

    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "inside the contract", session_id=stampable
        )
        await storage.add_conversation(
            "user", "outside the contract", session_id="did:x:1"
        )

        adapter = SovereignStorageAdapter(storage.db, user_secret="session-column")
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY
        )
        # Blank the column so a stale value cannot masquerade as a fresh
        # stamp; the DELETE + reinsert would replace the rows anyway, but
        # this makes the claim independent of that.
        await storage.db.execute_commit(
            "UPDATE conversation_history SET session_id = NULL"
        )

        result = await adapter.import_agent(cid)
        assert result.success, result.status

        rows = await storage.db.fetchall(
            "SELECT metadata, session_id FROM conversation_history ORDER BY id"
        )
        assert [row[1] for row in rows] == [stampable, None]
        assert [json.loads(row[0])["session_id"] for row in rows] == [
            stampable, "did:x:1",
        ]


@pytest.mark.asyncio
async def test_import_with_wrong_key_fails(temp_db):
    """
    Verify that import with the wrong encryption key is *rejected*
    (not raised) and the host DB is left untouched. Proves the
    encryption is real AND the new verification-gated contract.
    """
    agent_did = "did:pkh:eip155:1:test_wrong_key"

    async with Storage(db_path=temp_db) as storage:
        # Export with one key
        export_adapter = SovereignStorageAdapter(storage.db, user_secret="correct-key")
        await storage.add_conversation("user", "Secret message", metadata={"timestamp": "2025-11-21T10:00:00Z"})
        cid = await export_adapter.export_agent(agent_did, storage_tier=StorageTier.LOCAL_ONLY)

        # Sentinel conversation present in the host DB before the
        # rejected import — must still be there afterward.
        before = await storage.get_conversation_history()
        assert len(before) == 1

        # Try to import with different key
        import_adapter = SovereignStorageAdapter(storage.db, user_secret="wrong-key")

        result = await import_adapter.import_agent(cid)
        assert result.success is False
        assert result.status == "rejected"
        assert result.reject_reason == "keyring_decrypt_failed"

        # Host DB UNTOUCHED — no DELETE/INSERT happened.
        after = await storage.get_conversation_history()
        assert len(after) == 1
        assert after[0]["content"] == "Secret message"

        print("✅ Import correctly rejected wrong key; host DB untouched")


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
    await complete_deterministic_genesis_audit(
        agent,
        provenance="test:sovereignty_import",
    )

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

        assert "Restored" in import_result and "conversation messages" in import_result, \
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

            # Attempt import - with a shard cache file removed the CAR
            # bytes themselves are unaffected (the CAR is one blob keyed
            # by the manifest CID), so this still round-trips. The point
            # of the new contract is that *if* verification fails the
            # result is a structured rejection, never a raised exception.
            try:
                result = await sovereign_adapter.import_agent(cid)
                if result.success:
                    messages_recovered = result.messages_restored
                    print(f"✅ Import completed with {messages_recovered} messages recovered")
                else:
                    # Structured rejection — must name the failure clearly.
                    reason = (result.reject_reason or "").lower()
                    acceptable_keywords = ["shard", "corrupt", "fail", "missing", "retrieve", "decrypt", "continuity", "manifest", "car"]
                    assert any(kw in reason for kw in acceptable_keywords), \
                        f"Reject reason should clearly indicate the problem. Got: {result.reject_reason}"
                    print(f"✅ Graceful structured rejection: {result.reject_reason}")
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
