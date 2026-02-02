"""
End-to-end tests for Context Preservation during Sovereignty Export/Import.

These tests verify that:
1. Context can be rebuilt after sovereignty export/import
2. Conversation history survives round-trip
3. ContextManager produces consistent results before/after migration
4. Episode triggers work on restored data
5. Budget allocations are preserved across export/import

Tests use real storage - NO MOCKS.
"""

import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path

from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter
from kestrel_sovereign.filecoin_adapter import StorageTier
from kestrel_sovereign.agent.context_manager import ContextManager, ContextResult
from kestrel_sovereign.agent.token_budget import create_budget


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest_asyncio.fixture
async def storage_with_context_data(temp_db_path):
    """Create storage with conversation data suitable for context building."""
    async with AsyncStorage(str(temp_db_path)) as storage:
        storage.agent_id = "test-context-agent"

        # Add emotionally rich conversation history
        conversations = [
            ("user", "Hi there! I'm so excited today."),
            ("assistant", "That's wonderful! What's making you feel excited?"),
            ("user", "I just got engaged! My mom is going to be so happy."),
            ("assistant", "Congratulations! That's such wonderful news. When's the big day?"),
            ("user", "We're thinking June next year. My favorite color is blue, so we'll have a blue theme."),
            ("assistant", "A June wedding with blue accents sounds beautiful. That gives you plenty of time to plan."),
            ("user", "Yes! I'm a bit nervous about the planning though."),
            ("assistant", "That's completely normal. Would you like some tips on wedding planning?"),
            ("user", "Please! I have no idea where to start."),
            ("assistant", "Let's start with the basics: venue, date, and budget. What's your budget range?"),
        ]

        for role, content in conversations:
            await storage.conversation.add_conversation(
                role=role,
                content=content,
                metadata={"test_context": True}
            )

        yield storage


class TestContextPreservationRoundTrip:
    """Tests for context preservation during sovereignty export/import."""

    @pytest.mark.asyncio
    async def test_context_survives_export_import(self, temp_db_path):
        """
        Test that context can be rebuilt identically after export/import.

        Steps:
        1. Create storage with conversation history
        2. Build context (before export)
        3. Export to sovereignty backup
        4. Clear database
        5. Import from backup
        6. Build context again (after import)
        7. Verify contexts match
        """
        agent_did = "did:pkh:eip155:1:context_roundtrip"

        async with AsyncStorage(str(temp_db_path)) as storage:
            storage.agent_id = "test-context-agent"

            # Add conversation history
            conversations = [
                ("user", "Hello, my name is Alice."),
                ("assistant", "Nice to meet you, Alice!"),
                ("user", "I love programming in Python."),
                ("assistant", "Python is a great choice! What kind of projects do you work on?"),
            ]

            for role, content in conversations:
                await storage.conversation.add_conversation(
                    role=role,
                    content=content,
                    metadata={"timestamp": "2025-12-01T10:00:00Z"}
                )

            # Build context BEFORE export
            manager_before = ContextManager(
                storage=storage,
                model="gpt-4",
                agent_id="test-context-agent"
            )

            context_before = await manager_before.build_context(
                query="Tell me about Alice",
                constitution="Test Constitution",
                privacy_mode="NORMAL"
            )

            messages_before = len(context_before.messages)
            tokens_before = context_before.total_tokens

            # Export to sovereignty backup
            sovereign_adapter = SovereignStorageAdapter(
                storage.db,
                user_secret="test-context-secret"
            )
            cid = await sovereign_adapter.export_agent(
                agent_did,
                storage_tier=StorageTier.LOCAL_ONLY
            )
            assert cid is not None

            # Clear database (simulate data loss)
            await storage.db.execute_commit("DELETE FROM conversation_history")

            # Verify deletion
            context_cleared = await manager_before.build_context(
                query="Tell me about Alice",
                constitution="Test Constitution",
                privacy_mode="NORMAL"
            )
            assert len(context_cleared.messages) == 0, "Database should be empty after clear"

            # Import from backup
            stats = await sovereign_adapter.import_agent(cid)
            assert stats['messages_restored'] == 4

            # Build context AFTER import
            manager_after = ContextManager(
                storage=storage,
                model="gpt-4",
                agent_id="test-context-agent"
            )

            context_after = await manager_after.build_context(
                query="Tell me about Alice",
                constitution="Test Constitution",
                privacy_mode="NORMAL"
            )

            # Verify context matches
            assert len(context_after.messages) == messages_before
            # Token count may vary slightly due to timestamp changes
            # but should be in the same ballpark
            assert abs(context_after.total_tokens - tokens_before) < 50

    @pytest.mark.asyncio
    async def test_long_conversation_episodes_after_import(self, temp_db_path):
        """
        Test that long conversations get proper episode allocation after import.

        This verifies the adaptive budget allocation works on imported data.
        """
        agent_did = "did:pkh:eip155:1:episode_test"

        async with AsyncStorage(str(temp_db_path)) as storage:
            storage.agent_id = "test-agent"

            # Add 35 messages (triggers long conversation allocation)
            for i in range(35):
                await storage.conversation.add_conversation(
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}: Discussion about important topic number {i}.",
                    metadata={"index": i}
                )

            # Export
            sovereign_adapter = SovereignStorageAdapter(
                storage.db,
                user_secret="episode-test-secret"
            )
            cid = await sovereign_adapter.export_agent(
                agent_did,
                storage_tier=StorageTier.LOCAL_ONLY
            )

            # Clear and restore
            await storage.db.execute_commit("DELETE FROM conversation_history")
            stats = await sovereign_adapter.import_agent(cid)
            assert stats['messages_restored'] == 35

            # Build context on restored data
            manager = ContextManager(
                storage=storage,
                model="gpt-4",
                agent_id="test-agent"
            )

            result = await manager.build_context(
                query="What did we discuss?",
                constitution="Test",
                privacy_mode="NORMAL"
            )

            # Verify adaptive allocation applied (long conversation)
            allocations = result.budget_summary.get("allocations", {})
            assert "history" in allocations
            assert "episodes" in allocations

            # Long conversations (>30 msgs) should have episodes > history allocation
            # History is 25%, Episodes is 35% for long conversations
            if result.budget_summary.get("context_limit") == 8192:
                total_budget = 8192 - 1024  # minus response reserve
                expected_history = int(total_budget * 0.25)
                expected_episodes = int(total_budget * 0.35)
                assert allocations["episodes"]["budget"] > allocations["history"]["budget"]


class TestContextMetadataPreservation:
    """Tests for metadata preservation during export/import."""

    @pytest.mark.asyncio
    async def test_message_metadata_preserved(self, temp_db_path):
        """
        Test that message metadata (timestamps, etc.) survives export/import.
        """
        agent_did = "did:pkh:eip155:1:metadata_test"

        async with AsyncStorage(str(temp_db_path)) as storage:
            storage.agent_id = "test-agent"

            # Add messages with metadata
            await storage.conversation.add_conversation(
                role="user",
                content="Important message",
                metadata={
                    "importance": 0.9,
                    "timestamp": "2025-12-01T12:00:00Z",
                    "emotional_valence": 0.7,
                    "test_key": "test_value"
                }
            )

            # Get original history
            original_history = await storage.conversation.get_full_history()
            original_metadata = original_history[0].get("metadata", {})

            # Export
            sovereign_adapter = SovereignStorageAdapter(
                storage.db,
                user_secret="metadata-test"
            )
            cid = await sovereign_adapter.export_agent(
                agent_did,
                storage_tier=StorageTier.LOCAL_ONLY
            )

            # Clear and restore
            await storage.db.execute_commit("DELETE FROM conversation_history")
            await sovereign_adapter.import_agent(cid)

            # Check restored metadata
            restored_history = await storage.conversation.get_full_history()
            restored_metadata = restored_history[0].get("metadata", {})

            # Core metadata should be preserved
            assert restored_metadata.get("importance") == original_metadata.get("importance")
            assert restored_metadata.get("timestamp") == original_metadata.get("timestamp")


class TestDifferentModelContextsAfterImport:
    """Tests for context building with different models after import."""

    @pytest.mark.asyncio
    async def test_context_adapts_to_model_after_import(self, temp_db_path):
        """
        Test that context properly adapts to different model context limits
        when building from imported data.
        """
        agent_did = "did:pkh:eip155:1:model_test"

        async with AsyncStorage(str(temp_db_path)) as storage:
            storage.agent_id = "test-agent"

            # Add moderate conversation
            for i in range(15):
                await storage.conversation.add_conversation(
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}: Some content for testing model context limits.",
                    metadata={"index": i}
                )

            # Export
            sovereign_adapter = SovereignStorageAdapter(
                storage.db,
                user_secret="model-test"
            )
            cid = await sovereign_adapter.export_agent(
                agent_did,
                storage_tier=StorageTier.LOCAL_ONLY
            )

            # Clear and restore
            await storage.db.execute_commit("DELETE FROM conversation_history")
            await sovereign_adapter.import_agent(cid)

            # Build context with small model (phi3 - 4K)
            manager_small = ContextManager(
                storage=storage,
                model="phi3:3.8b",
                agent_id="test-agent"
            )

            result_small = await manager_small.build_context(
                query="test",
                constitution="Test",
                privacy_mode="NORMAL"
            )

            # Build context with large model (Claude - 1M)
            manager_large = ContextManager(
                storage=storage,
                model="claude-opus-4-5-20251101",
                agent_id="test-agent"
            )

            result_large = await manager_large.build_context(
                query="test",
                constitution="Test",
                privacy_mode="NORMAL"
            )

            # Large model should have much larger budget
            assert result_large.budget_summary["context_limit"] > result_small.budget_summary["context_limit"]

            # Both should fit their content
            assert result_small.total_tokens < result_small.budget_summary["context_limit"]
            assert result_large.total_tokens < result_large.budget_summary["context_limit"]


class TestPrivacyModeAfterImport:
    """Tests for privacy mode behavior after sovereignty import."""

    @pytest.mark.asyncio
    async def test_ephemeral_mode_ignores_imported_history(self, temp_db_path):
        """
        Test that EPHEMERAL mode still ignores history after import.

        Even after importing conversation history, EPHEMERAL mode
        should not expose any of it.
        """
        agent_did = "did:pkh:eip155:1:ephemeral_test"

        async with AsyncStorage(str(temp_db_path)) as storage:
            storage.agent_id = "test-agent"

            # Add conversation
            await storage.conversation.add_conversation(
                role="user",
                content="My secret password is hunter2",
                metadata={"sensitive": True}
            )

            # Export and reimport
            sovereign_adapter = SovereignStorageAdapter(
                storage.db,
                user_secret="ephemeral-test"
            )
            cid = await sovereign_adapter.export_agent(
                agent_did,
                storage_tier=StorageTier.LOCAL_ONLY
            )

            await storage.db.execute_commit("DELETE FROM conversation_history")
            await sovereign_adapter.import_agent(cid)

            # Build context in EPHEMERAL mode
            manager = ContextManager(
                storage=storage,
                model="gpt-4",
                agent_id="test-agent"
            )

            result = await manager.build_context(
                query="What's my password?",
                constitution="Test",
                privacy_mode="EPHEMERAL"
            )

            # Should have no messages despite imported history
            assert result.messages == []
            assert any("EPHEMERAL" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_normal_mode_accesses_imported_history(self, temp_db_path):
        """
        Test that NORMAL mode can access imported history.
        """
        agent_did = "did:pkh:eip155:1:normal_test"

        async with AsyncStorage(str(temp_db_path)) as storage:
            storage.agent_id = "test-agent"

            # Add conversation with identifiable content
            await storage.conversation.add_conversation(
                role="user",
                content="My favorite programming language is Rust",
                metadata={}
            )
            await storage.conversation.add_conversation(
                role="assistant",
                content="Rust is a great systems programming language!",
                metadata={}
            )

            # Export and reimport
            sovereign_adapter = SovereignStorageAdapter(
                storage.db,
                user_secret="normal-test"
            )
            cid = await sovereign_adapter.export_agent(
                agent_did,
                storage_tier=StorageTier.LOCAL_ONLY
            )

            await storage.db.execute_commit("DELETE FROM conversation_history")
            await sovereign_adapter.import_agent(cid)

            # Build context in NORMAL mode
            manager = ContextManager(
                storage=storage,
                model="gpt-4",
                agent_id="test-agent"
            )

            result = await manager.build_context(
                query="What's my favorite language?",
                constitution="Test",
                privacy_mode="NORMAL"
            )

            # Should have messages with the Rust content
            assert len(result.messages) > 0
            contents = [m.get("content", "") for m in result.messages]
            assert any("Rust" in c for c in contents)


class TestBudgetConsistencyAfterImport:
    """Tests for budget allocation consistency after import."""

    @pytest.mark.asyncio
    async def test_budget_status_consistent_after_import(self, temp_db_path):
        """
        Test that budget status returns consistent values before and after import.
        """
        agent_did = "did:pkh:eip155:1:budget_test"

        async with AsyncStorage(str(temp_db_path)) as storage:
            storage.agent_id = "test-agent"

            # Add 20 messages (medium conversation)
            for i in range(20):
                await storage.conversation.add_conversation(
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}",
                    metadata={}
                )

            # Get budget status before
            manager = ContextManager(
                storage=storage,
                model="gpt-4",
                agent_id="test-agent"
            )
            budget_before = manager.get_budget_status(message_count=20)

            # Export
            sovereign_adapter = SovereignStorageAdapter(
                storage.db,
                user_secret="budget-test"
            )
            cid = await sovereign_adapter.export_agent(
                agent_did,
                storage_tier=StorageTier.LOCAL_ONLY
            )

            # Clear and restore
            await storage.db.execute_commit("DELETE FROM conversation_history")
            await sovereign_adapter.import_agent(cid)

            # Get budget status after
            budget_after = manager.get_budget_status(message_count=20)

            # Budget allocations should be identical (same model, same message count)
            assert budget_before["model"] == budget_after["model"]
            assert budget_before["context_limit"] == budget_after["context_limit"]
            assert budget_before["allocations"]["history"]["budget"] == budget_after["allocations"]["history"]["budget"]
            assert budget_before["allocations"]["episodes"]["budget"] == budget_after["allocations"]["episodes"]["budget"]
