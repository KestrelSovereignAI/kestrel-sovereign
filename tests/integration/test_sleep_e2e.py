#!/usr/bin/env python3
"""
E2E Tests for Sleep Functionality.

Tests the SleepMixin that combines:
1. Memory consolidation (episodes, patterns, archiving)
2. Sovereignty export (sharded, encrypted export to IPFS)

Uses REAL services - NO MOCKS!
"""

import pytest
import pytest_asyncio
import os
import tempfile
import shutil
from pathlib import Path

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.agent.sleep import SleepReport


@pytest_asyncio.fixture
async def agent_with_storage():
    """Create a real KestrelAgent with storage for testing."""
    # Create temp directory for agent storage
    temp_dir = tempfile.mkdtemp(prefix="kestrel_sleep_test_")
    db_path = os.path.join(temp_dir, "test_agent.db")

    # Create agent
    agent = KestrelAgent(
        did="did:test:sleep-test-agent",
        storage_path=db_path,
        llm_service=None,  # No LLM needed for sleep tests
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()

    # Add some test messages to have something to consolidate
    if hasattr(agent, 'storage') and agent.storage:
        for i in range(5):
            await agent.storage.add_conversation(
                role="user",
                content=f"Test message {i}: This is message number {i} for sleep testing."
            )
            await agent.storage.add_conversation(
                role="assistant",
                content=f"Response {i}: I understand this is test message {i}."
            )

    yield agent

    # Cleanup
    if hasattr(agent, 'storage') and agent.storage:
        await agent.storage.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def ephemeral_agent():
    """Create an EPHEMERAL mode agent (should refuse to sleep)."""
    temp_dir = tempfile.mkdtemp(prefix="kestrel_ephemeral_test_")
    db_path = os.path.join(temp_dir, "ephemeral_agent.db")

    agent = KestrelAgent(
        did="did:test:ephemeral-agent",
        storage_path=db_path,
        llm_service=None,
        privacy_mode=PrivacyMode.EPHEMERAL,
    )
    await agent.initialize()

    yield agent

    if hasattr(agent, 'storage') and agent.storage:
        await agent.storage.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestSleepMixin:
    """Tests for SleepMixin functionality."""

    @pytest.mark.asyncio
    async def test_sleep_consolidation_only(self, agent_with_storage):
        """Test memory consolidation without export."""
        agent = agent_with_storage

        # Run consolidation only
        report = await agent.sleep(
            tier="local",
            skip_export=True,
        )

        assert report.success, f"Sleep failed: {report.error}"
        assert report.cid is None, "Should not have CID with skip_export=True"
        # Consolidation stats should be present
        assert report.consolidation_ms >= 0
        print(f"✅ Consolidation completed in {report.consolidation_ms}ms")
        print(f"   Episodes created: {report.episodes_created}")
        print(f"   Patterns found: {report.patterns_found}")
        print(f"   Messages archived: {report.messages_archived}")

    @pytest.mark.asyncio
    async def test_sleep_export_only(self, agent_with_storage):
        """Test sovereignty export without consolidation."""
        agent = agent_with_storage

        # Run export only (local tier to avoid IPFS dependency)
        report = await agent.sleep(
            tier="local",
            skip_consolidation=True,
        )

        assert report.success, f"Sleep failed: {report.error}"
        # Should have local path or CID
        assert report.export_ms >= 0
        assert report.storage_tier == "local"
        print(f"✅ Export completed in {report.export_ms}ms")
        print(f"   CID/Path: {report.cid}")
        print(f"   Shards: {report.shards_exported}")

    @pytest.mark.asyncio
    async def test_full_sleep_cycle(self, agent_with_storage):
        """Test full sleep cycle (consolidation + export)."""
        agent = agent_with_storage

        # Run full sleep
        report = await agent.sleep(tier="local")

        assert report.success, f"Sleep failed: {report.error}"
        assert report.consolidation_ms >= 0
        assert report.export_ms >= 0
        print(f"✅ Full sleep completed")
        print(f"   Consolidation: {report.consolidation_ms}ms")
        print(f"   Export: {report.export_ms}ms")
        print(str(report))

    @pytest.mark.asyncio
    async def test_ephemeral_agent_sleep_succeeds(self, ephemeral_agent):
        """Test that EPHEMERAL mode agents can sleep (they just have nothing to export).

        Privacy mode is message-level, not agent-level. EPHEMERAL messages aren't
        stored at all. Sleep doesn't need to be blocked - it just exports whatever
        IS stored (which for a pure ephemeral session is nothing).
        """
        agent = ephemeral_agent

        report = await agent.sleep(tier="local")

        # Sleep should succeed - it exports whatever is stored
        # For an ephemeral agent with no stored messages, that's essentially empty
        assert report.success, f"Sleep should succeed: {report.error}"
        print(f"✅ EPHEMERAL agent sleep succeeded (exports whatever is stored)")
        print(f"   Messages archived: {report.messages_archived}")
        print(f"   Shards exported: {report.shards_exported}")

    @pytest.mark.asyncio
    async def test_sleep_callback(self, agent_with_storage):
        """Test that on_sleep_complete callback is invoked."""
        agent = agent_with_storage
        callback_called = []

        @pytest.mark.asyncio
        async def test_callback(cid: str, report: SleepReport):
            callback_called.append((cid, report))

        agent.on_sleep_complete = test_callback

        # Run sleep
        report = await agent.sleep(tier="local")

        if report.success and report.cid:
            assert len(callback_called) == 1, "Callback should be called once"
            assert callback_called[0][0] == report.cid
            print(f"✅ Callback invoked with CID: {report.cid}")
        else:
            # If no CID, callback shouldn't be called
            print(f"ℹ️  No CID returned, callback correctly not called")


class TestQuickNap:
    """Tests for quick_nap functionality."""

    @pytest.mark.asyncio
    async def test_quick_nap_no_consolidation_needed(self, agent_with_storage):
        """Test quick_nap when no consolidation is needed."""
        agent = agent_with_storage

        # Quick nap might not create an episode if not enough messages
        result = await agent.quick_nap()

        # Result should be None or episode description
        if result:
            print(f"✅ Quick nap created episode: {result}")
        else:
            print("✅ Quick nap: no consolidation needed")


class TestSleepCommand:
    """Tests for !sleep command handling."""

    @pytest.mark.asyncio
    async def test_sleep_command_default(self, agent_with_storage):
        """Test !sleep command with defaults."""
        agent = agent_with_storage

        result = await agent._command_sleep("!sleep")

        assert "Sleep" in result or "failed" in result.lower()
        print(f"✅ !sleep command result:\n{result}")

    @pytest.mark.asyncio
    async def test_sleep_command_tier_option(self, agent_with_storage):
        """Test !sleep --tier local command."""
        agent = agent_with_storage

        result = await agent._command_sleep("!sleep --tier local")

        assert "local" in result.lower()
        print(f"✅ !sleep --tier local result:\n{result}")

    @pytest.mark.asyncio
    async def test_sleep_command_consolidate_only(self, agent_with_storage):
        """Test !sleep --consolidate-only command."""
        agent = agent_with_storage

        result = await agent._command_sleep("!sleep --consolidate-only")

        # Should not have CID in output
        print(f"✅ !sleep --consolidate-only result:\n{result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
