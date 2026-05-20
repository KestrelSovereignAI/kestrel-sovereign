"""
Integration-style tests that verify the memory pipeline is actually WIRED,
not just that individual components work in isolation.

These tests exist because previous sessions claimed the memory system was
working when in reality MemoryConsolidator was never invoked and access_count
was never incremented. Unit tests of the components passed; integration was
never verified.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.memory_retriever import MemoryRetriever


class TestUpdateAccessActuallyWrites:
    """Verify update_access() actually persists, not just logs.

    Smoking-gun guard: a previous regression had ``update_access`` just
    logging without writing.  After #1326 the retriever delegates to
    the conversation store's ``atomic_increment_metadata_counter`` (a
    single atomic SQL statement that fixes the lost-update race the
    old read-modify-write had under concurrent retrievals).  Tests
    assert delegation; end-to-end "the counter actually moves" is
    covered against a real SQLite in
    ``tests/integration/test_atomic_increment_metadata.py``.
    """

    @pytest.mark.asyncio
    async def test_update_access_delegates_to_atomic_increment(self):
        conv_store = MagicMock()
        conv_store.agent_id = "test-agent"
        conv_store.atomic_increment_metadata_counter = AsyncMock(return_value=True)

        retriever = MemoryRetriever(conversation_store=conv_store)
        await retriever.update_access(message_id=42, agent_id="test-agent")

        conv_store.atomic_increment_metadata_counter.assert_awaited_once_with(
            42,
            counter_field="access_count",
            timestamp_field="last_accessed",
        )

    @pytest.mark.asyncio
    async def test_update_access_silent_on_db_error(self):
        """DB errors must not propagate — never break retrieval."""
        conv_store = MagicMock()
        conv_store.agent_id = "test-agent"
        conv_store.atomic_increment_metadata_counter = AsyncMock(
            side_effect=RuntimeError("db down")
        )

        retriever = MemoryRetriever(conversation_store=conv_store)
        # Must not raise even though DB fails.
        await retriever.update_access(message_id=42, agent_id="test-agent")


class TestRetrieveTriggersAccessUpdate:
    """Verify retrieve() actually fires update_access for surfaced memories."""

    @pytest.mark.asyncio
    async def test_retrieve_schedules_access_updates(self):
        """The OTHER smoking-gun: even if update_access works, was it
        ever called? Verifies the wiring from retrieve → update_access."""
        conv_store = MagicMock()
        conv_store.agent_id = "test-agent"
        conv_store.get_conversation_history = AsyncMock(return_value=[
            {"id": 1, "role": "assistant", "content": "I love sunny days",
             "metadata": {"importance": 0.8}, "created_at": "2026-04-18T10:00:00Z"},
            {"id": 2, "role": "assistant", "content": "Rain makes me sad",
             "metadata": {"importance": 0.7}, "created_at": "2026-04-18T11:00:00Z"},
        ])

        retriever = MemoryRetriever(conversation_store=conv_store)

        # Patch update_access so we can verify it's called
        update_access_calls = []

        async def fake_update_access(message_id, agent_id):
            update_access_calls.append((message_id, agent_id))

        retriever.update_access = fake_update_access

        results = await retriever.retrieve(
            query="sunny weather",
            agent_id="test-agent",
            min_score=0.0,  # don't filter anything out
            limit=2,
        )

        await retriever.drain_access_updates()

        assert len(results) > 0, "retrieve returned no results to verify against"
        # Each surfaced memory should have triggered an access update
        assert len(update_access_calls) == len(results)
        called_ids = {call[0] for call in update_access_calls}
        result_ids = {r["id"] for r in results}
        assert called_ids == result_ids

    @pytest.mark.asyncio
    async def test_shutdown_cancels_pending_access_updates(self):
        """Access-count bookkeeping is owned and cancelled during shutdown."""
        conv_store = MagicMock()
        conv_store.agent_id = "test-agent"
        conv_store.get_conversation_history = AsyncMock(return_value=[
            {"id": 1, "role": "assistant", "content": "I love sunny days",
             "metadata": {"importance": 0.8}, "created_at": "2026-04-18T10:00:00Z"},
        ])

        retriever = MemoryRetriever(conversation_store=conv_store)
        started = asyncio.Event()

        async def never_finishes(message_id, agent_id):
            started.set()
            await asyncio.Event().wait()

        retriever.update_access = never_finishes

        await retriever.retrieve(
            query="sunny weather",
            agent_id="test-agent",
            min_score=0.0,
            limit=1,
        )
        await started.wait()

        task = next(iter(retriever._access_update_tasks))

        await retriever.shutdown()

        assert task.done()
        assert task.cancelled()
        assert retriever._access_update_tasks == set()


class TestMemoryConsolidateToolExists:
    """Verify the memory_consolidate tool is actually registered on MemoryFeature.

    Previous state: MemoryConsolidator existed in code but had no tool,
    so the scheduler couldn't invoke it via feature tool lookup.
    """

    def test_memory_feature_exposes_consolidate_tool(self):
        """The scheduler's _lookup_and_run_tool searches feature tools by
        name. If MemoryFeature doesn't expose memory_consolidate, the
        scheduled task fails with 'Unknown task'.

        (Phase 4 of #889 renamed `_execute_scheduled_task` →
        `_lookup_and_run_tool`; the search body it tests is unchanged.)"""
        from kestrel_sovereign.features.memory.feature import MemoryFeature

        # Find the @tool decorator on memory_consolidate
        method = getattr(MemoryFeature, "memory_consolidate", None)
        assert method is not None, "MemoryFeature.memory_consolidate method missing"

        # The @tool decorator attaches schema metadata
        assert hasattr(method, "_tool_schema") or hasattr(method, "schema") or hasattr(method, "_agent_skill"), (
            "memory_consolidate is not decorated with @tool — "
            "scheduler will not find it via feature tool lookup"
        )


class TestSchedulerDefaultsIncludeConsolidation:
    """Verify scheduler.feature.py includes memory_consolidate in defaults
    when MemoryFeature is loaded.

    Previous state: the cron job 'memory_consolidate' was being scheduled
    by some past code path, but never registered as a runnable task name —
    failing repeatedly in production logs with 'Unknown task'.
    """

    def test_defaults_include_memory_consolidate_when_memory_feature_present(self):
        """Read the source to verify the wiring exists."""
        from pathlib import Path
        scheduler_path = Path(__file__).parent.parent.parent / "kestrel_sovereign" / "features" / "scheduler" / "feature.py"
        source = scheduler_path.read_text()

        # The default-task block must conditionally add memory_consolidate
        assert '"MemoryFeature" in agent.features' in source, (
            "Scheduler does not check for MemoryFeature presence"
        )
        assert '"memory_consolidate"' in source, (
            "Scheduler defaults do not include memory_consolidate task"
        )
