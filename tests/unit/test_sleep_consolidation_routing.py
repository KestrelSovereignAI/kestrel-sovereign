"""#1674 P3: the sleep cycle consolidates through the single
MemorySystem.consolidate() chokepoint (so it inherits the forgetting deletion
tier), not the lower-level MemoryConsolidator.run_consolidation()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.sleep import SleepMixin


class _Agent(SleepMixin):
    """Minimal carrier for the mixin under test."""
    def __init__(self):
        self.memory_system = None
        self.memory_consolidator = None


@pytest.mark.asyncio
async def test_consolidate_routes_through_memory_system():
    agent = _Agent()
    agent.memory_system = MagicMock()
    agent.memory_system.consolidate = AsyncMock(return_value={
        "episodes_created": 1, "episodes_deleted": 4,
    })
    # Raw consolidator must NOT be used when memory_system is present.
    agent.memory_consolidator = MagicMock()
    agent.memory_consolidator.run_consolidation = AsyncMock(return_value={})

    result = await agent._consolidate_memories()

    agent.memory_system.consolidate.assert_awaited_once()
    agent.memory_consolidator.run_consolidation.assert_not_awaited()
    assert result["episodes_deleted"] == 4  # forgetting count flows through


@pytest.mark.asyncio
async def test_consolidate_falls_back_to_raw_consolidator():
    """No MemorySystem wired → fall back to the raw consolidator."""
    agent = _Agent()
    agent.memory_system = None
    agent.memory_consolidator = MagicMock()
    agent.memory_consolidator.run_consolidation = AsyncMock(
        return_value={"episodes_created": 0})

    result = await agent._consolidate_memories()

    agent.memory_consolidator.run_consolidation.assert_awaited_once()
    assert result == {"episodes_created": 0}
