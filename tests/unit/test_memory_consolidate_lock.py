"""#round3: the manual `!memory consolidate` tool must serialize against the
scheduled cron run by holding ResourceLock.MEMORY (the same lock the dispatcher
holds for the cron tick). Without it, a manual run racing the cron tick reads
the same empty covered-message-id set and both emit duplicate episodes.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.signals import ResourceLock


class _LockProbe:
    """Async-context-manager lock manager that records the names it was asked
    to acquire and tracks whether the lock is held while the body runs."""

    def __init__(self):
        self.acquired_with = None
        self.held = False

    def acquire(self, names):
        self.acquired_with = list(names)
        probe = self

        class _CM:
            async def __aenter__(self):
                probe.held = True
                return None

            async def __aexit__(self, *exc):
                probe.held = False
                return False

        return _CM()


def _make_memory_feature(*, dispatcher):
    from kestrel_sovereign.features.memory.feature import MemoryFeature

    agent = MagicMock()
    agent.memory_system = MagicMock()
    agent.dispatcher = dispatcher
    return MemoryFeature(agent), agent


@pytest.mark.asyncio
async def test_manual_consolidate_holds_memory_lock():
    probe = _LockProbe()
    dispatcher = MagicMock()
    dispatcher._locks = probe
    feature, agent = _make_memory_feature(dispatcher=dispatcher)

    async def _consolidate():
        # The MEMORY lock must be held while consolidation runs.
        assert probe.held, "MEMORY lock must be held during consolidation"
        return {"episodes_created": 1, "patterns_found": 0, "messages_archived": 0}

    agent.memory_system.consolidate = AsyncMock(side_effect=_consolidate)

    result = await feature.memory_consolidate()

    assert probe.acquired_with == [ResourceLock.MEMORY]
    assert probe.held is False  # released after
    assert result.data.get("episodes_created") == 1


@pytest.mark.asyncio
async def test_manual_consolidate_runs_without_dispatcher_lock():
    """No dispatcher / lock manager wired (standalone, tests) -> still runs."""
    feature, agent = _make_memory_feature(dispatcher=None)
    agent.memory_system.consolidate = AsyncMock(
        return_value={"episodes_created": 0, "patterns_found": 0, "messages_archived": 0}
    )

    result = await feature.memory_consolidate()

    agent.memory_system.consolidate.assert_awaited_once()
    assert result.data.get("episodes_created") == 0


# ---------------------------------------------------------------------------
# The tool SURFACES episodes_deleted from the consolidate() chokepoint (#1674 P3)
#
# Forgetting now lives in MemorySystem.consolidate() (one place, shared by the
# tool AND the nightly sleep cycle), so the tool's only job is to surface the
# count consolidate() returns. The forgetting LOGIC (enabled/disabled/errored/
# best-effort) is tested at its real home in test_memory_system.py.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_surfaces_episodes_deleted_from_consolidate():
    feature, agent = _make_memory_feature(dispatcher=None)
    agent.memory_system.consolidate = AsyncMock(return_value={
        "episodes_created": 2, "patterns_found": 0, "messages_archived": 0,
        "episodes_deleted": 3,
    })

    result = await feature.memory_consolidate()

    assert result.data.get("episodes_deleted") == 3
    assert "3 episode(s) forgotten" in result.confirmation


@pytest.mark.asyncio
async def test_tool_no_forgotten_clause_when_none_deleted():
    feature, agent = _make_memory_feature(dispatcher=None)
    agent.memory_system.consolidate = AsyncMock(return_value={
        "episodes_created": 1, "patterns_found": 0, "messages_archived": 0,
        "episodes_deleted": 0,
    })

    result = await feature.memory_consolidate()

    assert result.data.get("episodes_deleted") == 0
    assert "forgotten" not in result.confirmation
