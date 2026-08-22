"""#1674 P3: the sleep cycle consolidates through the single
MemorySystem.consolidate() chokepoint (so it inherits the forgetting deletion
tier), not the lower-level MemoryConsolidator.run_consolidation()."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
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
async def test_nightly_sleep_inherits_memory_system_consolidation_timeout(monkeypatch):
    """The production sleep path must use the chokepoint's hard deadline."""
    from kestrel_sovereign.storage.memory_system import MemorySystem

    entered = asyncio.Event()
    unwound = asyncio.Event()

    async def run_consolidation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            unwound.set()

    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 0.02},
    )
    memory_system = MemorySystem(storage=SimpleNamespace(), agent_id="did:test:sleep")
    memory_system.consolidator = SimpleNamespace(
        run_consolidation=run_consolidation
    )
    agent = _Agent()
    agent.memory_system = memory_system
    agent.sleep_hooks = []

    report = await agent.sleep(
        skip_export=True,
        skip_reflection=True,
    )

    assert entered.is_set()
    assert unwound.is_set()
    assert report.success is False
    assert report.error == "consolidation_failed"


@pytest.mark.asyncio
async def test_nightly_sleep_timeout_preserves_prior_retention_failure():
    """A consolidation deadline cannot hide an earlier retention failure."""
    from kestrel_sovereign.storage.memory_system import (
        MemoryConsolidationTimeoutError,
    )

    class _FailingSweepStorage:
        async def sweep_expired_governed_semantic_artifacts(self):
            raise RuntimeError("private retention detail")

    agent = _Agent()
    agent.storage = _FailingSweepStorage()
    agent.sleep_hooks = []
    agent._consolidate_memories = AsyncMock(
        side_effect=MemoryConsolidationTimeoutError(0.02)
    )

    report = await agent.sleep(skip_export=True, skip_reflection=True)

    assert report.success is False
    assert report.error == (
        "semantic_artifact_expiry_sweep_failed; consolidation_failed"
    )
    assert "private retention detail" not in report.error


@pytest.mark.asyncio
async def test_nightly_sleep_logs_invalid_timeout_cause(monkeypatch, caplog):
    """A bad operator value must remain visible after sleep records failure."""
    from kestrel_sovereign.storage.memory_system import MemorySystem

    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 0},
    )
    memory_system = MemorySystem(storage=SimpleNamespace(), agent_id="did:test:sleep")
    memory_system.consolidator = SimpleNamespace(run_consolidation=AsyncMock())
    agent = _Agent()
    agent.memory_system = memory_system
    agent.sleep_hooks = []

    with caplog.at_level(logging.ERROR):
        report = await agent.sleep(
            skip_export=True,
            skip_reflection=True,
        )

    record = next(
        record for record in caplog.records
        if record.message == "Consolidation failed"
    )
    assert report.error == "consolidation_failed"
    assert record.exc_info is not None
    assert (
        "retrieval.memory_consolidation_timeout_seconds must be positive"
        in str(record.exc_info[1])
    )


class _PreHook:
    """A sleep hook that only implements on_pre_sleep."""
    def __init__(self, insights: int):
        self.insights = insights
        self.called = False

    async def on_pre_sleep(self, agent):
        self.called = True
        return {"success": True, "insights_generated": self.insights}


@pytest.mark.asyncio
async def test_sleep_dispatches_all_registered_sleep_hooks():
    """The sleep cycle iterates the sleep_hooks list (not a single slot),
    firing every hook and aggregating their results."""
    agent = _Agent()
    h1, h2 = _PreHook(2), _PreHook(3)
    agent.sleep_hooks = [h1, h2]

    report = await agent.sleep(skip_consolidation=True, skip_export=True)

    assert h1.called and h2.called                 # both hooks fired
    assert len(report.pre_reflection) == 2          # one result dict per hook
    assert report.insights_generated == 5           # aggregated across hooks


@pytest.mark.asyncio
async def test_sleep_tolerates_no_sleep_hooks():
    """No hooks registered → sleep still runs, no reflection results."""
    agent = _Agent()
    report = await agent.sleep(skip_consolidation=True, skip_export=True)
    assert report.pre_reflection == []
    assert report.insights_generated == 0


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
