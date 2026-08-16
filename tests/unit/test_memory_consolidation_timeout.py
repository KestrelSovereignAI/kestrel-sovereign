"""Memory consolidation must never strand the MEMORY resource lock (#2907)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kestrel_sdk.signals import ResourceLock
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.storage.memory_system import (
    DEFAULT_CONSOLIDATION_TIMEOUT_SECONDS,
    MemoryConsolidationTimeoutError,
    MemorySystem,
    _consolidation_timeout_seconds,
)
from kestrel_sovereign.signals.lock_manager import OrderedLockManager


def _feature(memory_system, locks: OrderedLockManager) -> MemoryFeature:
    agent = SimpleNamespace(
        agent_name="Claw",
        memory_system=memory_system,
        dispatcher=SimpleNamespace(lock_manager=locks),
    )
    return MemoryFeature(agent)


def test_consolidation_timeout_uses_retrieval_config(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda section: {"memory_consolidation_timeout_seconds": 42.5}
        if section == "retrieval"
        else {},
    )

    assert _consolidation_timeout_seconds() == 42.5


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), "not-a-number"])
def test_invalid_consolidation_timeout_fails_fast(monkeypatch, value):
    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": value},
    )

    with pytest.raises(
        ValueError,
        match="retrieval.memory_consolidation_timeout_seconds must be positive",
    ):
        _consolidation_timeout_seconds()


def test_consolidation_timeout_has_safe_default(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {},
    )

    assert _consolidation_timeout_seconds() == DEFAULT_CONSOLIDATION_TIMEOUT_SECONDS


async def test_memory_system_chokepoint_times_out_in_owner_task(monkeypatch):
    entered = asyncio.Event()
    unwound = asyncio.Event()
    owner_task = asyncio.current_task()
    observed_tasks = []

    async def run_consolidation():
        observed_tasks.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            observed_tasks.append(asyncio.current_task())
            unwound.set()

    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 0.02},
    )
    memory_system = MemorySystem(storage=SimpleNamespace(), agent_id="did:test:claw")
    memory_system.consolidator = SimpleNamespace(
        run_consolidation=run_consolidation
    )

    with pytest.raises(MemoryConsolidationTimeoutError):
        await memory_system.consolidate()

    assert entered.is_set()
    assert unwound.is_set()
    assert observed_tasks == [owner_task, owner_task]


async def test_unrelated_timeout_is_not_reclassified_as_deadline(monkeypatch):
    async def run_consolidation():
        raise TimeoutError("embedding provider timed out")

    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 60},
    )
    memory_system = MemorySystem(storage=SimpleNamespace(), agent_id="did:test:io")
    memory_system.consolidator = SimpleNamespace(
        run_consolidation=run_consolidation
    )

    with pytest.raises(TimeoutError, match="embedding provider timed out") as caught:
        await memory_system.consolidate()

    assert not isinstance(caught.value, MemoryConsolidationTimeoutError)


async def test_tool_surfaces_unrelated_timeout_without_deadline_claim(monkeypatch):
    locks = OrderedLockManager()
    memory_system = SimpleNamespace(
        consolidate=AsyncMock(side_effect=TimeoutError("provider socket timeout"))
    )
    feature = _feature(memory_system, locks)
    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 60},
    )

    result = await feature.memory_consolidate()

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "provider socket timeout"
    assert "configured deadline" not in result.error


async def test_invalid_timeout_is_returned_as_tool_failure(monkeypatch):
    locks = OrderedLockManager()
    memory_system = SimpleNamespace(consolidate=AsyncMock())
    feature = _feature(memory_system, locks)
    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 0},
    )

    result = await feature.memory_consolidate()

    assert result.status is ToolResultStatus.ERROR
    assert (
        "retrieval.memory_consolidation_timeout_seconds must be positive"
        in result.error
    )
    memory_system.consolidate.assert_not_awaited()
    assert not locks.is_held(ResourceLock.MEMORY)


async def test_hung_consolidation_times_out_in_owner_task_and_releases_lock(
    monkeypatch,
):
    locks = OrderedLockManager()
    entered = asyncio.Event()
    unwound = asyncio.Event()
    owner_task = asyncio.current_task()
    observed_tasks = []
    holder_labels = []

    async def run_consolidation():
        observed_tasks.append(asyncio.current_task())
        holder_labels.append(locks.holder(ResourceLock.MEMORY).label)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            observed_tasks.append(asyncio.current_task())
            unwound.set()

    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 0.02},
    )
    memory_system = MemorySystem(storage=SimpleNamespace(), agent_id="did:test:tool")
    memory_system.consolidator = SimpleNamespace(
        run_consolidation=run_consolidation
    )
    feature = _feature(memory_system, locks)

    result = await feature.memory_consolidate()

    assert entered.is_set()
    assert unwound.is_set()
    assert observed_tasks == [owner_task, owner_task]
    assert holder_labels == ["Claw MemoryFeature.memory_consolidate"]
    assert result.status is ToolResultStatus.ERROR
    assert "configured deadline of 0.02 seconds" in result.error
    assert not locks.is_held(ResourceLock.MEMORY)
    assert locks.holder(ResourceLock.MEMORY) is None

    async with asyncio.timeout(0.2):
        async with locks.acquire(
            {ResourceLock.MEMORY}, label="following acquirer"
        ):
            assert locks.is_held(ResourceLock.MEMORY)


async def test_parent_cancellation_unwinds_and_releases_memory_lock(monkeypatch):
    locks = OrderedLockManager()
    entered = asyncio.Event()
    unwound = asyncio.Event()
    observed_tasks = []

    async def consolidate():
        observed_tasks.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            observed_tasks.append(asyncio.current_task())
            unwound.set()

    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 60},
    )
    feature = _feature(SimpleNamespace(consolidate=consolidate), locks)
    call = asyncio.create_task(feature.memory_consolidate())

    async with asyncio.timeout(0.2):
        await entered.wait()
    assert locks.is_held(ResourceLock.MEMORY)
    assert locks.holder(ResourceLock.MEMORY).label == (
        "Claw MemoryFeature.memory_consolidate"
    )

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert unwound.is_set()
    assert observed_tasks == [call, call]
    assert not locks.is_held(ResourceLock.MEMORY)
    assert locks.holder(ResourceLock.MEMORY) is None

    async with asyncio.timeout(0.2):
        async with locks.acquire(
            {ResourceLock.MEMORY}, label="following cancellation"
        ):
            assert locks.is_held(ResourceLock.MEMORY)


async def test_tool_skips_same_task_dispatch_lock_reacquisition(monkeypatch):
    locks = OrderedLockManager()
    memory_system = SimpleNamespace(
        consolidate=AsyncMock(
            return_value={
                "episodes_created": 1,
                "patterns_found": 0,
                "messages_archived": 0,
                "episodes_deleted": 0,
            }
        )
    )
    feature = _feature(memory_system, locks)
    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 0.2},
    )

    async with locks.acquire(
        {ResourceLock.MEMORY}, label="cron memory_consolidate"
    ):
        result = await feature.memory_consolidate()
        assert locks.holder(ResourceLock.MEMORY).label == "cron memory_consolidate"

    memory_system.consolidate.assert_awaited_once()
    assert result.status is ToolResultStatus.OK


async def test_lock_wait_does_not_consume_consolidation_budget(monkeypatch):
    locks = OrderedLockManager()
    held = asyncio.Event()
    consolidation_started = asyncio.Event()

    async def consolidate():
        consolidation_started.set()
        await asyncio.sleep(0.08)
        return {"episodes_created": 1}

    async def hold_memory():
        async with locks.acquire({ResourceLock.MEMORY}, label="other task"):
            held.set()
            await asyncio.sleep(0.15)

    holder_task = asyncio.create_task(hold_memory())
    await held.wait()
    feature = _feature(SimpleNamespace(consolidate=consolidate), locks)
    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 0.1},
    )

    result = await feature.memory_consolidate()
    await holder_task

    assert consolidation_started.is_set()
    assert result.status is ToolResultStatus.OK
    assert result.data["episodes_created"] == 1
    assert not locks.is_held(ResourceLock.MEMORY)


async def test_blocked_lock_acquisition_has_its_own_failure(monkeypatch):
    locks = OrderedLockManager()
    held = asyncio.Event()
    release = asyncio.Event()
    memory_system = SimpleNamespace(consolidate=AsyncMock())
    feature = _feature(memory_system, locks)

    async def hold_memory():
        async with locks.acquire({ResourceLock.MEMORY}, label="other task"):
            held.set()
            await release.wait()

    holder_task = asyncio.create_task(hold_memory())
    await held.wait()
    monkeypatch.setattr(
        "kestrel_sovereign.storage.memory_system.load_section",
        lambda _section: {"memory_consolidation_timeout_seconds": 0.02},
    )

    result = await feature.memory_consolidate()

    assert result.status is ToolResultStatus.ERROR
    assert "could not acquire the memory lock" in result.error
    assert "consolidation did not start" in result.error
    assert "configured 0.02-second deadline" not in result.error
    memory_system.consolidate.assert_not_awaited()
    assert locks.holder(ResourceLock.MEMORY).label == "other task"

    release.set()
    await holder_task
    assert not locks.is_held(ResourceLock.MEMORY)
