"""A stalled lock holder must be legible from the logs (#2770).

Filed after a prod incident on 2026-07-27: an agent's turns all stalled inside
the CONVERSATION lock. The last INFO record for that agent was
``process_input called``, and ``turn_lifecycle`` logged begin/end at DEBUG, so
there was no log line on *either* side of the blocking call. Its non-turn
endpoints kept answering in ~4ms, so the agent looked simultaneously healthy and
completely dead, and locating the region cost a manual bisect.

These tests pin the diagnostics, not the enforcement. Acquisition still blocks
indefinitely on purpose: a turn can legitimately run for minutes (the Anthropic
SDK's default read timeout alone is 600s, retried up to 5 times), so a hard
acquire deadline would cancel healthy work.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from kestrel_sdk.signals import ResourceLock

from kestrel_sovereign.signals.lock_manager import (
    LockHolder,
    OrderedLockManager,
    lock_sort_key,
)


@pytest.fixture
def fast_thresholds(monkeypatch):
    """Shrink the real thresholds so tests assert behavior, not wall-clock."""
    monkeypatch.setattr(
        "kestrel_sovereign.signals.lock_manager.SLOW_WAIT_WARN_SECONDS", 0.02
    )
    monkeypatch.setattr(
        "kestrel_sovereign.signals.lock_manager.SLOW_HOLD_WARN_SECONDS", 0.02
    )


async def test_holder_is_recorded_and_cleared():
    mgr = OrderedLockManager()

    assert mgr.holder(ResourceLock.CONVERSATION) is None
    async with mgr.acquire({ResourceLock.CONVERSATION}, label="Nellie turn_abc"):
        holder = mgr.holder(ResourceLock.CONVERSATION)
        assert isinstance(holder, LockHolder)
        assert holder.label == "Nellie turn_abc"
        assert holder.held_seconds() >= 0
    assert mgr.holder(ResourceLock.CONVERSATION) is None


async def test_holder_defaults_to_lock_name_without_a_label():
    mgr = OrderedLockManager()
    async with mgr.acquire({ResourceLock.CONVERSATION}):
        assert mgr.holder(ResourceLock.CONVERSATION).label == (
            ResourceLock.CONVERSATION.value
        )


async def test_long_hold_is_reported_while_it_is_still_happening(
    fast_thresholds, caplog
):
    """The signal that was missing: a warning from the holder's side, naming it,
    emitted *during* the stall rather than only in hindsight."""
    mgr = OrderedLockManager()
    with caplog.at_level(logging.WARNING):
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="Nellie turn_abc"):
            await asyncio.sleep(0.1)

    assert "has held the conversation lock" in caplog.text
    assert "Nellie turn_abc" in caplog.text


async def test_a_blocked_waiter_reports_who_is_holding(fast_thresholds, caplog):
    """A waiter's warning must name the *holder* — that is the fact an operator
    cannot otherwise recover from the logs."""
    mgr = OrderedLockManager()
    released = asyncio.Event()

    async def holder():
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="holding-turn"):
            await released.wait()

    async def waiter():
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="blocked-turn"):
            pass

    with caplog.at_level(logging.WARNING):
        holding = asyncio.ensure_future(holder())
        await asyncio.sleep(0.01)
        blocked = asyncio.ensure_future(waiter())
        await asyncio.sleep(0.1)
        released.set()
        await asyncio.gather(holding, blocked)

    assert "blocked-turn" in caplog.text
    assert "waiting" in caplog.text
    assert "held" in caplog.text and "holding-turn" in caplog.text


async def test_quiet_when_nothing_is_slow(fast_thresholds, caplog):
    """Diagnostics must not narrate healthy operation."""
    mgr = OrderedLockManager()
    with caplog.at_level(logging.WARNING):
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="fast-turn"):
            pass

    assert caplog.text == ""


async def test_watchdogs_do_not_outlive_the_hold(fast_thresholds, caplog):
    """A cancelled watchdog must not keep reporting a hold that already ended,
    and must not leak a task past the context manager."""
    mgr = OrderedLockManager()
    before = len(asyncio.all_tasks())

    async with mgr.acquire({ResourceLock.CONVERSATION}, label="short-turn"):
        pass

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await asyncio.sleep(0.1)

    assert "short-turn" not in caplog.text
    assert len(asyncio.all_tasks()) <= before + 1


async def test_lock_is_released_when_the_body_raises(fast_thresholds):
    mgr = OrderedLockManager()

    with pytest.raises(RuntimeError):
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="boom"):
            raise RuntimeError("boom")

    assert not mgr.is_held(ResourceLock.CONVERSATION)
    assert mgr.holder(ResourceLock.CONVERSATION) is None


async def test_cancelling_a_blocked_waiter_does_not_strand_the_lock(fast_thresholds):
    """Regression guard for the shape this instrumentation deliberately avoids:
    cancelling a timed-out acquire that had already succeeded would drop the lock
    on the floor. The watchdog runs beside the acquire, never wrapping it."""
    mgr = OrderedLockManager()
    released = asyncio.Event()

    async def holder():
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="holding"):
            await released.wait()

    async def waiter():
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="doomed"):
            pass

    holding = asyncio.ensure_future(holder())
    await asyncio.sleep(0.01)
    blocked = asyncio.ensure_future(waiter())
    await asyncio.sleep(0.05)
    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked

    released.set()
    await holding

    # The holder released, and the cancelled waiter never took ownership.
    assert not mgr.is_held(ResourceLock.CONVERSATION)
    assert mgr.holder(ResourceLock.CONVERSATION) is None

    # Still usable afterwards.
    async with mgr.acquire({ResourceLock.CONVERSATION}, label="after"):
        assert mgr.holder(ResourceLock.CONVERSATION).label == "after"


async def test_multi_lock_acquire_still_orders_and_releases_all(fast_thresholds):
    """Instrumentation must not disturb the ordering invariant or release."""
    mgr = OrderedLockManager()
    names = {ResourceLock.CONVERSATION, ResourceLock.MEMORY, ResourceLock.SCHEDULER}

    async with mgr.acquire(names, label="multi"):
        for name in names:
            assert mgr.is_held(name)
            assert mgr.holder(name).label == "multi"

    for name in names:
        assert not mgr.is_held(name)
        assert mgr.holder(name) is None

    # CONVERSATION must still sort last.
    assert sorted(names, key=lock_sort_key)[-1] is ResourceLock.CONVERSATION


async def test_empty_acquire_is_a_noop(fast_thresholds):
    mgr = OrderedLockManager()
    async with mgr.acquire(set(), label="none"):
        pass
    assert mgr.holder(ResourceLock.CONVERSATION) is None
