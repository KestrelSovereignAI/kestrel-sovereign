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
    """The signal that was missing: a report from the holder's side, naming it,
    emitted *during* the stall rather than only in hindsight."""
    mgr = OrderedLockManager()
    with caplog.at_level(logging.INFO):
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="Nellie turn_abc"):
            await asyncio.sleep(0.1)

    assert "has held the conversation lock" in caplog.text
    assert "Nellie turn_abc" in caplog.text


async def test_a_slow_solo_turn_is_not_a_warning(fast_thresholds, caplog):
    """A long hold with nobody blocked is a slow turn, not a fault.

    Emitting WARNING here would fire on every legitimate long generation (the
    tolerated envelope is ~600s per attempt, retried), and WARNING-keyed alerting
    would learn to ignore the one line this change exists to make meaningful.
    """
    mgr = OrderedLockManager()
    with caplog.at_level(logging.INFO):
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="solo-turn"):
            await asyncio.sleep(0.1)

    assert "nothing is waiting on it" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_a_long_hold_that_blocks_someone_is_a_warning(fast_thresholds, caplog):
    """Contention is the incident shape, and only here may the log claim work is
    queued — with a real count, not an assumption."""
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

    assert "holding-turn" in caplog.text
    assert "1 acquirer(s) blocked behind it" in caplog.text


async def test_waiter_count_does_not_leak_when_a_waiter_is_cancelled(
    fast_thresholds, caplog
):
    """A cancelled waiter that stayed counted would make every later holder
    report phantom queued turns for the life of the process."""
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
    await asyncio.sleep(0.03)
    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked
    released.set()
    await holding

    # A later solo hold must report itself as unblocked, not as contended.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="later-solo"):
            await asyncio.sleep(0.1)

    assert "nothing is waiting on it" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


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
    and must not leak a task past the context manager.

    The task-count assertion is exact: ``acquire`` now awaits its cancelled
    watchdogs, so by the time the block exits none of them remain pending. A
    tolerance of +1 here would have let a single genuinely-leaked watchdog pass.
    """
    mgr = OrderedLockManager()
    before = asyncio.all_tasks()

    async with mgr.acquire({ResourceLock.CONVERSATION}, label="short-turn"):
        pass

    assert asyncio.all_tasks() == before

    caplog.clear()
    with caplog.at_level(logging.INFO):
        await asyncio.sleep(0.1)

    # A surviving watchdog would have logged within this window, since the
    # patched threshold is 0.02s.
    assert "short-turn" not in caplog.text


async def test_lock_is_released_when_the_holder_is_cancelled(fast_thresholds):
    """Release must survive cancellation of the holding task.

    The diagnostics must never be able to cause the failure they exist to
    report: if release were sequenced after an await in the cleanup path, a
    cancellation landing there would strand the lock and wedge every later turn
    for that agent — the exact shape of #2770.
    """
    mgr = OrderedLockManager()
    entered = asyncio.Event()

    async def holder():
        async with mgr.acquire({ResourceLock.CONVERSATION}, label="doomed-holder"):
            entered.set()
            await asyncio.sleep(3600)

    task = asyncio.ensure_future(holder())
    await entered.wait()
    assert mgr.is_held(ResourceLock.CONVERSATION)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not mgr.is_held(ResourceLock.CONVERSATION)
    assert mgr.holder(ResourceLock.CONVERSATION) is None

    # And the lock is genuinely reusable, not merely reported free.
    async with mgr.acquire({ResourceLock.CONVERSATION}, label="after-cancel"):
        assert mgr.holder(ResourceLock.CONVERSATION).label == "after-cancel"


async def test_release_precedes_any_await_in_cleanup():
    """Pin the ordering structurally, not just behaviorally.

    A future refactor could reintroduce an await before the release loop and
    still pass the cancellation test above under single-cancel timing, because
    the hazard needs a second cancellation to bite. Assert on the source that
    release comes first.
    """
    import inspect

    from kestrel_sovereign.signals import lock_manager

    source = inspect.getsource(lock_manager.OrderedLockManager.acquire.__wrapped__)
    cleanup = source.split("finally:", 1)[1]
    # Strip comments: the surrounding prose legitimately discusses awaiting, and
    # matching that would assert on documentation rather than on control flow.
    code = "\n".join(
        line for line in cleanup.splitlines() if not line.strip().startswith("#")
    )
    release_at = code.index("lock.release()")
    first_await = code.index("await ")

    assert release_at < first_await, (
        "locks must be released before any suspension point in cleanup; an "
        "await placed first makes release non-cancellation-safe"
    )


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
