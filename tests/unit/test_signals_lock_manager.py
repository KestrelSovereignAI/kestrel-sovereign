"""Unit tests for the OrderedLockManager.

The lock manager is the only new concurrency primitive in Phase 1 — the
deadlock-free invariant (single ordered lock manager, lex acquisition)
must hold under contention.
"""

import asyncio

import pytest

from kestrel_sdk.signals import ResourceLock
from kestrel_sovereign.signals import OrderedLockManager
from kestrel_sovereign.signals.lock_manager import lock_sort_key


def test_conversation_sorts_last_in_canonical_order():
    """Load-bearing invariant: CONVERSATION is acquired LAST regardless of
    its alphabetical position. Phase 2's turn lifecycle relies on this —
    the dispatcher acquires registered resources, then enters a code path
    that acquires CONVERSATION; if CONVERSATION sorted first by enum value
    (alphabetically before MEMORY/SCHEDULER/WALLET), the lock-order
    invariant in the design would be violated."""
    all_locks = list(ResourceLock)
    ordered = sorted(all_locks, key=lock_sort_key)
    assert ordered[-1] == ResourceLock.CONVERSATION, (
        f"CONVERSATION must sort last; got order: "
        f"{[l.name for l in ordered]}"
    )
    # Non-CONVERSATION locks lex-order on enum value among themselves.
    others = [l for l in ordered if l != ResourceLock.CONVERSATION]
    assert others == sorted(others, key=lambda l: l.value)


def test_subset_with_conversation_still_orders_it_last():
    """Even when CONVERSATION is the only lex-early item in the subset,
    it goes last."""
    subset = {ResourceLock.CONVERSATION, ResourceLock.MEMORY}
    ordered = sorted(subset, key=lock_sort_key)
    assert ordered == [ResourceLock.MEMORY, ResourceLock.CONVERSATION]


@pytest.mark.asyncio
async def test_acquire_empty_set_is_noop():
    mgr = OrderedLockManager()
    async with mgr.acquire([]):
        pass


@pytest.mark.asyncio
async def test_acquire_releases_on_normal_exit():
    mgr = OrderedLockManager()
    async with mgr.acquire({ResourceLock.MEMORY}):
        assert mgr.is_held(ResourceLock.MEMORY)
    assert not mgr.is_held(ResourceLock.MEMORY)


@pytest.mark.asyncio
async def test_acquire_releases_on_exception():
    mgr = OrderedLockManager()
    with pytest.raises(RuntimeError):
        async with mgr.acquire({ResourceLock.MEMORY, ResourceLock.WALLET}):
            raise RuntimeError("boom")
    assert not mgr.is_held(ResourceLock.MEMORY)
    assert not mgr.is_held(ResourceLock.WALLET)


@pytest.mark.asyncio
async def test_concurrent_holders_serialize():
    """Two coroutines acquiring the same single lock must serialize.
    Without this, the dispatcher's per-resource serialization invariant
    would be a lie."""
    mgr = OrderedLockManager()
    order: list[str] = []

    async def holder(name: str, hold_for: float) -> None:
        async with mgr.acquire({ResourceLock.MEMORY}):
            order.append(f"start:{name}")
            await asyncio.sleep(hold_for)
            order.append(f"end:{name}")

    await asyncio.gather(
        holder("A", 0.05),
        holder("B", 0.01),
    )
    # Whichever started first must end before the other starts.
    assert order in (
        ["start:A", "end:A", "start:B", "end:B"],
        ["start:B", "end:B", "start:A", "end:A"],
    ), order


@pytest.mark.asyncio
async def test_lex_order_invariant_no_deadlock_under_overlap():
    """Two coroutines acquiring overlapping sets in different orders must
    not deadlock — the lex sort inside `acquire` is the whole guarantee.
    Without it, A holding MEMORY then asking for WALLET while B holds
    WALLET asking for MEMORY would deadlock."""
    mgr = OrderedLockManager()

    async def task_a() -> str:
        async with mgr.acquire({ResourceLock.MEMORY, ResourceLock.WALLET}):
            await asyncio.sleep(0.02)
            return "A"

    async def task_b() -> str:
        # Same logical set, declared in different iteration order — the
        # manager normalizes via sorted().
        async with mgr.acquire({ResourceLock.WALLET, ResourceLock.MEMORY}):
            await asyncio.sleep(0.02)
            return "B"

    results = await asyncio.wait_for(
        asyncio.gather(task_a(), task_b()), timeout=1.0
    )
    assert sorted(results) == ["A", "B"]


@pytest.mark.asyncio
async def test_manager_can_acquire_conversation_with_others():
    """Smoke test that the manager accepts CONVERSATION as part of an
    acquisition set without erroring. (The dispatcher rejects sources that
    declare CONVERSATION; this test exercises the lower-level manager
    directly, since Phase 2's turn lifecycle will use it that way.)"""
    mgr = OrderedLockManager()
    async with mgr.acquire({ResourceLock.CONVERSATION, ResourceLock.MEMORY}):
        assert mgr.is_held(ResourceLock.MEMORY)
        assert mgr.is_held(ResourceLock.CONVERSATION)
    assert not mgr.is_held(ResourceLock.MEMORY)
    assert not mgr.is_held(ResourceLock.CONVERSATION)


@pytest.mark.asyncio
async def test_disjoint_sets_run_in_parallel():
    """Two coroutines holding disjoint locks should not block each other —
    parallel execution is the whole point of named locks vs. one big lock."""
    mgr = OrderedLockManager()
    started = asyncio.Event()
    can_finish = asyncio.Event()

    async def holds_memory() -> str:
        async with mgr.acquire({ResourceLock.MEMORY}):
            started.set()
            await can_finish.wait()
            return "memory"

    async def holds_wallet() -> str:
        # Wait until memory holder is in critical section, then we should
        # be able to acquire WALLET without blocking.
        await started.wait()
        async with mgr.acquire({ResourceLock.WALLET}):
            return "wallet"

    memory_task = asyncio.create_task(holds_memory())
    wallet_task = asyncio.create_task(holds_wallet())

    wallet_result = await asyncio.wait_for(wallet_task, timeout=1.0)
    assert wallet_result == "wallet"

    can_finish.set()
    memory_result = await asyncio.wait_for(memory_task, timeout=1.0)
    assert memory_result == "memory"
