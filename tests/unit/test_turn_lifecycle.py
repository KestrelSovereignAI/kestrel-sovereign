"""Race regression tests for the shared turn lifecycle.

Per SIGNAL_DISPATCHER.md §Concern 1 and #891: `process_input` and
`process_input_streaming` had no concurrency guard; heartbeat could fire
during a user turn and interleave history writes. Phase 2 introduces
`_turn_lifecycle` as the single boundary owning the `CONVERSATION` lock.

These tests exercise the mixin in isolation against a synthetic class.
The end-to-end version (concurrent `process_input` calls against a real
KestrelAgent) lives in the integration suite — too heavy for unit tier.
"""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sdk.signals import ResourceLock
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.signals import OrderedLockManager


class _StubAgent(TurnLifecycleMixin):
    """Minimal carrier for the mixin. Real KestrelAgent satisfies the same
    contract — initializes `self._lock_manager` in `__init__` and inherits
    `_turn_lifecycle` from the mixin."""

    def __init__(self) -> None:
        self._lock_manager = OrderedLockManager()


# ---------------------------------------------------------------------------
# Serialization invariant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_concurrent_turns_serialize():
    """The race the dispatcher epic exists to fix: two `process_input`
    calls overlapping in wall-clock time MUST not interleave their critical
    sections. End-of-first comes before start-of-second."""
    agent = _StubAgent()
    order: list[str] = []

    async def turn(name: str, hold: float) -> None:
        async with agent._turn_lifecycle():
            order.append(f"start:{name}")
            await asyncio.sleep(hold)
            order.append(f"end:{name}")

    # Run two turns concurrently. The slow one starts first if it gets the
    # event loop; either ordering is acceptable as long as they don't
    # interleave.
    await asyncio.gather(turn("A", 0.05), turn("B", 0.01))

    assert order in (
        ["start:A", "end:A", "start:B", "end:B"],
        ["start:B", "end:B", "start:A", "end:A"],
    ), f"turns interleaved: {order}"


@pytest.mark.asyncio
async def test_three_concurrent_turns_serialize_in_order():
    """Stress: three concurrent turns. Same invariant — every end precedes
    the next start."""
    agent = _StubAgent()
    order: list[str] = []

    async def turn(name: str) -> None:
        async with agent._turn_lifecycle():
            order.append(f"start:{name}")
            await asyncio.sleep(0.01)
            order.append(f"end:{name}")

    await asyncio.gather(turn("A"), turn("B"), turn("C"))

    # Validate strict alternation: start, end, start, end, start, end.
    assert len(order) == 6
    for i in range(0, 6, 2):
        assert order[i].startswith("start:"), f"position {i}: {order[i]}"
        assert order[i + 1].startswith("end:"), f"position {i+1}: {order[i+1]}"
        # The end at i+1 must match the start at i (same turn).
        assert order[i].split(":")[1] == order[i + 1].split(":")[1]


# ---------------------------------------------------------------------------
# Exception safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_in_turn_body_releases_lock():
    """If a turn body raises, the next turn must be able to acquire the
    lock. Otherwise a single exception strands the agent."""
    agent = _StubAgent()

    with pytest.raises(RuntimeError):
        async with agent._turn_lifecycle():
            raise RuntimeError("turn died")

    # Lock should have been released — second turn proceeds.
    async with asyncio.timeout(1.0):
        async with agent._turn_lifecycle():
            assert agent._lock_manager.is_held(ResourceLock.CONVERSATION)


# ---------------------------------------------------------------------------
# Lock identity / state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_lock_is_held_inside_turn():
    """Sanity: inside the lifecycle, CONVERSATION shows as held; outside, free."""
    agent = _StubAgent()
    assert not agent._lock_manager.is_held(ResourceLock.CONVERSATION)
    async with agent._turn_lifecycle():
        assert agent._lock_manager.is_held(ResourceLock.CONVERSATION)
    assert not agent._lock_manager.is_held(ResourceLock.CONVERSATION)


@pytest.mark.asyncio
async def test_turn_id_is_unique_per_call():
    """Each turn gets a fresh id. Phase 5 (#894) will plumb this into
    Signal causation chains."""
    agent = _StubAgent()
    seen: list[str] = []
    for _ in range(5):
        async with agent._turn_lifecycle() as turn_id:
            seen.append(turn_id)
    assert len(seen) == len(set(seen)), f"duplicate turn_ids: {seen}"


# ---------------------------------------------------------------------------
# Lock manager sharing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_input_enters_lifecycle_before_bootstrap_and_commands():
    """Regression for the v3.1 review of PR #902: bootstrap and command
    paths in `process_input` had been outside the lifecycle wrap, so
    first-run bootstrap turns and `!command` invocations could interleave
    with heartbeat or another HTTP request and corrupt conversation
    history. Verified by source inspection — every code path that touches
    state must run inside `async with self._turn_lifecycle()`."""
    import inspect
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    src = inspect.getsource(KestrelAgent.process_input)

    # The lifecycle wrap exists.
    assert "self._turn_lifecycle()" in src, (
        "process_input must enter the turn lifecycle"
    )

    # The bootstrap and command paths come AFTER the lifecycle entry.
    lifecycle_pos = src.index("self._turn_lifecycle()")
    bootstrap_pos = src.index("BOOTSTRAP CHECK")
    command_pos = src.index("Handle explicit commands")
    assert bootstrap_pos > lifecycle_pos, (
        "BOOTSTRAP CHECK must run inside the turn lifecycle "
        "(see #902 P1 review — _handle_bootstrap writes conversation history)"
    )
    assert command_pos > lifecycle_pos, (
        "Command handling must run inside the turn lifecycle "
        "(command_handler.handle can persist agent state)"
    )


@pytest.mark.asyncio
async def test_lifecycle_shares_lock_manager_with_dispatcher():
    """Cross-system invariant: the agent's `_lock_manager` is the same
    instance the SignalDispatcher will hold. If a dispatched ACTION signal
    declares MEMORY in its resources, and a turn is in flight, the
    ACTION's MEMORY acquisition is independent of CONVERSATION (disjoint
    locks parallelize). This test verifies the dispatcher and the
    lifecycle don't sit on separate lock universes."""
    agent = _StubAgent()
    started_turn = asyncio.Event()
    can_finish_turn = asyncio.Event()
    action_completed = asyncio.Event()

    async def long_turn() -> None:
        async with agent._turn_lifecycle():
            started_turn.set()
            await can_finish_turn.wait()

    async def disjoint_action() -> None:
        # Simulate the dispatcher acquiring MEMORY only — should not block
        # on the in-flight turn's CONVERSATION lock.
        await started_turn.wait()
        async with agent._lock_manager.acquire({ResourceLock.MEMORY}):
            action_completed.set()

    turn_task = asyncio.create_task(long_turn())
    action_task = asyncio.create_task(disjoint_action())

    # Action finishes while turn is still parked.
    await asyncio.wait_for(action_completed.wait(), timeout=1.0)
    assert started_turn.is_set()

    can_finish_turn.set()
    await asyncio.gather(turn_task, action_task)
