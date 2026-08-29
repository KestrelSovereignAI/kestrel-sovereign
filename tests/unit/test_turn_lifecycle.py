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
from datetime import datetime, timezone

import pytest
from kestrel_sdk.signals import CausationFrame, ResourceLock

from kestrel_sovereign.agent.invocation import invocation_scope
from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
from kestrel_sovereign.agent.turn_lifecycle import (
    TurnLifecycleMixin,
    bind_turn_session,
    capture_turn_session_binding,
)
from kestrel_sovereign.signals import OrderedLockManager


class _StubAgent(TurnLifecycleMixin):
    """Minimal carrier for the mixin. Real KestrelAgent satisfies the same
    contract — initializes `self._lock_manager` in `__init__` and inherits
    `_turn_lifecycle` from the mixin."""

    def __init__(self) -> None:
        self._lock_manager = OrderedLockManager()


class _PrivateAccessorOnlyAgent(_StubAgent):
    """Agent shape supported during the public-accessor migration window."""

    get_turn_bound_session_id = None

    def _get_turn_bound_session_id(self):
        return self._active_session_id


class _RequestTurnAgent(RequestLifecycleMixin, TurnLifecycleMixin):
    """Minimal carrier for the request-generation/turn-index seam."""

    def __init__(self) -> None:
        self._lock_manager = OrderedLockManager()
        self._current_request_id = None
        self._active_request_ids = set()
        self._active_request_counts = {}
        self._active_request_started_at = {}
        self._cancelled_requests = set()


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


@pytest.mark.asyncio
async def test_turn_lifecycle_indexes_task_local_request_until_exit():
    agent = _StubAgent()

    with invocation_scope("request-visible-to-stop"):
        async with agent._turn_lifecycle() as turn_id:
            assert (
                agent.resolve_turn_request_id(turn_id)
                == "request-visible-to-stop"
            )

    assert agent.resolve_turn_request_id(turn_id) is None


@pytest.mark.asyncio
async def test_turn_index_carries_the_exact_request_generation():
    agent = _RequestTurnAgent()

    with invocation_scope("reusable-request"):
        generation = agent.register_active_request("reusable-request")
        async with agent._turn_lifecycle() as turn_id:
            assert agent.active_turn_request_bindings() == {
                turn_id: ("reusable-request", generation)
            }
            assert agent.active_turn_request_ids() == {
                turn_id: "reusable-request"
            }

    assert agent.active_turn_request_bindings() == {}


@pytest.mark.asyncio
async def test_turn_request_index_cleanup_survives_turn_failure():
    agent = _StubAgent()
    turn_id = None

    with invocation_scope("request-that-fails"):
        with pytest.raises(RuntimeError, match="turn failed"):
            async with agent._turn_lifecycle() as turn_id:
                assert agent.resolve_turn_request_id(turn_id) == "request-that-fails"
                raise RuntimeError("turn failed")

    assert turn_id is not None
    assert agent.resolve_turn_request_id(turn_id) is None


@pytest.mark.asyncio
async def test_repeated_request_id_gets_distinct_live_turn_bindings():
    agent = _StubAgent()
    turn_ids = []

    for _ in range(2):
        with invocation_scope("retry-request"):
            async with agent._turn_lifecycle() as turn_id:
                turn_ids.append(turn_id)
                assert agent.resolve_turn_request_id(turn_id) == "retry-request"

    assert turn_ids[0] != turn_ids[1]
    assert all(agent.resolve_turn_request_id(turn_id) is None for turn_id in turn_ids)


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


# ---------------------------------------------------------------------------
# Causation chain isolation across concurrent COGNITION dispatches
# (#906 review P1)
# ---------------------------------------------------------------------------


def _frame(agent_id: str, source: str, depth: int) -> CausationFrame:
    return CausationFrame(
        agent_id=agent_id,
        source=source,
        signal_id=f"sig-{agent_id}-{depth}",
        turn_id=None,
        depth=depth,
        emitted_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_concurrent_set_current_chain_is_task_local():
    """Two coroutines each set/get/clear their own chain concurrently.
    Without ContextVar isolation (the #906 P1 fix), one would clobber
    the other's view because `_current_chain` was a shared agent
    attribute. With ContextVar, each task sees only its own chain."""
    agent = _StubAgent()

    chain_a = [_frame("agent-A", "heartbeat", 1)]
    chain_b = [_frame("agent-B", "a2a.task_complete", 1),
               _frame("agent-B", "a2a.task_complete", 2)]

    a_started = asyncio.Event()
    b_can_finish = asyncio.Event()

    saw_a: list = []
    saw_b: list = []

    async def task_a() -> None:
        # A enters first, captures its chain, parks.
        token = agent._set_current_chain(chain_a)
        try:
            a_started.set()
            await b_can_finish.wait()
            # While B was running concurrently with its own chain, A's
            # view must remain chain_a — not chain_b.
            saw_a.extend(agent._get_current_chain() or [])
        finally:
            agent._clear_current_chain(token)

    async def task_b() -> None:
        # B starts after A is parked, sets its OWN chain, captures
        # immediately, then signals A to wake.
        await a_started.wait()
        token = agent._set_current_chain(chain_b)
        try:
            saw_b.extend(agent._get_current_chain() or [])
        finally:
            agent._clear_current_chain(token)
        b_can_finish.set()

    await asyncio.gather(task_a(), task_b())

    assert saw_a == chain_a, (
        "task A must see its own chain after task B set a different one — "
        "without ContextVar isolation, B's set() would have clobbered A's"
    )
    assert saw_b == chain_b


@pytest.mark.asyncio
async def test_chain_clears_to_default_on_token_reset():
    """After clear, the next read returns None — the default empty
    chain (translated to None for falsy semantics by _get_current_chain)."""
    agent = _StubAgent()
    assert agent._get_current_chain() is None

    token = agent._set_current_chain([_frame("a", "x", 1)])
    assert agent._get_current_chain() is not None

    agent._clear_current_chain(token)
    assert agent._get_current_chain() is None


@pytest.mark.asyncio
async def test_clear_without_token_falls_back_safely():
    """Defensive call (clear with no token) shouldn't raise. Used by
    the dispatcher when set_chain returned None (e.g. because the
    agent didn't have the mixin)."""
    agent = _StubAgent()
    # Set then clear without the matching token — must not raise.
    agent._set_current_chain([_frame("a", "x", 1)])
    agent._clear_current_chain(None)
    assert agent._get_current_chain() is None


# ---------------------------------------------------------------------------
# Turn-bound session resolution (#2877)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_bound_session_is_the_live_turns_session():
    agent = _StubAgent()
    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-A"
        assert agent.get_turn_bound_session_id() == "chat-A"


@pytest.mark.asyncio
async def test_no_turn_means_no_session():
    """Out-of-turn work (a cron tick, a CLI-filed request) has no chat window,
    even if a turn left a value behind."""
    agent = _StubAgent()
    assert agent.get_turn_bound_session_id() is None

    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-A"
    assert agent.get_turn_bound_session_id() is None


@pytest.mark.asyncio
async def test_child_task_of_the_live_turn_sees_the_session():
    """Tools run in tasks the turn creates and awaits — they must still
    resolve the turn's session, or every tool-side capture breaks."""
    agent = _StubAgent()

    async def tool():
        return agent.get_turn_bound_session_id()

    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-A"
        assert await asyncio.create_task(tool()) == "chat-A"


@pytest.mark.asyncio
async def test_detached_task_does_not_inherit_a_later_turns_session():
    """#2877 P1: a ContextVar is COPIED into child tasks, so a task detached
    from turn A keeps reporting turn A's id after A exits. Pairing it with the
    agent-global `_active_session_id` — as a naive capture does — hands that
    task whatever turn is live *now*, cross-wiring background work into an
    unrelated chat window. Ownership of the LIVE turn is the discriminator.
    """
    agent = _StubAgent()
    turn_b_running = asyncio.Event()
    seen: dict = {}

    async def detached():
        await turn_b_running.wait()
        seen["turn_id"] = agent._get_current_turn_id()
        seen["session"] = agent.get_turn_bound_session_id()
        seen["agent_global"] = agent._active_session_id

    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-A"
        task = asyncio.create_task(detached())

    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-B"
        turn_b_running.set()
        await task

    assert seen["turn_id"], "precondition: the detached task still reports turn A"
    assert seen["agent_global"] == "chat-B", "precondition: turn B owns the agent"
    assert seen["session"] is None


@pytest.mark.asyncio
async def test_stale_explicit_binding_vetoes_ambient_turn_authority():
    """A callback carrying turn A's binding cannot borrow later turn B."""
    agent = _StubAgent()
    callback_ready = asyncio.Event()
    read_now = asyncio.Event()

    async def callback():
        callback_ready.set()
        await read_now.wait()
        captured = capture_turn_session_binding(agent)
        return agent.get_turn_bound_session_id(), captured.session_id

    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-A"
        binding = capture_turn_session_binding(agent)
        with bind_turn_session(binding):
            # The callback copies turn A's explicit binding but never enters a
            # turn of its own. It must retain that authority boundary while an
            # unrelated turn B is live.
            task = asyncio.create_task(callback())
            await callback_ready.wait()

    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-B"
        read_now.set()
        assert await task == (None, None)


@pytest.mark.asyncio
async def test_inherited_binding_is_cleared_when_descendant_owns_new_turn():
    """A signal task spawned under turn A owns its later wake turn B."""
    agent = _StubAgent()

    async def woken_turn():
        async with agent._turn_lifecycle():
            agent._active_session_id = "wake-session-B"
            captured = capture_turn_session_binding(agent)
            return agent.get_turn_bound_session_id(), captured.session_id

    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-A"
        binding = capture_turn_session_binding(agent)
        with bind_turn_session(binding):
            # SignalDispatcher creates its background dispatch task while the
            # inline tool's binding is active. The task waits for turn A's
            # conversation lock, then enters and owns wake turn B.
            task = asyncio.create_task(woken_turn())

    assert await task == ("wake-session-B", "wake-session-B")


@pytest.mark.asyncio
async def test_explicit_unbound_binding_vetoes_matching_ambient_turn():
    """An off-turn callback stays unbound even on its matching copied turn."""
    agent = _StubAgent()
    binding = capture_turn_session_binding(agent)

    async with agent._turn_lifecycle():
        agent._active_session_id = "unrelated-chat"
        with bind_turn_session(binding):
            nested = capture_turn_session_binding(agent)
            assert agent.get_turn_bound_session_id() is None

    assert nested.turn_id is None
    assert nested.session_id is None


@pytest.mark.asyncio
async def test_session_less_and_blank_sessions_resolve_to_none():
    agent = _StubAgent()
    async with agent._turn_lifecycle():
        agent._active_session_id = None
        assert agent.get_turn_bound_session_id() is None
        agent._active_session_id = "   "
        assert agent.get_turn_bound_session_id() is None


@pytest.mark.asyncio
async def test_private_turn_session_accessor_is_a_compatibility_alias():
    agent = _StubAgent()
    async with agent._turn_lifecycle():
        agent._active_session_id = "chat-A"
        with pytest.warns(DeprecationWarning, match="removed in 0.54.0"):
            assert (
                agent._get_turn_bound_session_id()
                == agent.get_turn_bound_session_id()
            )


@pytest.mark.asyncio
async def test_capture_supports_private_turn_session_accessor_compatibility():
    """Cross-task capture matches the direct feature-read compatibility path."""
    agent = _PrivateAccessorOnlyAgent()
    async with agent._turn_lifecycle():
        agent._active_session_id = "  chat-compat  "
        binding = capture_turn_session_binding(agent)

    assert binding.agent is agent
    assert binding.turn_id is not None
    assert binding.session_id == "chat-compat"


@pytest.mark.asyncio
async def test_live_turn_id_is_cleared_even_when_the_turn_raises():
    agent = _StubAgent()
    with pytest.raises(RuntimeError):
        async with agent._turn_lifecycle():
            agent._active_session_id = "chat-A"
            raise RuntimeError("boom")
    assert agent._live_turn_id is None
    assert agent.get_turn_bound_session_id() is None


# ---------------------------------------------------------------------------
# Stall observability (#2770)
# ---------------------------------------------------------------------------


class _NamedAgent(_StubAgent):
    """Carries ``agent_name`` like a real KestrelAgent, so the lock label is
    attributable to a specific agent rather than to "some turn"."""

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name


@pytest.mark.asyncio
async def test_turn_boundaries_log_at_info(caplog):
    """#2770: a turn that stalled inside this region left ``process_input
    called`` as the agent's last record — begin/end were DEBUG and therefore
    invisible in production. Both boundaries must be visible at INFO."""
    import logging

    agent = _NamedAgent("Nellie")
    with caplog.at_level(logging.INFO):
        async with agent._turn_lifecycle():
            pass

    assert "turn_lifecycle: Nellie turn_" in caplog.text
    assert "begin" in caplog.text
    assert "end after" in caplog.text


@pytest.mark.asyncio
async def test_exit_line_carries_the_turn_duration(caplog):
    """Duration on the exit line so a slow turn is measurable from the log
    alone, instead of correlating two timestamps by hand."""
    import logging
    import re

    agent = _NamedAgent("Nellie")
    with caplog.at_level(logging.INFO):
        async with agent._turn_lifecycle():
            await asyncio.sleep(0.05)

    assert re.search(r"end after \d+\.\ds", caplog.text)


@pytest.mark.asyncio
async def test_lock_holder_is_labelled_with_agent_and_turn():
    """The holder label is what a blocked waiter reports, so it must identify
    the agent AND the specific turn."""
    agent = _NamedAgent("Nellie")

    async with agent._turn_lifecycle() as turn_id:
        holder = agent._lock_manager.holder(ResourceLock.CONVERSATION)
        assert holder is not None
        assert holder.label == f"Nellie {turn_id}"


@pytest.mark.asyncio
async def test_label_degrades_gracefully_without_an_agent_name():
    """Callers that bypass ``__init__`` (tests using ``__new__``) still get a
    usable label rather than an AttributeError inside the lock path."""
    agent = _StubAgent()

    async with agent._turn_lifecycle() as turn_id:
        holder = agent._lock_manager.holder(ResourceLock.CONVERSATION)
        assert holder.label == f"agent {turn_id}"


@pytest.mark.asyncio
async def test_boundaries_still_log_when_the_turn_raises(caplog):
    """A turn that dies must still close its region in the log, otherwise an
    exception looks identical to a stall."""
    import logging

    agent = _NamedAgent("Nellie")
    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError):
            async with agent._turn_lifecycle():
                raise RuntimeError("boom")

    assert "begin" in caplog.text
    assert "end after" in caplog.text
    assert not agent._lock_manager.is_held(ResourceLock.CONVERSATION)
