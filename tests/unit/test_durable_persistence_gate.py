"""#3316 — durable signal persistence must not queue behind an in-flight turn.

`SignalDispatcher.dispatch_signal` projects a signal for the durable ledger
and commits it. That pair must not straddle a privacy transition: a NORMAL
projection computed, the mode flipped to EPHEMERAL while the commit is
blocked, and the stale plaintext projection committed afterwards. The
dispatcher used to close that race by holding the agent's privacy-transition
lock — the lock every turn holds for its whole body since #3310 — so every
signal (inbound channel/A2A ACK ingress, the scheduler's cron dispatches) was
head-of-line blocked behind whatever turn was in flight. On 2026-10-03 that
stalled one agent's scheduler for a full 20-minute wake turn and dropped its
cron runs as misfires.

The fix is a narrow gate owned by the agent: dispatch holds it shared around
projection and commit, `privacy_transition()` holds it exclusive after
CONVERSATION and the privacy-transition lock. These tests drive the real
`KestrelAgent` lock acquisitions (`process_input`, `privacy_transition()`)
against a real `SignalDispatcher` over SQLite.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sdk.signals import ResourceLock, Signal, SignalMode, Status

from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.signals import (
    DurableAdmissionDisposition,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.scheduler import (
    build_cron_registrations,
    cron_source_name,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate

# Bounds a deadlock, not the work: a passing run returns as soon as the awaited
# event fires. Kept well inside CI's 60s per-test timeout (see #3454).
_HANG_GUARD_SECONDS = 30

_CRON_TASK = "wait_reconcile"


class _Rig(SimpleNamespace):
    agent: KestrelAgent
    dispatcher: SignalDispatcher
    backend: SQLiteBackend
    tool_calls: list


def _privacy(agent: KestrelAgent, preset: str) -> None:
    """Install the privacy config the dispatcher's projection reads."""
    agent.privacy_agent = SimpleNamespace(privacy_config=get_privacy_preset(preset))


@pytest.fixture
async def rig(tmp_path):
    agent = KestrelAgent(did="did:test:3316", storage_path=":memory:")
    # Everything `process_input` touches before its lock span is stubbed, so
    # the turn exercises the real CONVERSATION -> privacy acquisition.
    agent.storage = object()
    agent.context_manager = object()
    agent.bootstrap_service = None
    agent._safe_mode = False
    agent._maybe_audit = AsyncMock()
    agent._maybe_refresh_user_byok_resolver = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    _privacy(agent, "normal")

    backend = SQLiteBackend(str(tmp_path / "gate.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    tool_calls: list = []

    async def lookup(task_name, payload):
        tool_calls.append((task_name, dict(payload)))
        return f"{task_name} ran"

    for registration in build_cron_registrations(
        tool_lookup=lookup,
        reason_codes_lookup=lambda _name: frozenset(),
        agent=agent,
    ):
        registry.register(registration)
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=agent._get_lock_manager(),
        store=store,
    )
    agent.dispatcher = dispatcher
    await dispatcher.initialize_durable_delivery()
    try:
        yield _Rig(
            agent=agent,
            dispatcher=dispatcher,
            backend=backend,
            tool_calls=tool_calls,
        )
    finally:
        await dispatcher.shutdown_durable_delivery()
        await backend.close()


def _cron_signal(agent: KestrelAgent, **payload) -> Signal:
    return Signal(
        source=cron_source_name(_CRON_TASK),
        kind="tick",
        mode=SignalMode.ACTION,
        payload=dict(payload),
        target_agent=agent.did,
    )


async def _stored_payloads(rig: _Rig) -> list[dict]:
    rows = await rig.backend.fetch_all(
        "SELECT payload FROM durable_signal_events WHERE agent_id = ? "
        "ORDER BY source_sequence",
        (rig.agent.did,),
    )
    return [json.loads(row[0]) for row in rows]


def _block_turn(agent: KestrelAgent) -> tuple[asyncio.Event, asyncio.Event]:
    """Make the next turn hold its span until ``release`` is set."""
    in_body = asyncio.Event()
    release = asyncio.Event()

    async def traced(user_input, *args, **kwargs):
        in_body.set()
        await release.wait()
        return "turn done"

    agent._process_input_traced_locked = traced
    return in_body, release


class _StalledPersist:
    """Hold each durable persist at its entry, under the dispatcher's gate."""

    def __init__(self, dispatcher: SignalDispatcher) -> None:
        self.entered: list[asyncio.Event] = []
        self.committed: list[asyncio.Event] = []
        self.release = asyncio.Event()
        self.observations: list = []
        self._dispatcher = dispatcher
        self._original = dispatcher._durable_store.persist_signal
        dispatcher._durable_store.persist_signal = self._persist
        self._observe = None

    def observe(self, fn) -> None:
        self._observe = fn

    async def _persist(self, *args, **kwargs):
        entered = asyncio.Event()
        committed = asyncio.Event()
        self.entered.append(entered)
        self.committed.append(committed)
        if self._observe is not None:
            self.observations.append(self._observe())
        entered.set()
        await self.release.wait()
        result = await self._original(*args, **kwargs)
        committed.set()
        return result

    async def wait_entered(self, count: int) -> None:
        async def reached() -> None:
            while len(self.entered) < count:
                await asyncio.sleep(0)
            await asyncio.gather(*(e.wait() for e in self.entered[:count]))

        await asyncio.wait_for(reached(), timeout=_HANG_GUARD_SECONDS)

    def restore(self) -> None:
        self._dispatcher._durable_store.persist_signal = self._original


# --------------------------------------------------------------------------
# 1. Signal persistence no longer waits on a turn
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cron_action_dispatch_completes_while_a_turn_holds_the_privacy_lock(rig):
    """The scheduler stall: a cron ACTION dispatched mid-turn runs mid-turn."""
    agent = rig.agent
    in_body, release = _block_turn(agent)
    turn = asyncio.create_task(agent.process_input("wake"))
    try:
        await asyncio.wait_for(in_body.wait(), timeout=_HANG_GUARD_SECONDS)
        assert agent._get_privacy_transition_lock().locked()

        dispatch = asyncio.create_task(
            rig.dispatcher.dispatch_signal(_cron_signal(agent))
        )
        # The turn never finishes on its own, so the first task to complete
        # must be the dispatch. Before #3316 neither completed until the hang
        # guard: the dispatch waited on the privacy lock the turn holds.
        done, _ = await asyncio.wait(
            {dispatch, turn},
            timeout=_HANG_GUARD_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
        assert done == {dispatch}, (
            "durable signal persistence queued behind the in-flight turn"
        )
        assert dispatch.result().status is Status.OK
        assert rig.tool_calls == [(_CRON_TASK, {})]
        assert len(await _stored_payloads(rig)) == 1
        assert agent._get_privacy_transition_lock().locked(), (
            "the turn keeps its whole-body privacy span (#3310)"
        )
    finally:
        release.set()
        assert await asyncio.wait_for(turn, timeout=_HANG_GUARD_SECONDS) == "turn done"


@pytest.mark.asyncio
async def test_ack_ingress_admission_resolves_while_a_turn_holds_the_privacy_lock(rig):
    """The ingress shape from the issue: the ACK receipt does not wait a turn."""
    agent = rig.agent
    in_body, release = _block_turn(agent)
    turn = asyncio.create_task(agent.process_input("wake"))
    try:
        await asyncio.wait_for(in_body.wait(), timeout=_HANG_GUARD_SECONDS)

        handle = await rig.dispatcher.enqueue_signal(
            _cron_signal(agent), source_event_id="provider-update-1"
        )
        admission = await asyncio.wait_for(
            handle.wait_for_durable_admission(), timeout=_HANG_GUARD_SECONDS
        )

        assert admission.disposition is DurableAdmissionDisposition.COMMITTED
        assert admission.acknowledged
        assert not turn.done()
        assert (await asyncio.wait_for(handle.task, _HANG_GUARD_SECONDS)).status is (
            Status.OK
        )
    finally:
        release.set()
        await asyncio.wait_for(turn, timeout=_HANG_GUARD_SECONDS)


# --------------------------------------------------------------------------
# 2. The projection-vs-commit race stays closed
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_privacy_transition_waits_for_in_flight_persist_and_later_persist_sees_new_mode(
    rig,
):
    """A transition cannot overtake a persist, and nothing persists across it."""
    agent = rig.agent
    gate = agent._get_durable_persistence_gate()
    stalled = _StalledPersist(rig.dispatcher)
    flipped = asyncio.Event()
    may_finish_transition = asyncio.Event()
    observed_committed: list[bool] = []

    async def transition_to_ephemeral() -> None:
        async with agent.privacy_transition():
            # Reachable only once the in-flight NORMAL persist has committed.
            observed_committed.append(stalled.committed[0].is_set())
            _privacy(agent, "ephemeral")
            flipped.set()
            await may_finish_transition.wait()

    try:
        before = asyncio.create_task(
            rig.dispatcher.dispatch_signal(
                _cron_signal(agent, note="normal-before-transition@example.com")
            )
        )
        await stalled.wait_entered(1)
        transition = asyncio.create_task(transition_to_ephemeral())
        for _ in range(20):
            await asyncio.sleep(0)
        assert not flipped.is_set(), (
            "the transition flipped the mode under an in-flight persist"
        )

        stalled.release.set()
        assert (await asyncio.wait_for(before, _HANG_GUARD_SECONDS)).status is Status.OK
        await asyncio.wait_for(flipped.wait(), timeout=_HANG_GUARD_SECONDS)
        assert observed_committed == [True]

        # Started while the transition still owns the gate: it must wait, and
        # then project under the mode the transition installed.
        after = asyncio.create_task(
            rig.dispatcher.dispatch_signal(
                _cron_signal(agent, note="ephemeral-after-transition@example.com")
            )
        )
        for _ in range(20):
            await asyncio.sleep(0)
        assert len(stalled.entered) == 1, (
            "a persist entered while the transition owned the gate"
        )
        assert gate.locked()

        may_finish_transition.set()
        await asyncio.wait_for(transition, timeout=_HANG_GUARD_SECONDS)
        assert (await asyncio.wait_for(after, _HANG_GUARD_SECONDS)).status is Status.OK
    finally:
        stalled.release.set()
        may_finish_transition.set()
        stalled.restore()

    first, second = await _stored_payloads(rig)
    assert first == {"note": "normal-before-transition@example.com"}
    assert "ephemeral-after-transition@example.com" not in json.dumps(second)
    assert set(second) == {"_privacy_gated"}


@pytest.mark.asyncio
async def test_signal_dispatched_inside_the_transition_persists_under_the_new_mode(rig):
    """The transition owner's own dispatch re-enters the gate instead of wedging.

    Before #3316 a dispatch awaited on the transition's task re-entered the
    task-reentrant privacy lock. The gate keeps that: its exclusive owner's
    shared acquisition is admitted, so the persist runs inside the transition
    under the mode it installed.
    """
    agent = rig.agent

    async def transition_that_dispatches():
        async with agent.privacy_transition():
            _privacy(agent, "ephemeral")
            return await rig.dispatcher.dispatch_signal(
                _cron_signal(agent, note="inside-transition@example.com")
            )

    result = await asyncio.wait_for(
        transition_that_dispatches(), timeout=_HANG_GUARD_SECONDS
    )

    assert result.status is Status.OK
    [payload] = await _stored_payloads(rig)
    assert set(payload) == {"_privacy_gated"}
    assert not agent._get_durable_persistence_gate().locked()


# --------------------------------------------------------------------------
# 3. Dispatches share the gate
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_dispatches_both_proceed_under_the_shared_gate(rig):
    agent = rig.agent
    gate = agent._get_durable_persistence_gate()
    stalled = _StalledPersist(rig.dispatcher)
    stalled.observe(gate.locked)
    try:
        dispatches = [
            asyncio.create_task(
                rig.dispatcher.dispatch_signal(_cron_signal(agent, n=str(n)))
            )
            for n in range(2)
        ]
        # Both are inside projection-and-commit at once; an exclusive gate
        # would admit only the first until the second's hang guard.
        await stalled.wait_entered(2)
        assert stalled.observations == [True, True]
        stalled.release.set()
        results = await asyncio.wait_for(
            asyncio.gather(*dispatches), timeout=_HANG_GUARD_SECONDS
        )
    finally:
        stalled.release.set()
        stalled.restore()

    assert [r.status for r in results] == [Status.OK, Status.OK]
    assert len(await _stored_payloads(rig)) == 2
    assert not gate.locked()


# --------------------------------------------------------------------------
# 4. Lock order: CONVERSATION -> privacy transition -> persistence gate
# --------------------------------------------------------------------------


def _record_exclusive_acquisitions(agent: KestrelAgent) -> list[tuple[bool, bool]]:
    """Record, at each exclusive gate entry, what the entering task holds."""
    gate = agent._get_durable_persistence_gate()
    original = gate.exclusive
    held: list[tuple[bool, bool]] = []

    def exclusive():
        held.append(
            (
                agent._get_lock_manager().is_owned_by_current_task(
                    ResourceLock.CONVERSATION
                )
                or agent._caller_belongs_to_live_turn(),
                agent._get_privacy_transition_lock().current_reentry_token()
                is not None,
            )
        )
        return original()

    gate.exclusive = exclusive
    return held


@pytest.mark.asyncio
async def test_external_transition_takes_the_gate_last(rig):
    agent = rig.agent
    held = _record_exclusive_acquisitions(agent)

    async with agent.privacy_transition():
        pass

    assert held == [(True, True)], (
        "the gate must be taken after CONVERSATION and the privacy lock"
    )


@pytest.mark.asyncio
async def test_in_turn_transition_takes_the_gate_last_without_wedging(rig):
    agent = rig.agent
    held = _record_exclusive_acquisitions(agent)

    class _CommandHandler:
        async def handle(self, user_input, caller=None):
            async with agent.privacy_transition():
                return "privacy mode updated"

    agent.command_handler = _CommandHandler()

    response = await asyncio.wait_for(
        agent.process_input("!privacy ephemeral"), timeout=_HANG_GUARD_SECONDS
    )

    assert response == "privacy mode updated"
    assert held == [(True, True)]
    assert not agent._get_durable_persistence_gate().locked()


@pytest.mark.asyncio
async def test_dispatch_holds_the_gate_as_a_leaf_and_releases_it_before_routing(rig):
    """Nothing is acquired under the shared gate, and handlers run outside it.

    A route may start a turn (COGNITION takes CONVERSATION and the privacy
    lock); holding the gate there would invert the global order.
    """
    agent = rig.agent
    gate = agent._get_durable_persistence_gate()
    stalled = _StalledPersist(rig.dispatcher)
    stalled.release.set()
    stalled.observe(
        lambda: (
            gate.locked(),
            agent._get_lock_manager().is_owned_by_current_task(
                ResourceLock.CONVERSATION
            ),
            agent._get_privacy_transition_lock().locked(),
        )
    )
    gate_at_route: list[bool] = []
    registration = rig.dispatcher._registry.get(cron_source_name(_CRON_TASK))
    original_handler = registration.handler

    async def handler(payload):
        gate_at_route.append(gate.locked())
        return await original_handler(payload)

    registration.handler = handler
    try:
        result = await asyncio.wait_for(
            rig.dispatcher.dispatch_signal(_cron_signal(agent)),
            timeout=_HANG_GUARD_SECONDS,
        )
    finally:
        registration.handler = original_handler
        stalled.restore()

    assert result.status is Status.OK
    assert stalled.observations == [(True, False, False)]
    assert gate_at_route == [False]


def test_privacy_transition_acquires_the_gate_after_the_privacy_lock():
    """Source-level tripwire for both branches of ``privacy_transition()``.

    A concurrent run cannot reliably show an inversion (it needs a precise
    interleaving), so the order is pinned at the source as in
    ``test_non_streaming_turn_privacy_span.py``.
    """
    src = inspect.getsource(TurnLifecycleMixin.privacy_transition)
    lines = [line.strip() for line in src.splitlines()]
    lock_entries = [i for i, l in enumerate(lines) if l == "async with transition_lock:"]
    gate_entries = [
        i for i, l in enumerate(lines)
        if l == "async with persistence_gate.exclusive():"
    ]
    assert len(lock_entries) == 2 and len(gate_entries) == 2
    for lock_line, gate_line in zip(lock_entries, gate_entries):
        assert gate_line == lock_line + 1, (
            "each branch must take the persistence gate directly inside the "
            "privacy-transition lock"
        )
    conversation = next(
        i for i, l in enumerate(lines)
        if l.startswith("async with mgr.acquire({ResourceLock.CONVERSATION}")
    )
    assert lock_entries[0] < conversation < lock_entries[1], (
        "the external branch must take CONVERSATION before the privacy lock"
    )


# --------------------------------------------------------------------------
# The gate primitive
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_exclusive_is_task_reentrant_and_releases_once():
    gate = DurablePersistenceGate()

    async def nested() -> None:
        async with gate.exclusive():
            async with gate.exclusive():
                async with gate.shared():
                    assert gate.locked()
            assert gate.locked()

    await asyncio.wait_for(nested(), timeout=_HANG_GUARD_SECONDS)
    assert not gate.locked()


@pytest.mark.asyncio
async def test_gate_exclusive_waits_for_shared_and_blocks_new_shared():
    gate = DurablePersistenceGate()
    reader_in = asyncio.Event()
    reader_release = asyncio.Event()
    order: list[str] = []

    async def reader(name, entered=None, release=None):
        async with gate.shared():
            order.append(name)
            if entered is not None:
                entered.set()
            if release is not None:
                await release.wait()

    async def writer():
        async with gate.exclusive():
            order.append("writer")

    first = asyncio.create_task(reader("reader-1", reader_in, reader_release))
    await asyncio.wait_for(reader_in.wait(), timeout=_HANG_GUARD_SECONDS)
    queued_writer = asyncio.create_task(writer())
    for _ in range(5):
        await asyncio.sleep(0)
    late_reader = asyncio.create_task(reader("reader-2"))
    for _ in range(5):
        await asyncio.sleep(0)
    assert order == ["reader-1"], "a queued transition must not be starved"

    reader_release.set()
    await asyncio.wait_for(
        asyncio.gather(first, queued_writer, late_reader),
        timeout=_HANG_GUARD_SECONDS,
    )
    assert order == ["reader-1", "writer", "reader-2"]
