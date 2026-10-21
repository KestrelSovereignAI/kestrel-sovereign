"""Original effect receipts survive native question recovery and service shutdown.

These exercise the replacement for draft receiver settlement, claim renewal,
and SSE supervision: the primary native ledger and durable admission outbox.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
from a2a.types import TaskState

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.a2a.question_supervisor import NativeQuestionSupervisor
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.features.peers.directory import PeerDirectoryError
from tests.unit.test_a2a_native_question_store import questions as questions  # noqa: PLC0414 - pytest fixture export
from tests.unit.test_a2a_native_question_store import reserve, response
from tests.unit.test_a2a_native_question_supervisor import dispatcher_for
from tests.unit.test_execution_custody_review48_native import terminal
from tests.unit.test_execution_custody_review49_native import KINDS, join


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("route", [
    "unsettled", "expire", "local_read", "open_peer", "reconcile",
    "read_peer", "cancel_peer", "pending_wakes", "claim_wake", "enqueue",
    "admission", "checkpoint",
])
async def test_actual_a2a_consumer_preserves_original_native_receipt(
    questions, monkeypatch, kind, wrapped, route,
):
    p = questions
    q, dispatch, request = await reserve(p)
    submitting = await p.outbound.begin_submission(dispatch, operation_id="send")
    handoff = route in {"pending_wakes", "claim_wake", "enqueue", "admission", "checkpoint"}
    if route == "reconcile":
        dispatch = await p.outbound.mark_unknown(submitting, operation_id="lost-ack")
    else:
        dispatch = await p.store.acknowledge(
            submitting,
            response(q, request, state=(TaskState.TASK_STATE_COMPLETED if handoff else TaskState.TASK_STATE_WORKING)),
            operation_id="receipt",
        )
    original = terminal(kind)
    if wrapped:
        wrapper = PeerDirectoryError("peer I/O retained its native effect receipt")
        wrapper.__cause__ = original
        original = wrapper
    await p.backend.execute("CREATE TABLE question_custody_effects (id INTEGER PRIMARY KEY)")

    async def effect(*args, **kwargs):
        await p.backend.execute("INSERT INTO question_custody_effects VALUES (1)")
        raise original

    class Peer:
        async def reconcile(self, retained):
            assert route == "reconcile"
            return await effect()
        async def get_task(self, retained):
            if route == "read_peer":
                return await effect()
            return SimpleNamespace(dispatch=retained)
        async def cancel_task(self, retained):
            assert route == "cancel_peer"
            return await effect()

    @asynccontextmanager
    async def open_peer(dispatch_id):
        assert dispatch_id == dispatch.dispatch_id
        if route == "open_peer":
            await effect()
        yield Peer(), dispatch

    agent, dispatcher = await dispatcher_for(p)
    supervisor = NativeQuestionSupervisor(
        store=p.store, open_dispatch=open_peer, dispatcher=dispatcher,
        interval=0.01, is_current=lambda: True,
    )
    if route == "unsettled":
        monkeypatch.setattr(p.store, "unsettled", effect)
    elif route in {"expire", "cancel_peer"}:
        p.clock.now += timedelta(seconds=11)
        if route == "expire":
            monkeypatch.setattr(p.store, "expire", effect)
    elif route == "local_read":
        monkeypatch.setattr(p.outbound, "get", effect)
    elif route in {"pending_wakes", "claim_wake"}:
        monkeypatch.setattr(p.store, route, effect)
    elif route == "enqueue":
        monkeypatch.setattr(dispatcher, "enqueue_durable_cognition", effect)
    elif route == "admission":
        async def enqueue(*args, **kwargs):
            return SimpleNamespace(wait_for_durable_admission=effect)
        monkeypatch.setattr(dispatcher, "enqueue_durable_cognition", enqueue)
    elif route == "checkpoint":
        monkeypatch.setattr(p.store, "confirm_handoff", effect)
    owner = create_owned_task(supervisor.recover_once())
    try:
        outcome = await join(owner)
        assert outcome.error is original, "original effect receipt was normalized or retried"
        assert (await p.backend.fetch_one("SELECT COUNT(*) FROM question_custody_effects"))[0] == 1
        if handoff and route != "checkpoint":
            assert (await p.backend.fetch_one("SELECT COUNT(*) FROM durable_signal_events"))[0] == 0
    finally:
        if not owner.done():
            owner.cancel()
        await await_owned_task(owner)
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("interrupt", ["caller", "sibling"])
async def test_original_a2a_child_physically_joins_after_service_cancellation(
    questions, monkeypatch, kind, interrupt,
):
    p = questions
    q, dispatch, request = await reserve(p)
    submitting = await p.outbound.begin_submission(dispatch, operation_id="send")
    dispatch = await p.store.acknowledge(
        submitting, response(q, request, state=TaskState.TASK_STATE_WORKING),
        operation_id="receipt",
    )
    original = terminal(kind)
    entered, interrupted, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    await p.backend.execute("CREATE TABLE question_custody_effects (id INTEGER PRIMARY KEY)")

    class Peer:
        async def get_task(self, retained):
            await p.backend.execute("INSERT INTO question_custody_effects VALUES (1)")
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                interrupted.set()
                await await_owned_task(create_owned_task(release.wait()))
            await p.backend.execute("INSERT INTO question_custody_effects VALUES (2)")
            raise original

    @asynccontextmanager
    async def open_peer(_dispatch_id):
        yield Peer(), dispatch

    supervisor = NativeQuestionSupervisor(
        store=p.store, open_dispatch=open_peer, dispatcher=None,
        interval=0.01, is_current=lambda: True,
    )
    if interrupt == "sibling":
        async def failed_deadline_lane():
            await entered.wait()
            raise ExecutionAuthorityError("other service lane retired")
        monkeypatch.setattr(supervisor, "_deadline_loop", failed_deadline_lane)
    owner = create_owned_task(supervisor.run())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if interrupt == "caller":
            owner.cancel()
        await asyncio.wait_for(interrupted.wait(), 2)
        owner.cancel()
        await asyncio.sleep(0.01)
        assert not owner.done(), "service released its physically running original child"
        assert (await p.backend.fetch_one("SELECT COUNT(*) FROM question_custody_effects"))[0] == 1
        release.set()
        outcome = await join(owner)
        assert outcome.error is original, "late effect receipt lost to service cancellation"
        assert (await p.backend.fetch_one("SELECT COUNT(*) FROM question_custody_effects"))[0] == 2
    finally:
        release.set()
        if not owner.done():
            owner.cancel()
        await await_owned_task(owner)
