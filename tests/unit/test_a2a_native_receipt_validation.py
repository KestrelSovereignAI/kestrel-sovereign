"""Untrusted native peer facts cannot escape as storage failures."""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from kestrel_sovereign._vendor.a2a.types import TaskState
from kestrel_sovereign._vendor.a2a.utils.errors import A2AError

from kestrel_sovereign.a2a.native_client import NativeDispatchReceipt
from kestrel_sovereign.a2a.question_supervisor import NativeQuestionSupervisor
from kestrel_sovereign.a2a.stores.unified.outbound_store import (
    DispatchState,
    OutboundDispatchConflict,
)
from tests.unit.test_a2a_native_question_store import (
    questions as questions,  # noqa: PLC0414 - expose the shared pytest fixture
)
from tests.unit.test_a2a_native_question_store import (
    reserve,
    response,
)


@pytest.mark.parametrize("state", [TaskState.TASK_STATE_UNSPECIFIED, 99])
async def test_unknown_peer_state_is_typed_before_question_transaction(
    questions, state
):
    p = questions
    q, d, request = await reserve(p)
    sending = await p.outbound.begin_submission(d, operation_id=uuid4().hex)
    with pytest.raises((OutboundDispatchConflict, A2AError)):
        await p.store.acknowledge(
            sending, response(q, request, state=state), operation_id=uuid4().hex
        )
    assert await p.outbound.get(d.dispatch_id) == sending
    assert not await p.store.pending_wakes()
    assert (await p.store.get(q.question_id)).status == "WAITING"


async def test_competing_root_claims_have_one_owner_without_storage_error(questions):
    p = questions
    attempts = [await reserve(p, name) for name in ("first", "second")]
    sending = [
        await p.outbound.begin_submission(d, operation_id=uuid4().hex)
        for _, d, _ in attempts
    ]
    outcomes = await asyncio.gather(
        *(
            p.store.acknowledge(
                d,
                response(q, r, state=TaskState.TASK_STATE_WORKING),
                operation_id=uuid4().hex,
            )
            for (q, _, r), d in zip(attempts, sending, strict=True)
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(value, OutboundDispatchConflict) for value in outcomes) == 1
    assert sum(not isinstance(value, BaseException) for value in outcomes) == 1
    rows = await p.outbound.list_recent()
    assert sum(row.state is DispatchState.ACKNOWLEDGED for row in rows) == 1
    assert sum(row.state is DispatchState.SUBMITTING for row in rows) == 1


async def test_postgres_receipt_holds_exact_remote_owner_lock(questions, monkeypatch):
    import json

    p = questions
    if not p.outbound.is_postgres:
        pytest.skip("PostgreSQL transaction advisory-lock proof")
    q, d, request = await reserve(p)
    sending = await p.outbound.begin_submission(d, operation_id=uuid4().hex)
    task = response(q, request, state=TaskState.TASK_STATE_WORKING)
    key = json.dumps(
        [
            "a2a-outbound-task-owner",
            d.sender_id,
            d.sender_tenant,
            d.recipient_id,
            d.tenant,
            task.id,
        ]
    )
    actual, observed = p.backend.execute, []

    async def inspect_lock(query, params=()):
        if query.startswith("UPDATE a2a_outbound_dispatches SET"):
            [(count,)] = await p.backend.fetch_all(
                "SELECT COUNT(*) FROM pg_locks WHERE pid = pg_backend_pid() "
                "AND locktype = 'advisory' AND mode = 'ExclusiveLock' AND granted "
                "AND objsubid = 1 "
                "AND classid::bigint = ((hashtextextended(?, 0) >> 32) & 4294967295) "
                "AND objid::bigint = (hashtextextended(?, 0) & 4294967295)",
                (key, key),
            )
            observed.append(count)
        return await actual(query, params)

    monkeypatch.setattr(p.backend, "execute", inspect_lock)
    await p.store.acknowledge(sending, task, operation_id=uuid4().hex)
    assert observed == [1]


@pytest.mark.parametrize("fault", ["unknown-state", "task-owner"])
async def test_bad_peer_read_leaves_unrelated_deadlines_running(questions, fault):
    p = questions
    q1, d1, r1 = await reserve(p, "first")
    s1 = await p.outbound.begin_submission(d1, operation_id=uuid4().hex)
    a1 = await p.store.acknowledge(
        s1,
        response(q1, r1, state=TaskState.TASK_STATE_WORKING),
        operation_id=uuid4().hex,
    )
    q2, d2, r2 = await reserve(p, "second")
    s2 = await p.outbound.begin_submission(d2, operation_id=uuid4().hex)
    await p.outbound.mark_unknown(s2, operation_id=uuid4().hex)
    unrelated, _, _ = await reserve(p, "unrelated")

    class Peer:
        async def reconcile(self, dispatch):
            task = response(
                q2,
                r2,
                state=(
                    99 if fault == "unknown-state" else TaskState.TASK_STATE_WORKING
                ),
            )
            return await p.store.acknowledge(dispatch, task, operation_id=uuid4().hex)

        async def get_task(self, dispatch):
            return NativeDispatchReceipt(
                a1, response(q1, r1, state=TaskState.TASK_STATE_WORKING)
            )

    @asynccontextmanager
    async def open_dispatch(dispatch_id):
        yield Peer(), await p.outbound.get(dispatch_id)

    supervisor = NativeQuestionSupervisor(
        store=p.store,
        open_dispatch=open_dispatch,
        dispatcher=None,
        interval=1,
        is_current=lambda: True,
    )
    await supervisor._read_pass()
    assert (await p.store.get(q2.question_id)).status == "WAITING"
    p.clock.now += timedelta(seconds=11)
    await supervisor._deadline_pass()
    assert (await p.store.get(unrelated.question_id)).status == "EXPIRED"


async def test_backoff_observes_available_answer_before_original_deadline(questions):
    p = questions
    q, d, request = await reserve(p, deadline_seconds=300)
    sending = await p.outbound.begin_submission(d, operation_id=uuid4().hex)
    await p.store.acknowledge(
        sending,
        response(q, request, state=TaskState.TASK_STATE_WORKING),
        operation_id=uuid4().hex,
    )
    start, sim, reads = p.clock.now, SimpleNamespace(t=0), []
    working = response(q, request, state=TaskState.TASK_STATE_WORKING)
    completed = response(q, request, state=TaskState.TASK_STATE_COMPLETED)

    class Peer:
        async def get_task(self, current):
            reads.append(sim.t)
            task = completed if sim.t >= 270 else working
            return NativeDispatchReceipt(
                await p.store.acknowledge(current, task, operation_id=uuid4().hex),
                task,
            )

        async def cancel_task(self, current):
            raise AssertionError("An answer available before the deadline was canceled")

    @asynccontextmanager
    async def open_dispatch(dispatch_id):
        yield Peer(), await p.outbound.get(dispatch_id)

    supervisor = NativeQuestionSupervisor(
        store=p.store,
        open_dispatch=open_dispatch,
        dispatcher=None,
        interval=1,
        max_interval=60,
        clock=lambda: sim.t,
        is_current=lambda: True,
    )
    for second in range(400):
        sim.t, p.clock.now = second, start + timedelta(seconds=second)
        await supervisor._deadline_pass()
        await supervisor._read_pass()
    assert any(270 <= observed < 300 for observed in reads)
    settled = await p.store.get(q.question_id)
    assert settled.status == "RESOLVED" and settled.receiver_settled
    assert [wake.wave for wake in await p.store.pending_wakes()] == ["answer"]


@pytest.mark.parametrize("fail_settlement", [False, True])
async def test_follow_up_receipt_settles_root_question_atomically(
    questions,
    monkeypatch,
    fail_settlement,
):
    from kestrel_sovereign.a2a.stores.unified.question_store import QuestionConflict
    from tests.unit.test_a2a_native_outbound_continuation import follow_up

    p = questions
    q, d, request = await reserve(p)
    sending = await p.outbound.begin_submission(d, operation_id=uuid4().hex)
    root = await p.store.acknowledge(
        sending,
        response(q, request, state=TaskState.TASK_STATE_WORKING),
        operation_id=uuid4().hex,
    )
    child, _ = await follow_up(p.outbound, root)
    sending = await p.outbound.begin_submission(child, operation_id=uuid4().hex)
    answer = response(q, request)
    if fail_settlement:

        async def reject(*args):
            raise QuestionConflict("Settlement lost its original authority")

        monkeypatch.setattr(p.store, "record_answer", reject)
        with pytest.raises(QuestionConflict, match="original authority"):
            await p.store.acknowledge(sending, answer, operation_id=uuid4().hex)
        assert await p.outbound.get(root.dispatch_id) == root
        assert await p.outbound.get(child.dispatch_id) == sending
        assert await p.store.pending_wakes() == []
    else:
        receipt = await p.store.acknowledge(sending, answer, operation_id=uuid4().hex)
        assert receipt.state is DispatchState.COMPLETE
        retained = await p.outbound.get(root.dispatch_id)
        assert retained.state is DispatchState.COMPLETE
        assert await p.store.retained_response(retained) == answer
        assert (await p.store.get(q.question_id)).status == "RESOLVED"
        assert [wake.wave for wake in await p.store.pending_wakes()] == ["answer"]
