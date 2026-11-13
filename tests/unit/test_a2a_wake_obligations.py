"""One durable obligation per A2A task input decides its wake (#4041).

R1-R10 of the #4001 family-B decision, on SQLite and PostgreSQL. Real task
stores, durable ledgers and dispatchers are used; the cognition body is the
recipient double from the input-fencing tests. Every PostgreSQL case
requests ``native_pg`` by name, so under ``KESTREL_REQUIRE_POSTGRES_TESTS=1``
a skipped PostgreSQL leg is a failure (tests/shared/postgres_requirement.py).
"""

import asyncio
import copy
from datetime import timedelta

import pytest
import pytest_asyncio
from kestrel_sdk.native_a2a.types import TaskState, TaskStatusUpdateEvent

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.wake_obligations import (
    BOUND,
    CONFLICT,
    ENDED,
    OWED,
    RAN,
    RELEASED,
    TABLE,
    wake_identity,
)
from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
from kestrel_sovereign.signals import DurableAdmissionDisposition
from kestrel_sovereign.signals.sources.a2a_task_submitted import (
    submitted_input_identity,
)
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError
from tests.unit.test_a2a_native_input_fencing import (
    close_input_dispatcher,
    followup,
    input_dispatcher,
    snapshot,
    wake,
)
from tests.unit.test_a2a_native_task_manager import request
from tests.utils.native_a2a import manager_for

LIVE = ("pending", "retry", "leased", "initial_reserved")


@pytest.fixture
def native_pg(native_a2a_backend_factory):
    """This case's backend factory; the name selects it as a PostgreSQL case."""
    return native_a2a_backend_factory


@pytest_asyncio.fixture
async def manager(native_pg):
    backend = native_pg()
    await backend.connect()
    instance = manager_for(backend, NativeRecipientScope("did:kestrel:recipient", "tenant"))
    try:
        await instance.initialize()
        yield instance
    finally:
        await instance.close()


async def obligation(manager, task, revision=0):
    return await manager.get_wake_obligation(task.id, revision)


async def persisted_wake(manager, dispatcher, task, identity):
    """Admit one submission wake through the durable store under ``identity``."""
    candidate = await snapshot(manager, task)
    persisted = await dispatcher._durable_store.persist_signal(
        copy.deepcopy(wake(candidate, manager)),
        agent_id=manager.did,
        source_event_id=identity,
        retention_days=14,
    )
    [delivery_id] = persisted.delivery_ids
    return delivery_id


async def claim(dispatcher, consumer, delivery_id):
    row = await dispatcher._durable_store.get_delivery(
        agent_id=dispatcher._agent.did, consumer_id=consumer, delivery_id=delivery_id
    )
    return await dispatcher.claim_durable_delivery_for_event(
        consumer_id=consumer, event_id=row.event_id,
        executor_id=dispatcher._durable_delivery_owner,
    )


async def fail_terminally(dispatcher, consumer, claimed, error="wake failed"):
    released = await dispatcher.nack_durable_delivery(
        consumer_id=consumer, delivery_id=claimed.delivery_id,
        lease_token=claimed.lease_token, error=error, terminal=True,
    )
    assert released is not None and released.status == "failed"


async def set_legacy_status(dispatcher, delivery_id, status):
    """Give a legacy row a state an earlier release could have left."""
    store = dispatcher._durable_store
    now = store.now_utc()
    leased = status == "leased"
    await store.backend.execute(
        f"UPDATE {store.DELIVERIES} SET status = ?, attempts = 1, lease_owner = ?, "
        "lease_token = ?, lease_expires_at = ?, acknowledged_at = ?, terminal_at = ? "
        "WHERE delivery_id = ?",
        (
            status,
            "legacy-worker" if leased else None,
            "legacy-token" if leased else None,
            store.to_timestamp_param(now + timedelta(minutes=5)) if leased else None,
            store.to_timestamp_param(now) if status == "acknowledged" else None,
            store.to_timestamp_param(now) if status in ("acknowledged", "failed") else None,
            delivery_id,
        ),
    )


async def forget_obligations(manager):
    """The database as an earlier release left it: no obligation rows."""
    await manager.task_store.backend.execute(f"DELETE FROM {TABLE}")


async def state_of(manager, task):
    return (await snapshot(manager, task)).task.status.state


async def assert_not_stranded(manager, task, revision=0):
    """A live task's input is owed, has a bound wake, or ran; else settled."""
    state = await state_of(manager, task)
    current = await obligation(manager, task, revision)
    if state in (TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING):
        assert current is not None and current.state in (OWED, BOUND, RAN), (
            f"STRANDED: task {TaskState.Name(state)} with obligation {current}"
        )


# ---------------------------------------------------------------------------
# R1 (I1): one obligation per input, by the primary key alone.
# ---------------------------------------------------------------------------


async def test_obligation_primary_key_rejects_second_obligation(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    armed = await obligation(manager, task)
    assert (armed.state, armed.generation, armed.live_delivery_id) == (OWED, 1, None)
    backend = manager.task_store.backend
    with pytest.raises((QueryError, TransactionError)):
        await backend.execute(
            f"INSERT INTO {TABLE} (agent_id, consumer_id, task_id, input_revision, "
            "generation, state) VALUES (?, ?, ?, 0, 7, 'owed')",
            (armed.agent_id, armed.consumer_id, task.id),
        )
    assert await backend.fetch_val(
        f"SELECT COUNT(*) FROM {TABLE} WHERE task_id = ?", (task.id,)
    ) == 1


# ---------------------------------------------------------------------------
# R2 (I2, finding 1): conflicting legacy live wakes never run.
# ---------------------------------------------------------------------------


async def test_conflicting_legacy_live_wakes_are_not_claimable(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        await forget_obligations(manager)
        retried = await persisted_wake(
            manager, dispatcher, task, submitted_input_identity(task.id, 0)
        )
        leased = await persisted_wake(manager, dispatcher, task, '["%s",0,1]' % task.id)
        await set_legacy_status(dispatcher, retried, "retry")
        await set_legacy_status(dispatcher, leased, "leased")
    finally:
        await close_input_dispatcher(agent, dispatcher)
    # Boot of the new release backfills the obligation from task state.
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        held = await obligation(manager, task)
        assert (held.state, held.live_delivery_id) == (CONFLICT, None)
        assert dispatcher._durable_store.wake_obligation_backfill.conflict == 1
        assert await dispatcher.claim_durable_delivery(
            consumer_id=consumer, executor_id=dispatcher._durable_delivery_owner,
        ) is None
        assert await claim(dispatcher, consumer, retried) is None
        assert await dispatcher.list_durable_deliveries(
            consumer_id=consumer, statuses=["pending", "retry"], claimable_only=True,
        ) == []
        # The wake an earlier release had already leased finishes, and
        # decides nothing: the input is still held for its operator.
        assert await dispatcher._durable_store.ack_delivery(
            agent_id=manager.did, consumer_id=consumer, delivery_id=leased,
            lease_token="legacy-token",
        )
        assert (await obligation(manager, task)).state == CONFLICT
        await dispatcher.run_wake_settlement_pass()
        assert await state_of(manager, task) == TaskState.TASK_STATE_SUBMITTED
    finally:
        await close_input_dispatcher(agent, dispatcher)


# ---------------------------------------------------------------------------
# R3 (backfill from task state, finding 2).
# ---------------------------------------------------------------------------


async def test_acked_recovery_before_backfill_never_fails_working_task(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        await forget_obligations(manager)
        original = await persisted_wake(
            manager, dispatcher, task, submitted_input_identity(task.id, 0)
        )
        await set_legacy_status(dispatcher, original, "acknowledged")
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        failed = await persisted_wake(manager, dispatcher, task, '["%s",0,1]' % task.id)
        await set_legacy_status(dispatcher, failed, "failed")
        later = await persisted_wake(manager, dispatcher, task, '["%s",0,3]' % task.id)
        await set_legacy_status(dispatcher, later, "acknowledged")
    finally:
        await close_input_dispatcher(agent, dispatcher)
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        assert (await obligation(manager, task)).state == OWED
        await agent.reconcile_a2a_cognition_wakes()
        await dispatcher.run_wake_settlement_pass()
        assert await state_of(manager, task) == TaskState.TASK_STATE_WORKING
        # Worst case is one more wake on the WORKING task, never a failure.
        current = await obligation(manager, task)
        assert current.state in (BOUND, RAN) and current.generation == 1
        fresh = await dispatcher._durable_store.get_delivery(
            agent_id=manager.did, consumer_id=consumer,
            delivery_id=current.live_delivery_id,
        )
        assert fresh.event.source_event_id == wake_identity(task.id, 0, 1)
        await assert_not_stranded(manager, task)
    finally:
        await close_input_dispatcher(agent, dispatcher)


# ---------------------------------------------------------------------------
# R4 (I3, finding 5): a re-armed input wakes under a fresh identity.
# ---------------------------------------------------------------------------


async def test_working_to_submitted_rearms_with_fresh_generation_identity(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        first = await obligation(manager, task)
        delivery = await persisted_wake(manager, dispatcher, task, first.identity)
        claimed = await claim(dispatcher, consumer, delivery)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        assert await dispatcher.ack_durable_delivery(
            consumer_id=consumer, delivery_id=claimed.delivery_id,
            lease_token=claimed.lease_token,
        )
        assert (await obligation(manager, task)).state == RAN
        await manager.update_status(
            task.id, TaskState.TASK_STATE_SUBMITTED, recipient_agent_id=manager.did
        )
        rearmed = await obligation(manager, task)
        assert (rearmed.state, rearmed.generation, rearmed.live_delivery_id) == (OWED, 2, None)
    finally:
        await close_input_dispatcher(agent, dispatcher)
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        await agent.reconcile_a2a_cognition_wakes()
        recovered = await obligation(manager, task)
        assert recovered.state == BOUND and recovered.live_delivery_id != delivery
        row = await dispatcher._durable_store.get_delivery(
            agent_id=manager.did, consumer_id=consumer,
            delivery_id=recovered.live_delivery_id,
        )
        assert row.event.source_event_id == wake_identity(task.id, 0, 2)
        assert row.status in LIVE
        await assert_not_stranded(manager, task)
    finally:
        await close_input_dispatcher(agent, dispatcher)


# ---------------------------------------------------------------------------
# R5 (findings 6 and 8): no writer's clock orders any decision.
# ---------------------------------------------------------------------------


def skew_clock(dispatcher, seconds):
    store = dispatcher._durable_store
    real = type(store).now_utc
    store.now_utc = lambda: real(store) + timedelta(seconds=seconds)


@pytest.mark.parametrize(
    "emitter_skew, acker_skew",
    [(0, 0), (-20, 0), (20, 0), (0, -20), (0, 20), (20, -20), (-20, 20)],
)
async def test_obligation_decisions_invariant_under_writer_clock_skew(
    manager, emitter_skew, acker_skew
):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    emitter_agent, emitter, consumer = await input_dispatcher(manager)
    acker_agent, acker, _ = await input_dispatcher(manager)
    skew_clock(emitter, emitter_skew)
    skew_clock(acker, acker_skew)
    try:
        # Generation 1 runs: admitted by one writer, acknowledged by another.
        first = await persisted_wake(
            manager, emitter, task, (await obligation(manager, task)).identity
        )
        claimed = await claim(acker, consumer, first)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        assert await acker.ack_durable_delivery(
            consumer_id=consumer, delivery_id=claimed.delivery_id,
            lease_token=claimed.lease_token,
        )
        # The input is re-armed; its generation-2 wake then ends undelivered.
        await manager.update_status(
            task.id, TaskState.TASK_STATE_SUBMITTED, recipient_agent_id=manager.did
        )
        second = await persisted_wake(
            manager, emitter, task, (await obligation(manager, task)).identity
        )
        await fail_terminally(acker, consumer, await claim(acker, consumer, second))
        await assert_not_stranded(manager, task)
        assert await emitter.run_wake_settlement_pass() == 1
        assert (await obligation(manager, task)).state == ENDED
        assert await state_of(manager, task) == TaskState.TASK_STATE_FAILED
    finally:
        await close_input_dispatcher(acker_agent, acker)
        await close_input_dispatcher(emitter_agent, emitter)


# ---------------------------------------------------------------------------
# R6 (Q1, finding 4): projection follows the committed disposition.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lost", ["acknowledgement", "none"])
async def test_settlement_retry_after_unknown_commit_projects_to_open_subscriber(
    manager, lost
):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    stream = manager.subscribe(task.id, recipient_agent_id=manager.did)
    try:
        assert (await anext(stream)).status.state == TaskState.TASK_STATE_SUBMITTED
        delivery = await persisted_wake(
            manager, dispatcher, task, (await obligation(manager, task)).identity
        )
        await fail_terminally(dispatcher, consumer, await claim(dispatcher, consumer, delivery))
        store = dispatcher._durable_store
        settle = store.settle_wake_obligation
        calls = []

        async def commit_then_lose_acknowledgement(**kwargs):
            outcome = await settle(**kwargs)
            calls.append(outcome)
            if lost == "acknowledgement" and len(calls) == 1:
                raise ExecutionCommitOutcomeError("unknown")
            return outcome

        store.settle_wake_obligation = commit_then_lose_acknowledgement
        await dispatcher.run_wake_settlement_pass()
        if lost == "acknowledgement":
            # The committed settlement's acknowledgement was lost; its
            # delivery alone keeps the control, and nothing was projected.
            assert set(dispatcher._source_settlement_control) == {delivery}
            assert dispatcher._outcome_log_control_error is None
            await dispatcher.run_wake_settlement_pass()
            assert calls[-1].disposition == ENDED and not calls[-1].recorded
        assert dispatcher._source_settlement_control == {}
        async with asyncio.timeout(5):
            event = await anext(stream)
        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.TASK_STATE_FAILED
    finally:
        await stream.aclose()
        await close_input_dispatcher(agent, dispatcher)


# ---------------------------------------------------------------------------
# R7 (Q1 lock order): settlement and a peer's re-arm and admission serialize.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("round_", range(6))
async def test_settlement_and_peer_release_serialize(manager, native_pg, round_):
    """Never a task ended by settlement while a live wake holds its input."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    peer_backend = native_pg()
    await peer_backend.connect()
    peer_manager = manager_for(peer_backend, manager.task_store.scope)
    await peer_manager.initialize()
    peer_agent, peer, _ = await input_dispatcher(peer_manager)
    try:
        delivery = await persisted_wake(
            manager, dispatcher, task, (await obligation(manager, task)).identity
        )
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        await fail_terminally(dispatcher, consumer, await claim(dispatcher, consumer, delivery))

        async def peer_rearms_and_admits():
            await asyncio.sleep(0.001 * round_)
            try:
                await peer_manager.update_status(
                    task.id, TaskState.TASK_STATE_SUBMITTED,
                    recipient_agent_id=manager.did,
                )
            except ValueError:
                return  # settlement ended the task first
            await persisted_wake(
                peer_manager, peer, task,
                (await obligation(peer_manager, task)).identity,
            )

        async def settles():
            await asyncio.sleep(0.001 * (5 - round_))
            await dispatcher.run_wake_settlement_pass()

        await asyncio.gather(peer_rearms_and_admits(), settles())
        await dispatcher.run_wake_settlement_pass()
        state = await state_of(manager, task)
        current = await obligation(manager, task)
        live = [
            row for row in await dispatcher.list_durable_deliveries(consumer_id=consumer)
            if row.status in LIVE
        ]
        if state == TaskState.TASK_STATE_FAILED:
            assert current.state == ENDED and live == []
        else:
            assert state == TaskState.TASK_STATE_SUBMITTED
            assert current.state == BOUND and current.generation == 2
            assert [row.delivery_id for row in live] == [current.live_delivery_id]
    finally:
        await close_input_dispatcher(peer_agent, peer)
        await peer_manager.close()
        await close_input_dispatcher(agent, dispatcher)


async def test_settlement_task_write_and_obligation_record_are_one_transaction(manager):
    """A failed obligation record rolls the task write back with it (Q1)."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    stream = manager.subscribe(task.id, recipient_agent_id=manager.did)
    try:
        assert (await anext(stream)).status.state == TaskState.TASK_STATE_SUBMITTED
        delivery = await persisted_wake(
            manager, dispatcher, task, (await obligation(manager, task)).identity
        )
        await fail_terminally(dispatcher, consumer, await claim(dispatcher, consumer, delivery))
        obligations = dispatcher._durable_store._obligations
        real = obligations.settle
        failures = []

        async def fails_once(*args, **kwargs):
            if not failures:
                failures.append(True)
                raise RuntimeError("obligation write failed")
            return await real(*args, **kwargs)

        obligations.settle = fails_once
        assert await dispatcher.run_wake_settlement_pass() == 0
        assert await state_of(manager, task) == TaskState.TASK_STATE_SUBMITTED
        assert (await obligation(manager, task)).state == BOUND
        assert await dispatcher.run_wake_settlement_pass() == 1
        assert (await obligation(manager, task)).state == ENDED
        async with asyncio.timeout(5):
            event = await anext(stream)
        assert event.status.state == TaskState.TASK_STATE_FAILED
    finally:
        await stream.aclose()
        await close_input_dispatcher(agent, dispatcher)


# ---------------------------------------------------------------------------
# R8 (finding 3): settlement runs in the owner-tick lane.
# ---------------------------------------------------------------------------


async def test_settlement_pass_runs_under_fenced_execution_scope(manager):
    """A refused durable admission cannot hold back an owed settlement."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        delivery = await persisted_wake(
            manager, dispatcher, task, (await obligation(manager, task)).identity
        )
        await fail_terminally(dispatcher, consumer, await claim(dispatcher, consumer, delivery))
        # A lost native COMMIT acknowledgement elsewhere latched the
        # dispatcher's writer control: durable admission now refuses all
        # ordinary maintenance.
        dispatcher._outcome_log_control_error = ExecutionCommitOutcomeError("unknown")
        with pytest.raises(BaseException):
            async with dispatcher._admit_durable_operation():
                pass
        dispatcher._start_runtime_owner_heartbeat()
        async with asyncio.timeout(10):
            while (await obligation(manager, task)).state != ENDED:
                await asyncio.sleep(0.01)
        assert await state_of(manager, task) == TaskState.TASK_STATE_FAILED
    finally:
        dispatcher._outcome_log_control_error = None
        await close_input_dispatcher(agent, dispatcher)


# ---------------------------------------------------------------------------
# R10 (Q3): per-delivery control never widens.
# ---------------------------------------------------------------------------


async def test_settlement_cancellation_carrier_stays_per_delivery(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    other = await manager.create_task(request("other-context"), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        delivery = await persisted_wake(
            manager, dispatcher, task, (await obligation(manager, task)).identity
        )
        await fail_terminally(dispatcher, consumer, await claim(dispatcher, consumer, delivery))
        real = agent.settle_wake_task

        async def task_write_cancelled(obligation_, reason):
            raise asyncio.CancelledError()

        agent.settle_wake_task = task_write_cancelled
        await dispatcher.run_wake_settlement_pass()
        assert dispatcher._outcome_log_control_error is None
        assert set(dispatcher._source_settlement_control) == {delivery}
        # Admission is not latched: another input's wake is still admitted.
        handle = await dispatcher.enqueue_durable_cognition(
            wake(await snapshot(manager, other), manager),
            source_event_id=(await obligation(manager, other)).identity,
            consumer_id=consumer,
        )
        assert (await handle.wait_for_durable_admission()).disposition is (
            DurableAdmissionDisposition.COMMITTED
        )
        await handle.wait()
        # The retained entry keeps the owner unretired and fails close until
        # a pass reconciles it.
        assert dispatcher._runtime_owner_liveness_required()
        agent.settle_wake_task = real
        await dispatcher.run_wake_settlement_pass()
        assert dispatcher._source_settlement_control == {}
        assert (await obligation(manager, task)).state == ENDED
    finally:
        agent.settle_wake_task = real
        await close_input_dispatcher(agent, dispatcher)


async def test_cancelled_settlement_pass_retains_entry_and_stops_bare(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        delivery = await persisted_wake(
            manager, dispatcher, task, (await obligation(manager, task)).identity
        )
        await fail_terminally(dispatcher, consumer, await claim(dispatcher, consumer, delivery))
        entered = asyncio.Event()
        real = agent.settle_wake_task

        async def commit_unknown_while_cancelled(obligation_, reason):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                raise ExecutionCommitOutcomeError("unknown")

        agent.settle_wake_task = commit_unknown_while_cancelled
        running = asyncio.create_task(dispatcher.run_wake_settlement_pass())
        await entered.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError) as stopped:
            await running
        assert type(stopped.value) is asyncio.CancelledError
        assert stopped.value.__context__ is None and stopped.value.__cause__ is None
        [(key, carrier)] = dispatcher._source_settlement_control.values()
        assert isinstance(carrier, ExecutionCommitOutcomeError)
        assert dispatcher._outcome_log_control_error is None
        assert dispatcher.is_settlement_control(carrier)
        dispatcher._retain_durable_writer_control(carrier)
        assert dispatcher._outcome_log_control_error is None
        agent.settle_wake_task = real
        await dispatcher.run_wake_settlement_pass()
        assert dispatcher._source_settlement_control == {}
    finally:
        agent.settle_wake_task = real
        await close_input_dispatcher(agent, dispatcher)


# ---------------------------------------------------------------------------
# I3 on every transition into SUBMITTED, and the live producer.
# ---------------------------------------------------------------------------


async def test_new_input_arms_its_own_obligation_and_old_wake_is_released(manager):
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        task = await manager.create_task(request(), creator_agent_id="did:creator")
        delivery = await persisted_wake(
            manager, dispatcher, task, (await obligation(manager, task)).identity
        )
        assert (await obligation(manager, task)).state == BOUND
        await manager.continue_task(followup(manager, task), creator_agent_id="did:creator")
        assert (await obligation(manager, task, 1)).state == OWED
        # The superseded input's wake ends undelivered; its task holds a
        # newer input, so settlement lets the old obligation go.
        await fail_terminally(dispatcher, consumer, await claim(dispatcher, consumer, delivery))
        await dispatcher.run_wake_settlement_pass()
        assert (await obligation(manager, task)).state == RELEASED
        assert (await obligation(manager, task, 1)).state == OWED
        assert await state_of(manager, task) == TaskState.TASK_STATE_SUBMITTED
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def test_live_submission_callback_admits_at_the_obligation_generation(manager):
    agent, dispatcher, consumer = await input_dispatcher(manager, held=True)
    manager._on_task_submitted = agent._on_task_submitted
    try:
        task = await manager.create_task(request(), creator_agent_id="did:creator")
        await asyncio.gather(*agent.tasks)
        current = await obligation(manager, task)
        assert current.state == BOUND
        [row] = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        assert row.delivery_id == current.live_delivery_id
        assert row.event.source_event_id == wake_identity(task.id, 0, 1)
        # Boot finds nothing owed, so it admits no second wake.
        await agent.reconcile_a2a_cognition_wakes()
        assert len(await dispatcher.list_durable_deliveries(consumer_id=consumer)) == 1
    finally:
        manager._on_task_submitted = None
        await close_input_dispatcher(agent, dispatcher)


async def test_stale_generation_admission_is_refused(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        stale = (await obligation(manager, task)).identity
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        await manager.update_status(
            task.id, TaskState.TASK_STATE_SUBMITTED, recipient_agent_id=manager.did
        )
        handle = await dispatcher.enqueue_durable_cognition(
            wake(await snapshot(manager, task), manager),
            source_event_id=stale, consumer_id=consumer,
        )
        assert (await handle.wait_for_durable_admission()).disposition is (
            DurableAdmissionDisposition.REFUSED
        )
        await handle.wait()
        assert await dispatcher.list_durable_deliveries(consumer_id=consumer) == []
        assert (await obligation(manager, task)).state == OWED
    finally:
        await close_input_dispatcher(agent, dispatcher)
