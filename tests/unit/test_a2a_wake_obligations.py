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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from kestrel_sdk.native_a2a.types import TaskState, TaskStatusUpdateEvent

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.wake_obligation_store import (
    BOUND,
    CONFLICT,
    CONSUMER_ID,
    ENDED,
    OWED,
    RAN,
    RELEASED,
    TABLE,
    WakeObligation,
    wake_identity,
)
from kestrel_sovereign.agent.event_manager import EventManagerMixin
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
    # The completion wake is a durable projection only the settlement's own
    # projection raises; an open subscriber is also reached by publication.
    completions = []
    manager._on_task_complete = completions.append
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
            assert completions == []
            await dispatcher.run_wake_settlement_pass()
            assert calls[-1].disposition == ENDED and not calls[-1].recorded
        assert dispatcher._source_settlement_control == {}
        # The committed disposition was projected exactly once.
        assert len(completions) == 1
        async with asyncio.timeout(5):
            event = await anext(stream)
        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.TASK_STATE_FAILED
    finally:
        manager._on_task_complete = None
        await stream.aclose()
        await close_input_dispatcher(agent, dispatcher)


async def test_settlement_on_a_peer_reaches_this_managers_open_subscriber(
    manager, native_pg
):
    """A settlement committed elsewhere still ends this manager's open stream.

    Subscribers are this manager's own queues, and the peer that settles
    projects only to its own. This manager's pass has nothing left to settle,
    yet it publishes the committed terminal status to its subscriber, and
    re-runs none of the peer's durable projections.
    """
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    peer_backend = native_pg()
    await peer_backend.connect()
    peer_manager = manager_for(peer_backend, manager.task_store.scope)
    await peer_manager.initialize()
    peer_agent, peer, _ = await input_dispatcher(peer_manager)
    completions, peer_completions = [], []
    manager._on_task_complete = completions.append
    peer_manager._on_task_complete = peer_completions.append
    stream = manager.subscribe(task.id, recipient_agent_id=manager.did)
    try:
        assert (await anext(stream)).status.state == TaskState.TASK_STATE_SUBMITTED
        delivery = await persisted_wake(
            manager, dispatcher, task, (await obligation(manager, task)).identity
        )
        await fail_terminally(peer, consumer, await claim(peer, consumer, delivery))
        assert await peer.run_wake_settlement_pass() == 1
        assert (await obligation(manager, task)).state == ENDED
        assert len(peer_completions) == 1
        assert await dispatcher.run_wake_settlement_pass() == 0
        await dispatcher.run_wake_settlement_pass()
        # Published once, however many passes find it before it is read.
        assert [queue.qsize() for queue in manager._subscribers[task.id]] == [1]
        async with asyncio.timeout(5):
            event = await anext(stream)
        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.TASK_STATE_FAILED
        # Publication only: the peer's completion wake is not raised again.
        await dispatcher.run_wake_settlement_pass()
        assert completions == [] and len(peer_completions) == 1
    finally:
        await stream.aclose()
        manager._on_task_complete = None
        await close_input_dispatcher(peer_agent, peer)
        await peer_manager.close()
        await close_input_dispatcher(agent, dispatcher)


async def test_settled_wake_projection_is_retried_after_an_ordinary_failure(manager):
    """A committed settlement whose projection failed is projected next pass."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    completions = []
    manager._on_task_complete = completions.append
    try:
        delivery = await persisted_wake(
            manager, dispatcher, task, (await obligation(manager, task)).identity
        )
        await fail_terminally(dispatcher, consumer, await claim(dispatcher, consumer, delivery))
        real = manager.project_wake_settlement
        attempts = []

        async def fails_once(outcome):
            attempts.append(outcome)
            if len(attempts) == 1:
                raise RuntimeError("projection read failed")
            await real(outcome)

        manager.project_wake_settlement = fails_once
        # The settlement commits; its failed projection is not mistaken for
        # an input that stays owed, nor kept as control.
        assert await dispatcher.run_wake_settlement_pass() == 1
        assert (await obligation(manager, task)).state == ENDED
        assert await state_of(manager, task) == TaskState.TASK_STATE_FAILED
        assert dispatcher._source_settlement_control == {}
        assert completions == []
        assert await dispatcher.run_wake_settlement_pass() == 0
        assert [(a.disposition, a.recorded) for a in attempts] == [
            (ENDED, True), (ENDED, False),
        ]
        assert len(completions) == 1
        # Projected once: a later pass has nothing left to project.
        await dispatcher.run_wake_settlement_pass()
        assert len(attempts) == 2 and len(completions) == 1
    finally:
        manager.project_wake_settlement = real
        manager._on_task_complete = None
        await close_input_dispatcher(agent, dispatcher)


async def test_settlement_projection_failure_still_ends_the_open_stream(manager):
    """An open subscriber is not left waiting on a failing projection."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    stream = manager.subscribe(task.id, recipient_agent_id=manager.did)
    real = manager.project_wake_settlement

    async def always_fails(outcome):
        raise RuntimeError("projection read failed")

    try:
        assert (await anext(stream)).status.state == TaskState.TASK_STATE_SUBMITTED
        delivery = await persisted_wake(
            manager, dispatcher, task, (await obligation(manager, task)).identity
        )
        await fail_terminally(dispatcher, consumer, await claim(dispatcher, consumer, delivery))
        manager.project_wake_settlement = always_fails
        assert await dispatcher.run_wake_settlement_pass() == 1
        async with asyncio.timeout(5):
            event = await anext(stream)
        assert event.status.state == TaskState.TASK_STATE_FAILED
    finally:
        manager.project_wake_settlement = real
        await stream.aclose()
        await close_input_dispatcher(agent, dispatcher)


# ---------------------------------------------------------------------------
# R7 (Q1 lock order): two dispatchers on one database serialize every write
# to one input. Each case holds one writer inside its transaction at a
# barrier, then checks what the other dispatcher sees and does meanwhile.
# The races are the three wo-4001-9 boot races in obligation terms, plus
# settlement against a peer's re-arm in both orders.
# ---------------------------------------------------------------------------

# A peer that does not wait for the held writer's locks finishes well inside
# this; one that waits stays blocked until the barrier is released.
BLOCKED_WINDOW = 0.5


class Barrier:
    """Hold one writer at a point inside its transaction until released."""

    def __init__(self):
        self.reached, self.released = asyncio.Event(), asyncio.Event()

    def before(self, target, name):
        real = getattr(target, name)

        async def held(*args, **kwargs):
            self.reached.set()
            await self.released.wait()
            return await real(*args, **kwargs)

        setattr(target, name, held)

    def after(self, target, name):
        real = getattr(target, name)

        async def held(*args, **kwargs):
            result = await real(*args, **kwargs)
            self.reached.set()
            await self.released.wait()
            return result

        setattr(target, name, held)

    async def wait_reached(self, operation):
        reached = asyncio.ensure_future(self.reached.wait())
        try:
            await asyncio.wait({reached, operation}, timeout=10,
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            reached.cancel()
        if operation.done():
            operation.result()
            raise AssertionError("the held writer finished without reaching its barrier")
        assert self.reached.is_set(), "the held writer never reached its barrier"


async def assert_blocked(operation, what):
    await asyncio.wait({operation}, timeout=BLOCKED_WINDOW)
    if operation.done():
        operation.result()
        raise AssertionError(f"{what} did not wait for the held writer")


async def live_wakes(dispatcher, consumer):
    return [
        row for row in await dispatcher.list_durable_deliveries(consumer_id=consumer)
        if row.status in LIVE
    ]


async def assert_serial_view(manager, dispatcher, consumer, task, revision=0):
    """A reader sees one input in a state some serial order of writes left."""
    state = await state_of(manager, task)
    current = await obligation(manager, task, revision)
    live = await live_wakes(dispatcher, consumer)
    if state == TaskState.TASK_STATE_FAILED:
        assert current.state == ENDED and live == [], (state, current, live)
    else:
        assert state in (TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING)
        assert current.state in (OWED, BOUND, RAN), (state, current)
    assert [row.delivery_id for row in live] in (
        [], [current.live_delivery_id]
    ), (current, live)


async def bound_wake_failed_on_working_task(manager, dispatcher, consumer, task):
    """Generation 1's bound wake ended undelivered while its task WORKS."""
    delivery = await persisted_wake(
        manager, dispatcher, task, (await obligation(manager, task)).identity
    )
    await manager.update_status(
        task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
    )
    await fail_terminally(dispatcher, consumer, await claim(dispatcher, consumer, delivery))
    return delivery


async def admit(dispatcher, manager, task, identity, consumer):
    handle = await dispatcher.enqueue_durable_cognition(
        wake(await snapshot(manager, task), manager),
        source_event_id=identity, consumer_id=consumer,
    )
    receipt = await handle.wait_for_durable_admission()
    await handle.wait()
    return receipt.disposition


async def rearm(manager, task):
    await manager.update_status(
        task.id, TaskState.TASK_STATE_SUBMITTED, recipient_agent_id=manager.did
    )


R7_RACES = [
    # Settlement holds the input before its task write: a peer's re-arm
    # waits, then finds the task ended.
    "settlement_first_at_task_write",
    # Settlement holds the input between its task write and its obligation
    # record: no reader sees one without the other.
    "settlement_first_at_obligation_record",
    # A peer's re-arm holds the input: settlement waits, then finds nothing
    # owed, and the re-armed generation's wake runs (wo-4001-9 attempt 1).
    "rearm_first",
    # Boot read an owed generation; a peer re-armed and admitted the next
    # one before boot's admission (wo-4001-9 attempt 2).
    "stale_scan_admits_after_peer",
    # Two dispatchers admit the same owed generation (wo-4001-8).
    "both_admit_the_owed_generation",
]


@pytest.mark.parametrize("race", R7_RACES)
async def test_settlement_and_peer_release_serialize(manager, native_pg, race):
    """Never a task ended while a live wake holds its input, nor two wakes."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    peer_backend = native_pg()
    await peer_backend.connect()
    peer_manager = manager_for(peer_backend, manager.task_store.scope)
    await peer_manager.initialize()
    peer_agent, peer, _ = await input_dispatcher(peer_manager)
    barrier = Barrier()
    running = []

    def start(coroutine):
        running.append(asyncio.ensure_future(coroutine))
        return running[-1]

    try:
        if race.startswith("settlement_first"):
            await bound_wake_failed_on_working_task(manager, dispatcher, consumer, task)
            if race == "settlement_first_at_task_write":
                barrier.before(agent, "settle_wake_task")
            else:
                barrier.before(dispatcher._durable_store._obligations, "settle")
            settling = start(dispatcher.settle_wake_obligation(
                task_id=task.id, input_revision=0
            ))
            await barrier.wait_reached(settling)
            await assert_serial_view(peer_manager, peer, consumer, task)
            rearming = start(rearm(peer_manager, task))
            await assert_blocked(rearming, "the peer's re-arm")
            barrier.released.set()
            outcome = await settling
            assert (outcome.disposition, outcome.recorded) == (ENDED, True)
            with pytest.raises(ValueError):
                await rearming
            assert await state_of(manager, task) == TaskState.TASK_STATE_FAILED
        elif race == "rearm_first":
            failed = await bound_wake_failed_on_working_task(
                manager, dispatcher, consumer, task
            )
            barrier.after(peer_manager.task_store.obligations, "arm")
            rearming = start(rearm(peer_manager, task))
            await barrier.wait_reached(rearming)
            await assert_serial_view(manager, dispatcher, consumer, task)
            settling = start(dispatcher.settle_wake_obligation(
                task_id=task.id, input_revision=0
            ))
            await assert_blocked(settling, "the settlement")
            barrier.released.set()
            await rearming
            outcome = await settling
            assert (outcome.disposition, outcome.recorded) == ("not_owed", False)
            rearmed = await obligation(peer_manager, task)
            assert (rearmed.state, rearmed.generation) == (OWED, 2)
            assert await admit(peer, peer_manager, task, rearmed.identity, consumer) is (
                DurableAdmissionDisposition.COMMITTED
            )
            # The re-armed generation's own wake was admitted and ran.
            current = await obligation(manager, task)
            assert (current.state, current.generation) == (RAN, 2)
            assert current.live_delivery_id != failed
            assert await state_of(manager, task) == TaskState.TASK_STATE_SUBMITTED
        elif race == "stale_scan_admits_after_peer":
            scanned = (await obligation(manager, task)).identity
            await peer_manager.update_status(
                task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
            )
            await rearm(peer_manager, task)
            rearmed = await obligation(peer_manager, task)
            assert await admit(peer, peer_manager, task, rearmed.identity, consumer) is (
                DurableAdmissionDisposition.COMMITTED
            )
            assert await admit(dispatcher, manager, task, scanned, consumer) is (
                DurableAdmissionDisposition.REFUSED
            )
            # Only the peer's generation-2 wake was admitted, and it ran.
            current = await obligation(manager, task)
            assert (current.state, current.generation) == (RAN, 2)
        else:
            identity = (await obligation(manager, task)).identity
            barrier.after(dispatcher._durable_store._obligations, "bind")
            admitting = start(persisted_wake(manager, dispatcher, task, identity))
            await barrier.wait_reached(admitting)
            await assert_serial_view(peer_manager, peer, consumer, task)
            peer_admitting = start(admit(peer, peer_manager, task, identity, consumer))
            await assert_blocked(peer_admitting, "the peer's admission")
            barrier.released.set()
            first = await admitting
            assert await peer_admitting is DurableAdmissionDisposition.DUPLICATE
            assert (await obligation(manager, task)).live_delivery_id == first
        await assert_serial_view(manager, dispatcher, consumer, task)
        await assert_serial_view(peer_manager, peer, consumer, task)
        await assert_not_stranded(manager, task)
    finally:
        barrier.released.set()
        for operation in running:
            if not operation.done():
                operation.cancel()
        await asyncio.gather(*running, return_exceptions=True)
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


async def test_rearmed_input_resumes_after_hold_release_without_restart(manager):
    """A re-arm hands its input to the live producer; no reboot is needed.

    The generation-1 wake is admitted under Hold. The task then goes WORKING
    and back to SUBMITTED, which re-arms the input at generation 2 and makes
    that wake unclaimable. The re-arm's own wake must be admitted at once, so
    releasing Hold resumes the task in this process.
    """
    from kestrel_sovereign.hold import EffectiveHoldState
    from tests.unit.test_durable_signal_delivery import _HoldSnapshots

    agent, dispatcher, consumer = await input_dispatcher(manager, held=True)
    manager._on_task_submitted = agent._on_task_submitted
    prompts = []

    async def process(prompt):
        prompts.append(prompt)
        return "Handled the re-armed input"

    agent.process_input = process
    try:
        task = await manager.create_task(request(), creator_agent_id="did:creator")
        await asyncio.gather(*agent.tasks)
        first = await obligation(manager, task)
        assert (first.state, first.generation) == (BOUND, 1)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        await manager.update_status(
            task.id, TaskState.TASK_STATE_SUBMITTED, recipient_agent_id=manager.did
        )
        await asyncio.gather(*agent.tasks)
        rearmed = await obligation(manager, task)
        assert (rearmed.state, rearmed.generation) == (BOUND, 2)
        assert rearmed.live_delivery_id != first.live_delivery_id
        live = await dispatcher._durable_store.get_delivery(
            agent_id=manager.did, consumer_id=consumer,
            delivery_id=rearmed.live_delivery_id,
        )
        assert live.event.source_event_id == wake_identity(task.id, 0, 2)
        assert prompts == []
        agent._hold_store = _HoldSnapshots(EffectiveHoldState(host=None, agent=None))
        await dispatcher._drain_durable_cognition_consumer(consumer)
        assert len(prompts) == 1
        ran = await obligation(manager, task)
        assert (ran.state, ran.generation) == (RAN, 2)
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


# ---------------------------------------------------------------------------
# Boot pages owed obligations by key, and refuses a cursor that stands still.
# ---------------------------------------------------------------------------


def owed(task_id, input_revision=0):
    return WakeObligation(
        agent_id="did:kestrel:recipient", consumer_id=CONSUMER_ID, task_id=task_id,
        input_revision=input_revision, generation=1, state=OWED, live_delivery_id=None,
    )


class SamePageEveryTime:
    """An obligation store whose keyset ignores its cursor.

    It stops the test after ``bound`` pages, so a loop that never notices
    fails here instead of running the host out of memory.
    """

    def __init__(self, page, bound=50):
        self.page, self.bound, self.cursors = page, bound, []

    async def list_by_state(self, *, agent_id, state, after=None, limit=500):
        assert state == OWED
        self.cursors.append(after)
        if len(self.cursors) > self.bound:
            raise AssertionError(f"boot paged {self.bound} times without stopping")
        return list(self.page)


class KeysetPages:
    """An obligation store that honours its cursor, one obligation per page."""

    def __init__(self, obligations):
        self.obligations, self.cursors = obligations, []

    async def list_by_state(self, *, agent_id, state, after=None, limit=500):
        self.cursors.append(after)
        return [
            item for item in self.obligations
            if after is None or (item.task_id, item.input_revision) > after
        ][:1]


def paging_agent(store, admitted):
    async def admit(obligation_):
        admitted.append((obligation_.task_id, obligation_.input_revision))
        return None

    return SimpleNamespace(
        did="did:kestrel:recipient",
        signal_registry=SimpleNamespace(get=lambda source: SimpleNamespace(retention_days=14)),
        dispatcher=SimpleNamespace(run_wake_settlement_pass=AsyncMock(return_value=0)),
        task_manager=SimpleNamespace(
            task_store=SimpleNamespace(obligations=store),
            list_cognition_wake_candidates=AsyncMock(return_value=[]),
        ),
        _admit_owed_wake=admit,
    )


async def test_boot_paging_refuses_a_cursor_that_does_not_advance():
    store = SamePageEveryTime([owed("task-a"), owed("task-b", 2)])
    admitted = []
    with pytest.raises(RuntimeError, match="paging did not advance"):
        await EventManagerMixin.reconcile_a2a_cognition_wakes(
            paging_agent(store, admitted)
        )
    assert store.cursors == [None, ("task-b", 2)]
    # The repeated page is refused before any of it is admitted again.
    assert admitted == [("task-a", 0), ("task-b", 2)]


async def test_boot_paging_admits_every_page_of_an_advancing_cursor():
    store = KeysetPages([owed("task-a"), owed("task-a", 1), owed("task-b")])
    admitted = []
    await EventManagerMixin.reconcile_a2a_cognition_wakes(paging_agent(store, admitted))
    assert admitted == [("task-a", 0), ("task-a", 1), ("task-b", 0)]
    assert store.cursors == [None, ("task-a", 0), ("task-a", 1), ("task-b", 0)]
