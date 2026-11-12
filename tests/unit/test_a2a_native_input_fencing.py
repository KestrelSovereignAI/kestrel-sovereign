"""A later creator input must fence every old recipient execution and wake."""

import asyncio
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_data_artifact, new_text_message
from kestrel_sdk.native_a2a.types import Role, SendMessageRequest, Task, TaskState, TaskStatus

from kestrel_sovereign.a2a.stores.unified.task_store import (
    TaskInputConflictError,
    TaskInputTerminalError,
    TaskMutationAuthorizationError,
)
from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.signals.context import bind_current_signal
from kestrel_sovereign.signals.durable import DurableSignalEvent
from kestrel_sovereign.signals.rehydration import DurableCognitionWithdrawal
from kestrel_sovereign.signals.sources.a2a_task_submitted import (
    build_signal_for_submitted_task,
    parse_submitted_input_identity,
    submitted_input_identity,
)
from tests.unit.test_a2a_native_task_manager import manager as manager
from tests.unit.test_a2a_native_task_manager import request


def followup(manager, task, text="Refined request", **kwargs):
    return SendMessageRequest(
        tenant=manager.task_store.scope.tenant,
        message=new_text_message(text, role=Role.ROLE_USER, task_id=task.id, **kwargs),
        configuration={"return_immediately": True},
    )


async def snapshot(manager, task):
    return await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)


def wake(candidate, manager):
    return build_signal_for_submitted_task(candidate, manager.did)


def owner(manager):
    agent = EventManagerMixin()
    agent.did = manager.did
    agent.task_manager = manager
    return agent


def private_event(signal):
    now = datetime.now(UTC)
    return DurableSignalEvent(
        event_id=signal.id,
        source_event_id=signal.dedupe_key,
        agent_id=signal.target_agent,
        target_agent=signal.target_agent,
        source=signal.source,
        kind=signal.kind,
        mode=signal.mode.value,
        payload={"_privacy_gated": "none"},
        session_id=None,
        caller_identity=None,
        visibility=signal.visibility.value,
        urgency=signal.urgency.value,
        dedupe_key=signal.dedupe_key,
        causation_chain=[],
        arrived_at=now,
        committed_at=now,
        retention_until=now + timedelta(days=1),
    )


def identity(event):
    """The durable identity the settlement hook names a wake by."""
    return {
        "source": event.source,
        "target_agent": event.target_agent,
        "source_event_id": event.source_event_id,
    }


async def test_old_private_event_never_rehydrates_as_new_input(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    first = wake(await snapshot(manager, task), manager)
    recipient = owner(manager)
    event = private_event(first)
    assert await recipient.rehydrate_durable_cognition_signal(
        event, dispatch_signal=first
    )
    await manager.continue_task(followup(manager, task), creator_agent_id="did:creator")
    second = wake(await snapshot(manager, task), manager)
    assert first.dedupe_key != second.dedupe_key
    assert parse_submitted_input_identity(first.dedupe_key) == (task.id, 0)
    assert parse_submitted_input_identity(second.dedupe_key) == (task.id, 1)
    assert second.payload["request_content"] == "Please do this\n\nRefined request"
    assert isinstance(
        await recipient.rehydrate_durable_cognition_signal(
            event, dispatch_signal=first
        ),
        DurableCognitionWithdrawal,
    )
    recovered = await recipient.rehydrate_durable_cognition_signal(
        private_event(second), dispatch_signal=second
    )
    assert recovered.payload == second.payload
    refusal = await recipient.validate_cognition_signal_execution(first)
    assert refusal is not None and "superseded" in refusal
    assert "superseded" in await recipient.monitor_cognition_signal_execution(first)
    assert await recipient.validate_cognition_signal_execution(second) is None


async def test_execution_monitor_withdraws_original_after_admission(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    signal = wake(await snapshot(manager, task), manager)
    recipient = owner(manager)
    monitor = asyncio.create_task(recipient.monitor_cognition_signal_execution(signal))
    try:
        await asyncio.sleep(0)
        assert not monitor.done()
        await manager.continue_task(
            followup(manager, task), creator_agent_id="did:creator"
        )
        async with asyncio.timeout(3):
            assert "superseded" in await monitor
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)


@pytest.mark.parametrize(
    "operation",
    [
        "manager-status",
        "manager-artifact",
        "manager-decline",
        "store-status",
        "store-artifact",
        "store-lifecycle",
        "store-terminal",
        "store-decline",
    ],
)
async def test_superseded_input_cannot_mutate_the_new_input(manager, operation):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    signal = wake(await snapshot(manager, task), manager)
    await manager.continue_task(followup(manager, task), creator_agent_id="did:creator")
    expected = await snapshot(manager, task)
    artifact = new_data_artifact("old-result", {"old": True})
    stale = Task.FromString(task.SerializeToString())
    stale.status.state = TaskState.TASK_STATE_COMPLETED
    store = manager.task_store
    with bind_current_signal(signal):
        if operation == "manager-status":
            with pytest.raises(ValueError, match="already"):
                await manager.update_status(
                    task.id,
                    TaskState.TASK_STATE_COMPLETED,
                    recipient_agent_id=manager.did,
                )
        elif operation == "manager-artifact":
            with pytest.raises(TaskMutationAuthorizationError):
                await manager.add_artifact(
                    task.id, artifact, recipient_agent_id=manager.did
                )
        elif operation == "manager-decline":
            with pytest.raises(PermissionError):
                await manager.cancel_task(
                    task.id, agent_name=manager.did, recipient_agent_id=manager.did
                )
        elif operation == "store-status":
            assert not await store.update_status(
                task.id,
                TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                recipient_agent_id=manager.did,
                expected_state=TaskState.TASK_STATE_SUBMITTED,
            )
        elif operation == "store-artifact":
            with pytest.raises(TaskMutationAuthorizationError):
                await store.add_artifact(
                    task.id, artifact, recipient_agent_id=manager.did
                )
        elif operation == "store-lifecycle":
            assert (
                await store.save_recipient_lifecycle(
                    stale,
                    baseline=task,
                    recipient_agent_id=manager.did,
                    expected_state=TaskState.TASK_STATE_SUBMITTED,
                )
                is None
            )
        elif operation == "store-terminal":
            assert (
                await store.save_recipient_terminal_outcome(
                    stale,
                    baseline=task,
                    recipient_agent_id=manager.did,
                    operation_id="old-outcome",
                )
                is None
            )
        else:
            assert (
                await store.cancel_if_authorized(task.id, actor_agent_id=manager.did)
                is None
            )
    actual = await snapshot(manager, task)
    assert actual == expected, "A superseded input changed the new authoritative task"
    # The current input can still publish legitimate progress and results.
    with bind_current_signal(wake(actual, manager)):
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        await manager.add_artifact(task.id, artifact, recipient_agent_id=manager.did)
    assert (await snapshot(manager, task)).input_revision == 1


async def test_same_request_retry_is_not_another_wake_or_input(manager):
    observed = []
    manager._on_task_submitted = observed.append
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    next_request = followup(manager, task)
    one = await manager.continue_task(next_request, creator_agent_id="did:creator")
    two = await manager.continue_task(next_request, creator_agent_id="did:creator")
    assert one == two and len(observed) == 2
    assert (await snapshot(manager, task)).input_revision == 1
    next_request.message.parts[0].text = "Reused ID with changed content"
    with pytest.raises(TaskInputConflictError, match="Message ID"):
        await manager.continue_task(next_request, creator_agent_id="did:creator")
    assert len(observed) == 2 and len(two.history) == 2


async def test_terminal_input_and_creator_context_boundaries(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    assert (
        await manager.continue_task(
            followup(manager, task), creator_agent_id="did:other"
        )
        is None
    )
    with pytest.raises(TaskInputConflictError, match="context"):
        await manager.continue_task(
            followup(manager, task, context_id="different"),
            creator_agent_id="did:creator",
        )
    await manager.update_status(
        task.id, TaskState.TASK_STATE_COMPLETED, recipient_agent_id=manager.did
    )
    with pytest.raises(TaskInputTerminalError):
        await manager.continue_task(
            followup(manager, task), creator_agent_id="did:creator"
        )
    assert (await snapshot(manager, task)).input_revision == 0


@pytest.mark.parametrize(
    "value",
    [
        "task:working:1",
        "[]",
        '["task",true]',
        '["task",-1]',
        '["task",0,null]',
        '["task",0,1,2]',
    ],
)
def test_private_input_identity_rejects_ambiguous_or_unversioned_values(value):
    with pytest.raises(ValueError):
        parse_submitted_input_identity(value)


def test_input_identity_handles_arbitrary_task_identifiers():
    task_id = 'task:working:3["\\'
    assert parse_submitted_input_identity(submitted_input_identity(task_id, 7)) == (
        task_id,
        7,
    )
    assert parse_submitted_input_identity(
        submitted_input_identity(task_id, 7, working_revision=9)
    ) == (task_id, 7)


async def input_dispatcher(manager, *, held=False, private=False):
    from kestrel_sovereign.privacy import get_privacy_preset
    from kestrel_sovereign.signals import (
        DurableConsumerRegistration,
        OrderedLockManager,
        SignalDispatcher,
        SignalLogStore,
        SourceRegistry,
    )
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        DURABLE_COGNITION_CONSUMER_ID,
        build_a2a_task_submitted_registration,
    )
    from tests.unit.test_durable_signal_delivery import (
        _Agent,
        _held_state,
        _HoldSnapshots,
    )

    class Recipient(_Agent, EventManagerMixin):
        pass

    agent = Recipient(manager.did)
    agent.task_manager = manager
    if private:
        agent.privacy_config = get_privacy_preset("ephemeral")
    if held:
        agent._hold_store = _HoldSnapshots(_held_state(agent.did))
    logs = SignalLogStore(manager.task_store.backend)
    await logs.initialize()
    registry = SourceRegistry()
    from kestrel_sovereign.signals.sources.a2a import (
        DURABLE_COGNITION_CONSUMER_ID as COMPLETE_CONSUMER_ID,
    )
    from kestrel_sovereign.signals.sources.a2a import (
        build_a2a_task_complete_registration,
    )

    registry.register(build_a2a_task_submitted_registration())
    registry.register(build_a2a_task_complete_registration())
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=logs
    )
    agent.dispatcher = dispatcher
    agent.signal_registry = registry
    await dispatcher.initialize_durable_delivery()
    await dispatcher.register_durable_consumer(
        DurableConsumerRegistration(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID,
            source="a2a.task_submitted",
            agent_id=agent.did,
            max_attempts=0,
        )
    )
    # A task a settlement ends wakes the agent through its completion
    # consumer, as a booted agent's does.
    await dispatcher.register_durable_consumer(
        DurableConsumerRegistration(
            consumer_id=COMPLETE_CONSUMER_ID,
            source="a2a.task_complete",
            agent_id=agent.did,
            max_attempts=0,
        )
    )
    return agent, dispatcher, DURABLE_COGNITION_CONSUMER_ID


async def close_input_dispatcher(agent, dispatcher):
    await dispatcher.shutdown_durable_delivery()
    await asyncio.gather(*agent.tasks, return_exceptions=True)


async def heartbeat(dispatcher):
    """One owner heartbeat, and the settlement pass it starts, awaited.

    The heartbeat only starts the pass, so that renewal never waits on it
    (#4001); a test that reads what the pass did joins it.
    """
    await dispatcher._heartbeat_runtime_owner()
    await dispatcher._join_owed_settlement_pass()


async def test_restarted_private_recovery_delivers_only_current_input_once(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(
        manager, held=True, private=True
    )
    try:
        await agent.reconcile_a2a_cognition_wakes()
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        # The original held input still owns this working revision.
        await agent.reconcile_a2a_cognition_wakes()
        assert len(await dispatcher.list_durable_deliveries(consumer_id=consumer)) == 1
        await manager.continue_task(
            followup(manager, task), creator_agent_id="did:creator"
        )
        await agent.reconcile_a2a_cognition_wakes()
        await agent.reconcile_a2a_cognition_wakes()
        deliveries = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        assert len(deliveries) == 2
        assert {d.event.source_event_id for d in deliveries} == {
            submitted_input_identity(task.id, 0),
            submitted_input_identity(task.id, 1),
        }
        assert all(d.event.payload == {"_privacy_gated": "none"} for d in deliveries)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        await agent.reconcile_a2a_cognition_wakes()
        assert len(await dispatcher.list_durable_deliveries(consumer_id=consumer)) == 2
    finally:
        await close_input_dispatcher(agent, dispatcher)

    resumed, dispatcher, consumer = await input_dispatcher(manager, private=True)
    prompts = []

    async def process(prompt):
        prompts.append(prompt)
        return "Handled the current input"

    resumed.process_input = process
    try:
        await dispatcher._drain_durable_cognition_consumer(consumer)
        deliveries = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        assert all(d.is_terminal for d in deliveries), [
            (d.status, d.last_error) for d in deliveries
        ]
        assert len(prompts) == 1
        assert "Please do this" in prompts[0] and "Refined request" in prompts[0]
        assert (await snapshot(manager, task)).input_revision == 1
    finally:
        await close_input_dispatcher(resumed, dispatcher)


async def test_dispatcher_physically_joins_withdrawn_input_before_terminal_receipt(
    manager,
):
    from kestrel_sdk.signals import Status

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, _ = await input_dispatcher(manager)
    started, stopping, release, stopped = (asyncio.Event() for _ in range(4))

    async def process(prompt):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopping.set()
            await release.wait()
            stopped.set()

    agent.process_input = process
    delivery = asyncio.create_task(
        dispatcher.dispatch_signal(wake(await snapshot(manager, task), manager))
    )
    try:
        async with asyncio.timeout(5):
            await started.wait()
            await manager.continue_task(
                followup(manager, task), creator_agent_id="did:creator"
            )
            await stopping.wait()
            assert not delivery.done(), (
                "The dispatcher abandoned the original physical execution"
            )
            release.set()
            result = await delivery
        assert stopped.is_set()
        assert (
            result.status is Status.DROPPED_VALIDATION and "superseded" in result.error
        )
    finally:
        release.set()
        delivery.cancel()
        await asyncio.gather(delivery, return_exceptions=True)
        await close_input_dispatcher(agent, dispatcher)


async def test_immediate_callback_admits_distinct_inputs_before_restart(manager):
    agent, dispatcher, consumer = await input_dispatcher(manager, held=True)
    manager._on_task_submitted = agent._on_task_submitted

    try:
        task = await manager.create_task(request(), creator_agent_id="did:creator")
        await asyncio.gather(*agent.tasks)
        assert len(await dispatcher.list_durable_deliveries(consumer_id=consumer)) == 1
        await manager.continue_task(
            followup(manager, task), creator_agent_id="did:creator"
        )
        await asyncio.gather(*agent.tasks)
        deliveries = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        assert len(deliveries) == 2
        assert {d.event.source_event_id for d in deliveries} == {
            submitted_input_identity(task.id, 0),
            submitted_input_identity(task.id, 1),
        }
    finally:
        manager._on_task_submitted = None
        await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("reason", [None, "", " ", 0, False])
def test_source_withdrawal_requires_a_concrete_reason(reason):
    with pytest.raises(ValueError, match="concrete reason"):
        DurableCognitionWithdrawal(reason)


async def test_monitor_identifies_superseded_input_before_later_creator_cancellation(
    manager,
):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    original = wake(await snapshot(manager, task), manager)
    await manager.continue_task(followup(manager, task), creator_agent_id="did:creator")
    await manager.cancel_task(task.id, agent_name="did:creator")
    reason = await owner(manager).monitor_cognition_signal_execution(original)
    assert "superseded" in reason


@pytest.mark.parametrize("revision", [None, True, -1, "0", 0.5])
def test_submission_schema_requires_its_server_owned_integer_revision(revision):
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        _a2a_submitted_schema,
    )

    with pytest.raises(ValueError, match="input_revision"):
        _a2a_submitted_schema(
            {
                "task_id": "one",
                "session_id": "one",
                "sender": "did:sender",
                "input_revision": revision,
            }
        )


async def test_historical_task_cannot_be_rebuilt_as_a_new_input_wake(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    candidate = replace(await snapshot(manager, task), cognition_wake_eligible=False)
    with pytest.raises(ValueError, match="not eligible"):
        wake(candidate, manager)


async def settle_input(manager, event, *, stopped, cause="dropped_cycle"):
    """Apply one undelivered wake's settlement as the store's transaction does.

    The transition runs on its caller's transaction, here one the task store
    opens; the projection runs once that commits (#4001).
    """
    settlement = owner(manager).undelivered_cognition_wake_settlement(
        **identity(event), stopped=stopped, cause=cause
    )
    async with manager.task_store.write_transaction():
        kind, outcome = await settlement.apply()
    if kind == "ended":
        await settlement.project(outcome)
    return kind


@pytest.mark.parametrize("stopped", [True, False], ids=["stopped", "failed"])
async def test_an_undelivered_wake_ends_only_the_input_it_carried(manager, stopped):
    """#4001: a submission wake that ended without running ends its task, and
    only for the input it carried. A newer input the creator added meanwhile
    stays live for its own wake. Both wakes are payload-elided, as under a
    volatile privacy mode: the input identity alone settles them."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    stale = private_event(wake(await snapshot(manager, task), manager))
    await manager.continue_task(followup(manager, task), creator_agent_id="did:creator")

    assert await settle_input(manager, stale, stopped=stopped) == "released"
    current = await snapshot(manager, task)
    assert current.task.status.state == TaskState.TASK_STATE_SUBMITTED
    assert current.input_revision == 1

    current_wake = private_event(wake(current, manager))
    assert await settle_input(manager, current_wake, stopped=stopped) == "ended"
    settled = await snapshot(manager, task)
    text = settled.task.status.message.parts[0].text
    if stopped:
        assert settled.task.status.state == TaskState.TASK_STATE_CANCELED
        assert "Stop cancelled" in text
    else:
        assert settled.task.status.state == TaskState.TASK_STATE_FAILED
        assert "dropped_cycle" in text
    # Settling the same wake again finds nothing left to end.
    assert await settle_input(manager, current_wake, stopped=stopped) == "released"


async def test_an_undelivered_wake_leaves_a_task_awaiting_input_alone(manager):
    """A task that asked its creator for input is not waiting on the wake, so a
    stale wake for that input ending does not fail it. It may work on that
    input again, so the work stays owed rather than released."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    stale = private_event(wake(await snapshot(manager, task), manager))
    await manager.update_status(
        task.id, TaskState.TASK_STATE_INPUT_REQUIRED, recipient_agent_id=manager.did
    )

    assert await settle_input(
        manager, stale, stopped=False, cause="dropped_validation"
    ) == "owed"
    assert (await snapshot(manager, task)).task.status.state == (
        TaskState.TASK_STATE_INPUT_REQUIRED
    )


async def test_an_undelivered_wake_for_another_recipient_is_refused(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    foreign = replace(
        private_event(wake(await snapshot(manager, task), manager)),
        target_agent="did:kestrel:someone-else",
    )
    with pytest.raises(ValueError, match="another recipient"):
        await settle_input(manager, foreign, stopped=False)
    assert (await snapshot(manager, task)).task.status.state == (
        TaskState.TASK_STATE_SUBMITTED
    )


async def test_a_task_settlement_refuses_to_open_its_own_transaction(manager):
    """#4001 step 0: the task write may only join its caller's transaction,
    the delivery store's, so the two commit together. Outside one it
    refuses, writing nothing."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    settlement = owner(manager).undelivered_cognition_wake_settlement(
        **identity(private_event(wake(await snapshot(manager, task), manager))),
        stopped=False,
        cause="dropped_cycle",
    )
    with pytest.raises(RuntimeError, match="caller's"):
        await settlement.apply()
    assert (await snapshot(manager, task)).task.status.state == (
        TaskState.TASK_STATE_SUBMITTED
    )


async def claimed_submission_wake(manager, dispatcher, consumer, task, *, revision=0):
    """Persist a task input's submission wake and claim it for this owner."""
    import copy

    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    identity_key = submitted_input_identity(task.id, revision)
    await dispatcher.dispatch_signal(
        copy.deepcopy(wake(await snapshot(manager, task), manager)),
        source_event_id=identity_key,
    )
    original = await dispatcher.get_durable_delivery_for_source_event(
        consumer_id=consumer, source=SOURCE_NAME, source_event_id=identity_key
    )
    return await dispatcher.claim_durable_delivery_for_event(
        consumer_id=consumer,
        event_id=original.event_id,
        executor_id=dispatcher._durable_delivery_owner,
    )


async def earlier_terminal_row(dispatcher, consumer, claimed, *, error, terminal_ackable=False):
    """End a claimed wake's row as an earlier process commits it.

    The write goes to the ledger directly: this dispatcher made none, so it
    neither brings its own settlement pass forward nor learns of the row
    except by reading it.
    """
    return await dispatcher._durable_store.nack_delivery(
        agent_id=dispatcher._agent.did,
        consumer_id=consumer,
        delivery_id=claimed.delivery_id,
        lease_token=claimed.lease_token,
        error=error,
        terminal=True,
        terminal_ackable=terminal_ackable,
    )


async def failed_submission_wake(manager, dispatcher, consumer, task, *, revision=0):
    """A wake row a refused retry left failed, as an earlier process commits it."""
    claimed = await claimed_submission_wake(
        manager, dispatcher, consumer, task, revision=revision
    )
    failed = await earlier_terminal_row(
        dispatcher, consumer, claimed,
        error="execution_control_unresolved: tool_effect_reconciliation_required",
    )
    assert failed.status == "failed"
    return failed


def count_failed_row_writes(dispatcher):
    """Count every exact-token failed-row CAS, on either backend."""
    store = dispatcher._durable_store
    owner, name = (
        (store.backend, "fail_cognition_delivery")
        if store.backend.backend_type == "postgres"
        else (store, "release_managed_delivery_after_task")
    )
    write = getattr(owner, name)
    writes = []

    async def counted_write(**kwargs):
        writes.append(kwargs["delivery_id"])
        return await write(**kwargs)

    setattr(owner, name, counted_write)
    return writes


def fault_task_writes(manager, faults):
    """Raise each fault in turn from the settlement's real task write.

    The write runs inside the delivery store's settlement transaction, so a
    fault rolls that whole settlement back (#4001).
    """
    save = manager.task_store.apply_undelivered_wake_transition
    attempts = []

    async def save_with_faults(*args, **kwargs):
        attempts.append(len(faults))
        if faults:
            injected = faults.pop(0)
            if callable(injected):
                return await injected(save, *args, **kwargs)
            raise injected
        return await save(*args, **kwargs)

    manager.task_store.apply_undelivered_wake_transition = save_with_faults
    return attempts


def _cancellation_with_unknown_commit():
    cancelled = asyncio.CancelledError()
    cancelled._kestrel_unresolved_commit_outcome = "unknown"
    return cancelled


@pytest.mark.parametrize(
    "state",
    [TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING],
    ids=["submitted", "working"],
)
async def test_boot_reconcile_ends_a_task_whose_wake_already_failed(manager, state):
    """#4001: a process that stopped after failing a submission wake leaves
    the task live, with no wake left to run it. Boot reconciliation opens
    with the settlement pass, which ends it, rather than leaving it for its
    sender to wait on or minting another wake for the same refused input.
    The pass runs before boot reads its candidates, so boot sees the task
    ended and admits only the completion it is owed."""
    from kestrel_sovereign.signals import DurableAdmissionDisposition

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        await failed_submission_wake(manager, dispatcher, consumer, task)
        if state == TaskState.TASK_STATE_WORKING:
            await manager.update_status(task.id, state, recipient_agent_id=manager.did)
        admissions = []
        enqueue = dispatcher.enqueue_durable_cognition

        async def recorded_enqueue(signal, **kwargs):
            handle = await enqueue(signal, **kwargs)
            receipt = await handle.wait_for_durable_admission()
            admissions.append((signal.source, receipt.disposition))
            return handle

        dispatcher.enqueue_durable_cognition = recorded_enqueue

        await agent.reconcile_a2a_cognition_wakes()

        # The pass ends the task before boot reads its candidates, so boot
        # mints no submission wake for it: only the completion the ended
        # task is owed.
        assert admissions == [
            ("a2a.task_complete", DurableAdmissionDisposition.COMMITTED)
        ], admissions

        settled = await snapshot(manager, task)
        assert settled.task.status.state == TaskState.TASK_STATE_FAILED
        assert "tool_effect_reconciliation_required" in (
            settled.task.status.message.parts[0].text
        )
        assert [d.status for d in await dispatcher.list_durable_deliveries(
            consumer_id=consumer
        )] == ["failed"]
    finally:
        await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("when", ["after_keys", "after_pass"])
@pytest.mark.parametrize("ending", ["failed", "stopped", "acknowledged", "still_live"])
@pytest.mark.parametrize("held_by", ["original", "earlier_recovery"])
async def test_boot_mints_a_wake_only_for_an_input_no_wake_still_carries(
    manager, held_by, ending, when
):
    """Self-review of wo-4001-8 and of wo-4001-9: a WORKING task's input is
    held by a live wake when boot reads the ledger. That wake is its original
    submission wake, or an earlier recovery wake, minted at an older
    lifecycle revision of the same input; the task has since moved back to
    SUBMITTED and to WORKING again, so boot would mint under a new identity.
    A peer then ends the wake, just after boot's read or after its settlement
    pass too, or leaves it running. Before the fix boot looked again at the
    original wake alone: it minted a second wake beside a live earlier
    recovery wake, so one input ran twice, and beside one a peer had just
    failed, so the next settlement pass ended work the new wake still ran.
    Admission now decides as boot mints: it refuses a wake while another one
    that has not run carries the input. A wake that ended undelivered is
    settled by the next pass, and one that still runs is left to itself. One
    that ran leaves its input to recovery, so boot mints exactly one wake
    for it, as it does for any input whose wake ran (#4001)."""
    import copy

    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        held = await claimed_submission_wake(manager, dispatcher, consumer, task)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        if held_by == "earlier_recovery":
            assert await dispatcher.ack_durable_delivery(
                consumer_id=consumer, delivery_id=held.delivery_id,
                lease_token=held.lease_token,
            )
            earlier = await snapshot(manager, task)
            identity = submitted_input_identity(
                task.id, 0, working_revision=earlier.lifecycle_revision
            )
            await dispatcher.dispatch_signal(
                copy.deepcopy(wake(earlier, manager)), source_event_id=identity
            )
            held = await dispatcher.get_durable_delivery_for_source_event(
                consumer_id=consumer, source=SOURCE_NAME, source_event_id=identity
            )
            held = await dispatcher.claim_durable_delivery_for_event(
                consumer_id=consumer, event_id=held.event_id,
                executor_id=dispatcher._durable_delivery_owner,
            )
            # The same input, now at a later lifecycle revision.
            await manager.update_status(
                task.id, TaskState.TASK_STATE_SUBMITTED, recipient_agent_id=manager.did
            )
            await manager.update_status(
                task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
            )
        rows_before = {
            d.delivery_id
            for d in await dispatcher.list_durable_deliveries(consumer_id=consumer)
        }
        ended = []

        async def a_peer_ends_the_wake():
            if ended:
                return
            ended.append(held.delivery_id)
            if ending == "still_live":
                return
            if ending == "acknowledged":
                assert await dispatcher._durable_store.ack_delivery(
                    agent_id=agent.did, consumer_id=consumer,
                    delivery_id=held.delivery_id, lease_token=held.lease_token,
                )
                return
            await earlier_terminal_row(
                dispatcher, consumer, held,
                error=(
                    "stop_acknowledged: operator Stop" if ending == "stopped"
                    else "execution_control_unresolved: "
                    "tool_effect_reconciliation_required"
                ),
                terminal_ackable=ending == "stopped",
            )

        if when == "after_keys":
            assign = dispatcher.assign_durable_wake_exclusive_keys

            async def keys_then_end():
                changed = await assign()
                await a_peer_ends_the_wake()
                return changed

            dispatcher.assign_durable_wake_exclusive_keys = keys_then_end
        else:
            settle = dispatcher.settle_owed_cognition_sources

            async def settle_then_end():
                await settle()
                await a_peer_ends_the_wake()

            dispatcher.settle_owed_cognition_sources = settle_then_end
        await agent.reconcile_a2a_cognition_wakes()
        assert ended == [held.delivery_id]

        rows = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        minted = [d for d in rows if d.delivery_id not in rows_before]
        summary = [(d.event.source_event_id, d.status) for d in rows]
        if ending == "acknowledged":
            assert len(minted) == 1 and not minted[0].is_terminal, summary
        else:
            assert minted == [], (
                "boot minted a second wake for an input another wake carries",
                summary,
            )
        # Never two wakes that have not run for one input.
        assert sum(d.status != "acknowledged" for d in rows) <= 1, summary
        await heartbeat(dispatcher)
        assert (await snapshot(manager, task)).task.status.state == {
            "failed": TaskState.TASK_STATE_FAILED,
            "stopped": TaskState.TASK_STATE_CANCELED,
            "acknowledged": TaskState.TASK_STATE_WORKING,
            "still_live": TaskState.TASK_STATE_WORKING,
        }[ending]
        assert {
            d.delivery_id
            for d in await dispatcher.list_durable_deliveries(consumer_id=consumer)
        } == rows_before | {d.delivery_id for d in minted}
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def test_owed_settlements_are_terminal_wakes_whose_input_still_waits(
    manager, monkeypatch
):
    """#4001: what is owed is read from durable state alone. A task is owed a
    settlement exactly while its current input waits on a wake whose row is
    terminal and unacknowledged: not while that wake is still live or has
    run, not once the task ended, and not for an input the creator has since
    superseded. A WORKING task whose submission wake ran is owed through the
    WORKING wake boot mints for it. Every terminal row is read, page by page."""
    import copy

    from kestrel_sovereign.signals import dispatcher as dispatcher_module
    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    monkeypatch.setattr(dispatcher_module, "_UNDELIVERED_WAKE_PAGE_SIZE", 2)
    tasks = {
        name: await manager.create_task(request(), creator_agent_id="did:creator")
        for name in (
            "failed", "refused", "leased", "acknowledged", "ended", "superseded",
            "working",
        )
    }
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        failed = await failed_submission_wake(manager, dispatcher, consumer, tasks["failed"])
        refused = await claimed_submission_wake(manager, dispatcher, consumer, tasks["refused"])
        refused = await earlier_terminal_row(
            dispatcher, consumer, refused,
            error="stop_acknowledged: operator Stop", terminal_ackable=True,
        )
        assert refused.status == "terminal_ackable"
        await claimed_submission_wake(manager, dispatcher, consumer, tasks["leased"])
        ran = await claimed_submission_wake(manager, dispatcher, consumer, tasks["acknowledged"])
        assert await dispatcher.ack_durable_delivery(
            consumer_id=consumer, delivery_id=ran.delivery_id, lease_token=ran.lease_token,
        )
        await failed_submission_wake(manager, dispatcher, consumer, tasks["ended"])
        await manager.fail_task(
            tasks["ended"].id, "ended elsewhere", agent_name=manager.did,
            recipient_agent_id=manager.did,
        )
        await failed_submission_wake(manager, dispatcher, consumer, tasks["superseded"])
        await manager.continue_task(
            followup(manager, tasks["superseded"]), creator_agent_id="did:creator"
        )
        # A WORKING task whose submission wake ran, and whose recovery wake failed.
        ran = await claimed_submission_wake(manager, dispatcher, consumer, tasks["working"])
        assert await dispatcher.ack_durable_delivery(
            consumer_id=consumer, delivery_id=ran.delivery_id, lease_token=ran.lease_token,
        )
        await manager.update_status(
            tasks["working"].id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        working = await snapshot(manager, tasks["working"])
        working_identity = submitted_input_identity(
            working.task.id, working.input_revision,
            working_revision=working.lifecycle_revision,
        )
        await dispatcher.dispatch_signal(
            copy.deepcopy(wake(working, manager)), source_event_id=working_identity,
        )
        recovery = await dispatcher.get_durable_delivery_for_source_event(
            consumer_id=consumer, source=SOURCE_NAME, source_event_id=working_identity,
        )
        recovery = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=consumer, event_id=recovery.event_id,
            executor_id=dispatcher._durable_delivery_owner,
        )
        recovery = await earlier_terminal_row(
            dispatcher, consumer, recovery, error="durable consumer deactivated",
        )

        owed = await dispatcher.owed_cognition_wake_settlements()

        assert sorted(d.delivery_id for d in owed) == sorted(
            [failed.delivery_id, refused.delivery_id, recovery.delivery_id]
        )
        assert {
            dispatcher._undelivered_wake_disposition(d) for d in owed
        } == {
            (False, "tool_effect_reconciliation_required"),
            (True, "stopped"),
            (False, "wake failed"),
        }
    finally:
        await close_input_dispatcher(agent, dispatcher)


def _stopped(delivery):
    from kestrel_sdk.signals import Status

    from kestrel_sovereign.signals.dispatcher import _StoppedCognitionResult

    return _StoppedCognitionResult(
        signal_id=delivery.event.event_id,
        status=Status.COALESCED,
        mode=delivery.event.mode,
        duration_ms=0,
        error="stop_acknowledged: operator Stop",
    )


def _refused(delivery):
    from kestrel_sdk.signals import SignalResult, Status

    return SignalResult(
        signal_id=delivery.event.event_id,
        status=Status.DROPPED_VALIDATION,
        mode=delivery.event.mode,
        duration_ms=0,
        error="The authoritative A2A task is not eligible for cognition",
    )


@pytest.mark.parametrize("result", [_stopped, _refused], ids=["stopped", "refused"])
@pytest.mark.parametrize("caller", ["settlement", "retained_cleanup"])
async def test_a_stale_owner_leaves_the_task_its_successor_runs_alone(
    manager, result, caller
):
    """wo-4001-5 self-review: a wake's lease expired with its owner stale, and
    a successor reclaimed it. The stale owner's late Stop or refusal then
    reaches its settlement. Its terminal CAS loses to the successor's lease,
    so it may not touch the task: the input revision cannot tell the two
    owners apart. Settlement runs only from a committed terminal row, so the
    task is left for the successor, through either route the result arrives
    by and through the settlement pass that follows."""
    from datetime import UTC, datetime, timedelta

    from kestrel_sovereign.signals.durable import DurableSignalStore

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        stale = await claimed_submission_wake(manager, dispatcher, consumer, task)
        store = dispatcher._durable_store
        past = store.to_timestamp_param(datetime.now(UTC) - timedelta(hours=1))
        await store.backend.execute(
            f"UPDATE {DurableSignalStore.DELIVERIES} SET lease_expires_at = ? "
            "WHERE delivery_id = ?",
            (past, stale.delivery_id),
        )
        await store.backend.execute(
            f"UPDATE {DurableSignalStore.RUNTIME_OWNERS} SET heartbeat_at = ? "
            "WHERE agent_id = ? AND owner_id = ?",
            (past, agent.did, dispatcher._durable_delivery_owner),
        )
        successor = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=consumer,
            event_id=stale.event_id,
            executor_id="worker:successor",
        )
        assert successor is not None and successor.lease_token != stale.lease_token

        if caller == "settlement":
            assert not await dispatcher._settle_cognition_non_delivery(
                stale, result(stale), consumer_id=consumer, retry_delay=timedelta()
            )
        else:
            await dispatcher._release_retained_durable_cognition_task(
                stale, result=result(stale)
            )
        await dispatcher.settle_owed_cognition_sources()

        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )
        (row,) = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        assert (row.status, row.lease_owner, row.lease_token) == (
            "leased", "worker:successor", successor.lease_token,
        )
    finally:
        await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("fault", ["storage", "refusal", "rolled_back_write"])
async def test_every_ordinary_settlement_failure_leaves_the_task_owed(manager, fault):
    """#4001 wo-4001-4: no failure type decides a retry. A storage fault, an
    error the store has no class for, or one raised after the task write
    already ran inside the settlement transaction: each leaves the task
    owed, with no control debt, and the next pass finds it from durable
    state and settles it. The task write and the row's settlement are one
    transaction, so a failure after the task write leaves neither. Nothing
    in the process remembers the row in between."""
    from kestrel_sdk.storage.database.interface import QueryError

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        await failed_submission_wake(manager, dispatcher, consumer, task)

        async def write_then_fail(save, *args, **kwargs):
            outcome = await save(*args, **kwargs)
            assert outcome.kind == "ended"
            raise QueryError("connection reset after the task write")

        injected = {
            "storage": QueryError("database is locked"),
            "refusal": RuntimeError("an error the store has no class for"),
            "rolled_back_write": write_then_fail,
        }[fault]
        faults = [injected]
        attempts = fault_task_writes(manager, faults)

        assert await dispatcher._settle_owed_cognition_sources() is None
        assert faults == [] and len(attempts) == 1
        assert dispatcher.unresolved_cognition_control == ()
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )
        assert len(await dispatcher.owed_cognition_wake_settlements()) == 1

        assert await dispatcher._settle_owed_cognition_sources() is None

        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
        assert await dispatcher.owed_cognition_wake_settlements() == []
        assert dispatcher.unresolved_cognition_control == ()
    finally:
        await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("arrives_as", ["error", "cancellation"])
async def test_a_settlement_whose_commit_is_unknown_stops_the_pass_as_control(
    manager, arrives_as
):
    """#4001: a settlement write whose commit outcome is unknown, even one
    arriving as a cancellation, is native control, not an ordinary failure.
    The pass retains it as that delivery's settlement control and settles
    nothing beside it. The control guards no lease, so the owner heartbeat
    stays on its ordinary branch: its pass retries the fenced write, which
    here commits, ends the control, and settles the rest."""
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        execution_commit_outcome,
    )

    tasks = [
        await manager.create_task(request(), creator_agent_id="did:creator")
        for _ in range(2)
    ]
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        failed = [
            await failed_submission_wake(manager, dispatcher, consumer, task)
            for task in tasks
        ]
        uncertain = (
            ExecutionCommitOutcomeError("unknown")
            if arrives_as == "error"
            else _cancellation_with_unknown_commit()
        )
        faults = [uncertain]
        attempts = fault_task_writes(manager, faults)

        # Either way it comes back as that delivery's control, never raised.
        control = await dispatcher._settle_owed_cognition_sources()
        assert execution_commit_outcome(control.original) == "unknown"
        assert dispatcher._outcome_log_control_error is None
        assert faults == [] and len(attempts) == 1
        assert not dispatcher._retained_cognition_control_debt
        (debt,) = dispatcher._source_settlement_control.values()
        assert debt[0].delivery_id in {row.delivery_id for row in failed}
        assert not dispatcher._runtime_owner_cleanup_required()
        for task in tasks:
            assert (await snapshot(manager, task)).task.status.state == (
                TaskState.TASK_STATE_SUBMITTED
            )

        await heartbeat(dispatcher)

        assert len(attempts) == 3
        assert dispatcher.unresolved_cognition_control == ()
        for task in tasks:
            assert (await snapshot(manager, task)).task.status.state == (
                TaskState.TASK_STATE_FAILED
            )
    finally:
        await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("origin", ["live", "boot"])
async def test_an_unknown_settlement_commit_refuses_retirement_until_a_retry_returns(
    manager, origin
):
    """wo-4001-5 self-review: the write that settles a failed wake's task ends
    with its commit outcome unknown; the owner-retirement drain's retry then
    meets an ordinary database error. That error proves nothing about the
    earlier write, so the control stands: shutdown fails and keeps the owner,
    never reporting a deferred settlement as done. The next attempt settles
    the task. The control arises in the running process, after this
    dispatcher failed the row itself (``live``), or at boot's first pass over
    a row an earlier process failed (``boot``). The failed row's CAS is
    written once, by the process that failed it, and never again."""
    from kestrel_sdk.storage.database.interface import QueryError

    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        ToolAuditUnconfirmedError,
        execution_commit_outcome,
    )

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        failed_row_writes = count_failed_row_writes(dispatcher)
        faults = []
        attempts = fault_task_writes(manager, faults)
        if origin == "live":
            claimed = await claimed_submission_wake(manager, dispatcher, consumer, task)
            await dispatcher._terminalize_failed_cognition(
                claimed, ToolAuditUnconfirmedError("audit failed")
            )
            # The failed row commits; nothing settles before it does.
            assert failed_row_writes == [claimed.delivery_id]
            assert attempts == []
            delivery_id = claimed.delivery_id
            del claimed
            faults.append(ExecutionCommitOutcomeError("unknown"))
            control = await dispatcher._settle_owed_cognition_sources()
            assert execution_commit_outcome(control.original) == "unknown"
            del control
            expected_row_writes = [delivery_id]
        else:
            delivery_id = (
                await failed_submission_wake(manager, dispatcher, consumer, task)
            ).delivery_id
            faults.append(ExecutionCommitOutcomeError("unknown"))
            with pytest.raises(BaseException) as raised:
                await agent.reconcile_a2a_cognition_wakes()
            assert execution_commit_outcome(raised.value) == "unknown"
            del raised
            expected_row_writes = []
        assert faults == []
        assert dispatcher.unresolved_cognition_control == (delivery_id,)
        assert list(dispatcher._source_settlement_control) == [delivery_id]
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )

        faults.append(QueryError("database is locked"))
        with pytest.raises(BaseException) as refused:
            await close_input_dispatcher(agent, dispatcher)
        assert faults == []
        assert execution_commit_outcome(refused.value) == "unknown"
        del refused
        assert dispatcher.unresolved_cognition_control == (delivery_id,)
        assert list(dispatcher._source_settlement_control) == [delivery_id]
        assert dispatcher._durable_runtime_owner_registered
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )

        closed = True
        await close_input_dispatcher(agent, dispatcher)

        assert dispatcher.unresolved_cognition_control == ()
        assert not dispatcher._durable_runtime_owner_registered
        assert failed_row_writes == expected_row_writes
        settled = await snapshot(manager, task)
        assert settled.task.status.state == TaskState.TASK_STATE_FAILED
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


async def _owner_row(dispatcher):
    """This owner's ``(heartbeat_at, stopped_at)`` as stored."""
    from kestrel_sovereign.signals.durable import DurableSignalStore

    store = dispatcher._durable_store
    row = await store.backend.fetch_one(
        f"SELECT heartbeat_at, stopped_at FROM {DurableSignalStore.RUNTIME_OWNERS} "
        "WHERE agent_id = ? AND owner_id = ?",
        (dispatcher._agent.did, dispatcher._durable_delivery_owner),
    )
    return tuple(
        None if value is None else store.from_timestamp_field(value) for value in row
    )


@pytest.mark.parametrize("retry", ["unknown", "ordinary"])
async def test_an_owner_whose_retirement_refuses_keeps_heartbeating_until_it_retires(
    manager, monkeypatch, retry
):
    """Self-review of wo-4001-8: a settlement's commit outcome is unknown, so
    shutdown refuses to retire the owner. Its retry of that settlement meets
    the unknown outcome again (``unknown``: refused) or an ordinary database
    error (``ordinary``: failed). Either way the owner is not retired, and it
    still leases work a peer must not take. Before the fix its heartbeat
    stopped as soon as shutdown began, since settlement control alone no
    longer kept it running; once the stale threshold passed, a peer's
    recovery reclaimed the lease. Now the real timer keeps renewing an
    unretired owner, admitting no work, and stops only once it retires
    (#4001)."""
    from kestrel_sdk.storage.database.interface import QueryError

    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
    from kestrel_sovereign.signals import SignalDispatcher

    stale_after = timedelta(seconds=1.2)
    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__, "runtime_owner_stale_after", stale_after
    )
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        # Every settlement write raises the current fault until it is cleared,
        # including one a real heartbeat runs before this test's own pass.
        fault = {"error": ExecutionCommitOutcomeError("unknown")}
        attempts = []
        save = manager.task_store.apply_undelivered_wake_transition

        async def faulted(*args, **kwargs):
            attempts.append(fault["error"])
            if fault["error"] is not None:
                raise fault["error"]
            return await save(*args, **kwargs)

        monkeypatch.setattr(
            manager.task_store, "apply_undelivered_wake_transition", faulted
        )
        failed, held = await _held_workflow_lease(manager, dispatcher, consumer, task)
        await heartbeat(dispatcher)
        assert dispatcher.unresolved_cognition_control == (failed.delivery_id,)

        if retry == "ordinary":
            fault["error"] = QueryError("database is locked")
        before_shutdown = len(attempts)
        with pytest.raises(BaseException):
            await close_input_dispatcher(agent, dispatcher)
        assert len(attempts) > before_shutdown
        after_refusal = len(attempts)
        assert dispatcher._durable_shutdown
        assert dispatcher.unresolved_cognition_control == (failed.delivery_id,)
        assert dispatcher._durable_runtime_owner_registered

        await _assert_unretired_owner_stays_live(dispatcher, held, stale_after)
        # Shutdown admits nothing: no settlement pass ran while it stood.
        assert len(attempts) == after_refusal
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )

        fault["error"] = None
        closed = True
        await close_input_dispatcher(agent, dispatcher)
        assert dispatcher.unresolved_cognition_control == ()
        assert not dispatcher._durable_runtime_owner_registered
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
        await _assert_retired_owner_stays_retired(dispatcher, stale_after)
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("mark", ["failed", "unacknowledged"])
async def test_an_owner_whose_stop_mark_fails_keeps_heartbeating_until_it_retires(
    manager, monkeypatch, mark
):
    """Self-review of wo-4001-8: shutdown stops the owner heartbeat, then
    marks the owner stopped. If that mark fails (``failed``), the owner is not
    retired and still leases work, so its heartbeat resumes until a retry
    retires it, rather than leaving a peer free to reclaim the lease. If the
    mark committed but its acknowledgement was lost (``unacknowledged``), the
    owner is retired: the resumed heartbeat must not renew or revive it. No
    heartbeat starts while the owner is being marked stopped (#4001)."""
    from kestrel_sdk.storage.database.interface import QueryError

    from kestrel_sovereign.signals import SignalDispatcher

    stale_after = timedelta(seconds=1.2)
    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__, "runtime_owner_stale_after", stale_after
    )
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        _, held = await _held_workflow_lease(manager, dispatcher, consumer, task)
        store = dispatcher._durable_store
        release = store.release_initial_reservations
        marks = []

        async def faulted_stop_mark(**kwargs):
            if not kwargs.get("mark_owner_stopped", True):
                return await release(**kwargs)
            # A timer firing now finds the owner being retired.
            dispatcher._start_runtime_owner_heartbeat()
            marks.append(dispatcher._runtime_owner_heartbeat_task)
            if len(marks) > 1:
                return await release(**kwargs)
            if mark == "unacknowledged":
                await release(**kwargs)
            raise QueryError("database is locked")

        monkeypatch.setattr(store, "release_initial_reservations", faulted_stop_mark)
        with pytest.raises(BaseException):
            await close_input_dispatcher(agent, dispatcher)
        assert marks == [None]
        assert dispatcher._durable_runtime_owner_registered

        if mark == "failed":
            await _assert_unretired_owner_stays_live(dispatcher, held, stale_after)
        else:
            stopped = await _owner_row(dispatcher)
            assert stopped[1] is not None
            await asyncio.sleep(stale_after.total_seconds() * 2)
            assert await _owner_row(dispatcher) == stopped, (
                "a heartbeat renewed an owner whose retirement committed"
            )

        closed = True
        await close_input_dispatcher(agent, dispatcher)
        assert marks == [None, None]
        assert not dispatcher._durable_runtime_owner_registered
        await _assert_retired_owner_stays_retired(dispatcher, stale_after)
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


async def _held_workflow_lease(manager, dispatcher, consumer, task):
    """A failed submission wake, and a live lease this owner holds on its
    event that a peer would take were the owner dead."""
    from kestrel_sovereign.signals import DurableConsumerRegistration

    workflow = "workflows:wait:4001"
    await dispatcher.register_durable_consumer(
        DurableConsumerRegistration(
            consumer_id=workflow, source="a2a.task_submitted",
            agent_id=dispatcher._agent.did, max_attempts=0, lease_seconds=3600,
        )
    )
    failed = await failed_submission_wake(manager, dispatcher, consumer, task)
    held = await dispatcher.claim_durable_delivery_for_event(
        consumer_id=workflow, event_id=failed.event_id,
        executor_id=dispatcher._durable_delivery_owner,
    )
    assert held is not None and held.status == "leased"
    return failed, held


async def _assert_unretired_owner_stays_live(dispatcher, held, stale_after):
    """Past the stale threshold the owner was renewed and kept its lease."""
    refused = await _owner_row(dispatcher)
    assert refused[1] is None
    await asyncio.sleep(stale_after.total_seconds() * 2)
    renewed = await _owner_row(dispatcher)
    assert renewed[1] is None
    assert renewed[0] > refused[0], "the unretired owner stopped heartbeating"
    store = dispatcher._durable_store
    recovered = await store.recover_abandoned_leases(
        agent_id=dispatcher._agent.did,
        recovering_owner_id="dispatcher:peer-4001",
        stale_before=store.now_utc() - stale_after,
    )
    assert recovered == 0, "a peer reclaimed a lease the unretired owner holds"
    (lease,) = await store.list_deliveries(
        agent_id=dispatcher._agent.did, consumer_id=held.consumer_id
    )
    assert (lease.status, lease.lease_owner, lease.lease_token) == (
        "leased", dispatcher._durable_delivery_owner, held.lease_token,
    )


async def _assert_retired_owner_stays_retired(dispatcher, stale_after):
    """Once retired, nothing renews the owner or revives it."""
    retired = await _owner_row(dispatcher)
    assert retired[1] is not None
    assert dispatcher._runtime_owner_heartbeat_timer is None
    assert dispatcher._runtime_owner_heartbeat_task is None
    await asyncio.sleep(stale_after.total_seconds())
    assert await _owner_row(dispatcher) == retired


def stall_settlement_reads(dispatcher, monkeypatch):
    """Hold every settlement pass at its first page read until released.

    The read runs outside any transaction, as a slow read of a large backlog
    would; the settlement transactions after it are short.
    """
    store = dispatcher._durable_store
    read = store.list_unsettled_terminal_deliveries_page
    entered, release = asyncio.Event(), asyncio.Event()

    async def stalled_read(**kwargs):
        entered.set()
        await release.wait()
        return await read(**kwargs)

    monkeypatch.setattr(store, "list_unsettled_terminal_deliveries_page", stalled_read)
    return entered, release


async def test_a_stalled_settlement_pass_never_holds_back_owner_renewal(
    manager, monkeypatch
):
    """Self-review of wo-4001-9: the owner heartbeat awaited its whole
    settlement pass before it scheduled the next renewal. A pass whose task
    read stalled past the stale threshold left the owner unrenewed, and a
    peer's recovery reclaimed a lease this live owner still held. The pass
    now runs in a task of its own: the real timer renews the owner on its
    cadence while the pass stalls, and the pass still settles its row once
    the read returns (#4001)."""
    from kestrel_sovereign.signals import SignalDispatcher

    stale_after = timedelta(seconds=0.9)
    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__, "runtime_owner_stale_after", stale_after
    )
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    entered, release = stall_settlement_reads(dispatcher, monkeypatch)
    try:
        failed, held = await _held_workflow_lease(manager, dispatcher, consumer, task)
        dispatcher._request_owed_settlement_pass(consumer_id=consumer, status="failed")
        await asyncio.wait_for(entered.wait(), 5)

        await _assert_unretired_owner_stays_live(dispatcher, held, stale_after)
        # It was renewed while the pass was still stalled.
        stalled = dispatcher._owed_settlement_task
        assert stalled is not None and not stalled.done()

        release.set()
        await dispatcher._join_owed_settlement_pass()
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
        assert await dispatcher.unsettled_cognition_wakes() == []
    finally:
        release.set()
        await close_input_dispatcher(agent, dispatcher)


async def test_a_row_that_ends_during_a_pass_gets_a_prompt_pass_of_its_own(
    manager, monkeypatch
):
    """Self-review of wo-4001-9: with the pass in a task of its own, a wake
    that ends while one runs must not wait for the next ordinary beat. The
    running pass read its page before that row committed, so as it ends it
    asks for another, and the row's task is settled promptly (#4001)."""
    from kestrel_sovereign.signals import SignalDispatcher

    # An ordinary beat is 10 s away; only a prompt pass settles in time.
    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__, "runtime_owner_stale_after",
        timedelta(seconds=30),
    )
    first = await manager.create_task(request(), creator_agent_id="did:creator")
    second = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    entered, release = stall_settlement_reads(dispatcher, monkeypatch)
    loop = asyncio.get_running_loop()
    try:
        async def ended_by_this_dispatcher(task):
            claimed = await claimed_submission_wake(manager, dispatcher, consumer, task)
            row = await dispatcher.nack_durable_delivery(
                consumer_id=consumer, delivery_id=claimed.delivery_id,
                lease_token=claimed.lease_token,
                error="execution_control_unresolved: tool_effect_reconciliation_required",
                terminal=True,
            )
            assert row.status == "failed"

        await ended_by_this_dispatcher(first)
        await asyncio.wait_for(entered.wait(), 5)
        stalled = dispatcher._owed_settlement_task
        assert stalled is not None and not stalled.done()
        await ended_by_this_dispatcher(second)

        release.set()
        deadline = loop.time() + 2.0
        while (await snapshot(manager, second)).task.status.state != (
            TaskState.TASK_STATE_FAILED
        ):
            assert loop.time() < deadline, "a row that ended mid-pass waited for the next beat"
            await asyncio.sleep(0.02)
        assert (await snapshot(manager, first)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
    finally:
        release.set()
        await close_input_dispatcher(agent, dispatcher)


_RENEWAL_OBSTACLES = (
    "refused_shutdown",
    "failed_shutdown",
    "stalled_settlement",
    "stalled_maintenance",
    "native_denial",
    "lost_settlement_ack",
    "retained_control",
)


@pytest.mark.parametrize("obstacle", _RENEWAL_OBSTACLES)
async def test_an_unretired_owner_is_renewed_whatever_else_stands(
    manager, monkeypatch, obstacle
):
    """Self-review of wo-4001-8 and of both wo-4001-9 attempts: each round
    found another way the owner's renewal stopped while the owner still
    leased work. A shutdown that refused (an unknown settlement commit), one
    that failed (an ordinary error on its retry), a settlement pass or a
    maintenance sweep that stalled, native denial of the agent's own
    execution scope, a lost acknowledgement of a settlement's commit, and a
    plain retained control error all held
    renewal back, because renewal shared a path with work that can fail or
    wait. Renewal is now one fixed write, in a task of its own every tick,
    outside durable admission and on no execution scope the agent's work can
    fence. Whatever stands, the real timer renews the unretired owner past
    the stale threshold and a peer's recovery reclaims nothing it holds
    (#4001)."""
    from kestrel_sdk.storage.database.interface import QueryError

    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        ExecutionCustody,
        ProcessRuntimeExecutionFence,
    )
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.signals import SignalDispatcher

    stale_after = timedelta(seconds=1.2)
    monkeypatch.setitem(
        SignalDispatcher.__init__.__kwdefaults__, "runtime_owner_stale_after", stale_after
    )
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    released = asyncio.Event()
    fault = {"error": None}
    closed = False
    try:
        if obstacle in ("native_denial", "lost_settlement_ack"):
            # The agent's own execution scope, which a booted agent binds on
            # every resident timer and so on every heartbeat it starts.
            agent._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(agent.did))
            monkeypatch.setattr(
                type(agent), "_runtime_owner_context",
                KestrelAgent._runtime_owner_context, raising=False,
            )
            if dispatcher._runtime_owner_heartbeat_timer is not None:
                dispatcher._runtime_owner_heartbeat_timer.cancel()
                dispatcher._runtime_owner_heartbeat_timer = None
            dispatcher._schedule_runtime_owner_heartbeat()

        save = manager.task_store.apply_undelivered_wake_transition

        async def faulted_save(*args, **kwargs):
            if fault["error"] is not None:
                raise fault["error"]
            return await save(*args, **kwargs)

        monkeypatch.setattr(
            manager.task_store, "apply_undelivered_wake_transition", faulted_save
        )
        failed, held = await _held_workflow_lease(manager, dispatcher, consumer, task)

        if obstacle in ("refused_shutdown", "failed_shutdown"):
            fault["error"] = ExecutionCommitOutcomeError("unknown")
            await heartbeat(dispatcher)
            assert dispatcher.unresolved_cognition_control == (failed.delivery_id,)
            if obstacle == "failed_shutdown":
                fault["error"] = QueryError("database is locked")
            with pytest.raises(BaseException):
                await close_input_dispatcher(agent, dispatcher)
            assert dispatcher._durable_shutdown
            assert dispatcher._durable_runtime_owner_registered
        elif obstacle == "stalled_settlement":
            entered, released = stall_settlement_reads(dispatcher, monkeypatch)
            dispatcher._request_owed_settlement_pass(consumer_id=consumer, status="failed")
            await asyncio.wait_for(entered.wait(), 5)
        elif obstacle == "stalled_maintenance":
            entered = asyncio.Event()

            async def stalled_sweep():
                entered.set()
                await released.wait()
                return 0

            monkeypatch.setattr(dispatcher, "_recover_abandoned_leases", stalled_sweep)
            await asyncio.wait_for(entered.wait(), 5)
        elif obstacle == "native_denial":
            # What either backend does to every original scope when a COMMIT
            # acknowledgement is lost.
            agent._execution_custody.preserve_commit_uncertainty("unknown")
        elif obstacle == "lost_settlement_ack":
            store = dispatcher._durable_store
            settle = store.settle_terminal_delivery

            lost = []

            async def lost_ack(**kwargs):
                outcome = await settle(**kwargs)
                if lost:
                    return outcome
                lost.append(outcome)
                agent._execution_custody.preserve_commit_uncertainty("unknown")
                raise ExecutionCommitOutcomeError("unknown")

            monkeypatch.setattr(store, "settle_terminal_delivery", lost_ack)
            await heartbeat(dispatcher)
            # The settlement's own control, retained for its row, never the
            # dispatcher's writer latch.
            assert dispatcher.unresolved_cognition_control == (failed.delivery_id,)
            assert dispatcher._outcome_log_control_error is None
        else:
            dispatcher._outcome_log_control_error = ExecutionCommitOutcomeError("unknown")

        await _assert_unretired_owner_stays_live(dispatcher, held, stale_after)

        # The obstacle still stands: renewal went on beside it.
        if obstacle in ("refused_shutdown", "failed_shutdown"):
            assert dispatcher._durable_runtime_owner_registered
        elif obstacle == "stalled_settlement":
            assert not dispatcher._owed_settlement_task.done()
        elif obstacle == "stalled_maintenance":
            assert not dispatcher._runtime_owner_heartbeat_task.done()
        elif obstacle in ("native_denial", "lost_settlement_ack"):
            with pytest.raises(ExecutionCommitOutcomeError):
                agent._execution_custody.require_work()
        else:
            assert dispatcher._outcome_log_control_error is not None
    finally:
        released.set()
        fault["error"] = None
        dispatcher._outcome_log_control_error = None
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize(
    "race",
    ["earlier_identity_live", "earlier_identity_fails", "original_admitted_meanwhile"],
)
async def test_admission_refuses_a_wake_for_an_input_another_wake_carries(manager, race):
    """Self-review of wo-4001-8 and of both wo-4001-9 attempts: boot decided
    from a snapshot whether to mint a recovery wake for a WORKING task's
    input, and each round found the snapshot stale one way more. It checked
    only the original identity, while an earlier recovery wake still ran
    (``earlier_identity_live``). An earlier recovery wake a peer failed after
    boot read the ledger, and boot minted a replacement the next settlement
    pass then ended (``earlier_identity_fails``). A peer admitted the original
    wake after boot read, and boot minted a second leased wake for the same
    input (``original_admitted_meanwhile``). Admission now decides, in the
    store, in the transaction that admits the wake: it refuses a wake while
    another wake of the same input has not run, whatever its identity. Here
    two dispatchers share one store, and the peer acts while boot is between
    its reads and its admission (#4001)."""
    import copy

    from kestrel_sovereign.signals import DurableAdmissionDisposition
    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    peer_agent, peer, consumer = await input_dispatcher(manager)
    agent, dispatcher, _ = await input_dispatcher(manager)
    held = None
    try:
        if race != "original_admitted_meanwhile":
            ran = await claimed_submission_wake(manager, peer, consumer, task)
            assert await peer.ack_durable_delivery(
                consumer_id=consumer, delivery_id=ran.delivery_id,
                lease_token=ran.lease_token,
            )
            await manager.update_status(
                task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
            )
            earlier = await snapshot(manager, task)
            identity = submitted_input_identity(
                task.id, 0, working_revision=earlier.lifecycle_revision
            )
            await peer.dispatch_signal(
                copy.deepcopy(wake(earlier, manager)), source_event_id=identity
            )
            held = await peer.get_durable_delivery_for_source_event(
                consumer_id=consumer, source=SOURCE_NAME, source_event_id=identity
            )
            held = await peer.claim_durable_delivery_for_event(
                consumer_id=consumer, event_id=held.event_id,
                executor_id=peer._durable_delivery_owner,
            )
            # The same input, now at a later lifecycle revision.
            await manager.update_status(
                task.id, TaskState.TASK_STATE_SUBMITTED, recipient_agent_id=manager.did
            )
            await manager.update_status(
                task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
            )
        read = manager.list_cognition_wake_candidates
        read_revisions = []

        async def peer_acts_as_boot_reads(**kwargs):
            nonlocal held
            if race == "original_admitted_meanwhile":
                held = await claimed_submission_wake(manager, peer, consumer, task)
                await manager.update_status(
                    task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
                )
            candidates = await read(**kwargs)
            read_revisions.extend(
                c.lifecycle_revision for c in candidates if c.task.id == task.id
            )
            if race == "earlier_identity_fails":
                await earlier_terminal_row(
                    peer, consumer, held,
                    error="execution_control_unresolved: tool_effect_reconciliation_required",
                )
            return candidates

        manager.list_cognition_wake_candidates = peer_acts_as_boot_reads
        admissions = []
        enqueue = dispatcher.enqueue_durable_cognition

        async def recorded_enqueue(signal, **kwargs):
            handle = await enqueue(signal, **kwargs)
            admissions.append(
                (kwargs["source_event_id"], (await handle.wait_for_durable_admission()).disposition)
            )
            return handle

        dispatcher.enqueue_durable_cognition = recorded_enqueue
        await agent.reconcile_a2a_cognition_wakes()
        manager.list_cognition_wake_candidates = read

        rows = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        summary = [(d.event.source_event_id, d.status, d.lease_owner) for d in rows]
        assert [d.delivery_id for d in rows if d.status != "acknowledged"] == [
            held.delivery_id
        ], summary
        (revision,) = read_revisions
        assert admissions == [
            (
                submitted_input_identity(task.id, 0, working_revision=revision),
                DurableAdmissionDisposition.DUPLICATE,
            )
        ], admissions

        await heartbeat(dispatcher)
        # A live wake keeps its work; a failed one, the only wake of its
        # input, leaves it to the settlement pass, which ends it.
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED if race == "earlier_identity_fails"
            else TaskState.TASK_STATE_WORKING
        ), summary
    finally:
        await close_input_dispatcher(agent, dispatcher)
        await close_input_dispatcher(peer_agent, peer)


async def _claimed_legacy_wake(manager, dispatcher, consumer, task, *, recovery, claim=True):
    """Admit and claim a submission wake as a release before exclusive keys did.

    Its delivery holds no key, and nothing checked whether another wake of
    the same input was live: an earlier boot minted a recovery wake at the
    task's current lifecycle revision beside one that had not run. With
    ``claim=False`` the admitted delivery is returned still pending.
    """
    import copy

    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    current = await snapshot(manager, task)
    identity_key = (
        submitted_input_identity(
            task.id, current.input_revision, working_revision=current.lifecycle_revision
        )
        if recovery
        else submitted_input_identity(task.id, current.input_revision)
    )
    dispatcher._agent.durable_wake_exclusive_key = lambda **_: None
    try:
        await dispatcher.dispatch_signal(
            copy.deepcopy(wake(current, manager)), source_event_id=identity_key
        )
    finally:
        del dispatcher._agent.durable_wake_exclusive_key
    admitted = await dispatcher.get_durable_delivery_for_source_event(
        consumer_id=consumer, source=SOURCE_NAME, source_event_id=identity_key
    )
    if not claim:
        return admitted
    return await dispatcher.claim_durable_delivery_for_event(
        consumer_id=consumer,
        event_id=admitted.event_id,
        executor_id=dispatcher._durable_delivery_owner,
    )


async def _next_working_revision(manager, task):
    """Move a WORKING task to a later lifecycle revision of the same input."""
    await manager.update_status(
        task.id, TaskState.TASK_STATE_SUBMITTED, recipient_agent_id=manager.did
    )
    await manager.update_status(
        task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
    )


async def _delivery_keys(dispatcher, delivery_id):
    """A row's ``(exclusive_key, exclusive_conflict_key, settlement_disposition,
    settled_at is set)`` as stored."""
    from kestrel_sovereign.signals.durable import DurableSignalStore

    row = await dispatcher._durable_store.backend.fetch_one(
        "SELECT exclusive_key, exclusive_conflict_key, settlement_disposition, "
        f"settled_at FROM {DurableSignalStore.DELIVERIES} WHERE delivery_id = ?",
        (delivery_id,),
    )
    return row[0], row[1], row[2], row[3] is not None


@pytest.mark.parametrize("shape", ["two_failed", "live_beside_failed"])
async def test_legacy_duplicate_wakes_are_held_not_co_keyed(manager, shape):
    """Meridian's ruling on wo-4001-10: wakes admitted before exclusive keys
    existed can share one input, and boot gave each of them the same key
    from a snapshot. Every later path then decided from that snapshot, and
    three rounds found a door each (a peer's release, a lost release ACK, an
    ACK before the release). Key backfill now resolves each group in one
    store transaction that reads its rows again. Two failed wakes for one
    input (``two_failed``) are held: neither is keyed, a new wake for that
    input is refused, and no automatic path writes the task. One live wake
    beside a failed one (``live_beside_failed``) keys the live one and
    records the failed one carried by it, touching no task: the live wake
    alone carries the input, so once it has run the input is free and boot
    may wake the WORKING task again (#4001)."""
    from kestrel_sovereign.signals import DurableAdmissionDisposition

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    peer_agent, peer, consumer = await input_dispatcher(manager)
    agent, dispatcher, _ = await input_dispatcher(manager)
    reconciliation = "execution_control_unresolved: tool_effect_reconciliation_required"
    try:
        original = await _claimed_legacy_wake(manager, peer, consumer, task, recovery=False)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        recovery = await _claimed_legacy_wake(manager, peer, consumer, task, recovery=True)
        await earlier_terminal_row(peer, consumer, recovery, error=reconciliation)
        if shape == "two_failed":
            await earlier_terminal_row(peer, consumer, original, error=reconciliation)
        # A later lifecycle revision: boot's recovery identity is new, so only
        # the input's key, or its conflict, can refuse it.
        await _next_working_revision(manager, task)
        before = await snapshot(manager, task)
        key = submitted_input_identity(task.id, 0)
        for row in (original, recovery):
            assert await _delivery_keys(dispatcher, row.delivery_id) == (
                None, None, None, False,
            )

        admissions = []
        enqueue = dispatcher.enqueue_durable_cognition

        async def recorded_enqueue(signal, **kwargs):
            handle = await enqueue(signal, **kwargs)
            admissions.append((await handle.wait_for_durable_admission()).disposition)
            return handle

        dispatcher.enqueue_durable_cognition = recorded_enqueue
        await agent.reconcile_a2a_cognition_wakes()
        await heartbeat(dispatcher)

        after = await snapshot(manager, task)
        assert after.task.status.state == TaskState.TASK_STATE_WORKING
        assert after.lifecycle_revision == before.lifecycle_revision, (
            "an automatic path wrote the task of a legacy group"
        )
        assert admissions == [DurableAdmissionDisposition.DUPLICATE], admissions
        if shape == "two_failed":
            for row in (original, recovery):
                assert await _delivery_keys(dispatcher, row.delivery_id) == (
                    None, key, "held", False,
                )
            assert await dispatcher.held_cognition_wake_groups() == 1
            assert await dispatcher.unsettled_cognition_wakes() == []
            return
        assert await _delivery_keys(dispatcher, original.delivery_id) == (
            key, None, None, False,
        )
        assert await _delivery_keys(dispatcher, recovery.delivery_id) == (
            None, None, "carried", True,
        )
        assert await dispatcher.held_cognition_wake_groups() == 0
        assert await dispatcher.unsettled_cognition_wakes() == []

        # The live wake runs; nothing it carried comes back owed, and the
        # input's key is held by nothing, so boot wakes the input again.
        assert await peer.ack_durable_delivery(
            consumer_id=consumer, delivery_id=original.delivery_id,
            lease_token=original.lease_token,
        )
        await heartbeat(dispatcher)
        assert (await snapshot(manager, task)).task.status.state == TaskState.TASK_STATE_WORKING
        admissions.clear()
        await agent.reconcile_a2a_cognition_wakes()
        assert admissions == [DurableAdmissionDisposition.COMMITTED], admissions
    finally:
        await close_input_dispatcher(agent, dispatcher)
        await close_input_dispatcher(peer_agent, peer)


async def test_a_live_legacy_wake_beside_a_failed_one_still_owes_its_own_input(manager):
    """The failed legacy row is carried by the live one, not released for
    good: if the live wake then ends undelivered too, its own row owes the
    input, and the pass ends the task so its sender sees it fail (#4001)."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    peer_agent, peer, consumer = await input_dispatcher(manager)
    agent, dispatcher, _ = await input_dispatcher(manager)
    reconciliation = "execution_control_unresolved: tool_effect_reconciliation_required"
    try:
        original = await _claimed_legacy_wake(manager, peer, consumer, task, recovery=False)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        recovery = await _claimed_legacy_wake(manager, peer, consumer, task, recovery=True)
        await earlier_terminal_row(peer, consumer, recovery, error=reconciliation)
        await agent.reconcile_a2a_cognition_wakes()
        await heartbeat(dispatcher)
        assert (await snapshot(manager, task)).task.status.state == TaskState.TASK_STATE_WORKING

        await earlier_terminal_row(peer, consumer, original, error=reconciliation)
        await heartbeat(dispatcher)
        seen = (await snapshot(manager, task)).task
        assert seen.status.state == TaskState.TASK_STATE_FAILED
        assert "tool_effect_reconciliation_required" in seen.status.message.parts[0].text
        # A settled row keeps its key, so no later wake carries this input.
        assert await _delivery_keys(dispatcher, original.delivery_id) == (
            submitted_input_identity(task.id, 0), None, "ended", True,
        )
        assert await dispatcher.unsettled_cognition_wakes() == []
    finally:
        await close_input_dispatcher(agent, dispatcher)
        await close_input_dispatcher(peer_agent, peer)


async def _set_delivery_row(dispatcher, delivery_id, **columns):
    """Write a delivery row's columns as stored, to stage a state under test."""
    from kestrel_sovereign.signals.durable import DurableSignalStore

    store = dispatcher._durable_store
    assignments = ", ".join(f"{name} = ?" for name in columns)
    values = tuple(
        store.to_timestamp_param(value) if isinstance(value, datetime) else value
        for value in columns.values()
    )
    assert await store.backend.execute(
        f"UPDATE {DurableSignalStore.DELIVERIES} SET {assignments} WHERE delivery_id = ?",
        (*values, delivery_id),
    ) == 1


async def _delivery_state(dispatcher, delivery_id):
    """A row's ``(status, attempts)`` as stored."""
    from kestrel_sovereign.signals.durable import DurableSignalStore

    return tuple(await dispatcher._durable_store.backend.fetch_one(
        f"SELECT status, attempts FROM {DurableSignalStore.DELIVERIES} "
        "WHERE delivery_id = ?",
        (delivery_id,),
    ))


async def test_held_legacy_wakes_are_never_claimed(manager):
    """Self-review of the Meridian implementation: key backfill holds two
    live legacy wakes for one input (``exclusive_conflict_key``, disposition
    ``held``) but left their statuses claimable, and no claim excluded a
    held row. A real SQLite reproduction backfilled two RETRY wakes into one
    held group, then claimed both through ``claim_durable_delivery``: the
    resident drainer would run work kept for an operator. No claim, exact
    claim, initial transfer or activation now takes a held row, and neither
    the drainer's window nor the due-consumer hint lists one, so a started
    consumer neither runs them nor re-arms itself to poll them (#4001)."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    peer_agent, peer, consumer = await input_dispatcher(manager)
    agent, dispatcher, _ = await input_dispatcher(manager)
    peer_open = True
    store = dispatcher._durable_store
    try:
        original = await _claimed_legacy_wake(manager, peer, consumer, task, recovery=False)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        recovery = await _claimed_legacy_wake(
            manager, peer, consumer, task, recovery=True, claim=False,
        )
        # The original is retried and the recovery never claimed, by an
        # earlier process that has since stopped.
        assert await peer._durable_store.nack_delivery(
            agent_id=agent.did, consumer_id=consumer, delivery_id=original.delivery_id,
            lease_token=original.lease_token, error="interrupted",
        ) is not None
        await close_input_dispatcher(peer_agent, peer)
        peer_open = False
        rows = (original.delivery_id, recovery.delivery_id)
        before = {row: await _delivery_state(dispatcher, row) for row in rows}
        assert {state for state, _ in before.values()} == {"retry", "pending"}

        await dispatcher.assign_durable_wake_exclusive_keys()
        key = submitted_input_identity(task.id, 0)
        for row in rows:
            assert await _delivery_keys(dispatcher, row) == (None, key, "held", False)
        assert await dispatcher.held_cognition_wake_groups() == 1

        owner = dispatcher._durable_delivery_owner
        assert await dispatcher.claim_durable_delivery(
            consumer_id=consumer, executor_id=owner
        ) is None
        for held in (original, recovery):
            assert await dispatcher.claim_durable_delivery_for_event(
                consumer_id=consumer, event_id=held.event_id, executor_id=owner,
            ) is None
        assert await store.due_consumer_ids(
            agent_id=agent.did, source="a2a.task_submitted", limit=10,
        ) == []
        assert await dispatcher._drainable_durable_deliveries(consumer) == []
        await dispatcher.start_durable_cognition_consumer(consumer)
        drainer = dispatcher._durable_cognition_drainers.get(consumer)
        if drainer is not None:
            await asyncio.wait_for(asyncio.shield(drainer), timeout=10)
        assert consumer not in dispatcher._durable_cognition_drain_timers, (
            "a held row re-armed the drainer"
        )
        assert {row: await _delivery_state(dispatcher, row) for row in rows} == before

        # The emitting dispatcher's first lease is refused for a held row too.
        future = datetime.now(UTC) + timedelta(minutes=5)
        await _set_delivery_row(
            dispatcher, recovery.delivery_id, status="initial_reserved",
            lease_owner=owner, lease_token="activate", lease_expires_at=None,
        )
        assert await store.activate_initial_delivery(
            agent_id=agent.did, consumer_id=consumer, delivery_id=recovery.delivery_id,
            initial_lease_owner=owner, initial_lease_token="activate",
        ) is None
        await _set_delivery_row(
            dispatcher, recovery.delivery_id, status="leased",
            lease_token="transfer", lease_expires_at=future,
        )
        assert await store.claim_initial_delivery(
            agent_id=agent.did, consumer_id=consumer, delivery_id=recovery.delivery_id,
            initial_lease_owner=owner, initial_lease_token="transfer", executor_id=owner,
        ) is None

        # Control: the same rows, once no conflict holds them, are claimed by
        # each of the same paths, so the hold is what refused them above.
        for row in rows:
            await _set_delivery_row(
                dispatcher, row, exclusive_conflict_key=None, settlement_disposition=None,
            )
        await _set_delivery_row(
            dispatcher, recovery.delivery_id, status="initial_reserved",
            lease_token="activate", lease_expires_at=None,
        )
        activated = await store.activate_initial_delivery(
            agent_id=agent.did, consumer_id=consumer, delivery_id=recovery.delivery_id,
            initial_lease_owner=owner, initial_lease_token="activate",
        )
        assert activated is not None and activated.status == "leased"
        assert await store.claim_initial_delivery(
            agent_id=agent.did, consumer_id=consumer, delivery_id=recovery.delivery_id,
            initial_lease_owner=owner, initial_lease_token="activate", executor_id=owner,
        ) is not None
        assert await store.due_consumer_ids(
            agent_id=agent.did, source="a2a.task_submitted", limit=10,
        ) == [consumer]
        claimed = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=consumer, event_id=original.event_id, executor_id=owner,
        )
        assert claimed is not None and claimed.delivery_id == original.delivery_id
    finally:
        await close_input_dispatcher(agent, dispatcher)
        if peer_open:
            await close_input_dispatcher(peer_agent, peer)


@pytest.mark.parametrize("acknowledged", ["after_the_failed_wake", "before_it"])
async def test_backfill_reads_an_acknowledged_legacy_wake_of_the_same_input(
    manager, acknowledged
):
    """Self-review of the Meridian implementation: key backfill read only
    unacknowledged rows, and the ACK guard skips a row with no key. When a
    later legacy recovery wake for an input was acknowledged before the
    first boot that keys wakes, its older failed sibling looked like the
    input's only wake, was keyed, and the settlement pass failed a WORKING
    task whose input that later wake had delivered: a real SQLite
    reproduction saw ``WORKING -> FAILED``.

    Under exclusive keys no wake can be admitted beside an unacknowledged
    one, so a legacy wake acknowledged at or after a failed row was admitted
    ran the work that row carried: backfill records the row ``carried`` and
    touches no task (``after_the_failed_wake``). A wake acknowledged before
    the failed one was admitted is the ordinary history of a recovery wake
    that then failed (``before_it``): that row still owes its input, and the
    pass ends the task so its sender sees it fail (#4001)."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    peer_agent, peer, consumer = await input_dispatcher(manager)
    agent, dispatcher, _ = await input_dispatcher(manager)
    reconciliation = "execution_control_unresolved: tool_effect_reconciliation_required"
    try:
        original = await _claimed_legacy_wake(manager, peer, consumer, task, recovery=False)
        assert await peer.ack_durable_delivery(
            consumer_id=consumer, delivery_id=original.delivery_id,
            lease_token=original.lease_token,
        )
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        failed = await _claimed_legacy_wake(manager, peer, consumer, task, recovery=True)
        await earlier_terminal_row(peer, consumer, failed, error=reconciliation)
        if acknowledged == "after_the_failed_wake":
            await _next_working_revision(manager, task)
            later = await _claimed_legacy_wake(manager, peer, consumer, task, recovery=True)
            assert await peer.ack_durable_delivery(
                consumer_id=consumer, delivery_id=later.delivery_id,
                lease_token=later.lease_token,
            )
        assert await _delivery_keys(dispatcher, failed.delivery_id) == (
            None, None, None, False,
        )
        before = await snapshot(manager, task)

        await agent.reconcile_a2a_cognition_wakes()
        await heartbeat(dispatcher)

        seen = await snapshot(manager, task)
        key = submitted_input_identity(task.id, 0)
        assert await dispatcher.held_cognition_wake_groups() == 0
        assert await dispatcher.unsettled_cognition_wakes() == []
        if acknowledged == "after_the_failed_wake":
            assert seen.task.status.state == TaskState.TASK_STATE_WORKING
            assert seen.lifecycle_revision == before.lifecycle_revision, (
                "settlement wrote a task whose input a later wake delivered"
            )
            assert await _delivery_keys(dispatcher, failed.delivery_id) == (
                None, None, "carried", True,
            )
            return
        assert seen.task.status.state == TaskState.TASK_STATE_FAILED
        assert "tool_effect_reconciliation_required" in seen.task.status.message.parts[0].text
        assert await _delivery_keys(dispatcher, failed.delivery_id) == (
            key, None, "ended", True,
        )
    finally:
        await close_input_dispatcher(agent, dispatcher)
        await close_input_dispatcher(peer_agent, peer)


async def failed_recovery_wake(manager, dispatcher, consumer, task):
    """A WORKING task whose submission wake ran and whose recovery wake failed.

    Boot mints a WORKING wake under the task's lifecycle revision when its
    submission wake already ran; this one is then refused, as a retry whose
    tool operation needs reconciliation is.
    """
    import copy

    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    ran = await claimed_submission_wake(manager, dispatcher, consumer, task)
    assert await dispatcher.ack_durable_delivery(
        consumer_id=consumer, delivery_id=ran.delivery_id, lease_token=ran.lease_token,
    )
    await manager.update_status(
        task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
    )
    working = await snapshot(manager, task)
    identity_key = submitted_input_identity(
        task.id, working.input_revision, working_revision=working.lifecycle_revision,
    )
    await dispatcher.dispatch_signal(
        copy.deepcopy(wake(working, manager)), source_event_id=identity_key,
    )
    recovery = await dispatcher.get_durable_delivery_for_source_event(
        consumer_id=consumer, source=SOURCE_NAME, source_event_id=identity_key,
    )
    recovery = await dispatcher.claim_durable_delivery_for_event(
        consumer_id=consumer,
        event_id=recovery.event_id,
        executor_id=dispatcher._durable_delivery_owner,
    )
    failed = await earlier_terminal_row(
        dispatcher, consumer, recovery,
        error="execution_control_unresolved: tool_effect_reconciliation_required",
    )
    assert failed.status == "failed"
    return failed


async def test_a_failed_recovery_wake_still_owes_its_input_after_the_lifecycle_moves(
    manager,
):
    """Final self-review of wo-4001-6: a WORKING task's recovery wake fails,
    then another recipient writer moves the task back to SUBMITTED without a
    new input. The failed row still names the input it carried; the task's
    live lifecycle since is not part of what it owes. Before the fix the owed
    query looked wakes up from the task's current state and lifecycle
    revision, found only the acknowledged submission wake, and the sender
    saw ``submitted`` forever. The running heartbeat settles it."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        failed = await failed_recovery_wake(manager, dispatcher, consumer, task)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_SUBMITTED, recipient_agent_id=manager.did
        )
        assert [d.delivery_id for d in await dispatcher.owed_cognition_wake_settlements()] == [
            failed.delivery_id
        ]

        await heartbeat(dispatcher)

        seen = await manager.get_task_for_creator(
            task.id, "did:creator", recipient_agent_id=manager.did
        )
        assert seen.status.state == TaskState.TASK_STATE_FAILED
        assert "tool_effect_reconciliation_required" in seen.status.message.parts[0].text
        assert await dispatcher.owed_cognition_wake_settlements() == []
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def _age_task_past_the_replay_horizon(manager, task):
    old = manager.task_store.to_timestamp_param(datetime.now(UTC) - timedelta(days=15))
    await manager.task_store.backend.execute(
        "UPDATE a2a_tasks SET lifecycle_updated_at = ?, updated_at = ? WHERE id = ?",
        (old, old, task.id),
    )


async def _expire_wake_event(dispatcher, delivery):
    from kestrel_sovereign.signals.durable import DurableSignalStore

    store = dispatcher._durable_store
    await store.backend.execute(
        f"UPDATE {DurableSignalStore.EVENTS} SET retention_until = ? WHERE event_id = ?",
        (
            store.to_timestamp_param(datetime.now(UTC) - timedelta(days=1)),
            delivery.event_id,
        ),
    )


async def test_a_terminal_wake_older_than_the_replay_horizon_is_still_settled(manager):
    """Final self-review of wo-4001-6: a wake can stay pending or leased past
    its source's retention, which never purges live work, and turn terminal
    later. Its task's last lifecycle change is then older than the replay
    horizon. Before the fix the owed query read only tasks changed within
    that horizon, so it skipped this one while its failed row stood, and the
    sender saw ``submitted`` forever. The running heartbeat settles it."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        failed = await failed_submission_wake(manager, dispatcher, consumer, task)
        await _age_task_past_the_replay_horizon(manager, task)
        await _expire_wake_event(dispatcher, failed)

        await heartbeat(dispatcher)

        seen = await manager.get_task_for_creator(
            task.id, "did:creator", recipient_agent_id=manager.did
        )
        assert seen.status.state == TaskState.TASK_STATE_FAILED
        assert await dispatcher.owed_cognition_wake_settlements() == []
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def test_retention_holds_a_terminal_wake_until_its_work_is_settled(manager):
    """Final self-review of wo-4001-6: a terminal wake row is the only record
    that its task is owed a settlement, and the retention sweep deletes any
    expired event whose deliveries are all terminal. A settling consumer's
    terminal row is kept past its deadline until the settlement pass records
    it settled, which it does once the row's work can never be owed again:
    its task ended, by the pass or elsewhere. Not owed is not enough: a task
    that asked for input can work on the same input again, and then owes it.
    The sweep itself reads no task (#4001)."""
    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    owed_task, settled_task, asking_task = [
        await manager.create_task(request(), creator_agent_id="did:creator")
        for _ in range(3)
    ]
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        owed = await failed_submission_wake(manager, dispatcher, consumer, owed_task)
        settled = await failed_submission_wake(manager, dispatcher, consumer, settled_task)
        asking = await failed_submission_wake(manager, dispatcher, consumer, asking_task)
        await manager.fail_task(
            settled_task.id, "ended elsewhere", agent_name=manager.did,
            recipient_agent_id=manager.did,
        )
        await manager.update_status(
            asking_task.id, TaskState.TASK_STATE_INPUT_REQUIRED,
            recipient_agent_id=manager.did,
        )
        for delivery in (owed, settled, asking):
            await _expire_wake_event(dispatcher, delivery)

        async def rows():
            return {
                d.event_id
                for d in await dispatcher.list_durable_deliveries(consumer_id=consumer)
            }

        assert await dispatcher._durable_store.purge_expired(
            agent_id=agent.did, held_consumers=[consumer],
        ) == ()
        assert await dispatcher.purge_expired_durable_deliveries() == 0
        assert await rows() == {owed.event_id, settled.event_id, asking.event_id}
        assert [d.event_id for d in await dispatcher.owed_cognition_wake_settlements()] == [
            owed.event_id
        ]

        await heartbeat(dispatcher)
        assert (await snapshot(manager, owed_task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
        assert await dispatcher.purge_expired_durable_deliveries() == 2
        assert await rows() == {asking.event_id}

        # The task works on the same input again; its wake is long gone.
        await manager.update_status(
            asking_task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        await heartbeat(dispatcher)
        assert (await snapshot(manager, asking_task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
        assert await dispatcher.purge_expired_durable_deliveries() == 1
        assert await rows() == set()
        assert await dispatcher.get_durable_delivery_for_source_event(
            consumer_id=consumer,
            source=SOURCE_NAME,
            source_event_id=submitted_input_identity(owed_task.id, 0),
        ) is None
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def test_a_refused_duplicate_leaves_the_work_another_executor_owns_alone(
    manager,
):
    """wo-4001-6 self-review: a rebuilt wake refused by a dispatcher with a
    lower TTL used to fail its task at once, although the same input's
    earlier attempt was already admitted and leased to another executor. A
    refusal is now only recorded, under the source identity: the admitted
    attempt keeps it, so nothing is recorded, and that delivery's own lease
    and terminal CAS decide its work (#4001)."""
    from kestrel_sdk.signals import CausationFrame

    from kestrel_sovereign.a2a.extensions import (
        causation_metadata,
        serialize_chain_for_metadata,
    )
    from kestrel_sovereign.signals import OrderedLockManager, SignalDispatcher
    from kestrel_sovereign.signals.dispatcher import DurableAdmissionDisposition
    from kestrel_sdk.signals import Status

    metadata = causation_metadata(serialize_chain_for_metadata([
        CausationFrame(agent_id="did:creator", source="heartbeat", signal_id="earlier",
                       turn_id=None, depth=1, emitted_at=datetime.now(UTC)),
    ]))
    task = await manager.create_task(request(**metadata), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    refusing = SignalDispatcher(
        agent=agent, registry=agent.signal_registry, lock_manager=OrderedLockManager(),
        store=dispatcher._store, ttl=1,
    )
    try:
        signal = wake(await snapshot(manager, task), manager)
        identity_key = signal.dedupe_key
        await dispatcher.dispatch_signal(signal, source_event_id=identity_key)
        admitted = await dispatcher.get_durable_delivery_for_source_event(
            consumer_id=consumer, source=signal.source, source_event_id=identity_key,
        )
        successor = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=consumer, event_id=admitted.event_id, executor_id="successor",
        )
        assert successor is not None

        retry = await refusing.enqueue_durable_cognition(
            wake(await snapshot(manager, task), manager),
            source_event_id=identity_key,
            consumer_id=consumer,
        )
        assert (await retry.wait()).status == Status.DROPPED_CYCLE
        assert (await retry.wait_for_durable_admission()).disposition == (
            DurableAdmissionDisposition.TERMINAL
        )
        await heartbeat(dispatcher)
        await heartbeat(refusing)

        (delivery,) = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        assert delivery.event_id == admitted.event_id
        assert (delivery.status, delivery.lease_owner) == ("leased", "successor")
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )
        assert await dispatcher.owed_cognition_wake_settlements() == []
    finally:
        await refusing.shutdown_durable_delivery()
        await close_input_dispatcher(agent, dispatcher)


async def test_a_late_row_cleanup_keeps_the_settlement_control_it_did_not_create(
    manager, monkeypatch
):
    """wo-4001-6 self-review: the row debt of a delivery and the settlement
    control of the work its wake carried shared one map. A late observer of
    the row's own terminalizer, joining after the heartbeat's settlement met
    an unknown commit, cleared that newer control, and shutdown reported
    success with the task still submitted and still owed. Each is now its
    own ledger, and a row's cleanup clears only the row's debt (#4001)."""
    from kestrel_sovereign import _async_ownership
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        ToolAuditUnconfirmedError,
    )

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        original = await claimed_submission_wake(manager, dispatcher, consumer, task)
        row_error = ToolAuditUnconfirmedError("required audit unavailable")
        native_join = _async_ownership.await_owned_task
        observer_joined, release_observer = asyncio.Event(), asyncio.Event()

        async def join(owned, *args, **kwargs):
            result = await native_join(owned, *args, **kwargs)
            if asyncio.current_task().get_name() == "late-original-cleanup":
                observer_joined.set()
                await release_observer.wait()
            return result

        monkeypatch.setattr(_async_ownership, "await_owned_task", join)
        initial = asyncio.create_task(
            dispatcher._terminalize_failed_cognition(original, row_error),
        )
        late = asyncio.create_task(
            dispatcher._release_retained_durable_cognition_task(original, row_error),
            name="late-original-cleanup",
        )
        await asyncio.wait_for(observer_joined.wait(), 3)
        await initial
        assert not dispatcher._retained_cognition_control_debt

        fault_task_writes(manager, [ExecutionCommitOutcomeError("unknown")])
        await heartbeat(dispatcher)
        assert dispatcher.unresolved_cognition_control == (original.delivery_id,)

        release_observer.set()
        await late

        assert not dispatcher._retained_cognition_control_debt
        assert list(dispatcher._source_settlement_control) == [original.delivery_id]
        assert dispatcher.unresolved_cognition_control == (original.delivery_id,)
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )

        # Retirement retries only the settlement's fenced write, and clears
        # the control only once that write returns.
        closed = True
        await close_input_dispatcher(agent, dispatcher)
        assert dispatcher.unresolved_cognition_control == ()
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("row", ["recorded", "purged"])
async def test_a_peer_settlement_ends_the_control_this_owner_retains(manager, row):
    """Self-review of wo-4001-7: one owner's settlement of a failed wake meets
    an unknown commit and retains it as control. A peer owner sharing the
    database then settles the same task and records the row settled; once
    recorded, retention may remove the row. Either takes the row out of the
    unsettled scan, so before the fix the first owner's heartbeat never
    retried its control and the owner stayed unready for good. Its next pass
    now settles the control from the task's own state: it ends the control
    and writes nothing, since nothing is owed (#4001)."""
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    peer_agent, peer, _ = await input_dispatcher(manager)
    try:
        assert peer._durable_delivery_owner != dispatcher._durable_delivery_owner
        failed = await failed_submission_wake(manager, dispatcher, consumer, task)
        faults = [ExecutionCommitOutcomeError("unknown")]
        attempts = fault_task_writes(manager, faults)
        await heartbeat(dispatcher)
        assert faults == []
        assert dispatcher.unresolved_cognition_control == (failed.delivery_id,)

        await heartbeat(peer)
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
        assert peer.unresolved_cognition_control == ()
        assert await peer.unsettled_cognition_wakes() == []
        if row == "purged":
            await _expire_wake_event(peer, failed)
            assert await peer.purge_expired_durable_deliveries() == 1
            assert await peer.list_durable_deliveries(consumer_id=consumer) == []
        writes = len(attempts)

        await heartbeat(dispatcher)

        assert dispatcher.unresolved_cognition_control == ()
        assert not dispatcher._source_settlement_control
        assert len(attempts) == writes
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
    finally:
        await close_input_dispatcher(peer_agent, peer)
        await close_input_dispatcher(agent, dispatcher)


async def test_a_settlement_page_reads_only_the_rows_it_returns(manager):
    """wo-4001-6 self-review: the settlement pass paged terminal rows by
    delivery ID through an index ordered by retry time, so every page
    sorted the consumer's whole terminal history. Each page is now one
    terminal status as a literal, a range on the terminal index that also
    serves its order (#4001)."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        await failed_submission_wake(manager, dispatcher, consumer, task)
        store = dispatcher._durable_store
        statements = []
        fetch_all = store.backend.fetch_all

        async def recorded(sql, params=()):
            statements.append((sql, params))
            return await fetch_all(sql, params)

        store.backend.fetch_all = recorded
        try:
            assert len(await dispatcher.owed_cognition_wake_settlements()) == 1
        finally:
            store.backend.fetch_all = fetch_all
        pages = [
            statement for statement in statements
            if "ORDER BY d.delivery_id LIMIT" in statement[0]
        ]
        assert {"'failed'", "'terminal_ackable'"} == {
            literal for sql, _ in pages
            for literal in ("'failed'", "'terminal_ackable'")
            if f"d.status = {literal}" in sql
        }
        index = re.compile(rf"\bidx_{store.DELIVERIES}_settlement_scan\b")
        for sql, params in pages:
            if store.backend.backend_type == "postgres":
                async with store.backend.transaction():
                    await store.backend.execute("SET LOCAL enable_seqscan = off")
                    rows = await store.backend.fetch_all("EXPLAIN " + sql, params)
                plan = "\n".join(str(row[0]) for row in rows)
                assert index.search(plan), plan
                assert "Sort" not in plan, plan
            else:
                rows = await store.backend.fetch_all("EXPLAIN QUERY PLAN " + sql, params)
                plan = "\n".join(str(row[-1]) for row in rows)
                assert index.search(plan), plan
                assert "TEMP B-TREE" not in plan, plan
    finally:
        await close_input_dispatcher(agent, dispatcher)


def count_task_reads(manager, *, poisoned=frozenset()):
    """Count the settlement's task reads; raise a storage error for some tasks.

    The settlement reads its task once, locked, inside the transaction that
    settles the row (#4001).
    """
    from kestrel_sdk.storage.database.interface import QueryError

    read = manager.task_store.apply_undelivered_wake_transition
    reads = []

    async def counted(transition):
        reads.append(transition.task_id)
        if transition.task_id in poisoned:
            raise QueryError("task row cannot be read")
        return await read(transition)

    manager.task_store.apply_undelivered_wake_transition = counted
    return reads


async def test_a_settled_wake_leaves_the_settlement_pass(manager):
    """PR #4029 merge review: every heartbeat paged through every terminal
    row of the submission consumer and read its task, so the pass grew with
    the whole retention window's terminal history. Once durable state shows a
    row's work can never be owed again, the pass records it settled and never
    reads it again. A row that may still be owed (a task that asked for input
    at that input) stays, so a pass reads what may be owed (#4001)."""
    settled = [
        await manager.create_task(request(), creator_agent_id="did:creator")
        for _ in range(3)
    ]
    asking = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        for task in (*settled, asking):
            await failed_submission_wake(manager, dispatcher, consumer, task)
        await manager.update_status(
            asking.id, TaskState.TASK_STATE_INPUT_REQUIRED, recipient_agent_id=manager.did
        )

        await heartbeat(dispatcher)
        for task in settled:
            assert (await snapshot(manager, task)).task.status.state == (
                TaskState.TASK_STATE_FAILED
            )

        reads = count_task_reads(manager)
        await heartbeat(dispatcher)

        assert set(reads) == {asking.id}
        assert [d.event.source_event_id for d in await dispatcher.unsettled_cognition_wakes()] == [
            submitted_input_identity(asking.id, 0)
        ]
        assert [d.status for d in await dispatcher.list_durable_deliveries(
            consumer_id=consumer
        )] == ["failed"] * 4
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def test_a_task_that_cannot_be_read_neither_stops_the_pass_nor_refuses_boot(
    manager,
):
    """PR #4029 merge review: one terminal row whose task read keeps raising
    aborted the whole settlement pass, and refused every boot. It now stays
    owed and the pass goes on: the other tasks are settled, boot completes
    without minting a second wake for that input, and the heartbeat settles
    it once it can be read (#4001)."""
    tasks = [
        await manager.create_task(request(), creator_agent_id="did:creator")
        for _ in range(3)
    ]
    poisoned = tasks[1]
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        for task in tasks:
            await failed_submission_wake(manager, dispatcher, consumer, task)
        reads = count_task_reads(manager, poisoned={poisoned.id})

        await agent.reconcile_a2a_cognition_wakes()

        assert poisoned.id in reads
        states = {
            task.id: (await snapshot(manager, task)).task.status.state for task in tasks
        }
        assert states == {
            tasks[0].id: TaskState.TASK_STATE_FAILED,
            poisoned.id: TaskState.TASK_STATE_SUBMITTED,
            tasks[2].id: TaskState.TASK_STATE_FAILED,
        }
        assert len(await dispatcher.list_durable_deliveries(consumer_id=consumer)) == 3
        assert dispatcher.unresolved_cognition_control == ()

        del manager.task_store.apply_undelivered_wake_transition
        await heartbeat(dispatcher)

        assert (await snapshot(manager, poisoned)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
        assert await dispatcher.unsettled_cognition_wakes() == []
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def test_only_a_settling_consumer_row_brings_the_settlement_pass_forward(
    manager,
):
    """PR #4029 merge review: every terminal row of any consumer brought the
    owner heartbeat forward, while a submission row that failed by spending
    its last attempt brought nothing. A terminal row of a consumer that
    settles nothing leaves the heartbeat at its own beat; a submission row
    that exhausts its attempts is settled promptly, without waiting a beat
    (#4001)."""
    from kestrel_sovereign.signals import DurableConsumerRegistration
    from kestrel_sovereign.signals.durable import DurableSignalStore

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    loop = asyncio.get_running_loop()
    try:
        await dispatcher.register_durable_consumer(
            DurableConsumerRegistration(
                consumer_id="workflows:wait:4001",
                source="a2a.task_submitted",
                agent_id=agent.did,
                max_attempts=0,
            )
        )
        claimed = await claimed_submission_wake(manager, dispatcher, consumer, task)
        other = await dispatcher.claim_durable_delivery_for_event(
            consumer_id="workflows:wait:4001", event_id=claimed.event_id,
            executor_id="workflows-worker",
        )
        due = dispatcher._runtime_owner_heartbeat_timer.when()
        assert due - loop.time() > 1.0

        ended = await dispatcher.nack_durable_delivery(
            consumer_id="workflows:wait:4001", delivery_id=other.delivery_id,
            lease_token=other.lease_token, error="workflow gave up", terminal=True,
        )
        assert ended.status == "failed"
        assert not dispatcher._owed_settlement_pass_requested
        assert dispatcher._runtime_owner_heartbeat_timer.when() == due

        # The submission row was persisted with a finite budget, and this
        # ordinary retry spends its last attempt.
        store = dispatcher._durable_store
        await store.backend.execute(
            f"UPDATE {DurableSignalStore.DELIVERIES} SET max_attempts = 1 "
            "WHERE delivery_id = ?",
            (claimed.delivery_id,),
        )
        exhausted = await dispatcher.nack_durable_delivery(
            consumer_id=consumer, delivery_id=claimed.delivery_id,
            lease_token=claimed.lease_token, error="turn failed",
        )
        assert exhausted.status == "failed"
        assert dispatcher._owed_settlement_pass_requested

        deadline = loop.time() + 5.0
        while (await snapshot(manager, task)).task.status.state != (
            TaskState.TASK_STATE_FAILED
        ):
            assert loop.time() < deadline, "the exhausted wake was never settled"
            await asyncio.sleep(0.02)
    finally:
        await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("refused_as", ["dropped_cycle", "dropped_validation"])
async def test_a_wake_refused_after_it_persisted_is_settled_under_its_own_status(
    manager, refused_as,
):
    """PR #4029 merge review: a terminal no-op row that was not a Stop told
    the sender ``dropped_validation`` whatever refused it, a cycle drop
    included. The row now records the status that refused it, and the
    settlement names that status (#4001)."""
    from kestrel_sdk.signals import SignalResult, Status

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        claimed = await claimed_submission_wake(manager, dispatcher, consumer, task)
        result = SignalResult(
            signal_id=claimed.event.event_id,
            status=Status(refused_as),
            mode=claimed.event.mode,
            duration_ms=0,
            error="refused after the wake persisted",
        )
        assert await dispatcher._settle_cognition_non_delivery(
            claimed, result, consumer_id=consumer, retry_delay=timedelta()
        )
        (row,) = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        assert row.status == "terminal_ackable"
        assert dispatcher._undelivered_wake_disposition(row) == (False, refused_as)

        await dispatcher._settle_owed_cognition_sources()

        seen = await manager.get_task_for_creator(
            task.id, "did:creator", recipient_agent_id=manager.did
        )
        assert seen.status.state == TaskState.TASK_STATE_FAILED
        assert f"({refused_as})" in seen.status.message.parts[0].text
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def _row_settlement(dispatcher, delivery_id):
    """A row's ``(settlement_disposition, settled_at)`` as stored."""
    from kestrel_sovereign.signals.durable import DurableSignalStore

    return await dispatcher._durable_store.backend.fetch_one(
        "SELECT settlement_disposition, settled_at FROM "
        f"{DurableSignalStore.DELIVERIES} WHERE delivery_id = ?",
        (delivery_id,),
    )


async def test_settlement_retried_after_unknown_commit_returns_committed_disposition(
    manager, monkeypatch
):
    """Meridian's ruling on wo-4001-10 (Q1): the settlement of a failed wake
    commits, and its acknowledgement is lost. The retry must not decide
    again. The row's state, its key and the task change commit together, so
    the retry reads the committed outcome: the same disposition, the task
    changed exactly once, and ``settled_at`` untouched (#4001)."""
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        execution_commit_outcome,
    )

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        failed = await failed_submission_wake(manager, dispatcher, consumer, task)
        before = (await snapshot(manager, task)).lifecycle_revision
        store = dispatcher._durable_store
        settle = store.settle_terminal_delivery
        outcomes = []

        async def lost_ack(**kwargs):
            outcome = await settle(**kwargs)
            outcomes.append((outcome.disposition, outcome.recorded))
            if len(outcomes) == 1:
                raise ExecutionCommitOutcomeError("unknown")
            return outcome

        monkeypatch.setattr(store, "settle_terminal_delivery", lost_ack)
        control = await dispatcher._settle_owed_cognition_sources()
        assert control.delivery_id == failed.delivery_id
        assert execution_commit_outcome(control.original) == "unknown"
        committed = await snapshot(manager, task)
        assert committed.task.status.state == TaskState.TASK_STATE_FAILED
        stamp = await _row_settlement(dispatcher, failed.delivery_id)

        # The row has left the scan; the retained control retries it.
        assert await dispatcher._settle_owed_cognition_sources() is None

        assert outcomes == [("ended", True), ("ended", False)]
        assert dispatcher.unresolved_cognition_control == ()
        settled = await snapshot(manager, task)
        assert settled.lifecycle_revision == committed.lifecycle_revision == before + 1
        assert await _row_settlement(dispatcher, failed.delivery_id) == stamp
        assert stamp[0] == "ended" and stamp[1] is not None
    finally:
        await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("peer", ["admits_the_input", "settles_the_row"])
async def test_settlement_and_peer_release_serialize(
    manager, native_a2a_backend_factory, peer
):
    """Meridian's ruling on wo-4001-10 (Q1): two stores on one database. A
    owner settles a failed wake and is held inside its settlement, between
    ending the task and recording the row. A peer then admits a wake for the
    same input (``admits_the_input``), or settles the same row
    (``settles_the_row``). The peer waits for A's one transaction, and
    afterwards finds the work settled: its wake is refused, or it reads A's
    committed disposition. A state where the task ended and a live wake for
    that input committed, or where the row records an outcome other than the
    one that ended the task, is never reached (#4001)."""
    import copy

    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.signals import DurableAdmissionDisposition
    from tests.utils.native_a2a import manager_for

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    other = manager_for(
        native_a2a_backend_factory(),
        NativeRecipientScope(manager.did, manager.task_store.scope.tenant),
    )
    await other.task_store.backend.connect()
    await other.initialize()
    peer_agent, peer_dispatcher, _ = await input_dispatcher(other)
    try:
        failed = await failed_submission_wake(manager, dispatcher, consumer, task)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        before = await snapshot(manager, task)
        backend = dispatcher._durable_store.backend
        execute = backend.execute
        holding, release = asyncio.Event(), asyncio.Event()

        async def held_before_the_record(sql, params=()):
            if "SET settled_at = ?, settlement_disposition = ?" in sql and not holding.is_set():
                holding.set()
                await release.wait()
            return await execute(sql, params)

        backend.execute = held_before_the_record
        settling = asyncio.create_task(dispatcher._settle_owed_cognition_sources())
        await asyncio.wait_for(holding.wait(), 10)

        peer_outcomes = []
        if peer == "admits_the_input":
            identity = submitted_input_identity(
                task.id, 0, working_revision=before.lifecycle_revision
            )

            async def admit():
                handle = await peer_dispatcher.enqueue_durable_cognition(
                    copy.deepcopy(wake(before, other)),
                    source_event_id=identity,
                    consumer_id=consumer,
                )
                return (await handle.wait_for_durable_admission()).disposition

            acting = asyncio.create_task(admit())
        else:
            peer_store = peer_dispatcher._durable_store
            peer_settle = peer_store.settle_terminal_delivery

            async def observed(**kwargs):
                outcome = await peer_settle(**kwargs)
                peer_outcomes.append((outcome.disposition, outcome.recorded))
                return outcome

            peer_store.settle_terminal_delivery = observed
            acting = asyncio.create_task(peer_dispatcher._settle_owed_cognition_sources())
        await asyncio.sleep(0.5)
        assert not acting.done(), "the peer did not wait for the settlement transaction"
        release.set()
        assert await asyncio.wait_for(settling, 10) is None
        result = await asyncio.wait_for(acting, 10)

        settled = await snapshot(manager, task)
        assert settled.task.status.state == TaskState.TASK_STATE_FAILED
        assert settled.lifecycle_revision == before.lifecycle_revision + 1
        stored = await _row_settlement(dispatcher, failed.delivery_id)
        assert stored[0] == "ended" and stored[1] is not None
        rows = await dispatcher.list_durable_deliveries(consumer_id=consumer)
        assert [(d.delivery_id, d.status) for d in rows] == [
            (failed.delivery_id, "failed")
        ], "a live wake was committed for work a settlement ended"
        if peer == "admits_the_input":
            assert result is DurableAdmissionDisposition.DUPLICATE
        else:
            assert result is None
            assert peer_outcomes == [("ended", False)]
    finally:
        release.set()
        await close_input_dispatcher(peer_agent, peer_dispatcher)
        await other.close()
        await close_input_dispatcher(agent, dispatcher)


async def test_a_task_settlement_runs_on_the_delivery_settlement_transaction(
    manager, monkeypatch
):
    """Meridian's ruling on wo-4001-10, step 0: the task statement runs on
    the same connection and transaction as the delivery row it settles, on
    either backend. A failure after the task write rolls both back (#4001)."""
    from kestrel_sdk.storage.database.interface import QueryError

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        failed = await failed_submission_wake(manager, dispatcher, consumer, task)
        backend = dispatcher._durable_store.backend
        assert backend is manager.task_store.backend
        postgres = backend.backend_type == "postgres"

        async def transaction_identity():
            if postgres:
                return await backend.fetch_val("SELECT txid_current()")
            assert backend.owns_open_transaction
            return id(backend._txn_owner)

        observed = []
        store = dispatcher._durable_store
        lock = store._lock_scope_handoff

        async def handoff(**kwargs):
            await lock(**kwargs)
            observed.append(("handoff", await transaction_identity()))

        monkeypatch.setattr(store, "_lock_scope_handoff", handoff)
        apply = manager.task_store.apply_undelivered_wake_transition
        fail_after_write = {"armed": True}

        async def task_write(transition):
            outcome = await apply(transition)
            observed.append(("task", await transaction_identity()))
            if fail_after_write["armed"]:
                fail_after_write["armed"] = False
                raise QueryError("connection reset after the task write")
            return outcome

        monkeypatch.setattr(manager.task_store, "apply_undelivered_wake_transition", task_write)

        assert await dispatcher._settle_owed_cognition_sources() is None
        assert [label for label, _ in observed] == ["handoff", "task"]
        assert observed[0][1] == observed[1][1]
        # Rolled back together: the task and the row are both untouched.
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )
        assert await _row_settlement(dispatcher, failed.delivery_id) == (None, None)

        observed.clear()
        assert await dispatcher._settle_owed_cognition_sources() is None
        assert observed[0][1] == observed[1][1]
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
        assert (await _row_settlement(dispatcher, failed.delivery_id))[0] == "ended"
    finally:
        await close_input_dispatcher(agent, dispatcher)


def _cancellation_carrying_unknown_commit():
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    cancelled = asyncio.CancelledError()
    cancelled.__cause__ = ExecutionCommitOutcomeError("unknown")
    return cancelled


@pytest.mark.parametrize(
    "carrier",
    ["bare", "carrying_unknown_commit"],
)
async def test_settlement_cancellation_carrier_stays_per_delivery(manager, carrier):
    """Meridian's ruling on wo-4001-10 (Q3): one row's task write raises a
    cancellation while the pass itself is not being cancelled, bare or
    carrying an unknown commit. Before the fix the pass re-raised it, and the
    pass's completion handler promoted it into the dispatcher's writer latch,
    which refuses all durable admission, so no later settlement could run.
    It is now that delivery's control, returned through the real heartbeat
    pass: admission stays open, the control is retained for exactly that
    delivery, and shutdown surfaces it until a retry settles it (#4001)."""
    tasks = [
        await manager.create_task(request(), creator_agent_id="did:creator")
        for _ in range(2)
    ]
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        failed = [
            await failed_submission_wake(manager, dispatcher, consumer, task)
            for task in tasks
        ]
        injected = (
            asyncio.CancelledError() if carrier == "bare"
            else _cancellation_carrying_unknown_commit()
        )
        faults = [injected]
        fault_task_writes(manager, faults)

        await heartbeat(dispatcher)

        assert faults == []
        assert dispatcher._outcome_log_control_error is None
        (retained,) = dispatcher._source_settlement_control
        assert retained in {row.delivery_id for row in failed}
        # Admission is open: durable operations still run.
        assert len(await dispatcher.list_durable_deliveries(consumer_id=consumer)) == 2

        # Shutdown surfaces it while the retry fails.
        faults.append(injected)
        with pytest.raises(BaseException):
            await close_input_dispatcher(agent, dispatcher)
        assert list(dispatcher._source_settlement_control) == [retained]

        closed = True
        await close_input_dispatcher(agent, dispatcher)
        assert dispatcher.unresolved_cognition_control == ()
        # The pass stopped at that delivery; the retirement drain settled it,
        # and the other row stays owed in durable storage for the next pass.
        states = {
            row.delivery_id: (await snapshot(manager, task)).task.status.state
            for row, task in zip(failed, tasks)
        }
        assert states[retained] == TaskState.TASK_STATE_FAILED
        assert [
            state for delivery_id, state in states.items() if delivery_id != retained
        ] == [TaskState.TASK_STATE_SUBMITTED]
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


async def test_a_cancelled_settlement_pass_retains_its_row_and_latches_nothing(
    manager, monkeypatch
):
    """The companion case: the pass itself is cancelled while a row's
    settlement runs. The pass stops, the row's control is retained, and
    durable admission is still not latched; the next pass reconciles the
    row from its stored outcome (#4001)."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        failed = await failed_submission_wake(manager, dispatcher, consumer, task)
        project = manager.project_undelivered_wake_settlement
        entered, release = asyncio.Event(), asyncio.Event()

        async def held_projection(*args, **kwargs):
            entered.set()
            await release.wait()
            return await project(*args, **kwargs)

        monkeypatch.setattr(manager, "project_undelivered_wake_settlement", held_projection)
        dispatcher._request_owed_settlement_pass(consumer_id=consumer, status="failed")
        await asyncio.wait_for(entered.wait(), 5)
        running = dispatcher._owed_settlement_task
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dispatcher._join_owed_settlement_pass()

        assert dispatcher._outcome_log_control_error is None
        assert list(dispatcher._source_settlement_control) == [failed.delivery_id]
        assert len(await dispatcher.list_durable_deliveries(consumer_id=consumer)) == 1

        release.set()
        monkeypatch.setattr(manager, "project_undelivered_wake_settlement", project)
        await heartbeat(dispatcher)
        assert dispatcher.unresolved_cognition_control == ()
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
    finally:
        release.set()
        await close_input_dispatcher(agent, dispatcher)


async def test_bootstrap_demotes_duplicate_holders_before_enforcing_one(manager):
    """A database an earlier build of this branch keyed can hold two wakes
    under one key. Schema bootstrap resolves them by the backfill rule, one
    live beside a failed one, before it creates the unique holder index; the
    index then refuses a second holder outright (#4001)."""
    from kestrel_sovereign.signals.durable import DurableSignalStore

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        live = await claimed_submission_wake(manager, dispatcher, consumer, task)
        await manager.update_status(
            task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        recovery = await _claimed_legacy_wake(manager, dispatcher, consumer, task, recovery=True)
        await earlier_terminal_row(
            dispatcher, consumer, recovery,
            error="execution_control_unresolved: tool_effect_reconciliation_required",
        )
        store = dispatcher._durable_store
        key = submitted_input_identity(task.id, 0)
        index = f"idx_{DurableSignalStore.DELIVERIES}_exclusive_holder"
        await store.backend.execute(f"DROP INDEX {index}")
        await store.backend.execute(
            f"UPDATE {DurableSignalStore.DELIVERIES} SET exclusive_key = ? "
            "WHERE delivery_id = ?",
            (key, recovery.delivery_id),
        )

        await store.initialize()

        assert await _delivery_keys(dispatcher, live.delivery_id) == (key, None, None, False)
        assert await _delivery_keys(dispatcher, recovery.delivery_id) == (
            None, None, "carried", True,
        )
        with pytest.raises(Exception):
            await store.backend.execute(
                f"UPDATE {DurableSignalStore.DELIVERIES} SET exclusive_key = ?, "
                "settled_at = NULL WHERE delivery_id = ?",
                (key, recovery.delivery_id),
            )
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def test_the_writer_latch_refuses_a_deliverys_settlement_control(manager, caplog):
    """Defence in depth for Q3: should an error carrying a delivery's retained
    settlement control ever reach the dispatcher's writer latch, the latch
    logs it and refuses it, so it cannot refuse all durable admission
    (#4001)."""
    import logging

    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        failed = await failed_submission_wake(manager, dispatcher, consumer, task)
        original = ExecutionCommitOutcomeError("unknown")
        dispatcher._retain_source_settlement_control(failed, original)
        carrier = asyncio.CancelledError()
        carrier.__cause__ = original

        with caplog.at_level(logging.ERROR, logger="kestrel_sovereign.signals.dispatcher"):
            dispatcher._retain_durable_writer_control(carrier)

        assert dispatcher._outcome_log_control_error is None
        assert "Refusing to retain a delivery's settlement control" in caplog.text
        # Control that no delivery retains still latches, as before.
        dispatcher._retain_durable_writer_control(ExecutionCommitOutcomeError("unknown"))
        assert dispatcher._outcome_log_control_error is not None
    finally:
        dispatcher._outcome_log_control_error = None
        dispatcher._source_settlement_control.clear()
        await close_input_dispatcher(agent, dispatcher)
