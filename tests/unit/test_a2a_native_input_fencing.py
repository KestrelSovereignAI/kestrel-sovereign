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
    return agent, dispatcher, DURABLE_COGNITION_CONSUMER_ID


async def close_input_dispatcher(agent, dispatcher):
    await dispatcher.shutdown_durable_delivery()
    await asyncio.gather(*agent.tasks, return_exceptions=True)


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


@pytest.mark.parametrize("stopped", [True, False], ids=["stopped", "failed"])
async def test_an_undelivered_wake_ends_only_the_input_it_carried(manager, stopped):
    """#4001: a submission wake that ended without running ends its task, and
    only for the input it carried. A newer input the creator added meanwhile
    stays live for its own wake. Both wakes are payload-elided, as under a
    volatile privacy mode: the input identity alone settles them."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    stale = private_event(wake(await snapshot(manager, task), manager))
    await manager.continue_task(followup(manager, task), creator_agent_id="did:creator")
    recipient = owner(manager)

    assert not await recipient.settle_undelivered_cognition_wake(
        **identity(stale), stopped=stopped, cause="dropped_cycle"
    )
    current = await snapshot(manager, task)
    assert current.task.status.state == TaskState.TASK_STATE_SUBMITTED
    assert current.input_revision == 1

    assert await recipient.settle_undelivered_cognition_wake(
        **identity(private_event(wake(current, manager))),
        stopped=stopped,
        cause="dropped_cycle",
    )
    settled = await snapshot(manager, task)
    text = settled.task.status.message.parts[0].text
    if stopped:
        assert settled.task.status.state == TaskState.TASK_STATE_CANCELED
        assert "Stop cancelled" in text
    else:
        assert settled.task.status.state == TaskState.TASK_STATE_FAILED
        assert "dropped_cycle" in text
    # Settling the same wake again finds nothing left to end.
    assert not await recipient.settle_undelivered_cognition_wake(
        **identity(private_event(wake(current, manager))),
        stopped=stopped,
        cause="dropped_cycle",
    )


@pytest.mark.parametrize("stopped", [True, False], ids=["stopped", "failed"])
async def test_an_undelivered_wake_spares_an_input_that_lands_while_it_settles(
    manager, stopped
):
    """The creator adds an input after the settlement read the task and before
    it wrote: the write is fenced to the wake's own input, so the newer input
    stays live for its own wake."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    stale = private_event(wake(await snapshot(manager, task), manager))
    recipient = owner(manager)
    read = recipient._input_awaits_its_wake
    reads = []

    async def read_then_continue(task_id, input_revision):
        awaits = await read(task_id, input_revision)
        if not reads:
            await manager.continue_task(
                followup(manager, task), creator_agent_id="did:creator"
            )
        reads.append(awaits)
        return awaits

    recipient._input_awaits_its_wake = read_then_continue

    assert not await recipient.settle_undelivered_cognition_wake(
        **identity(stale), stopped=stopped, cause="dropped_cycle"
    )
    assert reads == [True, False]
    current = await snapshot(manager, task)
    assert current.task.status.state == TaskState.TASK_STATE_SUBMITTED
    assert current.input_revision == 1


async def test_an_undelivered_wake_leaves_a_task_awaiting_input_alone(manager):
    """A task that asked its creator for input is not waiting on the wake, so a
    stale wake for that input ending does not fail it."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    stale = private_event(wake(await snapshot(manager, task), manager))
    await manager.update_status(
        task.id, TaskState.TASK_STATE_INPUT_REQUIRED, recipient_agent_id=manager.did
    )

    assert not await owner(manager).settle_undelivered_cognition_wake(
        **identity(stale), stopped=False, cause="dropped_validation"
    )
    assert (await snapshot(manager, task)).task.status.state == (
        TaskState.TASK_STATE_INPUT_REQUIRED
    )


@pytest.mark.parametrize("stopped", [True, False], ids=["stopped", "failed"])
async def test_an_undelivered_wake_spares_a_task_that_asks_for_input_while_it_settles(
    manager, stopped
):
    """The recipient's worker asks the creator for input after the settlement
    read the task and before it wrote. The store refuses the write in the same
    statement, so the task stays input-required instead of being cancelled or
    failed out from under the question its creator now sees."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    stale = private_event(wake(await snapshot(manager, task), manager))
    recipient = owner(manager)
    read = recipient._input_awaits_its_wake
    reads = []

    async def read_then_ask_for_input(task_id, input_revision):
        awaits = await read(task_id, input_revision)
        if not reads:
            await manager.update_status(
                task.id,
                TaskState.TASK_STATE_INPUT_REQUIRED,
                recipient_agent_id=manager.did,
            )
        reads.append(awaits)
        return awaits

    recipient._input_awaits_its_wake = read_then_ask_for_input

    assert not await recipient.settle_undelivered_cognition_wake(
        **identity(stale), stopped=stopped, cause="dropped_cycle"
    )
    assert reads == [True, False]
    current = await snapshot(manager, task)
    assert current.task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
    assert current.input_revision == 0


async def test_an_undelivered_wake_for_another_recipient_is_refused(manager):
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    foreign = replace(
        private_event(wake(await snapshot(manager, task), manager)),
        target_agent="did:kestrel:someone-else",
    )
    with pytest.raises(ValueError, match="another recipient"):
        await owner(manager).settle_undelivered_cognition_wake(
            **identity(foreign), stopped=False, cause="dropped_cycle"
        )
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


async def failed_submission_wake(manager, dispatcher, consumer, task, *, revision=0):
    """A wake row a refused retry left failed, as an earlier process commits it."""
    claimed = await claimed_submission_wake(
        manager, dispatcher, consumer, task, revision=revision
    )
    failed = await dispatcher.nack_durable_delivery(
        consumer_id=consumer,
        delivery_id=claimed.delivery_id,
        lease_token=claimed.lease_token,
        error="execution_control_unresolved: tool_effect_reconciliation_required",
        terminal=True,
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
    """Raise each fault in turn from the recipient's real lifecycle write."""
    save = manager.task_store.save_recipient_lifecycle
    attempts = []

    async def save_with_faults(*args, **kwargs):
        attempts.append(len(faults))
        if faults:
            injected = faults.pop(0)
            if callable(injected):
                return await injected(save, *args, **kwargs)
            raise injected
        return await save(*args, **kwargs)

    manager.task_store.save_recipient_lifecycle = save_with_faults
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
    sender to wait on or minting another wake for the same refused input."""
    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        await failed_submission_wake(manager, dispatcher, consumer, task)
        if state == TaskState.TASK_STATE_WORKING:
            await manager.update_status(task.id, state, recipient_agent_id=manager.did)

        await agent.reconcile_a2a_cognition_wakes()

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
        refused = await dispatcher.nack_durable_delivery(
            consumer_id=consumer,
            delivery_id=refused.delivery_id,
            lease_token=refused.lease_token,
            error="stop_acknowledged: operator Stop",
            terminal=True,
            terminal_ackable=True,
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
        recovery = await dispatcher.nack_durable_delivery(
            consumer_id=consumer, delivery_id=recovery.delivery_id,
            lease_token=recovery.lease_token, error="durable consumer deactivated",
            terminal=True,
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


@pytest.mark.parametrize(
    "fault",
    ["storage", "lifecycle_conflict", "refusal", "cancellation"],
)
async def test_every_ordinary_settlement_failure_leaves_the_task_owed(manager, fault):
    """#4001 wo-4001-4: no failure type decides a retry. A storage fault, a
    lifecycle write that lost its compare-and-set to a concurrent
    SUBMITTED -> WORKING writer, an error the store has no class for, or a
    cancellation: each leaves the task owed, with no control debt, and the
    next pass finds it from durable state and settles it. Nothing in the
    process remembers it in between."""
    from kestrel_sdk.storage.database.interface import QueryError

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        await failed_submission_wake(manager, dispatcher, consumer, task)

        async def concurrent_working_writer(save, *args, **kwargs):
            baseline = kwargs["baseline"]
            working = Task.FromString(baseline.SerializeToString())
            working.status.state = TaskState.TASK_STATE_WORKING
            assert await save(
                working,
                baseline=baseline,
                recipient_agent_id=kwargs["recipient_agent_id"],
                expected_state=baseline.status.state,
            ) is not None
            return await save(*args, **kwargs)

        injected = {
            "storage": QueryError("database is locked"),
            "lifecycle_conflict": concurrent_working_writer,
            "refusal": RuntimeError("an error the store has no class for"),
            "cancellation": asyncio.CancelledError(),
        }[fault]
        faults = [injected]
        attempts = fault_task_writes(manager, faults)

        if fault == "cancellation":
            with pytest.raises(asyncio.CancelledError):
                await dispatcher._settle_owed_cognition_sources()
        else:
            assert await dispatcher._settle_owed_cognition_sources() is None
        assert faults == [] and len(attempts) == 1
        assert dispatcher.unresolved_cognition_control == ()
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_WORKING
            if fault == "lifecycle_conflict"
            else TaskState.TASK_STATE_SUBMITTED
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
    The pass retains it as that delivery's control debt and settles nothing
    beside it, and the owner heartbeat takes its cleanup branch, running no
    further settlement, until owner retirement resolves it."""
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

        if arrives_as == "error":
            control = await dispatcher._settle_owed_cognition_sources()
            assert execution_commit_outcome(control) == "unknown"
        else:
            with pytest.raises(asyncio.CancelledError):
                await dispatcher._settle_owed_cognition_sources()
        assert faults == [] and len(attempts) == 1
        assert not dispatcher._retained_cognition_control_debt
        (debt,) = dispatcher._source_settlement_control.values()
        assert debt[0].delivery_id in {row.delivery_id for row in failed}
        assert dispatcher._runtime_owner_cleanup_required()

        await dispatcher._heartbeat_runtime_owner()

        assert len(attempts) == 1
        for task in tasks:
            assert (await snapshot(manager, task)).task.status.state == (
                TaskState.TASK_STATE_SUBMITTED
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
            assert execution_commit_outcome(control) == "unknown"
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
    failed = await dispatcher.nack_durable_delivery(
        consumer_id=consumer,
        delivery_id=recovery.delivery_id,
        lease_token=recovery.lease_token,
        error="execution_control_unresolved: tool_effect_reconciliation_required",
        terminal=True,
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

        await dispatcher._heartbeat_runtime_owner()

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

        await dispatcher._heartbeat_runtime_owner()

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
    expired event whose deliveries are all terminal. A row past its deadline
    whose work is still owed is kept until that work is settled; one whose
    task ended is released. Not owed is not enough to release: a task that
    asked for input can work on the same input again, and then owes it.
    Held is the default: a sweep that releases nothing keeps every such row,
    so one that turns terminal while the sweep decides is kept for the next."""
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
        assert await rows() == {owed.event_id, settled.event_id, asking.event_id}

        assert await dispatcher.purge_expired_durable_deliveries() == 1
        assert await rows() == {owed.event_id, asking.event_id}
        assert [d.event_id for d in await dispatcher.owed_cognition_wake_settlements()] == [
            owed.event_id
        ]

        await dispatcher._heartbeat_runtime_owner()
        assert (await snapshot(manager, owed_task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
        assert await dispatcher.purge_expired_durable_deliveries() == 1
        assert await rows() == {asking.event_id}

        # The task works on the same input again; its wake is long gone.
        await manager.update_status(
            asking_task.id, TaskState.TASK_STATE_WORKING, recipient_agent_id=manager.did
        )
        await dispatcher._heartbeat_runtime_owner()
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
        await dispatcher._heartbeat_runtime_owner()
        await refusing._heartbeat_runtime_owner()

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
        await dispatcher._heartbeat_runtime_owner()
        assert dispatcher.unresolved_cognition_control == (original.delivery_id,)

        release_observer.set()
        await late

        assert not dispatcher._retained_cognition_control_debt
        assert list(dispatcher._source_settlement_control) == [original.delivery_id]
        assert dispatcher.unresolved_cognition_control == (original.delivery_id,)
        assert dispatcher._runtime_owner_cleanup_required()
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
        index = re.compile(rf"\bidx_{store.DELIVERIES}_terminal_page\b")
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
