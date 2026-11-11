"""A later creator input must fence every old recipient execution and wake."""

import asyncio
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


@pytest.mark.parametrize(
    "state",
    [TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING],
    ids=["submitted", "working"],
)
async def test_boot_reconcile_ends_a_task_whose_wake_already_failed(manager, state):
    """#4001 backstop: a process that stopped between failing a submission
    wake and ending its task leaves the task live, with no wake left to run
    it. Boot reconciliation ends it, rather than leaving it for its sender to
    wait on or minting another wake for the same refused input."""
    import copy

    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        identity = submitted_input_identity(task.id, 0)
        await dispatcher.dispatch_signal(
            copy.deepcopy(wake(await snapshot(manager, task), manager)),
            source_event_id=identity,
        )
        original = await dispatcher.get_durable_delivery_for_source_event(
            consumer_id=consumer, source=SOURCE_NAME, source_event_id=identity
        )
        claimed = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=consumer,
            event_id=original.event_id,
            executor_id=dispatcher._durable_delivery_owner,
        )
        failed = await dispatcher.nack_durable_delivery(
            consumer_id=consumer,
            delivery_id=claimed.delivery_id,
            lease_token=claimed.lease_token,
            error="execution_control_unresolved: tool_effect_reconciliation_required",
            terminal=True,
        )
        assert failed.status == "failed"
        if state == TaskState.TASK_STATE_WORKING:
            await manager.update_status(task.id, state, recipient_agent_id=manager.did)

        await agent.reconcile_a2a_cognition_wakes()

        settled = await snapshot(manager, task)
        assert settled.task.status.state == TaskState.TASK_STATE_FAILED
        assert "wake failed" in settled.task.status.message.parts[0].text
        assert [d.status for d in await dispatcher.list_durable_deliveries(
            consumer_id=consumer
        )] == ["failed"]
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def test_failed_wake_settlement_retains_native_commit_uncertainty(manager):
    """#4001 round 3: the task write that settles a failed submission wake can
    end with its commit outcome unknown. That is native control, not an
    ordinary settlement failure: it propagates, and stays retained as the
    delivery's debt. The retry, at the owner's shutdown drain here, settles
    only the task. The failed row's CAS committed once and is never written
    again."""
    import copy
    import gc

    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        ToolAuditUnconfirmedError,
        execution_commit_outcome,
    )
    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        identity_key = submitted_input_identity(task.id, 0)
        await dispatcher.dispatch_signal(
            copy.deepcopy(wake(await snapshot(manager, task), manager)),
            source_event_id=identity_key,
        )
        original = await dispatcher.get_durable_delivery_for_source_event(
            consumer_id=consumer, source=SOURCE_NAME, source_event_id=identity_key
        )
        claimed = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=consumer,
            event_id=original.event_id,
            executor_id=dispatcher._durable_delivery_owner,
        )
        # Each backend has its own exact-token failed-row CAS; count both.
        store = dispatcher._durable_store
        failed_row_writes = []

        def counted(owner, name):
            write = getattr(owner, name)

            async def counted_write(**kwargs):
                failed_row_writes.append(kwargs["delivery_id"])
                return await write(**kwargs)

            setattr(owner, name, counted_write)

        if store.backend.backend_type == "postgres":
            counted(store.backend, "fail_cognition_delivery")
        else:
            counted(store, "release_managed_delivery_after_task")
        uncertain = ExecutionCommitOutcomeError("unknown")
        save = manager.task_store.save_recipient_lifecycle
        writes = {"uncertain": 1}

        async def save_once_uncertain(*args, **kwargs):
            if writes["uncertain"]:
                writes["uncertain"] -= 1
                raise uncertain
            return await save(*args, **kwargs)

        manager.task_store.save_recipient_lifecycle = save_once_uncertain

        with pytest.raises(BaseException) as raised:
            await dispatcher._terminalize_failed_cognition(
                claimed, ToolAuditUnconfirmedError("audit failed")
            )
        assert execution_commit_outcome(raised.value) == "unknown"
        delivery_id = claimed.delivery_id
        assert failed_row_writes == [delivery_id]
        assert [d.status for d in await dispatcher.list_durable_deliveries(
            consumer_id=consumer
        )] == ["failed"]
        assert delivery_id in dispatcher._retained_cognition_control_debt
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )
        # Only the dispatcher's own retained debt may keep this delivery's
        # snapshot, and with it the failed row's receipt, alive.
        del claimed, original, raised
        gc.collect()

        closed = True
        await close_input_dispatcher(agent, dispatcher)

        assert writes["uncertain"] == 0
        assert failed_row_writes == [delivery_id]
        assert not dispatcher._retained_cognition_control_debt
        assert not dispatcher._failed_cognition_source_debt
        settled = await snapshot(manager, task)
        assert settled.task.status.state == TaskState.TASK_STATE_FAILED
        assert "tool_effect_outcome_unconfirmed" in settled.task.status.message.parts[0].text
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


async def claimed_submission_wake(manager, dispatcher, consumer, task):
    """Persist a task's submission wake and claim its delivery for this owner."""
    import copy

    from kestrel_sovereign.signals.sources.a2a_task_submitted import SOURCE_NAME

    identity_key = submitted_input_identity(task.id, 0)
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


async def test_a_heartbeat_settlement_retry_that_meets_native_uncertainty_retains_it(manager):
    """#4001 wo-4001-4: an ordinary failure leaves a failed wake's task owed,
    and the owner heartbeat retries it. A retry whose write ends with its
    commit outcome unknown is native control: it is retained as that
    delivery's control debt, the heartbeat runs no further ordinary
    settlement beside it, and the shutdown drain settles that task without
    writing its failed row again. The other owed task is left for boot."""
    from kestrel_sdk.storage.database.interface import QueryError

    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        ToolAuditUnconfirmedError,
    )

    tasks = [
        await manager.create_task(request(), creator_agent_id="did:creator")
        for _ in range(2)
    ]
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        claims = [
            await claimed_submission_wake(manager, dispatcher, consumer, task)
            for task in tasks
        ]
        store = dispatcher._durable_store
        failed_row_writes = []
        owner, name = (
            (store.backend, "fail_cognition_delivery")
            if store.backend.backend_type == "postgres"
            else (store, "release_managed_delivery_after_task")
        )
        write = getattr(owner, name)

        async def counted_write(**kwargs):
            failed_row_writes.append(kwargs["delivery_id"])
            return await write(**kwargs)

        setattr(owner, name, counted_write)
        faults = [
            QueryError("database is locked"),
            QueryError("database is locked"),
            ExecutionCommitOutcomeError("unknown"),
        ]
        save = manager.task_store.save_recipient_lifecycle
        task_writes = []

        async def save_with_faults(*args, **kwargs):
            task_writes.append(len(faults))
            if faults:
                raise faults.pop(0)
            return await save(*args, **kwargs)

        manager.task_store.save_recipient_lifecycle = save_with_faults

        for claimed in claims:
            await dispatcher._terminalize_failed_cognition(
                claimed, ToolAuditUnconfirmedError("audit failed")
            )
        delivery_ids = [claimed.delivery_id for claimed in claims]
        assert failed_row_writes == delivery_ids
        assert list(dispatcher._failed_cognition_source_debt) == delivery_ids
        assert not dispatcher._retained_cognition_control_debt

        await dispatcher._retry_failed_cognition_source_settlements()

        assert faults == [] and len(task_writes) == 3
        assert list(dispatcher._retained_cognition_control_debt) == delivery_ids[:1]
        assert list(dispatcher._failed_cognition_source_debt) == delivery_ids
        assert dispatcher._runtime_owner_cleanup_required()
        for task in tasks:
            assert (await snapshot(manager, task)).task.status.state == (
                TaskState.TASK_STATE_SUBMITTED
            )

        closed = True
        await close_input_dispatcher(agent, dispatcher)

        assert failed_row_writes == delivery_ids
        assert not dispatcher._retained_cognition_control_debt
        assert list(dispatcher._failed_cognition_source_debt) == delivery_ids[1:]
        assert (await snapshot(manager, tasks[0])).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
        assert (await snapshot(manager, tasks[1])).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


def _cancellation_with_unknown_commit():
    cancelled = asyncio.CancelledError()
    cancelled._kestrel_unresolved_commit_outcome = "unknown"
    return cancelled


@pytest.mark.parametrize(
    "fault, outcome",
    [
        ("storage", "retried"),
        ("lifecycle_conflict", "retried"),
        ("refusal", "retried"),
        ("cancellation", "cancelled_and_retried"),
        ("cancelled_commit", "control"),
    ],
)
async def test_each_failed_wake_settlement_failure_keeps_the_task_owed(manager, fault, outcome):
    """#4001: only the settlement that follows a committed failed row ends its
    source debt, by settling the task or finding it no longer waits on the
    wake. Every ordinary failure keeps the task owed to the owner heartbeat,
    whatever its type: a storage fault, a lifecycle write that lost its
    compare-and-set to a concurrent SUBMITTED -> WORKING writer, or any other
    error. A cancellation propagates with the debt kept. A cancellation
    carrying an unknown commit outcome is native control, and propagates
    composed with the failed row's own error."""
    from kestrel_sdk.storage.database.interface import QueryError

    from kestrel_sovereign.execution_custody import (
        ToolAuditUnconfirmedError,
        execution_commit_outcome,
    )

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        claimed = await claimed_submission_wake(manager, dispatcher, consumer, task)
        save = manager.task_store.save_recipient_lifecycle

        async def concurrent_working_writer(*args, **kwargs):
            # Another recipient writer wins the real store CAS for the same
            # input between the settlement's read and its write.
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

        raised = {
            "storage": QueryError("database is locked"),
            "lifecycle_conflict": concurrent_working_writer,
            "refusal": RuntimeError("an error the store has no class for"),
            "cancellation": asyncio.CancelledError(),
            "cancelled_commit": _cancellation_with_unknown_commit(),
        }[fault]
        faults = [raised]

        async def save_with_fault(*args, **kwargs):
            if faults:
                injected = faults.pop(0)
                if callable(injected):
                    return await injected(*args, **kwargs)
                raise injected
            return await save(*args, **kwargs)

        manager.task_store.save_recipient_lifecycle = save_with_fault
        error = None
        try:
            await dispatcher._terminalize_failed_cognition(
                claimed, ToolAuditUnconfirmedError("audit failed")
            )
        except BaseException as caught:
            error = caught

        delivery_id = claimed.delivery_id
        assert faults == []
        assert [d.status for d in await dispatcher.list_durable_deliveries(
            consumer_id=consumer
        )] == ["failed"]
        owed = dispatcher._failed_cognition_source_debt
        controlled = dispatcher._retained_cognition_control_debt
        if outcome == "retried":
            assert error is None
            assert list(owed) == [delivery_id] and not controlled
        elif outcome == "cancelled_and_retried":
            assert error is raised
            assert list(owed) == [delivery_id] and not controlled
        else:
            # Native control first: composed with the row's own failure, never
            # re-raised as a bare cancellation, and never left as ordinary debt
            # beside a write whose outcome is unknown.
            assert error is not None and error is not raised
            assert execution_commit_outcome(error) == "unknown"
            assert list(owed) == [delivery_id]
            # The heartbeat's retry does not swallow it either.
            faults.append(_cancellation_with_unknown_commit())
            with pytest.raises(asyncio.CancelledError):
                await dispatcher._retry_failed_cognition_source_settlements()
            assert faults == []
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_WORKING
            if fault == "lifecycle_conflict"
            else TaskState.TASK_STATE_SUBMITTED
        )
        if outcome in {"retried", "cancelled_and_retried"}:
            await dispatcher._retry_failed_cognition_source_settlements()
            assert not dispatcher._failed_cognition_source_debt
            assert (await snapshot(manager, task)).task.status.state == (
                TaskState.TASK_STATE_FAILED
            )
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def test_a_stopped_wake_whose_cancellation_commit_is_unknown_fails_its_row(manager):
    """#4001: Stop consumed a wake, and the write that cancels its task ends
    with its commit outcome unknown, arriving as a cancellation. That is
    native control, as it is where a failed row's settlement meets it: the
    delivery becomes a failed row, never left leased under a cancelled
    dispatch for a lease expiry to redeliver the stopped work, and the task
    it carried is settled as that row's source debt. The debt keeps the
    Stop's intent, so the task is cancelled as the Stop asked, not failed."""
    from kestrel_sdk.signals import Status

    from kestrel_sovereign.execution_custody import execution_commit_outcome
    from kestrel_sovereign.signals.dispatcher import _StoppedCognitionResult

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    try:
        claimed = await claimed_submission_wake(manager, dispatcher, consumer, task)
        faults = [_cancellation_with_unknown_commit()]
        cancel = manager.task_store.cancel_if_authorized

        async def cancel_once_uncertain(*args, **kwargs):
            if faults:
                raise faults.pop(0)
            return await cancel(*args, **kwargs)

        manager.task_store.cancel_if_authorized = cancel_once_uncertain
        stopped = _StoppedCognitionResult(
            signal_id=claimed.event.event_id,
            status=Status.COALESCED,
            mode=claimed.event.mode,
            duration_ms=0,
            error="stop_acknowledged: operator Stop",
        )

        with pytest.raises(BaseException) as raised:
            await dispatcher._settle_cognition_non_delivery(
                claimed, stopped, consumer_id=consumer, retry_delay=timedelta()
            )
        assert faults == []
        assert execution_commit_outcome(raised.value) == "unknown"
        assert [d.status for d in await dispatcher.list_durable_deliveries(
            consumer_id=consumer
        )] == ["failed"]
        settled = await snapshot(manager, task)
        assert settled.task.status.state == TaskState.TASK_STATE_CANCELED
        assert not dispatcher._failed_cognition_source_debt
        assert not dispatcher._retained_cognition_control_debt
        assert not dispatcher._failed_cognition_stop_intents
    finally:
        await close_input_dispatcher(agent, dispatcher)


async def test_a_stopped_wake_whose_unknown_cancellation_retry_fails_stays_controlled(manager):
    """#4001 wo-4001-5 self-review, the same rule one door over: a Stop's
    task cancellation whose commit outcome is unknown, made before the
    delivery's failed row existed, becomes that row's source control as the
    row commits. The retry that follows the row meets a database error, which
    proves nothing about the earlier write: the control stands, so owner
    retirement refuses, and the next attempt cancels the task as the Stop
    asked."""
    from kestrel_sdk.signals import Status
    from kestrel_sdk.storage.database.interface import QueryError

    from kestrel_sovereign.execution_custody import execution_commit_outcome
    from kestrel_sovereign.signals.dispatcher import _StoppedCognitionResult

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        claimed = await claimed_submission_wake(manager, dispatcher, consumer, task)
        delivery_id = claimed.delivery_id
        faults = [_cancellation_with_unknown_commit(), QueryError("database is locked")]
        cancel = manager.task_store.cancel_if_authorized

        async def cancel_with_faults(*args, **kwargs):
            if faults:
                raise faults.pop(0)
            return await cancel(*args, **kwargs)

        manager.task_store.cancel_if_authorized = cancel_with_faults
        stopped = _StoppedCognitionResult(
            signal_id=claimed.event.event_id,
            status=Status.COALESCED,
            mode=claimed.event.mode,
            duration_ms=0,
            error="stop_acknowledged: operator Stop",
        )

        with pytest.raises(BaseException) as raised:
            await dispatcher._settle_cognition_non_delivery(
                claimed, stopped, consumer_id=consumer, retry_delay=timedelta()
            )
        assert faults == []
        assert execution_commit_outcome(raised.value) == "unknown"
        del raised, claimed
        assert [d.status for d in await dispatcher.list_durable_deliveries(
            consumer_id=consumer
        )] == ["failed"]
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )
        assert list(dispatcher._retained_cognition_control_debt) == [delivery_id]
        assert list(dispatcher._failed_cognition_source_debt) == [delivery_id]
        assert dispatcher._failed_cognition_source_debt[delivery_id][2] is True
        assert not dispatcher._failed_cognition_pending_source_control
        assert dispatcher._runtime_owner_cleanup_required()

        # The owner-retirement drain's retry meets a database error too.
        faults.append(QueryError("database is locked"))
        with pytest.raises(BaseException) as refused:
            await close_input_dispatcher(agent, dispatcher)
        assert faults == []
        assert execution_commit_outcome(refused.value) == "unknown"
        del refused
        assert list(dispatcher._retained_cognition_control_debt) == [delivery_id]
        assert dispatcher._durable_runtime_owner_registered

        closed = True
        await close_input_dispatcher(agent, dispatcher)

        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_CANCELED
        )
        assert not dispatcher._retained_cognition_control_debt
        assert not dispatcher._failed_cognition_source_debt
        assert not dispatcher._failed_cognition_source_control
        assert not dispatcher._failed_cognition_stop_intents
        assert not dispatcher._durable_runtime_owner_registered
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


async def test_boot_settlement_of_a_failed_wake_retains_native_uncertainty(manager):
    """#4001 wo-4001-4 self-review: boot reconciliation settles a wake whose
    failed row an earlier process committed without settling its task. That
    row joins the dispatcher's source debt with its committed receipt. A task
    write whose commit outcome is unknown propagates from boot and is retained
    as the delivery's control debt; the shutdown drain then settles only the
    task. No failed-row CAS is written for a row this process never owned."""
    import gc

    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError,
        execution_commit_outcome,
    )

    task = await manager.create_task(request(), creator_agent_id="did:creator")
    agent, dispatcher, consumer = await input_dispatcher(manager)
    closed = False
    try:
        claimed = await claimed_submission_wake(manager, dispatcher, consumer, task)
        # The earlier process's failed row, committed without a settlement.
        failed = await dispatcher.nack_durable_delivery(
            consumer_id=consumer,
            delivery_id=claimed.delivery_id,
            lease_token=claimed.lease_token,
            error="execution_control_unresolved: tool_effect_reconciliation_required",
            terminal=True,
        )
        assert failed.status == "failed"
        delivery_id = claimed.delivery_id
        del claimed, failed
        store = dispatcher._durable_store
        failed_row_writes = []
        owner, name = (
            (store.backend, "fail_cognition_delivery")
            if store.backend.backend_type == "postgres"
            else (store, "release_managed_delivery_after_task")
        )
        write = getattr(owner, name)

        async def counted_write(**kwargs):
            failed_row_writes.append(kwargs["delivery_id"])
            return await write(**kwargs)

        setattr(owner, name, counted_write)
        faults = [ExecutionCommitOutcomeError("unknown")]
        save = manager.task_store.save_recipient_lifecycle

        async def save_once_uncertain(*args, **kwargs):
            if faults:
                raise faults.pop(0)
            return await save(*args, **kwargs)

        manager.task_store.save_recipient_lifecycle = save_once_uncertain

        with pytest.raises(BaseException) as raised:
            await agent.reconcile_a2a_cognition_wakes()
        assert execution_commit_outcome(raised.value) == "unknown"
        assert faults == []
        assert list(dispatcher._retained_cognition_control_debt) == [delivery_id]
        assert list(dispatcher._failed_cognition_source_debt) == [delivery_id]
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )
        # The dispatcher's own debt keeps the receipt alive, not this test.
        del raised
        gc.collect()

        closed = True
        await close_input_dispatcher(agent, dispatcher)

        assert failed_row_writes == []
        assert not dispatcher._retained_cognition_control_debt
        assert not dispatcher._failed_cognition_source_debt
        settled = await snapshot(manager, task)
        assert settled.task.status.state == TaskState.TASK_STATE_FAILED
        assert "wake failed" in settled.task.status.message.parts[0].text
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)


@pytest.mark.parametrize("origin", ["live", "boot"])
async def test_an_ordinary_retry_failure_never_clears_an_unknown_source_commit(manager, origin):
    """#4001 wo-4001-5 self-review: a failed row's task settlement whose
    commit outcome is unknown is the source's control, not the row's. The
    row's receipt does not clear it, not even while a retry is running, and
    nor does a retry's ordinary failure, which proves nothing about the
    earlier write. An owner-retirement drain whose retry meets a database
    error therefore refuses to retire the owner, keeping both debts, and the
    next attempt settles the task. The control arises in the running process
    (``live``), or at the owner heartbeat's retry of a settlement boot could
    not make (``boot``). The failed row's CAS is never written again."""
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
        claimed = await claimed_submission_wake(manager, dispatcher, consumer, task)
        delivery_id = claimed.delivery_id
        faults = []
        controlled_at_write = []
        save = manager.task_store.save_recipient_lifecycle

        async def save_with_faults(*args, **kwargs):
            controlled_at_write.append(
                delivery_id in dispatcher._retained_cognition_control_debt
            )
            if faults:
                raise faults.pop(0)
            return await save(*args, **kwargs)

        manager.task_store.save_recipient_lifecycle = save_with_faults
        if origin == "live":
            faults.append(ExecutionCommitOutcomeError("unknown"))
            with pytest.raises(BaseException) as raised:
                await dispatcher._terminalize_failed_cognition(
                    claimed, ToolAuditUnconfirmedError("audit failed")
                )
            assert execution_commit_outcome(raised.value) == "unknown"
            del raised
        else:
            # The earlier process's failed row, committed without a settlement.
            await dispatcher.nack_durable_delivery(
                consumer_id=consumer,
                delivery_id=delivery_id,
                lease_token=claimed.lease_token,
                error="execution_control_unresolved: tool_effect_reconciliation_required",
                terminal=True,
            )
            faults.extend([QueryError("boot fault"), ExecutionCommitOutcomeError("unknown")])
            await agent.reconcile_a2a_cognition_wakes()
            assert not dispatcher._retained_cognition_control_debt
            # The real owner heartbeat retries it, and meets the unknown commit.
            await dispatcher._heartbeat_runtime_owner()
        del claimed
        assert faults == []
        assert list(dispatcher._retained_cognition_control_debt) == [delivery_id]
        assert list(dispatcher._failed_cognition_source_debt) == [delivery_id]

        store = dispatcher._durable_store
        failed_row_writes = []
        owner, name = (
            (store.backend, "fail_cognition_delivery")
            if store.backend.backend_type == "postgres"
            else (store, "release_managed_delivery_after_task")
        )
        write = getattr(owner, name)

        async def counted_write(**kwargs):
            failed_row_writes.append(kwargs["delivery_id"])
            return await write(**kwargs)

        setattr(owner, name, counted_write)
        controlled_at_write.clear()
        faults.append(QueryError("database is locked"))
        with pytest.raises(BaseException) as refused:
            await close_input_dispatcher(agent, dispatcher)

        assert faults == []
        # The retry ran with the control still standing, not cleared by the
        # row's receipt on its way to the write.
        assert controlled_at_write == [True]
        assert execution_commit_outcome(refused.value) == "unknown"
        del refused
        assert list(dispatcher._retained_cognition_control_debt) == [delivery_id]
        assert list(dispatcher._failed_cognition_source_debt) == [delivery_id]
        assert dispatcher._runtime_owner_cleanup_required()
        assert dispatcher._durable_runtime_owner_registered
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_SUBMITTED
        )

        closed = True
        await close_input_dispatcher(agent, dispatcher)

        assert failed_row_writes == []
        assert not dispatcher._retained_cognition_control_debt
        assert not dispatcher._failed_cognition_source_debt
        assert not dispatcher._failed_cognition_source_control
        assert not dispatcher._durable_runtime_owner_registered
        assert (await snapshot(manager, task)).task.status.state == (
            TaskState.TASK_STATE_FAILED
        )
    finally:
        if not closed:
            await close_input_dispatcher(agent, dispatcher)
