"""A later creator input must fence every old recipient execution and wake."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_data_artifact, new_text_message
from kestrel_sovereign._vendor.a2a.types import Role, SendMessageRequest, Task, TaskState, TaskStatus

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
