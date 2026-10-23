"""Local execution must not become incoming cognition during durable recovery."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from kestrel_sovereign._vendor.a2a.helpers.proto_helpers import new_text_message
from kestrel_sovereign._vendor.a2a.types import Role, SendMessageRequest, TaskState, TaskStatus
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.signals.context import bind_current_signal
from kestrel_sovereign.signals.rehydration import DurableCognitionWithdrawal
from kestrel_sovereign.signals.sources.a2a import build_signal_for_completed_task
from kestrel_sovereign.signals.sources.a2a_task_submitted import (
    build_signal_for_submitted_task,
)
from tests.unit.test_a2a_native_input_fencing import owner, private_event
from tests.unit.test_a2a_native_task_manager import manager as manager
from tests.unit.test_a2a_native_task_manager import request
from tests.utils.native_a2a import manager_for


class LocalWakeFeature(Feature):
    @property
    def tool_description(self):
        return "Hold one real local invocation for recovery assertions"

    async def initialize(self):
        pass

    @tool(name="hold_local", description="Complete after release")
    async def hold_local(self) -> ToolResult:
        self.effects += 1
        self.started.set()
        await self.release.wait()
        return ToolResult.ok(confirmation="Local invocation finished")


def provider(manager):
    feature = LocalWakeFeature(
        SimpleNamespace(did=manager.did, _execution_custody=None)
    )
    feature.name = "local-wake"
    feature.effects = 0
    feature.started = asyncio.Event()
    feature.release = asyncio.Event()
    manager.register_provider("local-wake", feature.get_agent_skills(), feature)
    return feature


async def candidates(manager):
    horizon = datetime.now(UTC) - timedelta(days=1)
    return await manager.list_cognition_wake_candidates(
        recipient_agent_id=manager.did,
        live_changed_since=horizon,
        terminal_changed_since=horizon,
    )


@pytest.mark.parametrize("sync", [True, False])
async def test_local_execution_recovery_only_owes_an_async_completion(
    manager, native_a2a_backend_factory, sync
):
    feature = provider(manager)
    submissions, completions = [], []
    manager._on_task_submitted = submissions.append
    manager._on_task_complete = completions.append
    execution = asyncio.create_task(
        manager.execute_skill("local-wake", "hold_local", {}, sync=sync)
    )
    restarted = None
    try:
        async with asyncio.timeout(3):
            await feature.started.wait()
        rows = await manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks")
        assert len(rows) == 1
        task_id = rows[0][0]
        assert await candidates(manager) == [], (
            "A local tool became an incoming cognition recovery candidate"
        )
        backend = native_a2a_backend_factory()
        await backend.connect()
        restarted = manager_for(backend, manager.task_store.scope)
        await restarted.initialize()
        assert await candidates(restarted) == []
        current = await restarted.get_task_snapshot(
            task_id, recipient_agent_id=manager.did
        )
        with pytest.raises(ValueError, match="eligible"):
            build_signal_for_submitted_task(current, manager.did)
        assert submissions == [] and completions == []
        feature.release.set()
        await execution
        await asyncio.gather(*tuple(manager._execution_tasks))
        recovered = await candidates(restarted)
        assert [item.task.id for item in recovered] == ([] if sync else [task_id])
        assert len(completions) == (0 if sync else 1)
        assert feature.effects == 1
        snapshot = await restarted.get_task_snapshot(
            task_id, recipient_agent_id=manager.did
        )
        with pytest.raises(ValueError, match="eligible"):
            build_signal_for_submitted_task(snapshot, manager.did)
        if sync:
            with pytest.raises(ValueError, match="eligible"):
                build_signal_for_completed_task(snapshot, manager.did)
        else:
            assert (
                build_signal_for_completed_task(snapshot, manager.did).payload[
                    "task_id"
                ]
                == task_id
            )
    finally:
        feature.release.set()
        await execution
        await asyncio.gather(*tuple(manager._execution_tasks))
        if restarted is not None:
            await restarted.close()


async def test_peer_metadata_cannot_disable_its_server_owned_wake_policy(manager):
    task = await manager.create_task(
        request(cognition_wake_eligible=False, input_wake_eligible=False),
        creator_agent_id="did:peer",
    )
    candidate = await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
    assert (
        build_signal_for_submitted_task(candidate, manager.did).payload["task_id"]
        == task.id
    )
    assert [item.task.id for item in await candidates(manager)] == [task.id]


async def test_interrupted_sync_command_is_terminal_and_absent_from_pending_inbox(
    manager,
):
    from google.protobuf.json_format import MessageToDict

    feature = provider(manager)
    execution = asyncio.create_task(
        manager.execute_skill("local-wake", "hold_local", {}, sync=True)
    )
    async with asyncio.timeout(3):
        await feature.started.wait()
    [(task_id,)] = await manager.task_store.backend.fetch_all(
        "SELECT id FROM a2a_tasks"
    )
    execution.cancel("interrupt local command")
    with pytest.raises(asyncio.CancelledError, match="interrupt local command"):
        await execution
    stored = await manager.get_task_for_recipient(task_id, manager.did)
    assert stored.status.state == TaskState.TASK_STATE_CANCELED
    assert (
        MessageToDict(stored.metadata)["cancellation_receipt"]["actor_agent_id"]
        == manager.did
    )
    assert await manager.get_pending_tasks(recipient_agent_id=manager.did) == []
    assert await candidates(manager) == []
    assert feature.effects == 1 and not feature.release.is_set()


async def test_sync_interruption_keeps_an_independent_terminal_winner(manager):
    feature = provider(manager)
    execution = asyncio.create_task(
        manager.execute_skill("local-wake", "hold_local", {}, sync=True)
    )
    async with asyncio.timeout(3):
        await feature.started.wait()
    [(task_id,)] = await manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks")
    winner = await manager.update_status(
        task_id, TaskState.TASK_STATE_COMPLETED,
        new_text_message("Independent completion"), recipient_agent_id=manager.did,
    )
    execution.cancel("observer interrupted")
    with pytest.raises(asyncio.CancelledError, match="observer interrupted"):
        await execution
    assert await manager.get_task_for_recipient(task_id, manager.did) == winner
    assert await manager.get_pending_tasks(recipient_agent_id=manager.did) == []
    assert await candidates(manager) == [] and feature.effects == 1


async def test_stale_input_wake_cannot_drive_or_mutate_a_local_async_invocation(
    manager,
    monkeypatch,
):
    feature = provider(manager)
    task = await manager.execute_skill("local-wake", "hold_local", {}, sync=False)
    try:
        async with asyncio.timeout(3):
            await feature.started.wait()
        candidate = await manager.get_task_snapshot(
            task.id, recipient_agent_id=manager.did
        )
        # Represent an old queued envelope. Policy is always re-read from the
        # authoritative task, regardless of what that envelope once claimed.
        admitted = replace(
            candidate, cognition_wake_eligible=True, input_wake_eligible=True
        )
        stale = build_signal_for_submitted_task(admitted, manager.did)
        recipient = owner(manager)
        assert await recipient.validate_cognition_signal_execution(stale) is not None
        read = asyncio.Event()
        canonical_read = manager.get_task_cancellation_snapshot

        async def observe_read(*args, **kwargs):
            snapshot = await canonical_read(*args, **kwargs)
            read.set()
            return snapshot

        with monkeypatch.context() as patch:
            patch.setattr(manager, "get_task_cancellation_snapshot", observe_read)
            monitor = asyncio.create_task(
                recipient.monitor_cognition_signal_execution(stale)
            )
            try:
                async with asyncio.timeout(3):
                    await read.wait()
                assert monitor.done(), (
                    "The monitor accepted a local tool as incoming cognition"
                )
                assert await monitor is not None
            finally:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
        assert isinstance(
            await recipient.rehydrate_durable_cognition_signal(
                private_event(stale), dispatch_signal=stale
            ),
            DurableCognitionWithdrawal,
        )
        with bind_current_signal(stale):
            assert not await manager.task_store.update_status(
                task.id,
                TaskStatus(state=TaskState.TASK_STATE_WORKING),
                recipient_agent_id=manager.did,
                expected_state=TaskState.TASK_STATE_SUBMITTED,
            )
        assert (
            await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
        ).input_revision == 0
        followup = SendMessageRequest(
            tenant=manager.task_store.scope.tenant,
            message=new_text_message(
                "Do not replay this tool", role=Role.ROLE_USER, task_id=task.id
            ),
        )
        with pytest.raises(ValueError, match="does not accept"):
            await manager.continue_task(followup, creator_agent_id=manager.did)
    finally:
        feature.release.set()
        await asyncio.gather(*tuple(manager._execution_tasks))


async def test_old_sync_completion_is_withdrawn_before_private_or_public_cognition(
    manager,
):
    feature = provider(manager)
    feature.release.set()
    task = await manager.execute_skill("local-wake", "hold_local", {}, sync=True)
    candidate = await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
    stale = build_signal_for_completed_task(
        replace(candidate, cognition_wake_eligible=True), manager.did
    )
    recipient = owner(manager)
    assert await recipient.validate_cognition_signal_execution(stale) is not None
    event = replace(private_event(stale), source_event_id=task.id)
    assert isinstance(
        await recipient.rehydrate_durable_cognition_signal(
            event, dispatch_signal=stale
        ),
        DurableCognitionWithdrawal,
    )


@pytest.mark.parametrize("sync", [True, False])
async def test_actual_signal_ledger_recovery_keeps_only_owed_local_completion(
    manager, sync
):
    from kestrel_sovereign.signals import DurableConsumerRegistration
    from kestrel_sovereign.signals.sources.a2a import (
        DURABLE_COGNITION_CONSUMER_ID as COMPLETE_CONSUMER,
    )
    from tests.unit.test_a2a_native_input_fencing import (
        close_input_dispatcher,
        input_dispatcher,
    )

    agent, dispatcher, input_consumer = await input_dispatcher(
        manager, held=True, private=True
    )
    agent._pending_task_notifications = []
    await dispatcher.register_durable_consumer(
        DurableConsumerRegistration(
            consumer_id=COMPLETE_CONSUMER,
            source="a2a.task_complete",
            agent_id=agent.did,
            max_attempts=0,
        )
    )
    manager._on_task_submitted = agent._on_task_submitted
    manager._on_task_complete = agent._on_background_task_complete
    feature = provider(manager)
    execution = asyncio.create_task(
        manager.execute_skill("local-wake", "hold_local", {}, sync=sync)
    )
    try:
        async with asyncio.timeout(5):
            await feature.started.wait()
            await agent.reconcile_a2a_cognition_wakes()
            assert (
                await dispatcher.list_durable_deliveries(consumer_id=input_consumer)
                == []
            )
            assert (
                await dispatcher.list_durable_deliveries(consumer_id=COMPLETE_CONSUMER)
                == []
            )
            feature.release.set()
            task = await execution
            await manager.drain_execution_tasks()
            await asyncio.gather(*agent.tasks)
            # Both a repeated boot pass and immediate publication converge on
            # the same retained completion, without an input delivery.
            await agent.reconcile_a2a_cognition_wakes()
            await agent.reconcile_a2a_cognition_wakes()
        assert (
            await dispatcher.list_durable_deliveries(consumer_id=input_consumer) == []
        )
        deliveries = await dispatcher.list_durable_deliveries(
            consumer_id=COMPLETE_CONSUMER
        )
        assert len(deliveries) == (0 if sync else 1)
        if not sync:
            assert deliveries[0].event.source_event_id == task.id
        assert feature.effects == 1
    finally:
        feature.release.set()
        await execution
        await manager.drain_execution_tasks()
        await close_input_dispatcher(agent, dispatcher)


async def test_canceling_a_local_sync_invocation_does_not_emit_completion_cognition(
    manager,
):
    feature = provider(manager)
    completions = []
    manager._on_task_complete = completions.append
    execution = asyncio.create_task(
        manager.execute_skill("local-wake", "hold_local", {}, sync=True)
    )
    try:
        async with asyncio.timeout(3):
            await feature.started.wait()
        rows = await manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks")
        assert len(rows) == 1
        canceled = await manager.cancel_task(rows[0][0], agent_name=manager.did)
        assert canceled.status.state == TaskState.TASK_STATE_CANCELED
        assert completions == [], (
            "A synchronous local cancellation created a cognition completion"
        )
        assert await candidates(manager) == []
        feature.release.set()
        result = await execution
        assert result.status.state == TaskState.TASK_STATE_CANCELED
        assert feature.effects == 1
    finally:
        feature.release.set()
        await execution
