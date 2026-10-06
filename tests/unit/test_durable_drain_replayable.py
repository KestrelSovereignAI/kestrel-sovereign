"""The durable cognition drain claims only rows it can rebuild (#3392).

A payload-elided row stores only a privacy marker and no caller; the caller
is bound to the live envelope. Unless the agent declares that its rehydrate
hook rebuilds the row's source from an authoritative store (A2A tasks), only
the provider's redelivery of the same update can run the row. The drain used to claim such
rows anyway, fail caller recovery, and spend an attempt once per retry delay
until a redelivery arrived. A rehydratable A2A row with its payload sidecar
still attached failed the same way, because the drain decided "elided" by the
marker the sidecar had replaced.

Every test runs a real dispatcher over a real SQLite ledger.
"""

from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.hold import EffectiveHoldState
from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.signals import (
    ACKNOWLEDGED,
    RETRY,
    DurableAdmissionDisposition,
    DurableConsumerRegistration,
)
from kestrel_sovereign.signals.sources.channels import (
    DURABLE_COGNITION_CONSUMER_ID,
    DURABLE_COGNITION_MARKER,
    DURABLE_COGNITION_MARKER_VALUE,
)
from tests.unit.test_durable_signal_delivery import (
    _a2a_submitted_dispatcher,
    _channel_dispatcher,
    _channel_signal,
    _close,
    _held_state,
    _HoldSnapshots,
)

NOT_HELD = EffectiveHoldState(host=None, agent=None)


def _channel_consumer(agent_id: str) -> DurableConsumerRegistration:
    return DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent_id,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=0,
    )


async def _drain_once(dispatcher, consumer_id: str) -> None:
    """Run one drain pass to completion, including any rerun it requested.

    A drain that keeps re-arming itself for work it never claims is the
    defect, so a bounded number of passes is part of the assertion.
    """

    dispatcher._start_durable_cognition_drain(consumer_id)
    for _ in range(20):
        drainer = dispatcher._durable_cognition_drainers.get(consumer_id)
        if drainer is None:
            return
        await asyncio.wait_for(asyncio.shield(drainer), timeout=5.0)
        await asyncio.sleep(0)
    pytest.fail("the durable cognition drain kept re-arming itself")


async def _held_channel_message(dispatcher, did: str, name: str):
    """Admit one channel message while held; its row is returned to RETRY."""

    handle = await dispatcher.enqueue_durable_cognition(
        _channel_signal(did, name),
        source_event_id=f"telegram:update:{name}",
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
    )
    admission = await asyncio.wait_for(handle.wait_for_durable_admission(), timeout=2.0)
    assert admission.disposition is DurableAdmissionDisposition.HELD
    await asyncio.wait_for(handle.wait(), timeout=2.0)


@pytest.mark.asyncio
async def test_drain_leaves_an_elided_row_only_a_redelivery_can_run(tmp_path):
    did = "did:agent:drain-replayable:channel"
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / "channel.db", did)
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = _channel_consumer(did)
    hold = _HoldSnapshots(_held_state(did))
    agent._hold_store = hold
    agent.process_input = AsyncMock(return_value="ran")
    try:
        await dispatcher.register_durable_consumer(consumer)
        await dispatcher.start_durable_cognition_consumer(consumer.consumer_id)
        await _held_channel_message(dispatcher, did, "elided")
        hold.snapshots[:] = [NOT_HELD]

        await _drain_once(dispatcher, consumer.consumer_id)

        [waiting] = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        assert waiting.status == RETRY
        assert waiting.attempts == 0
        assert waiting.last_error == "hold_deferred"
        agent.process_input.assert_not_awaited()
        # Nothing re-arms the drain for a row it will never claim.
        assert consumer.consumer_id not in dispatcher._durable_cognition_drain_timers

        # The provider's redelivery carries the live envelope and runs it once.
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(did, "elided"),
            source_event_id="telegram:update:elided",
            consumer_id=consumer.consumer_id,
        )
        assert (
            await asyncio.wait_for(handle.wait_for_durable_admission(), timeout=2.0)
        ).disposition is DurableAdmissionDisposition.DUPLICATE
        await asyncio.wait_for(handle.wait(), timeout=2.0)
        [ran] = await dispatcher.list_durable_deliveries(consumer_id=consumer.consumer_id)
        assert ran.status == ACKNOWLEDGED
        assert ran.attempts == 1
        agent.process_input.assert_awaited_once()
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_rows_only_a_redelivery_can_run_do_not_crowd_out_executable_work(
    tmp_path,
):
    """More than one listing page of waiting rows precedes an executable one."""

    did = "did:agent:drain-replayable:window"
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / "window.db", did)
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = _channel_consumer(did)
    hold = _HoldSnapshots(_held_state(did))
    agent._hold_store = hold
    prompts: list[str] = []

    async def record_turn(prompt: str):
        prompts.append(prompt)
        return "ran"

    agent.process_input = record_turn
    try:
        await dispatcher.register_durable_consumer(consumer)
        for index in range(101):
            await _held_channel_message(dispatcher, did, f"elided-{index}")
        # The operator lifts the volatile mode; this row keeps its envelope.
        agent.privacy_config = get_privacy_preset("normal")
        await _held_channel_message(dispatcher, did, "stored")
        hold.snapshots[:] = [NOT_HELD]

        await dispatcher.start_durable_cognition_consumer(consumer.consumer_id)
        await _drain_once(dispatcher, consumer.consumer_id)

        assert len(prompts) == 1
        assert "message stored" in prompts[0]
        rows = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id, limit=200
        )
        assert sum(row.status == ACKNOWLEDGED for row in rows) == 1
        assert all(row.attempts == 0 for row in rows if row.status == RETRY)
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


async def _held_elided_a2a_wake(tmp_path, name: str, rehydrate):
    from kestrel_sovereign.a2a.types import TaskState
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        DURABLE_COGNITION_CONSUMER_ID as A2A_CONSUMER,
    )
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        build_signal_for_submitted_task,
    )

    did = f"did:agent:drain-replayable:{name}"
    task = SimpleNamespace(
        id=f"task-{name}",
        sessionId="session",
        metadata={},
        history=[],
        status=SimpleNamespace(state=TaskState.SUBMITTED),
    )
    consumer = DurableConsumerRegistration(
        consumer_id=A2A_CONSUMER,
        source="a2a.task_submitted",
        agent_id=did,
        max_attempts=0,
    )
    backend, agent, dispatcher = await _a2a_submitted_dispatcher(
        tmp_path / f"{name}.db", did
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    hold = _HoldSnapshots(_held_state(did))
    agent._hold_store = hold
    agent.process_input = AsyncMock(return_value="ran")
    agent.task_manager = SimpleNamespace(
        get_task_for_recipient=AsyncMock(return_value=task)
    )
    agent.rehydrate_durable_cognition_signal = rehydrate(agent)
    agent.durable_rehydratable_sources = frozenset({"a2a.task_submitted"})
    await dispatcher.register_durable_consumer(consumer)
    handle = await dispatcher.enqueue_durable_cognition(
        build_signal_for_submitted_task(task, target_agent=did),
        source_event_id=task.id,
        consumer_id=A2A_CONSUMER,
    )
    assert (
        await asyncio.wait_for(handle.wait_for_durable_admission(), timeout=2.0)
    ).disposition is DurableAdmissionDisposition.HELD
    await asyncio.wait_for(handle.wait(), timeout=2.0)
    [row] = await dispatcher.list_durable_deliveries(consumer_id=A2A_CONSUMER)
    # The Hold deferral returned the reservation and kept its payload sidecar.
    assert row.status == RETRY
    assert row.delivery_id in dispatcher._transient_durable_handoffs
    hold.snapshots[:] = [NOT_HELD]
    return backend, agent, dispatcher, A2A_CONSUMER, task


@pytest.mark.asyncio
async def test_drain_rehydrates_a_held_elided_a2a_wake_with_its_sidecar_attached(
    tmp_path,
):
    from kestrel_sovereign.agent.event_manager import EventManagerMixin

    backend, agent, dispatcher, consumer_id, task = await _held_elided_a2a_wake(
        tmp_path,
        "a2a-sidecar",
        lambda agent: MethodType(
            EventManagerMixin.rehydrate_durable_cognition_signal, agent
        ),
    )
    try:
        await dispatcher.start_durable_cognition_consumer(consumer_id)
        await _drain_once(dispatcher, consumer_id)

        [ran] = await dispatcher.list_durable_deliveries(consumer_id=consumer_id)
        assert ran.status == ACKNOWLEDGED
        assert ran.attempts == 1
        agent.process_input.assert_awaited_once()
        assert task.id in agent.process_input.await_args.args[0]
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_drain_still_claims_a_row_whose_store_fails_to_answer(tmp_path):
    """A declared source whose store fails is still claimed and reported."""

    def failing_rehydrate(_agent):
        async def rehydrate(_event, *, dispatch_signal):
            raise RuntimeError("task store unavailable")

        return rehydrate

    backend, agent, dispatcher, consumer_id, _task = await _held_elided_a2a_wake(
        tmp_path, "a2a-store-fails", failing_rehydrate
    )
    try:
        await dispatcher.start_durable_cognition_consumer(consumer_id)
        await _drain_once(dispatcher, consumer_id)

        [failed] = await dispatcher.list_durable_deliveries(consumer_id=consumer_id)
        assert failed.status == RETRY
        assert failed.attempts == 1
        assert failed.last_error == "Durable cognition caller recovery failed"
        agent.process_input.assert_not_awaited()
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_a_declared_source_without_a_rehydrate_hook_is_not_drained(tmp_path):
    """The declaration names what the hook rebuilds; with no hook, nothing is."""

    did = "did:agent:drain-replayable:no-hook"
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / "no-hook.db", did)
    agent.privacy_config = get_privacy_preset("ephemeral")
    agent.durable_rehydratable_sources = frozenset({"channel.message"})
    consumer = _channel_consumer(did)
    hold = _HoldSnapshots(_held_state(did))
    agent._hold_store = hold
    agent.process_input = AsyncMock(return_value="ran")
    try:
        await dispatcher.register_durable_consumer(consumer)
        await dispatcher.start_durable_cognition_consumer(consumer.consumer_id)
        await _held_channel_message(dispatcher, did, "no-hook")
        hold.snapshots[:] = [NOT_HELD]

        await _drain_once(dispatcher, consumer.consumer_id)

        [waiting] = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        assert waiting.status == RETRY
        assert waiting.attempts == 0
        agent.process_input.assert_not_awaited()
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)
