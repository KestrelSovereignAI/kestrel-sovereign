"""Native failure receipts and renewal loss retain original control evidence."""

import asyncio
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.execution_custody import ExecutionCustody
from kestrel_sovereign.signals import OrderedLockManager, SignalDispatcher, SignalLogStore, SourceRegistry
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from kestrel_sdk.signals import SignalResult, Status
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as _native_pg
from tests.integration.test_execution_custody_review13_pg import _seed_owned_deliveries
from tests.unit.test_durable_signal_delivery import _Agent, _channel_signal, _registration
from tests.unit.test_execution_custody_review15 import control_error

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
native_pg = _native_pg


async def setup_delivery(native_pg):
    backend, controller, schema = native_pg
    await _seed_owned_deliveries(backend, controller, schema)
    agent = _Agent("original")
    dispatcher = SignalDispatcher(agent=agent, registry=SourceRegistry(),
                                  lock_manager=OrderedLockManager(), store=SignalLogStore(backend))
    dispatcher._durable_delivery_owner = "dispatcher:original"
    await dispatcher.initialize_durable_delivery()
    await dispatcher._stop_runtime_owner_heartbeat()
    await controller.execute("UPDATE durable_signal_deliveries SET lease_expires_at=NOW()+INTERVAL '10 minutes' WHERE delivery_id='delivery1'")
    delivery = await dispatcher._durable_store.get_delivery(
        agent_id=agent.did, consumer_id=DURABLE_COGNITION_CONSUMER_ID, delivery_id="delivery1",
    )
    assert delivery is not None
    signal = _channel_signal(agent.did, "review18")
    dispatcher._held_signal_result = AsyncMock(return_value=None)
    dispatcher._signal_for_durable_retry = AsyncMock(return_value=signal)
    dispatcher._begin_durable_cognition_execution = AsyncMock()
    dispatcher._finalize_deferred_durable_outcome = lambda *args, **kwargs: None
    return agent, dispatcher, delivery, signal


async def route(dispatcher, delivery, signal):
    return await dispatcher._route_durable_cognition_delivery(
        signal, _registration(dispatcher._agent), time.monotonic(),
        persisted_event_id=delivery.event_id, consumer_id=delivery.consumer_id,
        durable_admission=None, durable_created=False, use_live_signal=True, claimed_delivery=delivery,
    )


async def join_owned(agent, dispatcher):
    await dispatcher._stop_runtime_owner_heartbeat()
    await asyncio.gather(*tuple(dispatcher._retained_durable_cognition_cleanup_tasks), return_exceptions=True)
    for task in agent.tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*agent.tasks, return_exceptions=True)


@pytest.mark.parametrize("receipt", ["nack", "fallback", "committed-nack"])
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
async def test_native_failure_receipt_control_settles_or_retains_exact_delivery(native_pg, receipt, control, carrier):
    backend, controller, _schema = native_pg
    agent, dispatcher, delivery, signal = await setup_delivery(native_pg)
    foreign_before = await controller.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery2'")
    error = control_error(control, carrier)
    dispatcher._route_after_durable_persistence = AsyncMock(return_value=SignalResult(
        signal_id=signal.id, status=Status.FAILED, mode=signal.mode, duration_ms=0, error="ordinary failed cognition",
    ))
    original_nack = dispatcher.nack_durable_delivery
    scope = ExecutionCustody(GenerationFence())
    agent._execution_custody = backend._execution_custody = scope

    async def fail_receipt(**kwargs):
        if receipt == "committed-nack":
            await original_nack(**kwargs)
        scope.revoke("original receipt lost authority")
        raise error

    if receipt == "fallback":
        dispatcher.nack_durable_delivery = AsyncMock(return_value=None)
        dispatcher.release_durable_delivery_after_task = fail_receipt
    else:
        dispatcher.nack_durable_delivery = fail_receipt
    try:
        with pytest.raises(BaseException) as caught:
            async with asyncio.timeout(0.5):
                await route(dispatcher, delivery, signal)
        assert caught.value is error
        state = await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'")
        if receipt == "committed-nack":
            assert state == "retry"
            assert dispatcher._retained_cognition_control_debt == {delivery.delivery_id: (delivery, error)}
            with pytest.raises(BaseException) as blocked:
                await dispatcher.shutdown_durable_delivery()
            assert blocked.value is error
            assert await controller.fetchval("SELECT stopped_at IS NULL FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'")
        else:
            assert state == "failed"
            assert not dispatcher._retained_cognition_control_debt
        assert await controller.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery2'") == foreign_before
    finally:
        continuation = dispatcher._fenced_durable_shutdown_completion
        if continuation is not None:
            continuation.cancel()
            await asyncio.gather(continuation, return_exceptions=True)
        await join_owned(agent, dispatcher)


@pytest.mark.parametrize("stubborn", [False, True])
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
@pytest.mark.parametrize("late_commit", [False, True])
async def test_native_renewal_control_survives_cooperative_or_retained_cognition(native_pg, monkeypatch, stubborn, control, carrier, late_commit):
    backend, controller, _schema = native_pg
    agent, dispatcher, delivery, signal = await setup_delivery(native_pg)
    delivery = replace(delivery, lease_expires_at=datetime.now(timezone.utc)+timedelta(seconds=0.03))
    error = control_error(control, carrier)
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    renewing = asyncio.Event()
    scope = ExecutionCustody(GenerationFence())
    agent._execution_custody = backend._execution_custody = scope
    monkeypatch.setattr("kestrel_sovereign.signals.dispatcher._DURABLE_COGNITION_CANCELLATION_GRACE", 0.01)

    async def turn(*args):
        entered.set()
        if late_commit:
            await renewing.wait()
            await backend.execute("INSERT INTO effects VALUES (1, 'committed before renewal join')")
            return SignalResult(signal_id=signal.id, status=Status.OK, mode=signal.mode, duration_ms=0)
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            if not stubborn:
                raise
            await release.wait()
        return SignalResult(signal_id=signal.id, status=Status.FAILED, mode=signal.mode, duration_ms=0, error="late ordinary failure")

    async def renew(**kwargs):
        await entered.wait()
        if late_commit:
            renewing.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
        scope.revoke("renewal original authority lost")
        raise error

    dispatcher._route_after_durable_persistence = turn
    dispatcher.renew_durable_delivery_lease = renew
    try:
        with pytest.raises(BaseException) as caught:
            async with asyncio.timeout(0.5):
                await route(dispatcher, delivery, signal)
        assert caught.value is error
        assert cancelled.is_set() is (not late_commit)
        if stubborn and not late_commit:
            assert dispatcher._retained_cognition_control_debt == {delivery.delivery_id: (delivery, error)}
            assert len(dispatcher._retained_durable_cognition_tasks) == 1
            assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "leased"
            release.set()
            await asyncio.gather(*tuple(dispatcher._retained_durable_cognition_tasks), return_exceptions=True)
            await asyncio.sleep(0)
            await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        assert not dispatcher._retained_cognition_control_debt
        assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "failed"
        assert await controller.fetchval("SELECT count(*) FROM effects") == int(late_commit)
    finally:
        release.set()
        await asyncio.gather(*tuple(dispatcher._retained_durable_cognition_tasks), return_exceptions=True)
        await asyncio.sleep(0)
        await join_owned(agent, dispatcher)
