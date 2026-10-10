"""Native lost-CAS and first Stop admission preserve unresolved ownership."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kestrel_sovereign.agent.invocation import bind_async_invocation
from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError, ExecutionCustody, is_execution_control_error
from kestrel_sovereign.signals import OrderedLockManager, SignalDispatcher, SignalLogStore, SourceRegistry
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg, GenerationFence
from tests.integration.test_execution_custody_review13_pg import _seed_owned_deliveries
from tests.unit.test_durable_signal_delivery import _Agent

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
native_pg = _native_pg


@pytest.mark.parametrize("state", ["retry", "successor"])
async def test_native_lost_terminal_and_liveness_cas_keep_cleanup_debt_and_successor(native_pg, state):
    backend, controller, schema = native_pg
    await _seed_owned_deliveries(backend, controller, schema)
    agent = _Agent("original")
    dispatcher = SignalDispatcher(agent=agent, registry=SourceRegistry(),
                                  lock_manager=OrderedLockManager(), store=SignalLogStore(backend))
    dispatcher._durable_delivery_owner = "dispatcher:original"
    await dispatcher.initialize_durable_delivery()
    dispatcher._discard_transient_durable_handoff = Mock()
    error = ExecutionCommitOutcomeError("unknown")
    delivery = SimpleNamespace(delivery_id="delivery1", consumer_id=DURABLE_COGNITION_CONSUMER_ID, lease_token="token1")
    if state == "retry":
        await controller.execute("UPDATE durable_signal_deliveries SET status='retry',lease_owner=NULL,lease_token=NULL WHERE delivery_id='delivery1'")
    else:
        await controller.execute("UPDATE durable_signal_deliveries SET lease_owner='dispatcher:successor',lease_token='successor-token' WHERE delivery_id='delivery1'")
    before = await controller.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery1'")
    scope = ExecutionCustody(GenerationFence())
    agent._execution_custody = backend._execution_custody = scope
    scope.preserve_commit_uncertainty("unknown")
    dispatcher._durable_shutdown = True
    dispatcher._durable_shutdown_owner_fenced = True
    try:
        with pytest.raises(ExecutionCommitOutcomeError) as caught:
            await dispatcher._release_retained_durable_cognition_task(delivery, error)
        assert caught.value is error
        with pytest.raises(ExecutionCommitOutcomeError) as heartbeat:
            await dispatcher._heartbeat_runtime_owner()
        assert heartbeat.value is error
        with pytest.raises(ExecutionCommitOutcomeError):
            await dispatcher.shutdown_durable_delivery()
        assert dispatcher.durable_shutdown_owner_fenced
        assert dispatcher._retained_cognition_control_debt == {"delivery1": (delivery, error)}
        dispatcher._discard_transient_durable_handoff.assert_not_called()
        assert await controller.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == before
        assert await controller.fetchval("SELECT stopped_at IS NULL FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'")
        assert await controller.fetchval("SELECT heartbeat_at < NOW()-INTERVAL '30 minutes' FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:successor'")
    finally:
        # No success is fabricated to retire debt. The disposable fixture owns
        # its original backend/schema; explicitly join only this test's timers.
        await dispatcher._stop_runtime_owner_heartbeat()
        continuation = dispatcher._fenced_durable_shutdown_completion
        if continuation is not None:
            continuation.cancel()
            await asyncio.gather(continuation, return_exceptions=True)
        for task in agent.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def test_native_first_admission_poll_control_cannot_be_settled_completed(native_pg):
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.stop import DistributedInvocationRegistry, DistributedInvocationStore
    from tests.unit.test_distributed_stop_invocations import _ReplicaAgent

    backend, controller, schema = native_pg
    database = AsyncDatabase(backend)
    store = DistributedInvocationStore(database)
    await store.ensure_schema()
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    recorded = []
    original_register = store.register

    async def register_then_latch(**kwargs):
        admitted = await original_register(**kwargs)
        recorded.append(kwargs["generation_id"])
        scope.preserve_commit_uncertainty("unknown")
        return admitted

    store.register = register_then_latch
    registry = DistributedInvocationRegistry(store)

    class Agent(_ReplicaAgent):
        _execution_custody = scope

        @bind_async_invocation("request_id", track_request_lifecycle=True)
        async def turn(self, request_id=None):
            raise AssertionError("uncertain first admission may not enter cognition")

    agent = Agent("original-agent")
    registry.attach(agent)
    try:
        with pytest.raises(BaseException) as caught:
            await agent.turn(request_id="original-admission")
        assert is_execution_control_error(caught.value)
        await asyncio.gather(*tuple(registry._cleanup_tasks))
        assert len(recorded) == 1
        assert await controller.fetchval(f'SELECT generation_id FROM "{schema}".stop_unresolved_invocations WHERE generation_id=$1', recorded[0]) == recorded[0]
        assert await controller.fetchval(f'SELECT count(*) FROM "{schema}".stop_active_invocations WHERE generation_id=$1', recorded[0]) == 0
    finally:
        await registry.close()
