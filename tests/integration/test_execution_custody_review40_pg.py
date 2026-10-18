"""Real PostgreSQL commit acknowledgements through public dispatch lanes."""
import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    bind_execution_custody_snapshot,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_durable_signal_delivery import _Agent
from tests.unit.test_signals_dispatcher import _action_reg, _signal

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("ack", ["error", "cancel"])
async def test_actual_pg_commit_ack_loss_preserves_both_public_receipts(native_pg, monkeypatch, ack):
    backend, controller, schema = native_pg
    await controller.execute(f'SET search_path TO "{schema}"')
    store = SignalLogStore(backend)
    await store.initialize()
    agent, registry = _Agent("original"), SourceRegistry()
    actions = []
    async def action(payload):
        actions.append(payload)
    registry.register(_action_reg("pg_ack", handler=action))
    dispatcher = SignalDispatcher(agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=store)
    await dispatcher.initialize_durable_delivery()
    await dispatcher._stop_runtime_owner_heartbeat()
    scope = ExecutionCustody(GenerationFence())
    backend._execution_custody = scope
    native_exit = asyncpg.transaction.Transaction.__aexit__
    acknowledgements = []
    async def native_commit_then_lost_ack(transaction, error_type, error, traceback):
        result = await native_exit(transaction, error_type, error, traceback)
        if error_type is None and not acknowledgements:
            acknowledgements.append(True)
            if ack == "cancel":
                raise asyncio.CancelledError("native PostgreSQL COMMIT acknowledgement lost")
            raise OSError("native PostgreSQL COMMIT acknowledgement lost")
        return result
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_commit_then_lost_ack)
    handle = None
    try:
        with bind_execution_custody_snapshot((scope,)):
            handle = await dispatcher.enqueue_signal(_signal("pg_ack"), source_event_id="pg-native-ack")
        errors = []
        for wait in (handle.wait, handle.wait_for_durable_admission):
            with pytest.raises(BaseException) as caught:
                await wait()
            errors.append(caught.value)
            assert execution_commit_outcome(caught.value) == "unknown"
        assert errors[0] is errors[1]
        assert acknowledgements == [True]
        assert not actions, "uncertain commit was misclassified as safe handler admission"
        assert await controller.fetchval("SELECT COUNT(*) FROM durable_signal_events WHERE source_event_id='pg-native-ack'") == 1
    finally:
        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_exit)
        for task in agent.tasks:
            await await_owned_task(task)
        # All native work is joined; no stale authority is required for the
        # disposable fixture's ordinary lifecycle bookkeeping.
        backend._execution_custody = None
        try:
            await dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)
            await dispatcher._stop_runtime_owner_heartbeat()
