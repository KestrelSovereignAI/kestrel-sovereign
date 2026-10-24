"""Original PostgreSQL claim replies survive dispatcher close and retirement."""

from __future__ import annotations

import asyncio

import asyncpg
import pytest
from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Trust,
)

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    execution_commit_outcome,
    retire_process_runtime_custody,
)
from kestrel_sovereign.signals import (
    DurableConsumerRegistration,
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.unit.test_durable_signal_delivery import _Agent

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
native_pg = _native_pg


@pytest.mark.parametrize("retire_runtime", [False, True])
async def test_shutdown_joins_the_same_native_postgres_claim_ack(native_pg, monkeypatch, tmp_path, retire_runtime):
    backend, controller, _schema = native_pg
    await controller.execute(f'SET search_path TO "{_schema}"')
    agent = _Agent("original")
    prompt = tmp_path / "native-shutdown.md"
    prompt.write_text("Original native shutdown claim")
    registry = SourceRegistry()
    registry.register(SourceRegistration(
        name="native.shutdown", schema=dict, default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}), trust=Trust.TRUSTED,
        prompt_template=prompt,
        log_redaction=RedactionPolicy(summarize=lambda payload: "<shutdown probe>"),
        retention_days=1,
    ))
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    dispatcher = SignalDispatcher(agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=log_store)
    closing = None
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        await dispatcher.register_durable_consumer(DurableConsumerRegistration(
            consumer_id="native-shutdown", source="native.shutdown", agent_id=agent.did,
        ))
        ledger = dispatcher.durable_store
        signal = Signal(source="native.shutdown", kind="claim", mode=SignalMode.COGNITION,
                        payload={}, caller="did:kestrel:sender", target_agent=agent.did)
        await ledger.persist_signal(
            signal,
            agent_id=agent.did, source_event_id="original-native-shutdown-claim", retention_days=1,
            caller_identity_factory=lambda: dispatcher._protect_durable_caller_identity(signal),
        )
        assert await dispatcher._drainable_durable_deliveries("native-shutdown")
        scope = ExecutionCustody(ProcessRuntimeExecutionFence(agent.did)) if retire_runtime else None
        backend._execution_custody = scope
        native_exit = asyncpg.transaction.Transaction.__aexit__

        async def hold_original_ack(transaction, error_type, error, traceback):
            result = await native_exit(transaction, error_type, error, traceback)
            # The original native exit has its own shielded Task. Its copied
            # transaction context identifies the exact drainer that admitted
            # this connection; current_task() is the native exit owner itself.
            origin = backend._txn_conn_var.get()
            if error_type is None and origin is not None and origin[0] in set(dispatcher._durable_cognition_drainers.values()):
                entered.set()
                await release.wait()
            return result

        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", hold_original_ack)
        await dispatcher.start_durable_cognition_consumer("native-shutdown")
        await asyncio.wait_for(entered.wait(), 5)
        drainer = dispatcher._durable_cognition_drainers["native-shutdown"]
        # The same await-free lifecycle ordering used by Agent.shutdown.
        dispatcher.shutdown()
        retire_process_runtime_custody(scope)
        closing = asyncio.create_task(dispatcher.shutdown_durable_delivery())
        await asyncio.sleep(0)
        assert drainer.cancelling() == 0 and not drainer.done()
        assert not closing.done()
        release.set()
        await asyncio.gather(closing, return_exceptions=True)
        if retire_runtime:
            # Retiring authority after an acknowledged COMMIT remains a known
            # committed denial on the exact original owner, never unknown.
            # This dispatcher embedding has no Agent completion callback;
            # full Agent shutdown separately collects that owner's control.
            error = drainer.exception()
            assert error is not None and execution_commit_outcome(error) == "committed"
        else:
            error = closing.exception() or dispatcher._outcome_log_control_error
            assert error is None
            drainer.result()
        row = await controller.fetchrow(
            "SELECT status, attempts, last_error FROM durable_signal_deliveries WHERE consumer_id='native-shutdown'",
        )
        assert row["status"] == "leased"
        assert row["attempts"] == 1
        assert "admission pending" in row["last_error"]
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await dispatcher._stop_runtime_owner_heartbeat()
        await asyncio.gather(*tuple(agent.tasks), return_exceptions=True)
