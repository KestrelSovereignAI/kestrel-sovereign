"""Core cognition cleanup and original receipts on owned PostgreSQL.

The protocol agent is a declared stand-in; dispatcher Tasks, native leases,
COMMITs, separate pools and shutdown are real. No provider/genesis acceptance
is inferred from these source-specific custody cases.
"""

import asyncio
import contextvars
import os

import asyncpg
import pytest

from kestrel_sdk.signals import SignalResult, Status
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import ExecutionCustody, execution_commit_outcome, is_execution_control_error
from kestrel_sovereign.signals import OrderedLockManager, SignalDispatcher, SignalLogStore, SourceRegistry
from kestrel_sovereign.signals.durable import DurableConsumerRegistration, DurableSignalStore
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg
from tests.integration.test_execution_custody_review18_pg import join_owned
from tests.unit.test_durable_signal_delivery import _Agent
from tests.unit.test_sqlite_cognition_shutdown_terminal import (
    cognition_registration as cognition_registration,
    cognition_signal,
    cognition_source as cognition_source,
)
from tests.utils.original_failure import OriginalFailure

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.native_postgres]


@pytest.fixture
async def cognition_lease(native_pg, cognition_source, cognition_registration):
    backend, observer, schema = native_pg
    await observer.execute(f'SET search_path TO "{schema}"')
    agent = _Agent("original")
    registry = SourceRegistry()
    registry.register(cognition_registration)
    log = SignalLogStore(backend)
    await log.initialize()
    dispatcher = SignalDispatcher(agent=agent, registry=registry,
                                  lock_manager=OrderedLockManager(), store=log)
    dispatcher._durable_delivery_owner = "dispatcher:original"
    await dispatcher.initialize_durable_delivery()
    await dispatcher._stop_runtime_owner_heartbeat()
    ledger = dispatcher.durable_store
    try:
        await ledger.register_consumer(DurableConsumerRegistration(
            agent_id=agent.did, consumer_id=cognition_source.DURABLE_COGNITION_CONSUMER_ID,
            source=cognition_source.SOURCE_NAME, max_attempts=0,
        ))
        signal = cognition_signal(cognition_source, agent.did)
        cognition_registration.schema(dict(signal.payload))
        await ledger.persist_signal(signal, agent_id=agent.did, source_event_id="original", retention_days=1)
        delivery = await ledger.claim_delivery(
            agent_id=agent.did, consumer_id=cognition_source.DURABLE_COGNITION_CONSUMER_ID,
            executor_id=dispatcher._durable_delivery_owner,
        )
        assert delivery is not None and delivery.lease_token
        yield backend, observer, schema, agent, dispatcher, delivery
    finally:
        await join_owned(agent, dispatcher)


async def test_core_cognition_native_cleanup_supports_original_registered_source(cognition_lease):
    backend, _observer, _schema, _agent, _dispatcher, delivery = cognition_lease
    assert await backend.retain_cognition_cleanup_owner(
        agent_id="original", owner_id="dispatcher:original",
    ), "the original A2A cognition owner lost its cleanup heartbeat"
    assert await backend.fail_cognition_delivery(
        agent_id="original", consumer_id=delivery.consumer_id,
        delivery_id=delivery.delivery_id, owner_id="dispatcher:original",
        lease_token=delivery.lease_token, error="original cognition control lost",
    )
    assert await backend.fetch_one(
        "SELECT status, lease_owner, lease_token, acknowledged_at FROM durable_signal_deliveries"
    ) == ("failed", None, None, None)


@pytest.mark.parametrize("mismatch", ["agent_id", "consumer_id", "delivery_id", "owner_id", "lease_token"])
async def test_core_cognition_native_cleanup_refuses_other_capabilities(cognition_lease, mismatch):
    backend, observer, _schema, _agent, _dispatcher, delivery = cognition_lease
    before = await observer.fetchrow("SELECT * FROM durable_signal_deliveries")
    arguments = dict(agent_id="original", consumer_id=delivery.consumer_id,
                     delivery_id=delivery.delivery_id, owner_id="dispatcher:original",
                     lease_token=delivery.lease_token, error="unrelated cleanup")
    arguments[mismatch] = "dispatcher:unrelated" if mismatch == "owner_id" else "unrelated"
    if mismatch == "consumer_id":
        with pytest.raises(ValueError, match="managed consumer"):
            await backend.fail_cognition_delivery(**arguments)
    else:
        assert not await backend.fail_cognition_delivery(**arguments)
    assert await observer.fetchrow("SELECT * FROM durable_signal_deliveries") == before


async def release(ledger, delivery, kind):
    arguments = dict(agent_id="original", consumer_id=delivery.consumer_id,
                     delivery_id=delivery.delivery_id, lease_token=delivery.lease_token)
    if kind == "hold":
        return await ledger.release_delivery_for_hold(**arguments)
    if kind == "nack":
        return await ledger.nack_delivery(**arguments, error="original retry")
    return await ledger.release_managed_delivery_after_task(
        **arguments, owner_id="dispatcher:original", error="original retained retry",
    )


@pytest.mark.parametrize("kind", ["nack", "managed", "hold"])
async def test_original_retry_commit_lost_ack_excludes_fresh_runtime_and_settles(cognition_lease, monkeypatch, kind):
    backend, observer, schema, agent, dispatcher, delivery = cognition_lease
    ledger = dispatcher.durable_store
    scope = ExecutionCustody(GenerationFence())
    agent._execution_custody = backend._execution_custody = scope
    pool = await asyncpg.create_pool(os.environ["TEST_POSTGRES_URL"], min_size=0, max_size=1,
                                    server_settings={"search_path": schema})
    successor = PostgresBackend.from_pool(pool)
    successor._execution_custody = ExecutionCustody(GenerationFence())
    fresh = DurableSignalStore(successor)
    original_exit = asyncpg.transaction.Transaction.__aexit__
    original_release = contextvars.ContextVar("original_retry_release", default=False)
    injected, claims = False, []
    terminal_entered, release_terminal = asyncio.Event(), asyncio.Event()
    terminalize = backend.fail_cognition_delivery
    operation = None

    async def lost_ack(transaction, error_type, error, traceback):
        nonlocal injected
        result = await original_exit(transaction, error_type, error, traceback)
        if error_type is None and original_release.get() and not injected:
            injected = True
            claims.append(await fresh.claim_delivery_for_event(
                agent_id=agent.did, consumer_id=delivery.consumer_id,
                executor_id="dispatcher:successor", event_id=delivery.event_id,
            ))
            raise OSError("actual original retry COMMIT acknowledgement lost")
        return result

    async def hold_exact_terminal(**kwargs):
        terminal_entered.set()
        await release_terminal.wait()
        return await terminalize(**kwargs)

    async def release_and_settle():
        token = original_release.set(True)
        try:
            if kind == "managed":
                await dispatcher._release_retained_durable_cognition_task(delivery)
            else:
                try:
                    await release(ledger, delivery, kind)
                except BaseException as error:
                    await dispatcher._terminalize_failed_cognition(delivery, error)
                    raise
        finally:
            original_release.reset(token)

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", lost_ack)
    monkeypatch.setattr(backend, "fail_cognition_delivery", hold_exact_terminal)
    try:
        operation = create_owned_task(release_and_settle(), retain_originals=True)
        await asyncio.wait_for(terminal_entered.wait(), 5)
        assert injected and claims == [None], "fresh runtime stole an unacknowledged original retry"
        assert tuple(await observer.fetchrow(
            "SELECT status,lease_owner,lease_token FROM durable_signal_deliveries"
        )) == ("retry", "dispatcher:original", delivery.lease_token)
        assert dispatcher._retained_cognition_control_debt
        assert await backend.retain_cognition_cleanup_owner(agent_id=agent.did, owner_id="dispatcher:original")
        assert await fresh.claim_delivery(agent_id=agent.did, consumer_id=delivery.consumer_id,
                                          executor_id="dispatcher:successor") is None
        release_terminal.set()
        outcome = await await_owned_task(operation)
        assert execution_commit_outcome(outcome.error) == "unknown"
        for source in OriginalFailure.capture_sources(outcome.error):
            source.assert_retained(outcome.error)
        assert tuple(await observer.fetchrow(
            "SELECT status,lease_owner,lease_token,acknowledged_at FROM durable_signal_deliveries"
        )) == ("failed", None, None, None)
        assert not dispatcher._retained_cognition_control_debt
        with pytest.raises(BaseException) as denied:
            await backend.execute("INSERT INTO effects VALUES (2, 'forbidden fresh work')")
        assert is_execution_control_error(denied.value)
        assert await observer.fetchval("SELECT COUNT(*) FROM effects") == 0
    finally:
        release_terminal.set()
        if operation is not None:
            await await_owned_task(operation)
        await successor.close()
        await pool.close()


async def test_retained_core_cognition_keeps_owner_until_original_body_and_terminal_receipt(
    cognition_lease, monkeypatch,
):
    from kestrel_sovereign.signals import dispatcher as dispatcher_module

    backend, observer, _schema, agent, dispatcher, delivery = cognition_lease
    entered, cancelled, release_body = asyncio.Event(), asyncio.Event(), asyncio.Event()
    body = None
    monkeypatch.setattr(dispatcher_module, "_DURABLE_COGNITION_CANCELLATION_GRACE", 0.01)

    async def original_cognition():
        await backend.execute("INSERT INTO effects VALUES (1, 'original cognition effect')")
        entered.set()
        while not release_body.is_set():
            try:
                await release_body.wait()
            except asyncio.CancelledError:
                cancelled.set()
        return SignalResult(signal_id=delivery.event_id, status=Status.OK,
                            mode=delivery.event.mode, duration_ms=0)

    try:
        async with backend.advisory_locks([(3822, 1)]) as lease:
            scope = ExecutionCustody(GenerationFence(lease=lease))
            agent._execution_custody = backend._execution_custody = scope
            body = agent._track_background_task(original_cognition(), name="original-a2a-cognition")
            await asyncio.wait_for(entered.wait(), 5)
        try:
            lease.require_live()
        except BaseException as original:
            assert is_execution_control_error(original)
            source = OriginalFailure.capture(original)
            await dispatcher._settle_cognition_control(delivery, original, body)
        await asyncio.wait_for(cancelled.wait(), 5)
        assert not body.done() and body in dispatcher._retained_durable_cognition_tasks
        source.assert_retained(dispatcher._retained_cognition_control_debt[delivery.delivery_id][1])
        assert await dispatcher.shutdown_durable_delivery() is False
        assert dispatcher.durable_shutdown_owner_fenced
        await dispatcher._retain_runtime_owner_cleanup_liveness()
        assert await observer.fetchval("SELECT stopped_at FROM durable_signal_runtime_owners") is None
        assert await observer.fetchval("SELECT status FROM durable_signal_deliveries") == "leased"
        release_body.set()
        await asyncio.wait_for(dispatcher.wait_for_durable_shutdown_release(), 10)
        assert tuple(await observer.fetchrow(
            "SELECT status,lease_owner,lease_token,acknowledged_at FROM durable_signal_deliveries"
        )) == ("failed", None, None, None)
        assert await observer.fetchval("SELECT stopped_at IS NOT NULL FROM durable_signal_runtime_owners")
        assert await observer.fetchval("SELECT COUNT(*) FROM effects") == 1
        assert not dispatcher._retained_cognition_control_debt
    finally:
        release_body.set()
        if body is not None:
            await await_owned_task(body)
        await asyncio.wait_for(dispatcher.wait_for_durable_shutdown_release(), 10)
