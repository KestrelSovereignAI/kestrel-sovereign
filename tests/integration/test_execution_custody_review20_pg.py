"""Native retry receipt custody and cancelled successful terminalization."""

import asyncio
import os

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import ExecutionCustody, execution_commit_outcome
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as _native_pg
from tests.integration.test_execution_custody_review18_pg import setup_delivery as _setup_delivery, join_owned
from tests.unit.test_execution_custody_review15 import control_error

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def setup_delivery(native_pg):
    result = await _setup_delivery(native_pg)
    # The reused fixture also seeds unrelated initial handoffs. Leave their
    # identities intact, but do not let the polling claim select them instead
    # of the one retry whose ownership this test is measuring.
    await native_pg[1].execute("UPDATE durable_signal_deliveries SET status='failed' WHERE delivery_id <> 'delivery1'")
    return result


async def release(store, agent, delivery, kind):
    identities = dict(agent_id=agent.did, consumer_id=delivery.consumer_id,
                      delivery_id=delivery.delivery_id, lease_token=delivery.lease_token)
    if kind == "hold":
        return await store.release_delivery_for_hold(**identities)
    if kind == "nack":
        return await store.nack_delivery(**identities, error="ordinary retry")
    return await store.release_managed_delivery_after_task(
        **identities, owner_id="dispatcher:original", error="ordinary retained retry",
    )


async def claim(store, agent, delivery, owner, exact):
    kwargs = dict(agent_id=agent.did, consumer_id=delivery.consumer_id, executor_id=owner)
    if exact:
        return await store.claim_delivery_for_event(**kwargs, event_id=delivery.event_id)
    return await store.claim_delivery(**kwargs)


@pytest.mark.parametrize("kind", ["nack", "managed", "hold"])
@pytest.mark.parametrize("exact", [False, True])
async def test_native_retry_retains_original_capability_and_excludes_live_sibling(native_pg, kind, exact):
    backend, controller, _ = native_pg
    agent, dispatcher, delivery, _signal = await setup_delivery(native_pg)
    store = dispatcher._durable_store
    try:
        assert await release(store, agent, delivery, kind)
        row = await controller.fetchrow("SELECT status, lease_owner, lease_token, lease_expires_at FROM durable_signal_deliveries WHERE delivery_id='delivery1'")
        assert tuple(row) == ("retry", "dispatcher:original", delivery.lease_token, None)
        assert await backend.retain_cognition_cleanup_owner(agent_id=agent.did, owner_id="dispatcher:original")
        assert await claim(store, agent, delivery, "dispatcher:sibling", exact) is None
        original_retry = await claim(store, agent, delivery, "dispatcher:original", exact)
        assert original_retry is not None
        assert original_retry.lease_token != delivery.lease_token
        # Old cleanup is never allowed to terminalize even its own newer lease.
        assert not await backend.fail_cognition_delivery(
            agent_id=agent.did, consumer_id=delivery.consumer_id, delivery_id=delivery.delivery_id,
            owner_id="dispatcher:original", lease_token=delivery.lease_token, error="old cleanup",
        )
        assert await controller.fetchval("SELECT lease_token FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == original_retry.lease_token
    finally:
        await join_owned(agent, dispatcher)


@pytest.mark.parametrize("owner_state", ["stopped", "stale", "missing"])
@pytest.mark.parametrize("kind", ["nack", "managed", "hold"])
@pytest.mark.parametrize("exact", [False, True])
async def test_native_retry_recovers_unavailable_original_owner_without_extra_slots(native_pg, owner_state, kind, exact):
    backend, controller, _ = native_pg
    agent, dispatcher, delivery, _signal = await setup_delivery(native_pg)
    try:
        assert await release(dispatcher._durable_store, agent, delivery, kind)
        if owner_state == "missing":
            await controller.execute("DELETE FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'")
        elif owner_state == "stopped":
            await controller.execute("UPDATE durable_signal_runtime_owners SET stopped_at=NOW() WHERE owner_id='dispatcher:original'")
        else:
            await controller.execute("UPDATE durable_signal_runtime_owners SET heartbeat_at=NOW()-INTERVAL '10 minutes' WHERE owner_id='dispatcher:original'")
        recovered = await claim(dispatcher._durable_store, agent, delivery, "dispatcher:sibling", exact)
        assert recovered is not None
        assert recovered.lease_token != delivery.lease_token
        assert not await backend.fail_cognition_delivery(
            agent_id=agent.did, consumer_id=delivery.consumer_id, delivery_id=delivery.delivery_id,
            owner_id="dispatcher:original", lease_token=delivery.lease_token, error="late dead cleanup",
        )
        assert await controller.fetchval("SELECT lease_owner FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "dispatcher:sibling"
    finally:
        await join_owned(agent, dispatcher)


@pytest.mark.parametrize("kind", ["nack", "hold"])
@pytest.mark.parametrize("consumer_scope,owner", [
    ("other-consumer", "dispatcher:original"),
    ("other-consumer", "public-worker"),
    ("canonical", "public-worker"),
])
async def test_unmanaged_or_other_consumer_retry_keeps_public_transfer_contract(native_pg, kind, consumer_scope, owner):
    _backend, controller, _ = native_pg
    agent, dispatcher, delivery, _signal = await setup_delivery(native_pg)
    from dataclasses import replace
    try:
        if consumer_scope != "canonical":
            await controller.execute("INSERT INTO durable_signal_consumers (agent_id,consumer_id,source,max_attempts,lease_seconds,active) VALUES ('original','other-consumer','channel.message',0,10,TRUE)")
            await controller.execute("UPDATE durable_signal_deliveries SET consumer_id='other-consumer' WHERE delivery_id='delivery1'")
            delivery = replace(delivery, consumer_id="other-consumer")
        await controller.execute("UPDATE durable_signal_deliveries SET lease_owner=$1 WHERE delivery_id='delivery1'", owner)
        assert await release(dispatcher._durable_store, agent, delivery, kind)
        assert tuple(await controller.fetchrow("SELECT status,lease_owner,lease_token FROM durable_signal_deliveries WHERE delivery_id='delivery1'")) == ("retry", None, None)
        assert await claim(dispatcher._durable_store, agent, delivery, "public-successor", True) is not None
    finally:
        await join_owned(agent, dispatcher)


@pytest.mark.parametrize("kind", ["nack", "managed"])
@pytest.mark.parametrize("terminal", ["failed", "ackable", "budget"])
async def test_terminal_managed_retry_release_clears_capabilities(native_pg, kind, terminal):
    _backend, controller, _ = native_pg
    agent, dispatcher, delivery, _signal = await setup_delivery(native_pg)
    try:
        if terminal == "budget":
            await controller.execute("UPDATE durable_signal_deliveries SET max_attempts=1,attempts=1 WHERE delivery_id='delivery1'")
        method = (dispatcher._durable_store.nack_delivery if kind == "nack"
                  else dispatcher._durable_store.release_managed_delivery_after_task)
        kwargs = dict(agent_id=agent.did, consumer_id=delivery.consumer_id,
                      delivery_id=delivery.delivery_id, lease_token=delivery.lease_token,
                      error="terminal release", terminal=terminal != "budget",
                      terminal_ackable=terminal == "ackable")
        if kind == "managed":
            kwargs["owner_id"] = "dispatcher:original"
        result = await method(**kwargs)
        assert result.status == ("terminal_ackable" if terminal == "ackable" else "failed")
        assert result.lease_owner is result.lease_token is None
        assert await claim(dispatcher._durable_store, agent, delivery, "dispatcher:sibling", True) is None
    finally:
        await join_owned(agent, dispatcher)


@pytest.mark.parametrize("kind", ["nack", "managed", "hold"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_actual_committed_release_lost_ack_cannot_be_reclaimed_by_fresh_runtime(native_pg, monkeypatch, kind, cleanup_fails):
    backend, controller, schema = native_pg
    agent, dispatcher, delivery, _signal = await setup_delivery(native_pg)
    scope = ExecutionCustody(GenerationFence())
    agent._execution_custody = backend._execution_custody = scope
    native_exit = asyncpg.transaction.Transaction.__aexit__
    pool = await asyncpg.create_pool(os.environ["TEST_POSTGRES_URL"], min_size=0, max_size=1, server_settings={"search_path": schema})
    successor = PostgresBackend.from_pool(pool)
    successor._execution_custody = ExecutionCustody(GenerationFence())
    store = DurableSignalStore(successor)
    injected = False
    pre_error_claims = []

    async def lost_ack(transaction, error_type, error, traceback):
        nonlocal injected
        result = await native_exit(transaction, error_type, error, traceback)
        if error_type is None and not injected:
            injected = True
            # A different real pool/authority tries before the original
            # dispatcher even learns the ACK was lost or records local debt.
            pre_error_claims.append(await claim(store, agent, delivery, "dispatcher:successor", True))
            raise OSError("actual native release COMMIT lost its ACK")
        return result

    async def unavailable_cleanup(**kwargs):
        raise ConnectionError("exact cleanup temporarily unavailable")

    original_cleanup = backend.fail_cognition_delivery
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", lost_ack)
    if cleanup_fails:
        backend.fail_cognition_delivery = unavailable_cleanup
    try:
        with pytest.raises(BaseException) as caught:
            if kind == "managed":
                await dispatcher._release_retained_durable_cognition_task(delivery)
            else:
                try:
                    await release(dispatcher._durable_store, agent, delivery, kind)
                except BaseException as error:
                    await dispatcher._terminalize_failed_cognition(delivery, error)
                    raise
        assert execution_commit_outcome(caught.value) == "unknown"
        assert pre_error_claims == [None]
        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_exit)
        assert await claim(store, agent, delivery, "dispatcher:successor", False) is None
        assert await claim(store, agent, delivery, "dispatcher:successor", True) is None
        if cleanup_fails:
            assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "retry"
            assert dispatcher._retained_cognition_control_debt[delivery.delivery_id][1] is caught.value
            assert await backend.retain_cognition_cleanup_owner(agent_id=agent.did, owner_id="dispatcher:original")
            backend.fail_cognition_delivery = original_cleanup
            await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "failed"
        assert not dispatcher._retained_cognition_control_debt
    finally:
        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", native_exit)
        backend.fail_cognition_delivery = original_cleanup
        await join_owned(agent, dispatcher)
        await successor.close()
        await pool.close()


@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
@pytest.mark.parametrize("repeat", [False, True])
async def test_cancelled_successful_native_terminalizer_clears_confirmed_debt(native_pg, carrier, repeat):
    backend, controller, _ = native_pg
    agent, dispatcher, delivery, _signal = await setup_delivery(native_pg)
    original = control_error("unknown", carrier)
    scope = ExecutionCustody(GenerationFence())
    agent._execution_custody = backend._execution_custody = scope
    scope.revoke("original runtime retired before terminal cleanup")
    entered, finish = asyncio.Event(), asyncio.Event()
    native_fail = backend.fail_cognition_delivery

    async def delayed_fail(**kwargs):
        entered.set()
        await finish.wait()
        return await native_fail(**kwargs)

    backend.fail_cognition_delivery = delayed_fail
    task = asyncio.create_task(dispatcher._terminalize_failed_cognition(delivery, original))
    try:
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        if repeat:
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(BaseException) as caught:
            await task
        assert caught.value is original
        assert await controller.fetchval("SELECT status FROM durable_signal_deliveries WHERE delivery_id='delivery1'") == "failed"
        assert not dispatcher._retained_cognition_control_debt
        await dispatcher._drain_retained_durable_cognition_cleanup_tasks()
        await dispatcher.shutdown_durable_delivery()
        assert await controller.fetchval("SELECT stopped_at IS NOT NULL FROM durable_signal_runtime_owners WHERE owner_id='dispatcher:original'")
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        backend.fail_cognition_delivery = native_fail
        await join_owned(agent, dispatcher)
