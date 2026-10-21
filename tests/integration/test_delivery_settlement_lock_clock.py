"""Original delivery ACK/NACK must validate expiry AFTER native serialization."""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm import service as service_module
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review13_pg import _seed_owned_deliveries

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("method", ["ack", "nack"])
@pytest.mark.parametrize("expires_while_locked", [False, True])
async def test_public_delivery_settlement_samples_after_original_row_lock(
    native_pg, tmp_path, monkeypatch, method, expires_while_locked
):
    backend, observer, schema = native_pg
    monkeypatch.setattr(service_module, "load_section", lambda _: {})
    service = service_module.LLMService(
        usage_db=AsyncDatabase(backend), agent_data_dir=tmp_path
    )
    agent = KestrelAgent(
        did="original",
        storage_path=str(tmp_path / "unused.db"),
        llm_service=service,
        allowed_features=[],
    )
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=SourceRegistry(),
        lock_manager=OrderedLockManager(),
        store=SignalLogStore(backend),
    )
    settling, transaction = None, None
    try:
        await dispatcher.initialize_durable_delivery()
        await _seed_owned_deliveries(backend, observer, schema)
        # Seed after initial orphan recovery; keep the ORIGINAL lease owner
        # alive rather than accidentally testing recovery of the seed's old
        # one-hour heartbeat instead of settlement of a live token.
        await observer.execute(
            "UPDATE durable_signal_runtime_owners SET heartbeat_at=clock_timestamp()"
        )
        assert not agent._runtime_publication_ready.is_set()
        duration = 0.5 if expires_while_locked else 30
        expiry = await observer.fetchval(
            "UPDATE durable_signal_deliveries SET lease_expires_at=clock_timestamp()+$1*interval '1 second' "
            "WHERE delivery_id='delivery1' RETURNING lease_expires_at",
            duration,
        )
        transaction = observer.transaction()
        await transaction.start()
        await observer.fetchrow(
            "SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery1' FOR UPDATE"
        )
        args = dict(
            consumer_id=DURABLE_COGNITION_CONSUMER_ID,
            delivery_id="delivery1",
            lease_token="token1",
        )
        operation = (
            dispatcher.ack_durable_delivery(**args)
            if method == "ack"
            else dispatcher.nack_durable_delivery(
                **args, error="native settlement proof"
            )
        )
        settling = create_owned_task(operation)
        async with asyncio.timeout(5):
            while True:
                await observer.execute("SELECT pg_stat_clear_snapshot()")
                if await observer.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid<>pg_backend_pid() "
                    "AND wait_event_type='Lock' AND query LIKE '%durable_signal_deliveries%')"
                ):
                    break
                if settling.done():
                    pytest.fail(
                        f"settlement never reached actual delivery lock: {await await_owned_task(settling)}"
                    )
                await asyncio.sleep(0.01)
        assert not settling.done()
        if expires_while_locked:
            async with asyncio.timeout(5):
                while await observer.fetchval("SELECT clock_timestamp()") <= expiry:
                    await asyncio.sleep(0.01)
        await transaction.commit()
        transaction = None
        outcome = await await_owned_task(settling)
        assert outcome.error is None, outcome.error
        row = await observer.fetchrow(
            "SELECT lease_expires_at,lease_token,status FROM durable_signal_deliveries WHERE delivery_id='delivery1'"
        )
        if expires_while_locked:
            assert not outcome.result, (
                "expired original token settled after waiting on its native lock"
            )
            assert row["status"] == "leased" and row["lease_token"] == "token1"
            assert row["lease_expires_at"] == expiry
        else:
            assert outcome.result
            assert row["status"] == ("acknowledged" if method == "ack" else "retry")
        assert not agent._runtime_publication_ready.is_set()
    finally:
        if transaction is not None:
            await transaction.rollback()
        if settling is not None:
            await await_owned_task(settling)
        await dispatcher.shutdown_durable_delivery()
        await await_owned_task(create_owned_task(agent._shutdown_background_tasks()))
        await service.close()
        await service.wait_for_close_completion()
