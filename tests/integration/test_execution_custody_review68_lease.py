"""Native row-lock expiry cannot revive a Core delivery lease."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.signals.sources.channels import DURABLE_COGNITION_CONSUMER_ID
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review13_pg import _seed_owned_deliveries

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("expires_while_locked", [False, True])
async def test_native_delivery_renewal_samples_time_after_its_row_lock(native_pg, expires_while_locked):
    backend, observer, schema = native_pg
    await _seed_owned_deliveries(backend, observer, schema)
    duration = .5 if expires_while_locked else 30
    expiry = await observer.fetchval(
        "UPDATE durable_signal_deliveries SET lease_expires_at=clock_timestamp()+$1*interval '1 second' "
        "WHERE delivery_id='delivery1' RETURNING lease_expires_at", duration,
    )
    transaction = observer.transaction()
    await transaction.start()
    await observer.fetchrow("SELECT * FROM durable_signal_deliveries WHERE delivery_id='delivery1' FOR UPDATE")
    renewing = create_owned_task(DurableSignalStore(backend).renew_delivery_lease(
        agent_id="original", consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        delivery_id="delivery1", lease_token="token1",
    ))
    try:
        async with asyncio.timeout(5):
            while True:
                # PostgreSQL caches activity snapshots inside this lock-holding
                # transaction; refresh before inspecting the other session.
                await observer.execute("SELECT pg_stat_clear_snapshot()")
                if await observer.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid<>pg_backend_pid() "
                    "AND datname=current_database() "
                    "AND wait_event_type='Lock' AND query LIKE '%durable_signal_deliveries%')"
                ):
                    break
                if renewing.done():
                    pytest.fail(f"renewal did not wait on its delivery: {(await await_owned_task(renewing))}")
                await asyncio.sleep(.01)
        assert not renewing.done(), "renewal must actually await the native delivery lock"
        if expires_while_locked:
            async with asyncio.timeout(5):
                while await observer.fetchval("SELECT clock_timestamp()") <= expiry:
                    await asyncio.sleep(.01)
        released_at = datetime.now(timezone.utc)
        await transaction.commit()
        transaction = None
        outcome = await await_owned_task(renewing)
        assert outcome.error is None, outcome.error
        row = await observer.fetchrow("SELECT lease_expires_at,lease_token,status FROM durable_signal_deliveries WHERE delivery_id='delivery1'")
        if expires_while_locked:
            assert outcome.result is None, "renewal revived the expired original delivery token"
            assert row["lease_expires_at"] == expiry
        else:
            assert outcome.result is not None
            assert row["lease_expires_at"] >= released_at + timedelta(seconds=10)
        assert row["lease_token"] == "token1" and row["status"] == "leased"
    finally:
        if transaction is not None:
            await transaction.rollback()
        await await_owned_task(renewing)
