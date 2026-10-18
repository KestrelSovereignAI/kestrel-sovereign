"""SQLite's deferred transaction must lock before sampling renewal time."""

import asyncio
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_while_locked", [False, True])
async def test_native_sqlite_delivery_renewal_cannot_revive_expired_token(tmp_path, monkeypatch, expires_while_locked):
    path = tmp_path / "delivery.db"
    backend = SQLiteBackend(str(path))
    await backend.connect()
    locked = sqlite3.connect(path)
    renewing = None
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(seconds=.5 if expires_while_locked else 30)
        await backend.execute(
            "INSERT INTO durable_signal_consumers(agent_id,consumer_id,source,max_attempts,lease_seconds,active) "
            "VALUES ('original','consumer','channel.message',0,10,1)"
        )
        await backend.execute(
            "INSERT INTO durable_signal_events(event_id,agent_id,target_agent,source,kind,mode,visibility,payload,urgency,causation_chain,arrived_at,retention_until,source_sequence) "
            "VALUES ('event','original','original','channel.message','message','cognition','internal','{}','normal','[]',?,?,1)",
            (store.to_timestamp_param(now), store.to_timestamp_param(now + timedelta(days=1))),
        )
        await backend.execute(
            "INSERT INTO durable_signal_deliveries(delivery_id,event_id,agent_id,consumer_id,status,max_attempts,lease_owner,lease_token,lease_expires_at) "
            "VALUES ('delivery','event','original','consumer','leased',0,'owner','token',?)",
            (store.to_timestamp_param(expiry),),
        )
        locked.execute("BEGIN IMMEDIATE")
        entered = asyncio.Event()
        execute = backend.execute

        async def native_execute(query, params=()):
            if "UPDATE durable_signal_deliveries" in query:
                entered.set()
            return await execute(query, params)

        monkeypatch.setattr(backend, "execute", native_execute)
        renewing = create_owned_task(store.renew_delivery_lease(
            agent_id="original", consumer_id="consumer", delivery_id="delivery", lease_token="token",
        ))
        await asyncio.wait_for(entered.wait(), 2)
        assert not renewing.done()
        if expires_while_locked:
            async with asyncio.timeout(3):
                while datetime.now(timezone.utc) <= expiry:
                    await asyncio.sleep(.01)
        else:
            await asyncio.sleep(.05)
        released_at = datetime.now(timezone.utc)
        locked.commit()
        outcome = await await_owned_task(renewing)
        assert outcome.error is None, outcome.error
        row = await backend.fetch_one("SELECT lease_expires_at,lease_token,status FROM durable_signal_deliveries WHERE delivery_id='delivery'")
        actual_expiry = store.from_timestamp_field(row[0])
        if expires_while_locked:
            assert outcome.result is None, "SQLite renewal revived its expired original token"
            assert actual_expiry == expiry
        else:
            assert outcome.result is not None
            assert actual_expiry >= released_at + timedelta(seconds=10)
        assert row[1:] == ("token", "leased")
    finally:
        locked.rollback()
        locked.close()
        if renewing is not None:
            await await_owned_task(renewing)
        await backend.close()
