"""SQLite/PostgreSQL parity for owner-scoped delivery idempotency."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from kestrel_sovereign.features.delivery.queue import (
    DeliveryIdempotencyConflict,
    DeliveryIdempotencyTerminal,
    DeliveryQueue,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.interface import QueryError


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_delivery_enqueue_idempotency_backend_parity(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery:{uuid4().hex}"
    other_owner = f"did:test:delivery:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    other_queue = DeliveryQueue(database, other_owner)
    await queue._ensure_tables()

    try:
        request = (
            "email",
            "person@example.com",
            {"subject": "Check-in", "body": "Are you okay?"},
        )
        entry_ids = await asyncio.gather(
            *(
                queue.enqueue(
                    *request,
                    idempotency_key="workflow/run/stage/attempt",
                )
                for _ in range(12)
            )
        )

        assert len(set(entry_ids)) == 1
        row = await database.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (owner,),
        )
        assert row == (1,)

        assert (
            await queue.enqueue(
                *request,
                idempotency_key="workflow/run/stage/attempt",
            )
            == entry_ids[0]
        )
        with pytest.raises(DeliveryIdempotencyConflict):
            await queue.enqueue(
                "email",
                "person@example.com",
                {"subject": "Check-in", "body": "changed"},
                idempotency_key="workflow/run/stage/attempt",
            )

        other_id = await other_queue.enqueue(
            *request,
            idempotency_key="workflow/run/stage/attempt",
        )
        assert other_id != entry_ids[0]
    finally:
        await database.execute(
            "DELETE FROM delivery_queue WHERE agent_id IN (?, ?)",
            (owner, other_owner),
        )
        await database.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id IN (?, ?)",
            (owner, other_owner),
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_delivery_idempotency_lifecycle_backend_parity(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-lifecycle:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    await queue._ensure_tables()
    request = (
        "email",
        "person@example.com",
        {"body": "check in"},
    )

    try:
        live_id = await queue.enqueue(
            *request, idempotency_key="ordinary-ledger-delete"
        )
        await database.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id = ?",
            (owner,),
        )
        assert await database.fetchone(
            "SELECT id FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (live_id, owner),
        ) == (live_id,)

        dead_id = await queue.enqueue(
            "email",
            "dead@example.com",
            {"body": "dead"},
            max_retries=11,
            idempotency_key="dead-letter-replay",
        )
        await queue.move_to_dead_letter(dead_id, "terminal")
        with pytest.raises(DeliveryIdempotencyTerminal):
            await queue.enqueue(
                "email",
                "dead@example.com",
                {"body": "dead"},
                max_retries=11,
                idempotency_key="dead-letter-replay",
            )
        retry_results = await asyncio.gather(*(queue.retry(dead_id) for _ in range(8)))
        successful_retries = [result for result in retry_results if result["success"]]
        assert len(successful_retries) == 1
        retried = successful_retries[0]
        assert await database.fetchone(
            "SELECT max_retries FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (retried["entry_id"], owner),
        ) == (11,)
        assert await queue.enqueue(
            "email",
            "dead@example.com",
            {"body": "dead"},
            max_retries=11,
            idempotency_key="dead-letter-replay",
        ) == retried["entry_id"]

        delivered_id = await queue.enqueue(
            "email",
            "done@example.com",
            {"body": "done"},
            idempotency_key="purged-replay",
        )
        old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        await database.execute(
            """
            UPDATE delivery_queue SET status = ?, delivered_at = ?
            WHERE id = ? AND agent_id = ?
            """,
            ("delivered", old, delivered_id, owner),
        )
        assert await queue.purge_delivered(older_than_hours=24) == 1
        assert await queue.enqueue(
            "email",
            "done@example.com",
            {"body": "done"},
            idempotency_key="purged-replay",
        ) != delivered_id
    finally:
        await database.execute(
            "DELETE FROM delivery_dead_letter WHERE agent_id = ?", (owner,)
        )
        await database.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id = ?", (owner,)
        )
        await database.execute(
            "DELETE FROM delivery_queue WHERE agent_id = ?", (owner,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_idempotent_failure_uses_real_savepoint(db_backend):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL savepoint contract")

    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-savepoint:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    function_name = f"reject_delivery_{uuid4().hex}"
    trigger_name = f"reject_delivery_{uuid4().hex}"
    await queue._ensure_tables()
    await database.execute(
        f"""
        CREATE FUNCTION {function_name}() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.channel_type = 'reject-savepoint' THEN
                RAISE EXCEPTION 'injected delivery insert failure';
            END IF;
            RETURN NEW;
        END
        $$
        """
    )
    await database.execute(
        f"""
        CREATE TRIGGER {trigger_name}
        BEFORE INSERT ON delivery_queue
        FOR EACH ROW EXECUTE FUNCTION {function_name}()
        """
    )

    try:
        async with database.transaction(immediate=True):
            with pytest.raises(QueryError, match="injected delivery"):
                await queue.enqueue(
                    "reject-savepoint",
                    "person@example.com",
                    {"body": "hello"},
                    idempotency_key="savepoint-failure",
                )
            # This write proves the caller's outer transaction was not poisoned.
            await database.execute(
                """
                INSERT INTO delivery_queue
                    (id, agent_id, channel_type, recipient, content_json,
                     content_hash, status, attempts, max_retries,
                     next_retry_at, last_error, created_at, delivered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, ?, NULL, ?, NULL)
                """,
                (
                    str(uuid4()),
                    owner,
                    "sentinel",
                    "person@example.com",
                    "{}",
                    "sentinel",
                    "pending",
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        assert await database.fetchone(
            "SELECT COUNT(*) FROM delivery_idempotency WHERE agent_id = ?",
            (owner,),
        ) == (0,)
        assert await database.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (owner,),
        ) == (1,)
    finally:
        await database.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON delivery_queue")
        await database.execute(f"DROP FUNCTION IF EXISTS {function_name}()")
        await database.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id = ?", (owner,)
        )
        await database.execute(
            "DELETE FROM delivery_queue WHERE agent_id = ?", (owner,)
        )
