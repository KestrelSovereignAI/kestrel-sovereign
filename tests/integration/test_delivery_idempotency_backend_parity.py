"""SQLite/PostgreSQL parity for owner-scoped delivery idempotency."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from kestrel_sovereign.features.delivery.queue import (
    DeliveryIdempotencyConflict,
    DeliveryQueue,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


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
