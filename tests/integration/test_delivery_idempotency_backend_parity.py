"""SQLite/PostgreSQL parity for owner-scoped delivery idempotency."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from kestrel_sovereign.features.delivery.queue import (
    DeliveryIdempotencyConflict,
    DeliveryIdempotencyStateError,
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
async def test_distinct_replay_keys_share_content_dedup_gate(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-dedup:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    await queue._ensure_tables()

    try:
        entry_ids = await asyncio.gather(
            *(
                queue.enqueue(
                    "email",
                    "person@example.com",
                    {"subject": "Check-in", "body": "Are you okay?"},
                    idempotency_key=f"workflow-action-{index}",
                )
                for index in range(12)
            )
        )

        assert len(set(entry_ids)) == 1
        assert await database.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (owner,),
        ) == (1,)
        assert await database.fetchone(
            "SELECT COUNT(*) FROM delivery_idempotency WHERE agent_id = ?",
            (owner,),
        ) == (12,)
    finally:
        await database.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id = ?", (owner,)
        )
        await database.execute(
            "DELETE FROM delivery_queue WHERE agent_id = ?", (owner,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_keyed_and_plain_enqueues_share_content_gate(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-mixed:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    await queue._ensure_tables()

    try:
        entry_ids = await asyncio.gather(
            *(
                queue.enqueue(
                    "email",
                    "mixed@example.com",
                    {"body": "same"},
                    idempotency_key=(f"mixed-{index}" if index % 2 else None),
                )
                for index in range(12)
            )
        )

        assert len(set(entry_ids)) == 1
        assert await database.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (owner,),
        ) == (1,)
    finally:
        await database.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id = ?", (owner,)
        )
        await database.execute(
            "DELETE FROM delivery_queue WHERE agent_id = ?", (owner,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_stale_claim_repair_preserves_effective_retry_policy(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-stale-policy:{uuid4().hex}"
    original_queue = DeliveryQueue(database, owner, max_retries=5)
    restarted_queue = DeliveryQueue(database, owner, max_retries=99)
    await original_queue._ensure_tables()

    try:
        original_id = await original_queue.enqueue(
            "email",
            "stale-policy@example.com",
            {"body": "same"},
            idempotency_key="stale-policy",
        )
        await database.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, owner),
        )

        repaired_id = await restarted_queue.enqueue(
            "email",
            "stale-policy@example.com",
            {"body": "same"},
            idempotency_key="stale-policy",
        )

        assert repaired_id != original_id
        assert await database.fetchone(
            "SELECT max_retries FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (repaired_id, owner),
        ) == (5,)
    finally:
        await database.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id = ?", (owner,)
        )
        await database.execute(
            "DELETE FROM delivery_queue WHERE agent_id = ?", (owner,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_dead_letter_retry_preserves_legacy_content_hash(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-retry-hash:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    await queue._ensure_tables()
    payload = {"z": 1, "a": 2}

    try:
        original_id = await queue.enqueue(
            "email",
            "rolling-hash@example.com",
            payload,
            idempotency_key="rolling-hash",
        )
        original_hash = await database.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, owner),
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")

        retried = await queue.retry(original_id)

        assert retried["success"] is True
        assert await database.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (retried["entry_id"], owner),
        ) == original_hash
        assert await queue.enqueue(
            "email", "rolling-hash@example.com", payload
        ) == retried["entry_id"]
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
async def test_rolling_retry_metadata_recovers_from_replay_ledger(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-ledger-retry:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    await queue._ensure_tables()
    payload = {"z": 1, "a": 2}

    try:
        original_id = await queue.enqueue(
            "email",
            "ledger-retry@example.com",
            payload,
            max_retries=11,
            idempotency_key="ledger-retry",
        )
        original_hash = await database.fetchone(
            "SELECT content_hash FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, owner),
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")
        await database.execute(
            """
            UPDATE delivery_dead_letter
            SET max_retries = NULL, legacy_content_hash = NULL
            WHERE original_id = ? AND agent_id = ?
            """,
            (original_id, owner),
        )

        retried = await DeliveryQueue(database, owner, max_retries=99).retry(
            original_id
        )

        assert retried["success"] is True
        assert await database.fetchone(
            """
            SELECT max_retries, content_hash FROM delivery_queue
            WHERE id = ? AND agent_id = ?
            """,
            (retried["entry_id"], owner),
        ) == (11, original_hash[0])
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
async def test_unlinked_rolling_retry_fails_closed_backend_parity(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-unlinked-retry:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    await queue._ensure_tables()
    request = (
        "email",
        "unlinked-retry@example.com",
        {"z": 1, "a": 2},
    )

    try:
        original_id = await queue.enqueue(
            *request, idempotency_key="unlinked-retry"
        )
        original = await database.fetchone(
            """
            SELECT content_json, canonical_content_hash, max_retries
            FROM delivery_queue WHERE id = ? AND agent_id = ?
            """,
            (original_id, owner),
        )
        claim_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        candidate_time = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        ).isoformat()
        await database.execute(
            """
            UPDATE delivery_idempotency SET created_at = ?
            WHERE agent_id = ?
            """,
            (claim_time, owner),
        )
        await database.execute(
            "DELETE FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (original_id, owner),
        )
        await database.execute(
            """
            INSERT INTO delivery_queue
                (id, agent_id, channel_type, recipient, content_json,
                 content_hash, canonical_content_hash, status, attempts,
                 max_retries, next_retry_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL, 'pending', 0, ?, ?, ?)
            """,
            (
                f"rolling-{uuid4().hex}",
                owner,
                request[0],
                request[1],
                original[0],
                original[1],
                original[2],
                candidate_time,
                candidate_time,
            ),
        )

        with pytest.raises(DeliveryIdempotencyStateError, match="unlinked"):
            await queue.enqueue(*request, idempotency_key="unlinked-retry")
        assert await database.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?", (owner,)
        ) == (1,)
    finally:
        await database.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id = ?", (owner,)
        )
        await database.execute(
            "DELETE FROM delivery_queue WHERE agent_id = ?", (owner,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_keyed_adoption_requires_matching_delivery_semantics(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-semantics:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    await queue._ensure_tables()

    try:
        content = {"body": "same"}
        plain_channel = await queue.enqueue(
            "webhook", "channel@example.com", content
        )
        keyed_channel = await queue.enqueue(
            "email",
            "channel@example.com",
            content,
            idempotency_key="channel-key",
        )
        plain_policy = await queue.enqueue(
            "email", "policy@example.com", content
        )
        keyed_policy = await queue.enqueue(
            "email",
            "policy@example.com",
            content,
            max_retries=99,
            idempotency_key="policy-key",
        )

        assert keyed_channel != plain_channel
        assert keyed_policy != plain_policy
        assert await database.fetchone(
            "SELECT channel_type FROM delivery_queue WHERE id = ?",
            (keyed_channel,),
        ) == ("email",)
        assert await database.fetchone(
            "SELECT max_retries FROM delivery_queue WHERE id = ?",
            (keyed_policy,),
        ) == (99,)
    finally:
        await database.execute(
            "DELETE FROM delivery_idempotency WHERE agent_id = ?", (owner,)
        )
        await database.execute(
            "DELETE FROM delivery_queue WHERE agent_id = ?", (owner,)
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_dead_letter_retry_reconciles_residual_live_row(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-dual-state:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    await queue._ensure_tables()

    try:
        original_id = await queue.enqueue(
            "email",
            "dual-state@example.com",
            {"body": "same"},
            idempotency_key="dual-state",
        )
        dead_letter_id = str(uuid4())
        await database.execute(
            """
            INSERT INTO delivery_dead_letter
                (id, original_id, agent_id, channel_type, recipient,
                 content_json, error, attempts, created_at, max_retries)
            SELECT ?, id, agent_id, channel_type, recipient, content_json,
                   ?, attempts, created_at, max_retries
            FROM delivery_queue WHERE id = ? AND agent_id = ?
            """,
            (dead_letter_id, "injected transition failure", original_id, owner),
        )

        retried = await queue.retry(original_id)

        assert retried["success"] is True
        assert retried["entry_id"] != original_id
        assert await database.fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?",
            (owner,),
        ) == (1,)
        assert await database.fetchone(
            "SELECT COUNT(*) FROM delivery_dead_letter WHERE agent_id = ?",
            (owner,),
        ) == (0,)
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
async def test_enqueue_repairs_runtime_legacy_writer_hash(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-rolling:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    await queue._ensure_tables()

    try:
        original_id = await queue.enqueue(
            "email",
            "rolling@example.com",
            {"subject": "hello", "body": "world"},
        )
        await database.execute(
            "UPDATE delivery_queue SET canonical_content_hash = NULL WHERE id = ?",
            (original_id,),
        )

        assert await queue.enqueue(
            "email",
            "rolling@example.com",
            {"body": "world", "subject": "hello"},
        ) == original_id
    finally:
        await database.execute(
            "DELETE FROM delivery_queue WHERE agent_id = ?", (owner,)
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
