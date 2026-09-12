"""SQLite/PostgreSQL parity for owner-scoped delivery idempotency."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from kestrel_sovereign.features.delivery.queue import (
    DeliveryIdempotencyConflict,
    DeliveryIdempotencyStateError,
    DeliveryIdempotencyTerminal,
    DeliveryQueue,
)
from kestrel_sovereign.features.delivery.models import QueueEntry
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
async def test_dead_letter_retry_prefers_attached_ledger_policy(db_backend):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-retry-policy:{uuid4().hex}"
    queue = DeliveryQueue(database, owner, max_retries=2)
    await queue._ensure_tables()

    try:
        original_id = await queue.enqueue(
            "email",
            "rolling-policy@example.com",
            {"body": "same"},
            idempotency_key="rolling-policy",
        )
        await queue.move_to_dead_letter(original_id, "provider rejected")
        # Early prerelease schemas supplied DEFAULT 5 when a rolling old
        # writer omitted this column. The attached replay ledger is the
        # durable request policy and must win over that ambiguous value.
        await database.execute(
            "UPDATE delivery_dead_letter SET max_retries = 5 "
            "WHERE original_id = ? AND agent_id = ?",
            (original_id, owner),
        )

        retried = await DeliveryQueue(
            database, owner, max_retries=99
        ).retry(original_id)

        assert retried["success"] is True
        assert await database.fetchone(
            "SELECT max_retries FROM delivery_queue WHERE id = ? AND agent_id = ?",
            (retried["entry_id"], owner),
        ) == (2,)
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
async def test_replay_cannot_miss_concurrent_dead_letter_commit(
    db_backend, monkeypatch
):
    database = AsyncDatabase(db_backend)
    owner = f"did:test:delivery-dead-letter-race:{uuid4().hex}"
    queue = DeliveryQueue(database, owner)
    await queue._ensure_tables()
    request = (
        "email",
        "dead-letter-race@example.com",
        {"body": "terminal"},
    )

    try:
        original_id = await queue.enqueue(
            *request, idempotency_key="dead-letter-race"
        )
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        await database.execute(
            "UPDATE delivery_queue SET created_at = ? WHERE id = ?",
            (old, original_id),
        )

        if db_backend.backend_type == "sqlite":
            # BEGIN IMMEDIATE serializes the two writers. PostgreSQL's
            # statement snapshots are the backend-specific race under test.
            await queue.move_to_dead_letter(original_id, "provider rejected")
            with pytest.raises(DeliveryIdempotencyTerminal):
                await queue.enqueue(*request, idempotency_key="dead-letter-race")
            return

        first_tombstone_read = asyncio.Event()
        resume_replay = asyncio.Event()
        original_fetchone = database.fetchone
        replay_task = None
        paused = False

        async def pause_after_tombstone_miss(sql, params=()):
            nonlocal paused
            row = await original_fetchone(sql, params)
            if (
                asyncio.current_task() is replay_task
                and "AS matching_dead_letter" in sql
                and not paused
            ):
                paused = True
                assert row is None
                first_tombstone_read.set()
                await resume_replay.wait()
            return row

        monkeypatch.setattr(database, "fetchone", pause_after_tombstone_miss)
        replay_task = asyncio.create_task(
            queue.enqueue(*request, idempotency_key="dead-letter-race")
        )
        await asyncio.wait_for(first_tombstone_read.wait(), timeout=5)
        await queue.move_to_dead_letter(original_id, "provider rejected")
        resume_replay.set()

        with pytest.raises(DeliveryIdempotencyTerminal):
            await asyncio.wait_for(replay_task, timeout=5)
        assert await original_fetchone(
            "SELECT COUNT(*) FROM delivery_queue WHERE agent_id = ?", (owner,)
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
async def test_legacy_delivery_queue_schema_upgrade_converges(
    db_backend, tmp_path
):
    """Upgrade an actual pre-canonical-hash table on both supported backends."""
    scoped_backend = None
    schema = None
    if db_backend.backend_type == "sqlite":
        database = await AsyncDatabase.sqlite(str(tmp_path / "legacy-delivery.db"))
    else:
        from kestrel_sovereign.storage.db.postgres import PostgresBackend

        postgres_url = (
            os.environ.get("TEST_POSTGRES_URL")
            or os.environ.get("KESTREL_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
        )
        assert postgres_url
        schema = f"delivery_upgrade_{uuid4().hex}"
        await db_backend.execute(f'CREATE SCHEMA "{schema}"')
        separator = "&" if "?" in postgres_url else "?"
        scoped_dsn = (
            f"{postgres_url}{separator}options=-csearch_path%3D{schema}"
        )
        scoped_backend = PostgresBackend(
            scoped_dsn, min_pool_size=1, max_pool_size=1
        )
        await scoped_backend.connect()
        database = AsyncDatabase(scoped_backend)

    owner = f"did:test:delivery-schema-upgrade:{uuid4().hex}"
    content_json = json.dumps({"subject": "hello", "body": "world"})
    entry_id = f"legacy-{uuid4().hex}"
    try:
        await database.execute(
            """
            CREATE TABLE delivery_queue (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                channel_type TEXT NOT NULL,
                recipient TEXT NOT NULL,
                content_json TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 5,
                next_retry_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                delivered_at TEXT
            )
            """
        )
        await database.execute(
            """
            CREATE TABLE delivery_dead_letter (
                id TEXT PRIMARY KEY,
                original_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                channel_type TEXT NOT NULL,
                recipient TEXT NOT NULL,
                content_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                max_retries INTEGER NOT NULL DEFAULT 5
            )
            """
        )
        await database.execute(
            """
            INSERT INTO delivery_queue
                (id, agent_id, channel_type, recipient, content_json,
                 content_hash, status, attempts, max_retries,
                 next_retry_at, created_at)
            VALUES (?, ?, 'email', ?, ?, ?, 'pending', 0, 5, ?, ?)
            """,
            (
                entry_id,
                owner,
                "legacy-schema@example.com",
                content_json,
                QueueEntry.compute_content_hash(
                    "legacy-schema@example.com", content_json
                ),
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        queue = DeliveryQueue(database, owner)
        await queue._ensure_tables()
        assert await database.column_exists(
            "delivery_queue", "canonical_content_hash"
        )
        if database.backend_type == "postgres":
            assert await database.column_accepts_null(
                "delivery_dead_letter", "max_retries"
            )
            assert not await database.column_has_default(
                "delivery_dead_letter", "max_retries"
            )
        canonical_row = await database.fetchone(
            "SELECT canonical_content_hash FROM delivery_queue WHERE id = ?",
            (entry_id,),
        )
        assert canonical_row[0] is not None
        assert await queue.enqueue(
            "email",
            "legacy-schema@example.com",
            {"body": "world", "subject": "hello"},
        ) == entry_id
        assert await queue.enqueue(
            "email",
            "legacy-schema@example.com",
            {"body": "world", "subject": "hello"},
            idempotency_key="legacy-schema-adoption",
        ) == entry_id
        await queue._ensure_tables()
    finally:
        await database.close()
        if schema is not None:
            await db_backend.execute(f'DROP SCHEMA "{schema}" CASCADE')


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
