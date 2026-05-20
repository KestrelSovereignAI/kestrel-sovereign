"""Integration tests for ``AsyncConversationStore.atomic_increment_metadata_counter``.

Pre-#1326, both ``MemoryRetriever.update_access`` and the new
``MemoryRetriever.update_applied`` did a Python-level read-modify-write
of the metadata JSON to bump a counter.  Under concurrent dispatch
(parallel reflection hooks, parallel retrievals of the same memory)
two coroutines could both read the same old value and both write the
same successor, losing one increment — codex flagged this on the
first round of #1326 review.

The fix lives in the conversation store as
``atomic_increment_metadata_counter``: a single SQL UPDATE using
``json_set`` + ``json_extract`` (SQLite) or ``jsonb_set`` (PostgreSQL).
Both retriever bookkeeping methods route through it.

These tests run against a real SQLite database (no mocks) so the SQL
itself is exercised — that's the layer where the race would have been
observable.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from kestrel_sovereign.storage import AsyncStorage


AGENT_ID = "did:test:atomic-increment"


@pytest.mark.asyncio
async def test_increment_creates_counter_when_absent(tmp_path):
    """Calling the increment on metadata that has no prior counter
    field initializes it to 1 (not None+1).  Pre-#1326 metadata rows
    have no ``applied_count`` at all — the first attestation must
    upgrade them cleanly, not crash on the missing key."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("assistant", "hello")
        history = await storage.conversation.get_full_history_with_ids()
        msg_id = history[0]["id"]

        ok = await storage.conversation.atomic_increment_metadata_counter(
            msg_id,
            counter_field="applied_count",
            timestamp_field="last_applied",
        )
        assert ok is True

        history = await storage.conversation.get_full_history_with_ids()
        row = next(m for m in history if m["id"] == msg_id)
        meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        assert meta["applied_count"] == 1
        assert meta["last_applied"]  # ISO timestamp populated


@pytest.mark.asyncio
async def test_increment_is_monotonic_across_serial_calls(tmp_path):
    """N serial increments produce a counter of N — basic sanity that
    the SQL increment actually compounds rather than overwriting."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("assistant", "hello")
        history = await storage.conversation.get_full_history_with_ids()
        msg_id = history[0]["id"]
        for _ in range(5):
            await storage.conversation.atomic_increment_metadata_counter(
                msg_id, counter_field="access_count", timestamp_field="last_accessed",
            )
        history = await storage.conversation.get_full_history_with_ids()
        row = next(m for m in history if m["id"] == msg_id)
        meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        assert meta["access_count"] == 5


@pytest.mark.asyncio
async def test_increment_does_not_collide_concurrent_calls(tmp_path):
    """The whole point of the atomic SQL increment: N concurrent
    increments produce a counter of N, not <N.  Under the pre-#1326
    Python-level read-modify-write this test would fail with lost
    updates (two coroutines reading the same old value and writing
    the same successor).  The atomic JSON-set in a single UPDATE
    statement removes the read-modify-write entirely.

    Note for SQLite: each UPDATE is its own transaction at the DB
    level, and the in-process AsyncDatabase serializes writes; the
    real race lived in the Python-side READ that preceded the WRITE.
    Removing that READ closes it.  PostgreSQL is genuinely concurrent
    and would have shown lost updates without this fix."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("assistant", "hello")
        history = await storage.conversation.get_full_history_with_ids()
        msg_id = history[0]["id"]

        async def one_increment():
            await storage.conversation.atomic_increment_metadata_counter(
                msg_id, counter_field="applied_count", timestamp_field="last_applied",
            )

        # Fire 20 concurrent increments on the same row.
        await asyncio.gather(*(one_increment() for _ in range(20)))

        history = await storage.conversation.get_full_history_with_ids()
        row = next(m for m in history if m["id"] == msg_id)
        meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        assert meta["applied_count"] == 20, (
            f"Expected 20 concurrent increments to all register, got "
            f"{meta['applied_count']} — read-modify-write race not fully closed"
        )


@pytest.mark.asyncio
async def test_increment_preserves_other_metadata_fields(tmp_path):
    """The atomic UPDATE must touch ONLY the counter + timestamp keys;
    other metadata (importance, emotional_valence, etc.) must survive
    untouched.  Without ``json_set`` semantics this would be easy to
    get wrong — a naive UPDATE could clobber the whole metadata blob."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("assistant", "hello")
        history = await storage.conversation.get_full_history_with_ids()
        msg_id = history[0]["id"]
        # Seed unrelated metadata.
        await storage.conversation.update_message_metadata(
            msg_id,
            {"importance": 0.85, "emotional_valence": 0.4, "session_id": "abc"},
        )

        await storage.conversation.atomic_increment_metadata_counter(
            msg_id, counter_field="applied_count", timestamp_field="last_applied",
        )

        history = await storage.conversation.get_full_history_with_ids()
        row = next(m for m in history if m["id"] == msg_id)
        meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        assert meta["applied_count"] == 1
        # Untouched fields survive.
        assert meta["importance"] == 0.85
        assert meta["emotional_valence"] == 0.4
        assert meta["session_id"] == "abc"


@pytest.mark.asyncio
async def test_postgres_sql_shape_uses_jsonb_cast(tmp_path):
    """Regression guard for codex round-2 #1326: the PostgreSQL branch
    used ``metadata->>?`` directly, but ``conversation_history.metadata``
    is declared TEXT — the ``->>`` operator is undefined on TEXT and the
    UPDATE would raise at runtime.  The caller swallows the exception,
    so the bookkeeping write would silently no-op for every Postgres
    deployment.

    Without a running Postgres in CI, this test asserts the *shape* of
    the SQL string emitted by the helper when the backend reports
    Postgres — specifically that every read of ``metadata`` goes
    through ``metadata::jsonb`` and the extract uses
    ``(metadata::jsonb->>?)``, not ``metadata->>?``.
    """
    from unittest.mock import AsyncMock, MagicMock

    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    # Build a minimally-stubbed store with a Postgres-backend reporting db.
    # ``AsyncDatabase.execute_commit`` returns the affected-row count as
    # an int, not a result object — so the mock returns an int directly.
    fake_db = MagicMock()
    fake_db.backend_type = "postgres"
    fake_db.execute_commit = AsyncMock(return_value=1)

    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store.db = fake_db
    store.agent_id = "test-agent"

    # With timestamp.
    ok = await store.atomic_increment_metadata_counter(
        message_id=5,
        counter_field="applied_count",
        timestamp_field="last_applied",
    )
    assert ok is True
    sql, params = fake_db.execute_commit.await_args.args
    # Every read of `metadata` must cast to jsonb.
    assert "metadata::jsonb" in sql
    # The extract must NOT use the broken TEXT-column form.
    assert "metadata->>" not in sql.replace("metadata::jsonb->>", ""), (
        "PG branch must cast metadata to jsonb before using ->> — "
        "otherwise the UPDATE errors and the bookkeeping write is lost"
    )
    # Every JSON-key parameter (``?``) must carry an explicit ``::text``
    # cast.  ``jsonb ->>`` is overloaded between (jsonb, text) and
    # (jsonb, int) so asyncpg's prepared-statement type inference
    # rejects an untyped placeholder — without the cast the UPDATE
    # fails to prepare on Postgres and the caller silently drops the
    # bookkeeping write.  Codex round-4 catch on #1326.
    assert "?::text" in sql
    # ARRAY[…] feeding jsonb_set must also carry the typed cast for
    # the same reason — the array element type must be unambiguously
    # text for the statement to prepare.
    assert "ARRAY[?::text]" in sql
    # Counter field is parameter-bound (no SQL injection from caller).
    assert "applied_count" in params
    assert "last_applied" in params

    # Without timestamp — same correctness invariant.
    fake_db.execute_commit.reset_mock()
    await store.atomic_increment_metadata_counter(
        message_id=5,
        counter_field="access_count",
        timestamp_field=None,
    )
    sql2, _ = fake_db.execute_commit.await_args.args
    assert "metadata::jsonb" in sql2
    assert "metadata->>" not in sql2


@pytest.mark.asyncio
async def test_increment_returns_false_for_unknown_message(tmp_path):
    """Contract: ``True`` only when a row was actually updated.  Pre-
    codex-round-3 the helper returned ``True`` unconditionally on
    SQLite and used a broken ``getattr(int, 'rowcount', 1)`` default
    on Postgres — both reported success even when the message_id
    didn't exist.  Callers that want to detect a no-op (reflection
    hooks attesting against a stale message_id, audit code) need
    the truthful signal."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("assistant", "hello")
        ok = await storage.conversation.atomic_increment_metadata_counter(
            message_id=999_999,  # not present
            counter_field="applied_count",
            timestamp_field="last_applied",
        )
        assert ok is False


@pytest.mark.asyncio
async def test_increment_returns_true_for_real_message(tmp_path):
    """Companion to the unknown-message guard — when the row exists,
    the helper must report ``True``."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("assistant", "hello")
        history = await storage.conversation.get_full_history_with_ids()
        msg_id = history[0]["id"]
        ok = await storage.conversation.atomic_increment_metadata_counter(
            msg_id, counter_field="applied_count", timestamp_field="last_applied",
        )
        assert ok is True


@pytest.mark.asyncio
async def test_increment_without_timestamp_field(tmp_path):
    """``timestamp_field=None`` increments the counter without
    stamping a timestamp — useful for callers that don't care to
    record when an event last happened."""
    db_path = tmp_path / "kestrel.db"
    async with AsyncStorage(str(db_path), agent_id=AGENT_ID) as storage:
        await storage.conversation.add_conversation("assistant", "hello")
        history = await storage.conversation.get_full_history_with_ids()
        msg_id = history[0]["id"]
        await storage.conversation.atomic_increment_metadata_counter(
            msg_id, counter_field="ad_hoc_counter", timestamp_field=None,
        )
        history = await storage.conversation.get_full_history_with_ids()
        row = next(m for m in history if m["id"] == msg_id)
        meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        assert meta["ad_hoc_counter"] == 1
        # No timestamp written.
        assert "last_accessed" not in meta
        assert "last_applied" not in meta
