"""Exact-id hard-purge contracts for conversation history (#2509).

Every permanent conversation deletion must remove the keyed lexical tokens
that point at the destroyed message and must never re-evaluate a broad
predicate after its destructive audit snapshot.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_conversation_store import (
    AsyncConversationStore,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import TransactionError
from kestrel_sovereign.storage.destructive_audit import hash_rows
from kestrel_sovereign.storage.lexical_memory_index import (
    LexicalIndexReplacement,
)


@dataclass(frozen=True)
class _IndexedMessage:
    row_id: int
    lexical_index_id: str


def _timestamp_value(
    db: AsyncDatabase, value: datetime | str | None
) -> datetime | str | None:
    if value is None or isinstance(value, str) or db.backend_type == "postgres":
        return value
    return value.isoformat(sep=" ")


async def _storage_for_backend(db_backend, agent_id: str) -> AsyncStorage:
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    return storage


async def _seed_indexed_message(
    db: AsyncDatabase,
    agent_id: str,
    *,
    content: str,
    session_id: str | None = None,
    created_at: datetime | str | None = None,
    deleted_at: datetime | str | None = None,
    lexical_index_id: str | None = None,
) -> _IndexedMessage:
    lexical_index_id = (
        uuid4().hex if lexical_index_id is None else lexical_index_id
    )
    metadata = json.dumps({"session_id": session_id}) if session_id else "{}"
    created_at = created_at or datetime(2026, 7, 1, 12, 0, 0)
    await db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, lexical_index_id, "
        "lexical_index_version, created_at, deleted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            agent_id,
            "user",
            content,
            metadata,
            lexical_index_id,
            "v1:test",
            _timestamp_value(db, created_at),
            _timestamp_value(db, deleted_at),
        ),
    )
    row = await db.fetchone(
        "SELECT id FROM conversation_history "
        "WHERE agent_id = ? AND lexical_index_id = ? ORDER BY id DESC LIMIT 1",
        (agent_id, lexical_index_id),
    )
    assert row is not None
    await db.execute_many(
        "INSERT INTO conversation_lexical_tokens "
        "(agent_id, lexical_index_id, token_hash) VALUES (?, ?, ?) "
        "ON CONFLICT (agent_id, lexical_index_id, token_hash) DO NOTHING",
        [
            (agent_id, lexical_index_id, f"token-{index}-{lexical_index_id}")
            for index in range(3)
        ],
    )
    return _IndexedMessage(int(row[0]), lexical_index_id)


async def _cancel_and_observe(task: asyncio.Task[Any] | None) -> None:
    """Cancel a test-owned task and always retrieve its terminal outcome."""
    if task is None:
        return
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError):
        await task


class _TwoPartyCleanupGate:
    """Drive two transactions to one key boundary and expose lock overlap."""

    def __init__(self) -> None:
        self.arrivals = 0
        self.all_arrived = asyncio.Event()
        self.first_acquired = asyncio.Event()
        self.second_acquired = asyncio.Event()
        self.release_first = asyncio.Event()
        self.acquired: list[str] = []

    def wrapper_for(self, index: Any, label: str):
        original = index.serialized_token_cleanup

        @asynccontextmanager
        async def wrapper(keys):
            self.arrivals += 1
            if self.arrivals == 2:
                self.all_arrived.set()
            await self.all_arrived.wait()
            async with original(keys) as ordered_keys:
                self.acquired.append(label)
                if len(self.acquired) == 1:
                    self.first_acquired.set()
                    await self.release_first.wait()
                else:
                    self.second_acquired.set()
                yield ordered_keys

        return wrapper

    async def assert_exclusive_first_holder(self) -> None:
        await asyncio.wait_for(self.first_acquired.wait(), timeout=5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(self.second_acquired.wait(), timeout=0.1)
        assert len(self.acquired) == 1


async def _assert_destroyed(db: AsyncDatabase, message: _IndexedMessage) -> None:
    assert (
        await db.fetchval(
            "SELECT COUNT(*) FROM conversation_history WHERE id = ?",
            (message.row_id,),
        )
        == 0
    )
    assert (
        await db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE lexical_index_id = ?",
            (message.lexical_index_id,),
        )
        == 0
    )


async def _assert_present(db: AsyncDatabase, message: _IndexedMessage) -> None:
    assert (
        await db.fetchval(
            "SELECT COUNT(*) FROM conversation_history WHERE id = ?",
            (message.row_id,),
        )
        == 1
    )
    assert (
        await db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE lexical_index_id = ?",
            (message.lexical_index_id,),
        )
        == 3
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_purge_message_removes_its_blind_index(db_backend):
    agent_id = f"did:test:purge-message:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    target = await _seed_indexed_message(
        storage.db, agent_id, content="purge one message"
    )

    assert await storage.conversation.purge_message(target.row_id) is True
    await _assert_destroyed(storage.db, target)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_purge_message_reclaims_an_empty_legacy_lexical_key(db_backend):
    """Empty TEXT is a stored key, not the absence of a key."""
    agent_id = f"did:test:purge-empty-key:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    target = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="empty legacy key must be reclaimed",
        lexical_index_id="",
    )

    assert await storage.conversation.purge_message(target.row_id) is True
    assert (
        await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_history "
            "WHERE agent_id = ? AND id = ?",
            (agent_id, target.row_id),
        )
        == 0
    )
    assert (
        await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE agent_id = ? AND lexical_index_id = ''",
            (agent_id,),
        )
        == 0
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_purge_preserves_shared_blind_index_until_last_owner(db_backend):
    """Legacy duplicate ownership must not turn one purge into collateral loss."""
    agent_id = f"did:test:purge-shared-key:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    shared_key = uuid4().hex
    first = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="first shared-key owner",
        lexical_index_id=shared_key,
    )
    survivor = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="second shared-key owner",
        lexical_index_id=shared_key,
    )

    assert first.row_id != survivor.row_id
    assert await storage.conversation.purge_message(first.row_id) is True
    assert (
        await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_history WHERE id = ?",
            (first.row_id,),
        )
        == 0
    )
    await _assert_present(storage.db, survivor)

    assert await storage.conversation.purge_message(survivor.row_id) is True
    await _assert_destroyed(storage.db, survivor)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_concurrent_final_shared_key_purges_reclaim_tokens(
    db_backend, monkeypatch
):
    """The final concurrent owners serialize their MVCC owner checks."""
    agent_id = f"did:test:purge-shared-key-concurrent:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    first_store = storage.conversation
    second_store = AsyncConversationStore(storage.db, agent_id=agent_id)
    shared_key = uuid4().hex
    first = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="first concurrent shared-key owner",
        lexical_index_id=shared_key,
    )
    second = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="second concurrent shared-key owner",
        lexical_index_id=shared_key,
    )

    first_task: asyncio.Task[bool] | None = None
    second_task: asyncio.Task[bool] | None = None
    cleanup_gate: _TwoPartyCleanupGate | None = None
    if storage.db.backend_type == "postgres":
        # Hold both transactions after their SELECT FOR UPDATE snapshots, then
        # make them arrive at the key boundary after deleting their respective
        # history rows. Exactly one may cross while it owns the advisory lock.
        first_audit = _PausingAudit(None)
        second_audit = _PausingAudit(None)
        first_store._destructive_audit = first_audit
        second_store._destructive_audit = second_audit
        cleanup_gate = _TwoPartyCleanupGate()

        monkeypatch.setattr(
            first_store._lexical_index,
            "serialized_token_cleanup",
            cleanup_gate.wrapper_for(first_store._lexical_index, "first"),
        )
        monkeypatch.setattr(
            second_store._lexical_index,
            "serialized_token_cleanup",
            cleanup_gate.wrapper_for(second_store._lexical_index, "second"),
        )

    try:
        first_task = asyncio.create_task(first_store.purge_message(first.row_id))
        second_task = asyncio.create_task(second_store.purge_message(second.row_id))
        if storage.db.backend_type == "postgres":
            await asyncio.wait_for(first_audit.snapshot_ready.wait(), timeout=5)
            await asyncio.wait_for(second_audit.snapshot_ready.wait(), timeout=5)
            first_audit.resume.set()
            second_audit.resume.set()
            assert cleanup_gate is not None
            await cleanup_gate.assert_exclusive_first_holder()
            cleanup_gate.release_first.set()

        assert await asyncio.wait_for(first_task, timeout=5) is True
        assert await asyncio.wait_for(second_task, timeout=5) is True
        assert (
            await storage.db.fetchval(
                "SELECT COUNT(*) FROM conversation_history "
                "WHERE agent_id = ? AND lexical_index_id = ?",
                (agent_id, shared_key),
            )
            == 0
        )
        assert (
            await storage.db.fetchval(
                "SELECT COUNT(*) FROM conversation_lexical_tokens "
                "WHERE agent_id = ? AND lexical_index_id = ?",
                (agent_id, shared_key),
            )
            == 0
        )
    finally:
        if cleanup_gate is not None:
            cleanup_gate.release_first.set()
        if storage.db.backend_type == "postgres":
            first_audit.resume.set()
            second_audit.resume.set()
        await _cancel_and_observe(first_task)
        await _cancel_and_observe(second_task)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_session_and_full_purge_share_one_row_lock_order(
    db_backend, monkeypatch
):
    """A >500-row session purge cannot deadlock a concurrent full purge."""
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL row-lock ordering regression")

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    agent_id = f"did:test:purge-row-lock-order:{uuid4()}"
    session_id = f"session-{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    second_db = AsyncDatabase(
        PostgresBackend.from_pool(storage.db.backend._pool)
    )
    second_store = AsyncConversationStore(second_db, agent_id=agent_id)
    started_at = datetime(2026, 7, 1, 12, 0, 0)
    await storage.db.execute_many(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (
                agent_id,
                "user",
                f"session row {index}",
                json.dumps({"session_id": session_id}),
                started_at + timedelta(seconds=index),
            )
            for index in range(600)
        ]
        + [
            (
                agent_id,
                "user",
                "full purge must also remove this row",
                json.dumps({"session_id": f"other-{uuid4()}"}),
                started_at + timedelta(seconds=700),
            )
        ],
    )

    first_batch_locked = asyncio.Event()
    resume_session_purge = asyncio.Event()
    full_purge_started = asyncio.Event()
    first_batch_ids: list[int] = []
    original_first_fetchall = storage.db.fetchall
    original_second_fetchall = second_db.fetchall

    async def pause_after_first_session_lock(sql: str, params: tuple = ()):
        rows = await original_first_fetchall(sql, params)
        if (
            not first_batch_locked.is_set()
            and "lexical_index_id" in sql
            and "id IN (" in sql
            and "FOR UPDATE" in sql
        ):
            first_batch_ids.extend(int(row[0]) for row in rows)
            first_batch_locked.set()
            await resume_session_purge.wait()
        return rows

    async def observe_full_purge(sql: str, params: tuple = ()):
        if "lexical_index_id" in sql and "FOR UPDATE" in sql:
            full_purge_started.set()
        return await original_second_fetchall(sql, params)

    monkeypatch.setattr(storage.db, "fetchall", pause_after_first_session_lock)
    monkeypatch.setattr(second_db, "fetchall", observe_full_purge)
    session_task: asyncio.Task[int] | None = None
    full_task: asyncio.Task[int] | None = None
    try:
        session_task = asyncio.create_task(
            storage.conversation.purge_conversation_session(session_id)
        )
        await asyncio.wait_for(first_batch_locked.wait(), timeout=5)
        assert len(first_batch_ids) == 500
        assert first_batch_ids == sorted(first_batch_ids)

        full_task = asyncio.create_task(second_store.purge_all())
        await asyncio.wait_for(full_purge_started.wait(), timeout=5)
        done, _pending = await asyncio.wait({full_task}, timeout=0.1)
        assert not done, "full purge must wait for the first ascending row lock"

        resume_session_purge.set()
        session_count, full_count = await asyncio.wait_for(
            asyncio.gather(session_task, full_task),
            timeout=10,
        )
        assert session_count == 600
        assert full_count == 1
        assert (
            await storage.db.fetchval(
                "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?",
                (agent_id,),
            )
            == 0
        )
    finally:
        resume_session_purge.set()
        tasks = [task for task in (session_task, full_task) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await second_db.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_concurrent_backfills_reclaim_the_final_shared_key(
    db_backend, monkeypatch
):
    agent_id = f"did:test:backfill-shared-key-concurrent:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    first_index = storage.conversation._lexical_index
    second_store = AsyncConversationStore(storage.db, agent_id=agent_id)
    second_index = second_store._lexical_index
    shared_key = uuid4().hex
    first = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="first shared owner moving to a private key",
        lexical_index_id=shared_key,
    )
    second = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="second shared owner moving to a private key",
        lexical_index_id=shared_key,
    )
    first_key = first_index.backfill_message_key(first.row_id)
    second_key = second_index.backfill_message_key(second.row_id)
    first_entry = LexicalIndexReplacement(
        message_id=first.row_id,
        expected_key=shared_key,
        replacement_key=first_key,
        tokens=("first", "private"),
    )
    second_entry = LexicalIndexReplacement(
        message_id=second.row_id,
        expected_key=shared_key,
        replacement_key=second_key,
        tokens=("second", "private"),
    )

    cleanup_gate: _TwoPartyCleanupGate | None = None
    if storage.db.backend_type == "postgres":
        cleanup_gate = _TwoPartyCleanupGate()
        monkeypatch.setattr(
            first_index,
            "serialized_token_cleanup",
            cleanup_gate.wrapper_for(first_index, "first"),
        )
        monkeypatch.setattr(
            second_index,
            "serialized_token_cleanup",
            cleanup_gate.wrapper_for(second_index, "second"),
        )

    first_task = asyncio.create_task(
        first_index.replace_existing_messages([first_entry])
    )
    second_task = asyncio.create_task(
        second_index.replace_existing_messages([second_entry])
    )
    try:
        if cleanup_gate is not None:
            await cleanup_gate.assert_exclusive_first_holder()
            cleanup_gate.release_first.set()
        first_result = await asyncio.wait_for(first_task, timeout=5)
        second_result = await asyncio.wait_for(second_task, timeout=5)

        assert first_result.updated == second_result.updated == 1
        assert (
            first_result.garbage_collected
            + second_result.garbage_collected
            == 3
        )
        rows = await storage.db.fetchall(
            "SELECT id, lexical_index_id FROM conversation_history "
            "WHERE id IN (?, ?) ORDER BY id",
            (first.row_id, second.row_id),
        )
        assert rows == [(first.row_id, first_key), (second.row_id, second_key)]
        assert (
            await storage.db.fetchval(
                "SELECT COUNT(*) FROM conversation_lexical_tokens "
                "WHERE agent_id = ? AND lexical_index_id = ?",
                (agent_id, shared_key),
            )
            == 0
        )
        for replacement_key in (first_key, second_key):
            assert (
                await storage.db.fetchval(
                    "SELECT COUNT(*) FROM conversation_lexical_tokens "
                    "WHERE agent_id = ? AND lexical_index_id = ?",
                    (agent_id, replacement_key),
                )
                == 2
            )
    finally:
        if cleanup_gate is not None:
            cleanup_gate.release_first.set()
        await _cancel_and_observe(first_task)
        await _cancel_and_observe(second_task)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("owner_state", ["live", "deleted", "archived"])
async def test_backfill_uses_a_fresh_key_when_replacement_has_another_owner(
    db_backend, owner_state
):
    """Replacement never clobbers a current, trashed, or archived owner."""
    agent_id = f"did:test:backfill-replacement-owner:{owner_state}:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    index = storage.conversation._lexical_index
    target = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="target collision fact",
        lexical_index_id=f"old-target-key-{uuid4()}",
    )
    proposed_key = index.backfill_message_key(target.row_id)
    survivor = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="survivor collision fact",
        lexical_index_id=proposed_key,
        deleted_at=(
            datetime(2026, 7, 2, 12, 0, 0)
            if owner_state == "deleted"
            else None
        ),
    )
    await storage.db.execute(
        "UPDATE conversation_history SET lexical_index_version = ? "
        "WHERE id = ? AND agent_id = ?",
        (index.version, survivor.row_id, agent_id),
    )
    if owner_state == "archived":
        await storage.db.execute(
            "UPDATE conversation_history SET archived_at = ? "
            "WHERE id = ? AND agent_id = ?",
            (
                _timestamp_value(
                    storage.db,
                    datetime(2026, 7, 2, 12, 0, 0),
                ),
                survivor.row_id,
                agent_id,
            ),
        )
    await index.index_message(proposed_key, ("survivor", "collision"))

    if owner_state == "live":
        assert await index.candidate_message_ids(("survivor",), limit=10) == [
            survivor.row_id
        ]

    result = await storage.conversation.backfill_lexical_index(
        batch_size=1,
        max_rows=1,
    )
    target_marker = await storage.db.fetchone(
        "SELECT lexical_index_id, lexical_index_version "
        "FROM conversation_history WHERE id = ? AND agent_id = ?",
        (target.row_id, agent_id),
    )
    survivor_marker = await storage.db.fetchone(
        "SELECT lexical_index_id, lexical_index_version "
        "FROM conversation_history WHERE id = ? AND agent_id = ?",
        (survivor.row_id, agent_id),
    )

    assert result["indexed"] == 1
    assert target_marker is not None
    assert target_marker[0] not in {target.lexical_index_id, proposed_key}
    assert target_marker[1] == index.version
    assert survivor_marker == (proposed_key, index.version)
    assert (
        await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE agent_id = ? AND lexical_index_id = ? AND token_hash = ?",
            (agent_id, proposed_key, index.hash_token("survivor")),
        )
        == 1
    )
    assert (
        await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE agent_id = ? AND lexical_index_id = ? AND token_hash = ?",
            (agent_id, target_marker[0], index.hash_token("target")),
        )
        == 1
    )
    if owner_state == "live":
        assert await index.candidate_message_ids(("survivor",), limit=10) == [
            survivor.row_id
        ]
        assert await index.candidate_message_ids(("target",), limit=10) == [
            target.row_id
        ]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_backfill_rolls_back_if_fresh_collision_fallback_is_owned(
    db_backend, monkeypatch
):
    """An occupied fallback fails closed without changing owners or tokens."""
    from types import SimpleNamespace

    from kestrel_sovereign.storage import lexical_memory_index

    agent_id = f"did:test:backfill-fallback-owner:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    index = storage.conversation._lexical_index
    target = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="target must roll back",
        lexical_index_id=f"old-target-key-{uuid4()}",
    )
    proposed_key = index.backfill_message_key(target.row_id)
    fallback_key = f"occupied-fallback-{uuid4()}"
    proposed_owner = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="proposed key owner",
        lexical_index_id=proposed_key,
    )
    fallback_owner = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="fallback key owner",
        lexical_index_id=fallback_key,
    )
    await storage.db.execute_many(
        "UPDATE conversation_history SET lexical_index_version = ? "
        "WHERE id = ? AND agent_id = ?",
        [
            (index.version, proposed_owner.row_id, agent_id),
            (index.version, fallback_owner.row_id, agent_id),
        ],
    )
    await index.index_message(proposed_key, ("proposed",))
    await index.index_message(fallback_key, ("fallback",))
    monkeypatch.setattr(
        lexical_memory_index,
        "uuid4",
        lambda: SimpleNamespace(hex=fallback_key),
    )

    with pytest.raises(
        TransactionError,
        match="Fresh lexical replacement key is already owned",
    ):
        await index.replace_existing_messages(
            [
                LexicalIndexReplacement(
                    message_id=target.row_id,
                    expected_key=target.lexical_index_id,
                    replacement_key=proposed_key,
                    tokens=("target",),
                )
            ]
        )

    assert await storage.db.fetchone(
        "SELECT lexical_index_id, lexical_index_version "
        "FROM conversation_history WHERE id = ? AND agent_id = ?",
        (target.row_id, agent_id),
    ) == (target.lexical_index_id, "v1:test")
    assert (
        await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE agent_id = ? AND lexical_index_id = ? AND token_hash = ?",
            (agent_id, proposed_key, index.hash_token("proposed")),
        )
        == 1
    )
    assert (
        await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE agent_id = ? AND lexical_index_id = ? AND token_hash = ?",
            (agent_id, fallback_key, index.hash_token("fallback")),
        )
        == 1
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_backfill_repairs_and_reclaims_an_empty_legacy_key(db_backend):
    agent_id = f"did:test:backfill-empty-key:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    index = storage.conversation._lexical_index
    target = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="empty key backfill fact",
        lexical_index_id="",
    )

    result = await storage.conversation.backfill_lexical_index(
        batch_size=1,
        max_rows=1,
    )
    marker = await storage.db.fetchone(
        "SELECT lexical_index_id, lexical_index_version "
        "FROM conversation_history WHERE id = ? AND agent_id = ?",
        (target.row_id, agent_id),
    )

    assert result["indexed"] == 1
    assert marker is not None and marker[0]
    assert marker[1] == index.version
    assert (
        await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens "
            "WHERE agent_id = ? AND lexical_index_id = ''",
            (agent_id,),
        )
        == 0
    )
    assert await index.candidate_message_ids(("empty",), limit=10) == [
        target.row_id
    ]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_purge_session_removes_all_blind_indexes_and_retains_title(db_backend):
    agent_id = f"did:test:purge-session:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    session_id = f"session-{uuid4()}"
    first = await _seed_indexed_message(
        storage.db, agent_id, content="session first", session_id=session_id
    )
    second = await _seed_indexed_message(
        storage.db, agent_id, content="session second", session_id=session_id
    )
    survivor = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="different session",
        session_id=f"session-{uuid4()}",
    )
    await storage.conversation.set_conversation_name(session_id, "Retained title")

    assert await storage.conversation.purge_conversation_session(session_id) == 2
    await _assert_destroyed(storage.db, first)
    await _assert_destroyed(storage.db, second)
    await _assert_present(storage.db, survivor)
    assert (
        await storage.conversation.get_conversation_name(session_id) == "Retained title"
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_purge_all_removes_only_the_agents_rows_and_blind_indexes(db_backend):
    agent_id = f"did:test:purge-all:{uuid4()}"
    other_agent_id = f"did:test:purge-all-other:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    first = await _seed_indexed_message(storage.db, agent_id, content="first")
    second = await _seed_indexed_message(storage.db, agent_id, content="second")
    other = await _seed_indexed_message(
        storage.db, other_agent_id, content="other agent"
    )

    assert await storage.conversation.purge_all() == 2
    await _assert_destroyed(storage.db, first)
    await _assert_destroyed(storage.db, second)
    await _assert_present(storage.db, other)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_purge_all_batches_history_and_token_deletes(db_backend):
    agent_id = f"did:test:purge-batches:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    row_count = 501
    lexical_keys = [uuid4().hex for _ in range(row_count)]
    await storage.db.execute_many(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, lexical_index_id, "
        "lexical_index_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                agent_id,
                "user",
                f"batched purge row {index}",
                "{}",
                lexical_key,
                "v1:test",
                _timestamp_value(storage.db, datetime(2026, 7, 1, 12, 0, 0)),
            )
            for index, lexical_key in enumerate(lexical_keys)
        ],
    )
    await storage.db.execute_many(
        "INSERT INTO conversation_lexical_tokens "
        "(agent_id, lexical_index_id, token_hash) VALUES (?, ?, ?)",
        [
            (agent_id, lexical_key, f"token-{lexical_key}")
            for lexical_key in lexical_keys
        ],
    )

    assert await storage.conversation.purge_all() == row_count
    assert (
        await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        )
        == 0
    )
    assert (
        await storage.db.fetchval(
            "SELECT COUNT(*) FROM conversation_lexical_tokens WHERE agent_id = ?",
            (agent_id,),
        )
        == 0
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_purge_all_since_removes_only_snapshotted_rows_and_blind_indexes(
    db_backend,
):
    agent_id = f"did:test:purge-since:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    before = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="before watermark",
        created_at=datetime(2026, 5, 1, 12, 0, 0),
    )
    after = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="after watermark",
        created_at=datetime(2026, 7, 1, 12, 0, 0),
    )

    assert await storage.conversation.purge_all_since("2026-06-01T00:00:00") == 1
    await _assert_present(storage.db, before)
    await _assert_destroyed(storage.db, after)


@pytest.mark.asyncio
async def test_purge_all_since_compares_mixed_sqlite_timestamps_chronologically(
    tmp_path,
):
    db_path = tmp_path / "mixed-created-at.db"
    agent_id = f"did:test:purge-since-mixed:{uuid4()}"
    async with AsyncStorage(str(db_path), agent_id=agent_id) as storage:
        before = await _seed_indexed_message(
            storage.db,
            agent_id,
            content="before same-day cutoff",
            created_at="2026-06-01 11:59:59",
        )
        boundary = await _seed_indexed_message(
            storage.db,
            agent_id,
            content="at same-day cutoff",
            created_at="2026-06-01T12:00:00Z",
        )
        after = await _seed_indexed_message(
            storage.db,
            agent_id,
            content="after same-day cutoff in legacy format",
            created_at="2026-06-01 13:00:00",
        )

        assert await storage.conversation.purge_all_since("2026-06-01T12:00:00Z") == 2
        await _assert_present(storage.db, before)
        await _assert_destroyed(storage.db, boundary)
        await _assert_destroyed(storage.db, after)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_purge_old_trash_removes_only_bounded_rows_and_blind_indexes(
    db_backend,
):
    agent_id = f"did:test:purge-trash:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    oldest = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="oldest trash",
        deleted_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    old = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="old trash",
        deleted_at=datetime(2026, 2, 1, 12, 0, 0),
    )
    recent = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="recent trash",
        deleted_at=datetime(2026, 7, 1, 12, 0, 0),
    )

    assert (
        await storage.conversation.purge_trash_older_than(
            "2026-06-01T00:00:00", max_rows=1
        )
        == 1
    )
    await _assert_destroyed(storage.db, oldest)
    await _assert_present(storage.db, old)
    await _assert_present(storage.db, recent)


@pytest.mark.asyncio
async def test_purge_trash_compares_mixed_sqlite_timestamps_chronologically(tmp_path):
    db_path = tmp_path / "mixed-deleted-at.db"
    agent_id = f"did:test:purge-trash-mixed:{uuid4()}"
    async with AsyncStorage(str(db_path), agent_id=agent_id) as storage:
        before = await _seed_indexed_message(
            storage.db,
            agent_id,
            content="trash before same-day cutoff",
            deleted_at="2026-06-01T11:59:59Z",
        )
        boundary = await _seed_indexed_message(
            storage.db,
            agent_id,
            content="trash at same-day cutoff",
            deleted_at="2026-06-01 12:00:00",
        )
        after = await _seed_indexed_message(
            storage.db,
            agent_id,
            content="trash after same-day cutoff in legacy format",
            deleted_at="2026-06-01 13:00:00",
        )

        assert (
            await storage.conversation.purge_trash_older_than("2026-06-01T12:00:00Z")
            == 1
        )
        await _assert_destroyed(storage.db, before)
        await _assert_present(storage.db, boundary)
        await _assert_present(storage.db, after)


class _FailingAudit:
    async def append(self, _event: Any) -> int:
        raise RuntimeError("audit unavailable")


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_audit_failure_leaves_history_and_blind_index_untouched(db_backend):
    agent_id = f"did:test:purge-audit:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    target = await _seed_indexed_message(storage.db, agent_id, content="keep me")
    storage.conversation._destructive_audit = _FailingAudit()

    with pytest.raises(RuntimeError, match="audit unavailable") as raised:
        await storage.conversation.purge_all()

    assert type(raised.value) is RuntimeError
    await _assert_present(storage.db, target)


@pytest.mark.asyncio
async def test_history_delete_failure_rolls_back_blind_index_cleanup(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "transaction-rollback.db"
    agent_id = f"did:test:purge-rollback:{uuid4()}"
    async with AsyncStorage(str(db_path), agent_id=agent_id) as storage:
        target = await _seed_indexed_message(storage.db, agent_id, content="keep both")
        original_execute = storage.db.execute

        async def fail_history_delete(sql: str, params: tuple = ()) -> int:
            if sql.startswith("DELETE FROM conversation_history"):
                raise RuntimeError("forced history delete failure")
            return await original_execute(sql, params)

        monkeypatch.setattr(storage.db, "execute", fail_history_delete)

        with pytest.raises(TransactionError, match="forced history delete failure"):
            await storage.conversation.purge_all()

        await _assert_present(storage.db, target)


class _PausingAudit:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.snapshot_ready = asyncio.Event()
        self.resume = asyncio.Event()
        self.event: Any = None

    async def append(self, event: Any) -> int:
        self.event = event
        self.snapshot_ready.set()
        await self.resume.wait()
        if self.delegate is None:
            return 1
        return await self.delegate.append(event)


def _audit_snapshot(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "role": row[1],
        "content": row[2],
        "metadata": row[3],
        "created_at": row[4],
        "deleted_at": row[5],
    }


@pytest.mark.asyncio
async def test_concurrent_matching_insert_after_snapshot_survives_unaudited(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "concurrent-purge.db"
    agent_id = f"did:test:purge-race:{uuid4()}"
    async with AsyncStorage(str(db_path), agent_id=agent_id) as storage:
        target = await _seed_indexed_message(
            storage.db,
            agent_id,
            content="present at snapshot",
            created_at=datetime(2026, 7, 1, 12, 0, 0),
        )
        raw_target = await storage.db.fetchone(
            "SELECT id, role, content, metadata, created_at, deleted_at "
            "FROM conversation_history WHERE id = ?",
            (target.row_id,),
        )
        assert raw_target is not None

        writer = await AsyncDatabase.sqlite(str(db_path))
        pausing_audit = _PausingAudit(storage.conversation._destructive_audit)
        storage.conversation._destructive_audit = pausing_audit
        purge_task = asyncio.create_task(
            storage.conversation.purge_all_since("2026-06-01T00:00:00")
        )
        writer_task: asyncio.Task[_IndexedMessage] | None = None
        try:
            await asyncio.wait_for(pausing_audit.snapshot_ready.wait(), timeout=5)
            write_started = asyncio.Event()
            original_execute = writer.execute

            async def observe_insert(sql: str, params: tuple = ()) -> int:
                if sql.startswith("INSERT INTO conversation_history"):
                    write_started.set()
                return await original_execute(sql, params)

            monkeypatch.setattr(writer, "execute", observe_insert)
            writer_task = asyncio.create_task(
                _seed_indexed_message(
                    writer,
                    agent_id,
                    content="committed after snapshot",
                    created_at=datetime(2026, 7, 2, 12, 0, 0),
                )
            )
            await asyncio.wait_for(write_started.wait(), timeout=5)
            done, _pending = await asyncio.wait({writer_task}, timeout=0.05)
            assert not done, "SQLite writer must wait for the audited purge unit"

            pausing_audit.resume.set()

            assert await asyncio.wait_for(purge_task, timeout=5) == 1
            concurrent = await asyncio.wait_for(writer_task, timeout=5)
            await _assert_destroyed(storage.db, target)
            await _assert_present(storage.db, concurrent)
            assert pausing_audit.event.row_count == 1
            assert pausing_audit.event.pre_operation_hash == hash_rows(
                [_audit_snapshot(raw_target)]
            )
        finally:
            pausing_audit.resume.set()
            await _cancel_and_observe(purge_task)
            await _cancel_and_observe(writer_task)
            await writer.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_selected_row_cannot_change_after_its_audit_hash(db_backend, monkeypatch):
    """A same-ID writer must lose to deletion, not mutate audited bytes."""
    agent_id = f"did:test:purge-audited-update:{uuid4()}"
    storage = await _storage_for_backend(db_backend, agent_id)
    target = await _seed_indexed_message(
        storage.db,
        agent_id,
        content="bytes represented by audit hash",
    )
    raw_target = await storage.db.fetchone(
        "SELECT id, role, content, metadata, created_at, deleted_at "
        "FROM conversation_history WHERE id = ?",
        (target.row_id,),
    )
    assert raw_target is not None

    writer = (
        await AsyncDatabase.sqlite(db_backend.db_path)
        if storage.db.backend_type == "sqlite"
        else storage.db
    )
    close_writer = writer is not storage.db
    pausing_audit = _PausingAudit(storage.conversation._destructive_audit)
    storage.conversation._destructive_audit = pausing_audit
    purge_task = asyncio.create_task(storage.conversation.purge_message(target.row_id))
    writer_task: asyncio.Task[int] | None = None
    try:
        await asyncio.wait_for(pausing_audit.snapshot_ready.wait(), timeout=5)
        write_started = asyncio.Event()
        original_execute = writer.execute

        async def observe_update(sql: str, params: tuple = ()) -> int:
            if sql.startswith("UPDATE conversation_history SET content"):
                write_started.set()
            return await original_execute(sql, params)

        monkeypatch.setattr(writer, "execute", observe_update)
        writer_task = asyncio.create_task(
            writer.execute(
                "UPDATE conversation_history SET content = ?, metadata = ? "
                "WHERE id = ?",
                ("new unaudited bytes", '{"revision":2}', target.row_id),
            )
        )
        await asyncio.wait_for(write_started.wait(), timeout=5)
        done, _pending = await asyncio.wait({writer_task}, timeout=0.05)
        assert not done, "same-row writer bypassed the purge snapshot lock"

        pausing_audit.resume.set()
        assert await asyncio.wait_for(purge_task, timeout=5) is True
        assert await asyncio.wait_for(writer_task, timeout=5) == 0
        await _assert_destroyed(storage.db, target)
        assert pausing_audit.event.pre_operation_hash == hash_rows(
            [_audit_snapshot(raw_target)]
        )
    finally:
        pausing_audit.resume.set()
        await _cancel_and_observe(purge_task)
        await _cancel_and_observe(writer_task)
        if close_writer:
            await writer.close()


@pytest.mark.asyncio
async def test_stale_production_backfill_cannot_recreate_post_purge_tokens(
    tmp_path, monkeypatch
):
    """A hydrated pre-delete row is revalidated before backfill token writes."""
    db_path = tmp_path / "backfill-purge-race.db"
    agent_id = f"did:test:purge-backfill-race:{uuid4()}"
    async with AsyncStorage(str(db_path), agent_id=agent_id) as purger:
        async with AsyncStorage(str(db_path), agent_id=agent_id) as backfiller:
            target = await _seed_indexed_message(
                purger.db,
                agent_id,
                content="stale hydrated backfill content",
            )
            replacement_key = (
                backfiller.conversation._lexical_index.backfill_message_key(
                    target.row_id
                )
            )

            pausing_audit = _PausingAudit(purger.conversation._destructive_audit)
            purger.conversation._destructive_audit = pausing_audit
            hydrated = asyncio.Event()
            replacement_started = asyncio.Event()
            original_hydrate = backfiller.conversation.get_messages_by_ids
            original_replace = (
                backfiller.conversation._lexical_index.replace_existing_messages
            )

            async def observe_hydration(ids: list[int]) -> list[dict[str, Any]]:
                rows = await original_hydrate(ids)
                hydrated.set()
                return rows

            async def observe_replacement(entries):
                replacement_started.set()
                return await original_replace(entries)

            monkeypatch.setattr(
                backfiller.conversation, "get_messages_by_ids", observe_hydration
            )
            monkeypatch.setattr(
                backfiller.conversation._lexical_index,
                "replace_existing_messages",
                observe_replacement,
            )

            purge_task = asyncio.create_task(
                purger.conversation.purge_message(target.row_id)
            )
            backfill_task: asyncio.Task[dict[str, Any]] | None = None
            try:
                await asyncio.wait_for(pausing_audit.snapshot_ready.wait(), timeout=5)
                backfill_task = asyncio.create_task(
                    backfiller.conversation.backfill_lexical_index(
                        batch_size=1, max_rows=1
                    )
                )
                await asyncio.wait_for(hydrated.wait(), timeout=5)
                await asyncio.wait_for(replacement_started.wait(), timeout=5)
                done, _pending = await asyncio.wait({backfill_task}, timeout=0.05)
                assert not done, "backfill write should wait behind purge"

                pausing_audit.resume.set()
                assert await asyncio.wait_for(purge_task, timeout=5) is True
                result = await asyncio.wait_for(backfill_task, timeout=5)
                assert result["indexed"] == 0
                await _assert_destroyed(purger.db, target)
                assert (
                    await purger.db.fetchval(
                        "SELECT COUNT(*) FROM conversation_lexical_tokens "
                        "WHERE agent_id = ? AND lexical_index_id = ?",
                        (agent_id, replacement_key),
                    )
                    == 0
                )
            finally:
                pausing_audit.resume.set()
                await _cancel_and_observe(purge_task)
                await _cancel_and_observe(backfill_task)


@pytest.mark.asyncio
async def test_cancelled_backfill_token_write_rolls_back_before_purge(
    tmp_path, monkeypatch
):
    """Cancellation cannot commit token residue without its owner marker."""
    db_path = tmp_path / "cancelled-backfill-purge.db"
    agent_id = f"did:test:cancelled-backfill:{uuid4()}"
    loop = asyncio.get_running_loop()
    unobserved: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unobserved.append(context))
    try:
        async with AsyncStorage(str(db_path), agent_id=agent_id) as purger:
            async with AsyncStorage(str(db_path), agent_id=agent_id) as backfiller:
                target = await _seed_indexed_message(
                    purger.db,
                    agent_id,
                    content="cancel after replacement tokens are written",
                )
                await purger.db.execute(
                    "UPDATE conversation_history SET lexical_index_id = NULL, "
                    "lexical_index_version = NULL WHERE id = ?",
                    (target.row_id,),
                )
                await purger.db.execute(
                    "DELETE FROM conversation_lexical_tokens "
                    "WHERE agent_id = ? AND lexical_index_id = ?",
                    (agent_id, target.lexical_index_id),
                )
                replacement_key = (
                    backfiller.conversation._lexical_index.backfill_message_key(
                        target.row_id
                    )
                )

                token_write_complete = asyncio.Event()
                never_resume = asyncio.Event()
                original_execute_many = backfiller.db.execute_many

                async def pause_after_token_write(
                    sql: str, params_list: list[tuple]
                ) -> int:
                    affected = await original_execute_many(sql, params_list)
                    if sql.startswith("INSERT INTO conversation_lexical_tokens"):
                        token_write_complete.set()
                        await never_resume.wait()
                    return affected

                monkeypatch.setattr(
                    backfiller.db, "execute_many", pause_after_token_write
                )
                backfill_task = asyncio.create_task(
                    backfiller.conversation.backfill_lexical_index(
                        batch_size=1, max_rows=1
                    )
                )
                purge_task: asyncio.Task[bool] | None = None
                try:
                    await asyncio.wait_for(token_write_complete.wait(), timeout=5)
                    purge_task = asyncio.create_task(
                        purger.conversation.purge_message(target.row_id)
                    )
                    done, _pending = await asyncio.wait({purge_task}, timeout=0.05)
                    assert not done, "purge should wait for backfill transaction"

                    backfill_task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await backfill_task
                    assert await asyncio.wait_for(purge_task, timeout=5) is True

                    assert (
                        await purger.db.fetchval(
                            "SELECT COUNT(*) FROM conversation_history WHERE id = ?",
                            (target.row_id,),
                        )
                        == 0
                    )
                    assert (
                        await purger.db.fetchval(
                            "SELECT COUNT(*) FROM conversation_lexical_tokens "
                            "WHERE agent_id = ? AND lexical_index_id IN (?, ?)",
                            (agent_id, target.lexical_index_id, replacement_key),
                        )
                        == 0
                    )
                finally:
                    await _cancel_and_observe(backfill_task)
                    await _cancel_and_observe(purge_task)
        await asyncio.sleep(0)
        assert unobserved == []
    finally:
        loop.set_exception_handler(previous_handler)
