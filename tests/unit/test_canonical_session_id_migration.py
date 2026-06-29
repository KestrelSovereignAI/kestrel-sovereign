"""Tests for the canonical-session-id data migration (#2012).

``migrate_canonical_session_ids`` relinks conversation messages whose
``metadata.session_id`` was stored as a bare integer (the conversation-list
endpoint's old row-id key, echoed back by the web UI) to the canonical UUID
carried on the session's ``new_session`` marker — in both
``conversation_history`` and ``conversation_titles``.

Genuine legacy time-gap anchors (an integer naming a plain first message with
no marker UUID) are deliberately left alone. The migration runs on every
startup via ``AsyncDatabase._init_schema``, so idempotency is contractual.
"""

import json

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_canonical_session_ids,
)


async def _db(tmp_path):
    raw = SQLiteBackend(str(tmp_path / "canonical-session-migration.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    await db._init_schema()
    return db


async def _insert(db, role, content, metadata, agent_id="test-agent"):
    """Insert a row and return its autoincrement id."""
    await db.execute(
        "INSERT INTO conversation_history (agent_id, role, content, metadata) "
        "VALUES (?, ?, ?, ?)",
        (agent_id, role, content, json.dumps(metadata) if metadata is not None else None),
    )
    row = await db.fetchone(
        "SELECT id FROM conversation_history ORDER BY id DESC LIMIT 1", ()
    )
    return row[0]


async def _session_ids(db):
    rows = await db.fetchall(
        "SELECT id, metadata FROM conversation_history ORDER BY id", ()
    )
    out = {}
    for row_id, raw in rows:
        meta = json.loads(raw) if raw else {}
        out[row_id] = meta.get("session_id")
    return out


@pytest.mark.asyncio
async def test_relinks_integer_session_id_to_marker_uuid(tmp_path):
    db = await _db(tmp_path)
    uuid = "e1fd6fe5-885e-4d8b-9aaa-000000000001"
    marker_id = await _insert(
        db, "system", "", {"new_session": True, "session_id": uuid}
    )
    # Continued turns mis-filed under the marker's integer row-id.
    u1 = await _insert(db, "user", "hi", {"session_id": str(marker_id)})
    a1 = await _insert(db, "assistant", "hello", {"session_id": str(marker_id)})

    await migrate_canonical_session_ids(db)

    sids = await _session_ids(db)
    assert sids[marker_id] == uuid  # marker untouched (already UUID)
    assert sids[u1] == uuid
    assert sids[a1] == uuid


@pytest.mark.asyncio
async def test_leaves_legacy_timegap_anchor_alone(tmp_path):
    db = await _db(tmp_path)
    # A plain first message (NOT a new_session marker) — its row-id is a
    # legitimate legacy session key with no UUID to map to.
    anchor_id = await _insert(db, "user", "first", {"session_id": "0"})
    cont_id = await _insert(db, "assistant", "reply", {"session_id": str(anchor_id)})

    await migrate_canonical_session_ids(db)

    sids = await _session_ids(db)
    assert sids[cont_id] == str(anchor_id)  # unchanged


@pytest.mark.asyncio
async def test_leaves_uuid_session_ids_alone(tmp_path):
    db = await _db(tmp_path)
    uuid = "722e2ba0-0000-0000-0000-000000000002"
    marker_id = await _insert(
        db, "system", "", {"new_session": True, "session_id": uuid}
    )
    u1 = await _insert(db, "user", "q", {"session_id": uuid})

    await migrate_canonical_session_ids(db)

    sids = await _session_ids(db)
    assert sids[marker_id] == uuid
    assert sids[u1] == uuid


@pytest.mark.asyncio
async def test_remaps_conversation_title(tmp_path):
    db = await _db(tmp_path)
    uuid = "53519af8-0000-0000-0000-000000000003"
    marker_id = await _insert(
        db, "system", "", {"new_session": True, "session_id": uuid}
    )
    await db.execute(
        "INSERT INTO conversation_titles (agent_id, session_id, name) "
        "VALUES (?, ?, ?)",
        ("test-agent", str(marker_id), "My chat"),
    )

    await migrate_canonical_session_ids(db)

    rows = await db.fetchall(
        "SELECT session_id, name FROM conversation_titles", ()
    )
    assert rows == [(uuid, "My chat")]


@pytest.mark.asyncio
async def test_title_collision_drops_integer_keeps_uuid(tmp_path):
    db = await _db(tmp_path)
    uuid = "60402f43-0000-0000-0000-000000000004"
    marker_id = await _insert(
        db, "system", "", {"new_session": True, "session_id": uuid}
    )
    # Both an integer-keyed and a UUID-keyed name exist; the UUID wins.
    await db.execute(
        "INSERT INTO conversation_titles (agent_id, session_id, name) VALUES (?, ?, ?)",
        ("test-agent", str(marker_id), "old name"),
    )
    await db.execute(
        "INSERT INTO conversation_titles (agent_id, session_id, name) VALUES (?, ?, ?)",
        ("test-agent", uuid, "canonical name"),
    )

    await migrate_canonical_session_ids(db)

    rows = await db.fetchall(
        "SELECT session_id, name FROM conversation_titles ORDER BY session_id", ()
    )
    assert rows == [(uuid, "canonical name")]


@pytest.mark.asyncio
async def test_idempotent_second_run_is_noop(tmp_path):
    db = await _db(tmp_path)
    uuid = "e81797a9-0000-0000-0000-000000000005"
    marker_id = await _insert(
        db, "system", "", {"new_session": True, "session_id": uuid}
    )
    await _insert(db, "user", "hi", {"session_id": str(marker_id)})

    await migrate_canonical_session_ids(db)
    first = await _session_ids(db)
    await migrate_canonical_session_ids(db)
    second = await _session_ids(db)

    assert first == second
    assert all(v == uuid for v in first.values())


@pytest.mark.asyncio
async def test_init_schema_runs_migration_on_startup(tmp_path):
    raw = SQLiteBackend(str(tmp_path / "startup-canonical.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    await db._init_schema()
    uuid = "aaaaaaaa-0000-0000-0000-000000000006"
    marker_id = await _insert(
        db, "system", "", {"new_session": True, "session_id": uuid}
    )
    cont_id = await _insert(db, "user", "hi", {"session_id": str(marker_id)})

    # Next boot over the same backend relinks the row.
    await db._init_schema()

    sids = await _session_ids(db)
    assert sids[cont_id] == uuid
