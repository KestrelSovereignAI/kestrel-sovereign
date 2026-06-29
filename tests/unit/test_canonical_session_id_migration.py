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
async def test_skips_marker_that_inherited_prior_session_uuid(tmp_path):
    """A legacy new_session marker written without an explicit id could
    INHERIT the previous still-active session's UUID via the time-gap
    heuristic. The migration must NOT map the marker's row-id to that
    inherited UUID — doing so would merge two distinct conversations."""
    db = await _db(tmp_path)
    shared_uuid = "abababab-0000-0000-0000-000000000007"
    # Conversation A owns the UUID (earliest row carrying it).
    await _insert(db, "user", "A1", {"session_id": shared_uuid})
    # An inherited new_session marker carrying the SAME UUID.
    inherited_marker = await _insert(
        db, "system", "", {"new_session": True, "session_id": shared_uuid}
    )
    # Conversation B's continued turn, mis-filed under the marker row-id.
    b_turn = await _insert(db, "user", "B1", {"session_id": str(inherited_marker)})

    await migrate_canonical_session_ids(db)

    sids = await _session_ids(db)
    # B's turn must NOT have been merged into A's UUID — left as the integer.
    assert sids[b_turn] == str(inherited_marker)


@pytest.mark.asyncio
async def test_relinks_double_marker_same_session(tmp_path):
    """Two back-to-back new_session markers can share a UUID (the second
    inherited it from the first, with NO content row in between) — that is
    ONE session, not a prior conversation. Continued turns keyed by the
    second marker's row-id must relink to the shared UUID (the live-Emma
    1313/1314 MCP case)."""
    db = await _db(tmp_path)
    shared_uuid = "e1fd6fe5-885e-43b2-b6e2-cfbea64f66a2"
    await _insert(db, "system", "", {"new_session": True, "session_id": shared_uuid})
    marker2 = await _insert(
        db, "system", "", {"new_session": True, "session_id": shared_uuid}
    )
    u1 = await _insert(db, "user", "hi", {"session_id": str(marker2)})
    a1 = await _insert(db, "assistant", "hello", {"session_id": str(marker2)})

    await migrate_canonical_session_ids(db)

    sids = await _session_ids(db)
    assert sids[u1] == shared_uuid
    assert sids[a1] == shared_uuid


@pytest.mark.asyncio
async def test_skips_duplicate_uuid_markers_each_with_own_content(tmp_path):
    """codex: two markers share a UUID but EACH already has its own continued
    messages keyed by its integer row-id (two distinct conversations that
    collided on the UUID via the inheritance bug). The migration must NOT
    relink either — that would merge the two conversations."""
    db = await _db(tmp_path)
    shared_uuid = "dddddddd-0000-0000-0000-00000000000c"
    m1 = await _insert(db, "system", "", {"new_session": True, "session_id": shared_uuid})
    c1 = await _insert(db, "user", "conv1", {"session_id": str(m1)})
    m2 = await _insert(db, "system", "", {"new_session": True, "session_id": shared_uuid})
    c2 = await _insert(db, "user", "conv2", {"session_id": str(m2)})

    await migrate_canonical_session_ids(db)

    sids = await _session_ids(db)
    # Neither conversation merged into the shared UUID — both stay split.
    assert sids[c1] == str(m1)
    assert sids[c2] == str(m2)


@pytest.mark.asyncio
async def test_skips_two_marker_collision_uuid_then_integer(tmp_path):
    """The live-Emma 4152cc73 shape: marker A mints the UUID and owns turns
    filed UNDER the UUID; minutes later marker B inherits the same UUID and
    owns turns filed under B's integer row-id. Two distinct conversations
    share the UUID — the migration must NOT merge B's turns into A."""
    db = await _db(tmp_path)
    uuid = "ababcdcd-0000-0000-0000-00000000000e"
    marker_a = await _insert(db, "system", "", {"new_session": True, "session_id": uuid})
    a1 = await _insert(db, "user", "convA-1", {"session_id": uuid})
    a2 = await _insert(db, "assistant", "convA-2", {"session_id": uuid})
    marker_b = await _insert(db, "system", "", {"new_session": True, "session_id": uuid})
    b1 = await _insert(db, "user", "convB-1", {"session_id": str(marker_b)})

    await migrate_canonical_session_ids(db)

    sids = await _session_ids(db)
    # Conversation A keeps its UUID rows; conversation B stays under its
    # integer key — NOT merged.
    assert sids[a1] == uuid and sids[a2] == uuid
    assert sids[b1] == str(marker_b)


@pytest.mark.asyncio
async def test_collision_analysis_is_agent_scoped(tmp_path):
    """codex: a UUID reused across two agents (imported/restored data) is NOT
    a collision — each agent's session must still consolidate independently."""
    db = await _db(tmp_path)
    uuid = "5a5a5a5a-0000-0000-0000-00000000000f"
    # Agent A: marker + integer-keyed continuation.
    ma = await _insert(db, "system", "", {"new_session": True, "session_id": uuid}, agent_id="agent-A")
    a1 = await _insert(db, "user", "A-turn", {"session_id": str(ma)}, agent_id="agent-A")
    # Agent B: same UUID, its own marker + continuation.
    mb = await _insert(db, "system", "", {"new_session": True, "session_id": uuid}, agent_id="agent-B")
    b1 = await _insert(db, "user", "B-turn", {"session_id": str(mb)}, agent_id="agent-B")

    await migrate_canonical_session_ids(db)

    sids = await _session_ids(db)
    # Each agent's integer-keyed turn relinks to the (shared) UUID — neither
    # is skipped as a false cross-agent collision.
    assert sids[a1] == uuid
    assert sids[b1] == uuid


@pytest.mark.asyncio
async def test_consolidates_single_marker_mixed_key_session(tmp_path):
    """codex: a SINGLE-marker session whose turns are split between the
    canonical UUID and the marker's integer row-id is ONE conversation (one
    client wrote the UUID, the UI wrote the integer). It must consolidate —
    the integer-keyed rows relink to the UUID."""
    db = await _db(tmp_path)
    uuid = "cccccccc-0000-0000-0000-00000000000d"
    marker = await _insert(db, "system", "", {"new_session": True, "session_id": uuid})
    # Some turns already carry the canonical UUID (after the marker)...
    uuid_turn = await _insert(db, "user", "via-uuid", {"session_id": uuid})
    # ...others were mis-filed under the marker's integer row-id.
    int_turn = await _insert(db, "assistant", "via-int", {"session_id": str(marker)})

    await migrate_canonical_session_ids(db)

    sids = await _session_ids(db)
    assert sids[uuid_turn] == uuid
    assert sids[int_turn] == uuid  # relinked, not left split


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
