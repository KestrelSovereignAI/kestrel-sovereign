"""#2958: ``conversation_history.session_id`` becomes an indexed column.

Session identity has only ever lived inside each row's ``metadata`` JSON, so it
could not be indexed and the conversation list had to oversample and hope. This
covers the additive phase: the column, its index, the one-time backfill that
lifts legacy rows out of metadata, and the write paths that stamp it going
forward.

**Additive means additive.** Phase A changes what the database STORES, not what
any reader does with it and not what any caller may send. Metadata stays
authoritative; session grouping and ingress are untouched, and there are tests
below that fail if that stops being true.

The contract is stated as an equality: for every row, the column equals
:func:`~kestrel_sovereign.storage.session_id_column.column_session_id` of its
metadata. The rule that function implements — and the reason it is narrower
than session grouping's — lives in ``test_session_id_contract.py``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.session_grouping import group_messages_into_sessions
from kestrel_sovereign.storage.session_id_column import (
    SESSION_ID_MAX_LENGTH,
    column_session_id,
)


AGENT = "did:test:session-column"
UUID_A = "8f1d1c62-9b0e-4b2c-9a1d-000000000001"
UUID_B = "8f1d1c62-9b0e-4b2c-9a1d-000000000002"

# The shape before this migration: no session_id column, no dependent index.
PRE_MIGRATION_SCHEMA = """
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP DEFAULT NULL
);
CREATE INDEX idx_conversation_agent_id ON conversation_history(agent_id);
"""

# (label, stored metadata, expected session_id column) — every row a live
# database can actually hold.  ``expected`` is spelled out rather than computed
# so the test states the contract instead of restating the implementation.
LEGACY_ROWS = [
    ("uuid", json.dumps({"session_id": UUID_A}), UUID_A),
    ("uuid marker", json.dumps({"session_id": UUID_B, "new_session": True}), UUID_B),
    ("sess key", json.dumps({"session_id": "sess_9fk21xa"}), "sess_9fk21xa"),
    # #2012: the list endpoint keyed sessions by row id and the UI echoed the
    # integer back. Grouping ignores these, so the column must not adopt one.
    ("bare integer", json.dumps({"session_id": "1314"}), None),
    ("json number", json.dumps({"session_id": 1314}), None),
    ("absent", json.dumps({"role_hint": "user"}), None),
    ("empty string", json.dumps({"session_id": ""}), None),
    ("json null", json.dumps({"session_id": None}), None),
    ("no metadata", None, None),
    # Written by nothing, but a migration that RAISES here fails the whole
    # boot rather than one row — the P1 this ticket was rewritten around.
    ("malformed metadata", "{not json", None),
    ("nul escape", '{"session_id": "\\u0000"}', None),
    ("embedded nul escape", '{"session_id": "a\\u0000b"}', None),
    ("at the length limit", json.dumps({"session_id": "b" * SESSION_ID_MAX_LENGTH}),
     "b" * SESSION_ID_MAX_LENGTH),
    ("over the length limit",
     json.dumps({"session_id": "b" * (SESSION_ID_MAX_LENGTH + 1)}), None),
    # Values whose spelling depends on who read them.
    ("json true", json.dumps({"session_id": True}), None),
    ("json false", json.dumps({"session_id": False}), None),
    ("json object", json.dumps({"session_id": {"nested": UUID_A}}), None),
    ("json array", json.dumps({"session_id": [UUID_A]}), None),
    ("json float", json.dumps({"session_id": 1.5}), None),
    # Renders as ``-5``: inside the charset, not all digits. Only the JSON
    # type test keeps it out, which is what makes that clause load-bearing.
    ("json negative number", json.dumps({"session_id": -5}), None),
    # str.isdigit() calls this digits and neither SQL dialect does.
    ("unicode digits", json.dumps({"session_id": "١٢٣"}), None),
    ("outside the charset", json.dumps({"session_id": "did:x:1"}), None),
    # Metadata that is valid JSON but not an object has no session_id at all.
    ("metadata not an object", json.dumps([{"session_id": UUID_A}]), None),
]


def _seed_pre_migration_db(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(PRE_MIGRATION_SCHEMA)
        for label, metadata, _expected in LEGACY_ROWS:
            conn.execute(
                "INSERT INTO conversation_history (agent_id, role, content, metadata) "
                "VALUES (?, ?, ?, ?)",
                (AGENT, "user", label, metadata),
            )
        conn.commit()
    finally:
        conn.close()


def _grouped_session_id(metadata):
    """The session id ``group_messages_into_sessions`` files a lone row under.

    ``keep_empty_markers`` so a ``new_session`` marker — structural, and the
    only row in these fixtures — is still returned rather than dropped as an
    empty session.
    """
    sessions = group_messages_into_sessions(
        [{
            "id": 1,
            "role": "user",
            "content": "x",
            "metadata": metadata,
            "created_at": "2026-01-01T00:00:00+00:00",
        }],
        keep_empty_markers=True,
    )
    return sessions[0]["session_id"] if sessions else None


async def _rows(db):
    return await db.fetchall(
        "SELECT content, metadata, session_id FROM conversation_history "
        "ORDER BY id",
        (),
    )


async def _index_names(db):
    rows = await db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name='conversation_history'",
        (),
    )
    return {row[0] for row in rows}


@pytest.mark.asyncio
async def test_backfill_lifts_only_stampable_session_ids(tmp_path):
    db_path = tmp_path / "pre-migration.db"
    _seed_pre_migration_db(str(db_path))

    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        observed = {content: session_id for content, _meta, session_id in await _rows(db)}
        assert observed == {
            label: expected for label, _metadata, expected in LEGACY_ROWS
        }
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_boot_against_only_unparseable_metadata_still_completes(tmp_path):
    """Malformed and NUL-bearing metadata must not be able to brick startup.

    The migration is mandatory — ``migrate_columns_once`` raises if the column
    does not land — so a backfill that raised on a poison row would take the
    whole boot with it, not one row. A table made ENTIRELY of poison is the
    strongest form of that: nothing else can be carrying the migration.
    """
    db_path = tmp_path / "poison.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(PRE_MIGRATION_SCHEMA)
        for content, metadata in (
            ("malformed", "{not json"),
            ("truncated", '{"session_id": "abc'),
            ("nul", '{"session_id": "\\u0000"}'),
            ("not an object", "42"),
            ("empty text", ""),
        ):
            conn.execute(
                "INSERT INTO conversation_history (agent_id, role, content, metadata) "
                "VALUES (?, ?, ?, ?)",
                (AGENT, "user", content, metadata),
            )
        conn.commit()
    finally:
        conn.close()

    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        assert [session_id for _c, _m, session_id in await _rows(db)] == [None] * 5
        assert any(
        n.startswith("idx_conversation_agent_session_")
        for n in await _index_names(db)
    )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_backfilled_column_never_disagrees_with_session_grouping(tmp_path):
    """The acceptance statement, row by row against the real grouper.

    Not "equals" — the contract is deliberately narrower, so the column may be
    NULL for a row grouping still files under a session id. What it may never
    do is name a different one.
    """
    db_path = tmp_path / "pre-migration.db"
    _seed_pre_migration_db(str(db_path))

    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        rows = await _rows(db)
        assert len(rows) == len(LEGACY_ROWS)
        for content, metadata, session_id in rows:
            assert session_id == column_session_id(metadata), content
            if session_id is None:
                continue
            assert _grouped_session_id(json.loads(metadata)) == session_id, content
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_legacy_database_gains_the_column_and_its_index(tmp_path):
    db_path = tmp_path / "pre-migration.db"
    _seed_pre_migration_db(str(db_path))

    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        columns = {
            row[1] for row in await db.fetchall("PRAGMA table_info(conversation_history)")
        }
        assert "session_id" in columns
        assert any(
        n.startswith("idx_conversation_agent_session_")
        for n in await _index_names(db)
    )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fresh_database_gains_the_column_and_its_index(tmp_path):
    """A database created from CORE_SCHEMA must land in the same shape."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "fresh.db"))
    try:
        columns = {
            row[1] for row in await db.fetchall("PRAGMA table_info(conversation_history)")
        }
        assert "session_id" in columns
        assert any(
        n.startswith("idx_conversation_agent_session_")
        for n in await _index_names(db)
    )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_migration_is_idempotent_across_boots(tmp_path):
    """Re-opening must not re-run the backfill or disturb newer writes."""
    db_path = tmp_path / "pre-migration.db"
    _seed_pre_migration_db(str(db_path))

    db = await AsyncDatabase.sqlite(str(db_path))
    first = await _rows(db)
    await db.close()

    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        assert await _rows(db) == first
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_write_path_stamps_the_column(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "writes.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "hello", session_id=UUID_A)

        rows = await _rows(db)
        assert len(rows) == 1
        _content, metadata, session_id = rows[0]
        assert session_id == UUID_A
        assert json.loads(metadata)["session_id"] == UUID_A
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_write_path_stamps_the_derived_implicit_session(tmp_path):
    """No explicit id: the column follows whatever derivation put in metadata.

    Also pins that the minted UUIDs are inside the contract's charset — if they
    were not, every ordinary turn would land with a NULL column and the index
    would be worthless.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "implicit.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "hello")
        await store.add_conversation("assistant", "hi")

        for _content, metadata, session_id in await _rows(db):
            assert session_id == json.loads(metadata)["session_id"]
            assert session_id is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_write_path_leaves_a_bare_integer_session_id_null(tmp_path):
    """An echoed row-id that canonicalization cannot resolve stays out of the column."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "integer.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "first")
        # "1" names the row above, which is NOT a new_session marker, so
        # canonicalization leaves the integer in place (#2012).
        await store.add_conversation("user", "echoed", session_id="1")

        _content, metadata, session_id = (await _rows(db))[1]
        assert json.loads(metadata)["session_id"] == "1"
        assert session_id is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_an_unstampable_caller_id_is_stored_but_not_indexed(tmp_path):
    """Phase A adds a column. It does not add an ingress rule.

    The column contract is narrower than what callers may send, and closing
    that gap by rejecting or rewriting caller input would change endpoint
    behaviour under cover of an additive migration (#2958 Finding 5). So an id
    outside the charset still reaches metadata verbatim, still groups exactly
    as it did before, and simply leaves the column NULL — the state legacy rows
    are already in and the one Phase C has to tolerate anyway.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "ingress.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "hello", session_id="did:x:1")

        _content, metadata, session_id = (await _rows(db))[0]
        assert json.loads(metadata)["session_id"] == "did:x:1"
        assert session_id is None

        # ...and grouping still files it under the id it always did.
        assert _grouped_session_id(json.loads(metadata)) == "did:x:1"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_resolve_session_id_still_echoes_what_the_caller_sent(tmp_path):
    """``X-Session-Id`` names the session the row is filed under, as before.

    Filing is metadata's job in Phase A, so an id the column cannot hold is
    still the correct answer here. A Phase-A patch that started normalizing at
    this seam would be changing the endpoint contract, not adding a column.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "resolve.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        assert await store.resolve_session_id("did:x:1") == "did:x:1"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_boot_converges_a_relinked_legacy_row_onto_the_column(tmp_path):
    """#2012's relink rewrites metadata; the derived column must follow it.

    One boot runs the backfill and then the relink. The backfill correctly
    leaves the integer-keyed row NULL — but the relink then moves that row's
    metadata onto the marker's UUID. If the relink did not carry the column
    with it, the row would end the boot with metadata naming a canonical
    session and a column claiming it has none: exactly the drift the column
    exists to make impossible.
    """
    db_path = tmp_path / "relink.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(PRE_MIGRATION_SCHEMA)
        conn.execute(
            "INSERT INTO conversation_history (id, agent_id, role, content, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, AGENT, "user", "marker",
             json.dumps({"session_id": UUID_A, "new_session": True})),
        )
        # The UI echoed the marker's row id back as the session key.
        conn.execute(
            "INSERT INTO conversation_history (id, agent_id, role, content, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (2, AGENT, "user", "continued", json.dumps({"session_id": "1"})),
        )
        conn.commit()
    finally:
        conn.close()

    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        for _content, metadata, session_id in await _rows(db):
            assert json.loads(metadata)["session_id"] == UUID_A
            assert session_id == UUID_A
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_write_path_still_relinks_an_integer_session_id(tmp_path):
    """The #2958 column must not disturb the #2012 row-id echo it sits beside.

    The marker is deliberately STALE. With a recent one, the time-gap heuristic
    hands back the same UUID the relink would, so the test would pass whether
    or not canonicalization ran — it would assert nothing. Beyond the gap the
    fallback mints a fresh UUID, so only a genuine relink produces ``UUID_A``.
    """
    db_path = tmp_path / "echo.db"
    stale = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(PRE_MIGRATION_SCHEMA)
        conn.execute(
            "INSERT INTO conversation_history "
            "(id, agent_id, role, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (1, AGENT, "user", "marker",
             json.dumps({"session_id": UUID_A, "new_session": True}), stale),
        )
        conn.commit()
    finally:
        conn.close()

    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "continued", session_id="1")

        _content, metadata, session_id = (await _rows(db))[1]
        assert json.loads(metadata)["session_id"] == UUID_A
        assert session_id == UUID_A
    finally:
        await db.close()


@pytest.mark.parametrize(
    ("backend_type", "expected", "expected_params"),
    [
        (
            "sqlite",
            (
                "json_extract(metadata, '$.session_id')",
                "json_type(metadata, '$.session_id') = 'text'",
                "json_valid(metadata) = 1",
                # SQLite has no cast to fail, so the NUL is excluded by hand:
                # its length() and GLOB stop at the first one and would measure
                # a NUL-bearing id as short and in-charset.
                "instr(metadata, ?) = 0",
            ),
            ("\\u0000",),
        ),
        (
            "postgres",
            (
                "metadata::jsonb ->> 'session_id'",
                "jsonb_typeof(metadata::jsonb -> 'session_id') = 'string'",
                # Cast safety, and it must be pg_input_is_valid rather than an
                # enumeration of the documents jsonb rejects: `IS JSON OBJECT`
                # plus a NUL search passes a lone surrogate, which then raises
                # on the cast and rolls back the migration.
                "pg_input_is_valid(metadata, 'jsonb')",
                "metadata IS JSON OBJECT",
            ),
            (),
        ),
    ],
)
def test_backfill_reads_each_backend_with_its_own_json_dialect(
    backend_type, expected, expected_params
):
    """SQLite's json_extract and Postgres' jsonb operator are not interchangeable.

    Postgres cannot run ``json_extract`` at all, so the dialect choice is
    asserted here — the SQLite-only CI leg still proves the Postgres branch was
    written. What the two statements MEAN is settled against real engines in
    ``tests/integration/test_session_id_column_backend_parity.py``.

    The two dialects no longer take the same parameters, and that asymmetry is
    the point rather than an accident: PostgreSQL asks the parser whether the
    cast will succeed, so it needs no literal to search for, while SQLite has
    no cast to ask about and must exclude the NUL escape by hand.
    """
    db = AsyncDatabase(SimpleNamespace(backend_type=backend_type))
    sql, params = db._conversation_session_id_backfill()

    assert params == expected_params
    assert sql.count("?") == len(params)
    for fragment in expected:
        assert fragment in sql
    assert sql.startswith("UPDATE conversation_history SET session_id = ")
