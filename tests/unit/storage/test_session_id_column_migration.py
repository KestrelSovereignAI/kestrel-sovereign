"""#2958: ``conversation_history.session_id`` becomes an indexed column.

Session identity has only ever lived inside each row's ``metadata`` JSON, so it
could not be indexed and the conversation list had to oversample and hope. This
covers the additive phase: the column, its index, and the one-time backfill
that lifts legacy rows out of metadata.

The contract under test is stated as an equality, not a list of cases: for
every row, the column equals
:func:`~kestrel_sovereign.storage.session_grouping.canonical_session_id` of its
metadata — the same rule ``group_messages_into_sessions`` applies when it
decides which id a session is filed under. A bare-integer id is a mis-filed
legacy key (#2012) that grouping ignores, so it must stay NULL rather than be
promoted into the indexed column.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.session_grouping import canonical_session_id


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
    # #2012: the list endpoint keyed sessions by row id and the UI echoed the
    # integer back. Grouping ignores these, so the column must not adopt one.
    ("bare integer", json.dumps({"session_id": "1314"}), None),
    ("json number", json.dumps({"session_id": 1314}), None),
    ("absent", json.dumps({"role_hint": "user"}), None),
    ("empty string", json.dumps({"session_id": ""}), None),
    ("json null", json.dumps({"session_id": None}), None),
    ("no metadata", None, None),
    # Not written by any code path, but a migration that raises here would
    # fail the whole boot rather than one row.
    ("malformed metadata", "{not json", None),
    # ── Values that render DIFFERENTLY in each reader ────────────────────
    # A session id must be a JSON string. These are the shapes that made the
    # three implementations of the rule disagree before it was stated once:
    # ``true`` extracts as 1 in SQLite, 'true' in Postgres and 'True' under
    # Python's ``str()``; a JSON object comes back as compact JSON text from
    # SQLite and as a dict repr from Python. Nothing may be filed under an
    # identity whose spelling depends on who read it, so all of them are NULL.
    ("json true", json.dumps({"session_id": True}), None),
    ("json false", json.dumps({"session_id": False}), None),
    ("json object", json.dumps({"session_id": {"nested": UUID_A}}), None),
    ("json array", json.dumps({"session_id": [UUID_A]}), None),
    ("json float", json.dumps({"session_id": 1.5}), None),
    # Not a bare integer by the ASCII rule the SQL implements — and NOT
    # excludable via ``str.isdigit()``, which calls this digits. It is a string
    # containing no ASCII 0-9, so it is a (strange but usable) session id and
    # every backend agrees on its spelling.
    ("unicode digits", json.dumps({"session_id": "١٢٣"}), "١٢٣"),
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
async def test_backfill_lifts_only_canonical_session_ids(tmp_path):
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
async def test_backfilled_column_agrees_with_session_grouping(tmp_path):
    """The acceptance statement, asserted row by row against the shared rule."""
    db_path = tmp_path / "pre-migration.db"
    _seed_pre_migration_db(str(db_path))

    db = await AsyncDatabase.sqlite(str(db_path))
    try:
        rows = await _rows(db)
        assert len(rows) == len(LEGACY_ROWS)
        for content, metadata, session_id in rows:
            assert session_id == canonical_session_id(metadata), (
                f"column disagrees with session grouping for {content!r}"
            )
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
        assert "idx_conversation_agent_session" in await _index_names(db)
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
        assert "idx_conversation_agent_session" in await _index_names(db)
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
    """No explicit id: the column follows whatever derivation put in metadata."""
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


@pytest.mark.parametrize(
    ("backend_type", "expected"),
    [
        (
            "sqlite",
            (
                "json_extract(metadata, '$.session_id')",
                "json_type(metadata, '$.session_id') = 'text'",
                "GLOB '*[^0-9]*'",
            ),
        ),
        (
            "postgres",
            (
                "metadata::jsonb ->> 'session_id'",
                "jsonb_typeof(metadata::jsonb -> 'session_id') = 'string'",
                "~ '[^0-9]'",
            ),
        ),
    ],
)
def test_backfill_reads_each_backend_with_its_own_json_dialect(backend_type, expected):
    """SQLite's json_extract and Postgres' jsonb operator are not interchangeable.

    Postgres cannot run ``json_extract`` at all, and neither the type test nor
    the digit test has a portable spelling, so the dialect choice is asserted
    here — the SQLite-only CI run still proves the Postgres branch was written.
    """
    db = AsyncDatabase(SimpleNamespace(backend_type=backend_type))
    sql, params = db._conversation_session_id_backfill()

    assert params == ()
    for fragment in expected:
        assert fragment in sql
    assert sql.startswith("UPDATE conversation_history SET session_id = ")


@pytest.mark.asyncio
async def test_write_path_refuses_a_session_id_that_is_not_a_string(tmp_path):
    """Ingress owns the type, so no row is ever stored outside the rule's reach.

    A JSON body is unvalidated, so ``session_id`` can arrive as any JSON value.
    Stamping one into metadata would write a row whose metadata names a session
    the column (and every reader) refuses — so the write path drops it and the
    turn joins the time-gap session, exactly as an omitted field would.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "typed.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        await store.add_conversation("user", "first", session_id=UUID_A)
        for unsupported in (True, 1.5, {"nested": UUID_A}, [UUID_A]):
            await store.add_conversation("user", "garbage", session_id=unsupported)

        for content, metadata, session_id in await _rows(db):
            stored = json.loads(metadata)["session_id"]
            assert isinstance(stored, str), content
            assert session_id == canonical_session_id(metadata), content
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_write_path_still_relinks_an_integer_session_id_sent_as_a_number(
    tmp_path,
):
    """Normalizing the type must not break the #2012 row-id echo it exists for.

    Older UI clients round-trip the marker's row id. Sending it as a JSON
    number rather than a string is the same client mistake, so it must reach
    canonicalization and be relinked — dropping it as "not a string" would
    regress #2012 in the name of #2958.

    The marker is deliberately STALE. With a recent one, the time-gap heuristic
    hands back the same UUID the relink would, so the test passes whether or not
    the number ever reached canonicalization — it would assert nothing. Beyond
    the gap, the fallback mints a fresh UUID, so only a genuine relink produces
    ``UUID_A``.
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
        await store.add_conversation("user", "continued", session_id=1)

        _content, metadata, session_id = (await _rows(db))[1]
        assert json.loads(metadata)["session_id"] == UUID_A
        assert session_id == UUID_A
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_resolved_session_id_matches_what_the_write_path_will_file(tmp_path):
    """The echoed ``X-Session-Id`` may not name a session no row can carry."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "resolve.db"))
    try:
        store = AsyncConversationStore(db, agent_id=AGENT)
        resolved = await store.resolve_session_id(True)

        assert resolved is not True
        assert resolved is None or isinstance(resolved, str)

        await store.add_conversation("user", "hello", session_id=True)
        _content, metadata, session_id = (await _rows(db))[0]
        assert json.loads(metadata)["session_id"] != True  # noqa: E712
        assert session_id == canonical_session_id(metadata)
    finally:
        await db.close()
