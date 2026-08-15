"""#2958: the session_id column means the same thing on SQLite and PostgreSQL.

The backfill is the one part of this migration that cannot be written once —
SQLite reads metadata with ``json_extract`` and PostgreSQL with a ``jsonb``
operator, and none of the rule's clauses has a portable spelling. The rule
itself is authored once and compiled into both (see
``kestrel_sovereign/storage/session_id_column.py``); what *this* file does is
run those compilations against real engines, because a compiler that emits
plausible SQL is worth nothing until an engine agrees.

Two classes of failure live here and nowhere else:

* **Disagreement.** A value the two dialects extract differently means an
  upgraded row is filed under a different identity depending on which backend
  ran the migration.
* **Refusal.** PostgreSQL — not SQLite — raises on ``metadata::jsonb`` when the
  document is malformed or carries a NUL escape, and refuses a B-tree entry
  over ~2704 bytes. Inside a mandatory migration each of those is a failed
  boot, not a skipped row, which is why they are exercised against a server
  rather than reasoned about.

Run the PostgreSQL leg with::

    TEST_POSTGRES_URL=postgresql://u:p@127.0.0.1:5432/db uv run pytest \\
        tests/integration/test_session_id_column_backend_parity.py

A skipped PostgreSQL case is not a passing one. When a DSN is configured this
file must report zero skips.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.session_grouping import group_messages_into_sessions
from kestrel_sovereign.storage.session_id_column import (
    SESSION_ID_MAX_LENGTH,
    column_session_id,
)

pytestmark = pytest.mark.integration

POSTGRES_URL = (
    os.environ.get("TEST_POSTGRES_URL")
    or os.environ.get("KESTREL_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
)

UUID_A = "3a6c1f0e-2b7d-4c8a-9e10-000000000001"

# A JSON document whose session_id is a NUL. Written as the escape the way any
# ``json.dumps`` would store it — PostgreSQL accepts this text as JSON and then
# raises when asked to cast it to jsonb, which is precisely the trap.
NUL_METADATA = '{"session_id": "\\u0000"}'
EMBEDDED_NUL_METADATA = '{"session_id": "a\\u0000b"}'

# Longer than a PostgreSQL B-tree entry can hold. If this ever reached the
# column, ``CREATE INDEX idx_conversation_agent_session`` would fail — at boot.
OVERSIZED = "z" * 4000

# JSON permits a key twice, and each reader resolves it differently: SQLite's
# ``json_extract`` takes the FIRST occurrence, PostgreSQL's ``jsonb`` and
# Python's ``json.loads`` take the LAST. Written as text rather than dumped,
# because no ``json.dumps`` can produce it.
DUP_FIRST = "aaaa-1111"
DUP_LAST = "bbbb-2222"
DUPLICATED_METADATA = f'{{"session_id": "{DUP_FIRST}", "session_id": "{DUP_LAST}"}}'
# A duplicate one level down leaves the top level unambiguous, and a key that
# merely CONTAINS the key is a different key. Both are shapes real rows have,
# and both would be lost by a guard that counted substrings instead of keys.
NESTED_DUPLICATE_METADATA = (
    '{"nested": {"session_id": "x1", "session_id": "y2"}, '
    '"session_id": "cccc-3333"}'
)
NEIGHBOURING_KEY_METADATA = (
    '{"parent_session_id": "aaaa-1111", "session_id": "cccc-3333"}'
)

# (label, stored metadata, expected column value).
CASES = [
    ("uuid", json.dumps({"session_id": UUID_A}), UUID_A),
    ("sess key", json.dumps({"session_id": "sess_9fk21xa"}), "sess_9fk21xa"),
    ("bare integer", json.dumps({"session_id": "1314"}), None),
    ("json number", json.dumps({"session_id": 1314}), None),
    ("absent", json.dumps({"role_hint": "user"}), None),
    ("empty string", json.dumps({"session_id": ""}), None),
    ("json null", json.dumps({"session_id": None}), None),
    ("no metadata", None, None),
    # The three refusal shapes. Each one aborted the migration before this
    # ticket was rewritten around them.
    ("malformed metadata", '{"session_id": "oops', None),
    ("not json at all", "{not json", None),
    ("nul escape", NUL_METADATA, None),
    ("embedded nul escape", EMBEDDED_NUL_METADATA, None),
    ("oversized", json.dumps({"session_id": OVERSIZED}), None),
    ("at the length limit", json.dumps({"session_id": "b" * SESSION_ID_MAX_LENGTH}),
     "b" * SESSION_ID_MAX_LENGTH),
    ("one over the limit",
     json.dumps({"session_id": "b" * (SESSION_ID_MAX_LENGTH + 1)}), None),
    # Values that render differently in each reader: ``true`` extracts as 1 in
    # SQLite, 'true' in PostgreSQL and 'True' under Python's ``str()``; an
    # object comes back as compact JSON text from SQLite and a dict repr from
    # Python. Nothing may be filed under an identity whose spelling depends on
    # who read it.
    ("json true", json.dumps({"session_id": True}), None),
    ("json false", json.dumps({"session_id": False}), None),
    ("json object", json.dumps({"session_id": {"nested": UUID_A}}), None),
    ("json array", json.dumps({"session_id": [UUID_A]}), None),
    ("json float", json.dumps({"session_id": 1.5}), None),
    # ``-5`` is inside the charset and is not all digits, so the JSON type
    # test is the only thing keeping it out of the column — on BOTH dialects.
    # Without it SQLite and Postgres would each stamp a value Python declines.
    ("json negative number", json.dumps({"session_id": -5}), None),
    # ``str.isdigit()`` is true for these and neither dialect's digit test is.
    ("unicode digits", json.dumps({"session_id": "١٢٣"}), None),
    ("outside the charset", json.dumps({"session_id": "did:x:1"}), None),
    ("metadata not an object", json.dumps([{"session_id": UUID_A}]), None),
    ("metadata scalar", json.dumps("hello"), None),
    # A duplicated key has no answer the three readers share, so it has none
    # here either. The two near-misses beside it must survive: they are not
    # duplicates and a coarser guard would null them.
    ("duplicated key", DUPLICATED_METADATA, None),
    ("duplicated only in a nested object", NESTED_DUPLICATE_METADATA, "cccc-3333"),
    ("a longer key containing the key", NEIGHBOURING_KEY_METADATA, "cccc-3333"),
]


@pytest.fixture
async def postgres_db() -> AsyncIterator[AsyncDatabase]:
    """A PostgreSQL-only database, for the cases SQLite cannot exhibit.

    Separate from the ``db_backend`` parametrization on purpose: expressing
    "PostgreSQL only" as a ``pytest.skip`` inside a dual-backend test reports a
    skip on the SQLite leg, and this ticket's acceptance gate treats any skip in
    this file as a failure.
    """
    if not POSTGRES_URL:  # pragma: no cover - environment gate
        pytest.skip("TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL required")
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    backend = PostgresBackend(POSTGRES_URL)
    await backend.connect()
    try:
        yield AsyncDatabase(backend)
    finally:
        await backend.close()


@asynccontextmanager
async def _isolated_schema(db_backend) -> AsyncIterator[AsyncDatabase]:
    """Give the pre-migration table an isolated SQLite DB / Postgres schema."""
    db = AsyncDatabase(db_backend)
    if db.backend_type == "sqlite":
        yield db
        return

    schema = f"session_id_column_{uuid4().hex}"
    await db.execute(f'CREATE SCHEMA "{schema}"')
    try:
        async with db.transaction():
            await db.execute(f'SET LOCAL search_path TO "{schema}"')
            yield db
    finally:
        await db.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


async def _create_pre_migration_table(db: AsyncDatabase, table: str = "conversation_history") -> None:
    """The table as it stood before session_id was a column."""
    id_column = (
        "BIGSERIAL PRIMARY KEY"
        if db.backend_type == "postgres"
        else "INTEGER PRIMARY KEY AUTOINCREMENT"
    )
    await db.execute(
        f"CREATE TABLE {table} ("
        f"id {id_column}, agent_id TEXT NOT NULL DEFAULT '', "
        "role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "deleted_at TIMESTAMP DEFAULT NULL)"
    )


async def _seed(db: AsyncDatabase, agent_id: str, cases) -> None:
    for label, metadata, *_ in cases:
        await db.execute(
            "INSERT INTO conversation_history (agent_id, role, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            (agent_id, "user", label, metadata),
        )


def _grouped_session_id(metadata: str):
    """The id ``group_messages_into_sessions`` files this lone row under."""
    try:
        parsed = json.loads(metadata) if metadata else {}
    except ValueError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    sessions = group_messages_into_sessions(
        [{"id": 1, "role": "user", "content": "x", "metadata": parsed,
          "created_at": "2026-01-01T00:00:00+00:00"}],
        keep_empty_markers=True,
    )
    return sessions[0]["session_id"] if sessions else None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_backfill_means_the_same_thing_on_both_backends(db_backend):
    """Every case, against whichever dialect actually ran.

    Three assertions per row, and they are different claims: the spelled-out
    expectation (what the ticket agreed to), agreement with the Python
    rendering (no dialect drift), and the invariant against the real grouper
    (the column never names a session the transcript does not show).
    """
    agent_id = f"did:test:session-column:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        await _seed(db, agent_id, CASES)

        await db._migrate_conversation_session_id_column()

        rows = await db.fetchall(
            "SELECT content, metadata, session_id FROM conversation_history "
            "ORDER BY id",
            (),
        )
        assert [(content, session_id) for content, _meta, session_id in rows] == [
            (label, expected) for label, _metadata, expected in CASES
        ]
        for content, metadata, session_id in rows:
            assert session_id == column_session_id(metadata), content
            if session_id is not None:
                assert _grouped_session_id(metadata) == session_id, content


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_table_of_nothing_but_poison_still_migrates(db_backend):
    """Malformed, NUL-bearing and oversized metadata must not brick a boot.

    ``migrate_columns_once`` raises unless the column lands, so a backfill that
    raised on one of these would take the whole startup with it. A table made
    ENTIRELY of poison is the strongest form of the claim: nothing else is
    carrying the migration.
    """
    agent_id = f"did:test:session-column:{uuid4()}"
    poison = [
        ("malformed", '{"session_id": "oops', None),
        ("not json", "{not json", None),
        ("nul", NUL_METADATA, None),
        ("embedded nul", EMBEDDED_NUL_METADATA, None),
        ("oversized", json.dumps({"session_id": OVERSIZED}), None),
        ("scalar", "42", None),
        ("empty text", "", None),
    ]
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        await _seed(db, agent_id, poison)

        await db._migrate_conversation_session_id_column()

        values = await db.fetchall(
            "SELECT session_id FROM conversation_history ORDER BY id", ()
        )
        assert [row[0] for row in values] == [None] * len(poison)
        # The index is the second thing an oversized id would have broken.
        assert await db._index_exists(
            "idx_conversation_agent_session", "conversation_history"
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_engines_really_do_pick_different_halves_of_a_duplicate(db_backend):
    """The premise behind refusing duplicates, measured on each engine.

    Refusing costs a NULL, so it is only justified if the readers genuinely
    cannot be reconciled. They cannot: asked for the same document, SQLite's
    ``json_extract`` hands back the FIRST ``session_id`` and PostgreSQL's
    ``jsonb`` the LAST — and Python agrees with PostgreSQL, so no choice of
    occurrence makes all three agree. Pinned against the engines rather than
    quoted from their documentation, because if a future version changed its
    mind the contract would be over-conservative rather than wrong, and this is
    what would say so.
    """
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)

        if db.backend_type == "postgres":
            raw = await db.fetchval(
                "SELECT (?::jsonb) ->> 'session_id'", (DUPLICATED_METADATA,)
            )
            assert raw == DUP_LAST
        else:
            raw = await db.fetchval(
                "SELECT json_extract(?, '$.session_id')", (DUPLICATED_METADATA,)
            )
            assert raw == DUP_FIRST
        # Python sides with PostgreSQL, so SQLite is the odd one out either way.
        assert json.loads(DUPLICATED_METADATA)["session_id"] == DUP_LAST
        # ...and the column takes neither.
        assert column_session_id(DUPLICATED_METADATA) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_metadata_update_carries_the_column_with_it(db_backend):
    """The one door that can rewrite ``metadata.session_id`` after insertion.

    ``update_message_metadata`` takes an arbitrary key set, so a caller can
    move a row from one session to another. Before this it moved only the
    metadata, leaving the indexed column naming the session the row had left —
    the single state the column is not allowed to be in. Driven through the
    real store against the real engine, because the merge is hand-written SQL
    per dialect and the defect was invisible at the Python layer.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    agent_id = f"did:test:session-column:{uuid4()}"
    moved_to = "3a6c1f0e-2b7d-4c8a-9e10-000000000002"
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        await db.execute(
            "INSERT INTO conversation_history (agent_id, role, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            (agent_id, "user", "turn", json.dumps({"session_id": UUID_A})),
        )
        await db._migrate_conversation_session_id_column()
        store = AsyncConversationStore(db, agent_id=agent_id)
        message_id = await db.fetchval(
            "SELECT id FROM conversation_history WHERE agent_id = ?", (agent_id,)
        )
        assert await db.fetchval(
            "SELECT session_id FROM conversation_history WHERE id = ?", (message_id,)
        ) == UUID_A

        # 1. A replacement inside the contract moves both.
        assert await store.update_message_metadata(message_id, {"session_id": moved_to})
        metadata, column = await db.fetchone(
            "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
            (message_id,),
        )
        assert json.loads(metadata)["session_id"] == moved_to
        assert column == moved_to

        # 2. A replacement OUTSIDE the contract must blank the column rather
        #    than keep the previous id: metadata no longer says what it says.
        assert await store.update_message_metadata(message_id, {"session_id": "did:x:1"})
        metadata, column = await db.fetchone(
            "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
            (message_id,),
        )
        assert json.loads(metadata)["session_id"] == "did:x:1"
        assert column is None

        # 3. And an update that does not mention the key leaves the column
        #    alone — including when it is NULL for a reason.
        assert await store.update_message_metadata(message_id, {"importance": 0.9})
        metadata, column = await db.fetchone(
            "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
            (message_id,),
        )
        assert json.loads(metadata)["importance"] == 0.9
        assert json.loads(metadata)["session_id"] == "did:x:1"
        assert column is None

        # 4. Back inside the contract: the column comes back, not stays NULL.
        assert await store.update_message_metadata(message_id, {"session_id": UUID_A})
        metadata, column = await db.fetchone(
            "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
            (message_id,),
        )
        assert column == UUID_A
        assert column == column_session_id(metadata)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_updating_a_duplicated_key_never_stamps_what_only_one_reader_sees(
    db_backend,
):
    """The merge cannot always speak for the document, and says so with NULL.

    A legacy row carrying ``session_id`` twice is the one shape where the two
    dialects' merges genuinely differ, so the column cannot be derived from the
    incoming value alone:

    * PostgreSQL merges in ``jsonb``, which deduplicates on parse — the merge
      REPAIRS the row, leaving one key holding the new value.
    * SQLite merges the text as written, and ``json_set`` replaces only the
      FIRST occurrence. The document stays duplicated, ``json_extract`` reads
      the new value and ``json.loads`` — and therefore session grouping — still
      reads the second. Stamping the incoming value there would be the column
      disagreeing with the metadata its reader groups by, which is the one
      state it may never occupy.

    The invariant is asserted first and identically on both backends; the
    mechanism assertions after it are per-dialect because the mechanism is.
    Removing the guard in ``merged_column_assignment`` leaves the PostgreSQL
    leg green and fails here on SQLite.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    agent_id = f"did:test:session-column:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        await db.execute(
            "INSERT INTO conversation_history (agent_id, role, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            (agent_id, "user", "turn", DUPLICATED_METADATA),
        )
        await db._migrate_conversation_session_id_column()
        store = AsyncConversationStore(db, agent_id=agent_id)
        message_id = await db.fetchval(
            "SELECT id FROM conversation_history WHERE agent_id = ?", (agent_id,)
        )
        # The backfill already refused it, so this starts from NULL and the
        # test cannot pass by never having stamped anything.
        assert await db.fetchval(
            "SELECT session_id FROM conversation_history WHERE id = ?", (message_id,)
        ) is None

        assert await store.update_message_metadata(message_id, {"session_id": UUID_A})
        metadata, column = await db.fetchone(
            "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
            (message_id,),
        )

        # The invariant, against the real grouper: silent is allowed, wrong is
        # not. This is the assertion that must hold on every backend.
        assert column in (None, _grouped_session_id(metadata)), (
            f"column {column!r} names a session the grouper does not: "
            f"{_grouped_session_id(metadata)!r} from {metadata!r}"
        )
        # ...and the column is never something only ONE dialect's reader sees.
        assert column == column_session_id(metadata)

        if db.backend_type == "postgres":
            # Repaired: one key, the new value, everyone agrees.
            assert json.loads(metadata) == {"session_id": UUID_A}
            assert metadata.count('"session_id"') == 1
            assert column == UUID_A
        else:
            # Still duplicated, so still ambiguous — the update moved the first
            # occurrence and the grouper still reads the second.
            assert metadata.count('"session_id"') == 2
            assert json.loads(metadata)["session_id"] == DUP_LAST
            assert await db.fetchval(
                "SELECT json_extract(metadata, '$.session_id') "
                "FROM conversation_history WHERE id = ?",
                (message_id,),
            ) == UUID_A
            assert column is None

        # An unambiguous row is unaffected by the guard: the next update of a
        # document the merge CAN speak for stamps normally, so the guard is not
        # quietly nulling the ordinary path.
        await db.execute(
            "INSERT INTO conversation_history (agent_id, role, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            (agent_id, "user", "plain", json.dumps({"session_id": UUID_A})),
        )
        plain_id = await db.fetchval(
            "SELECT id FROM conversation_history WHERE agent_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (agent_id,),
        )
        moved_to = "3a6c1f0e-2b7d-4c8a-9e10-000000000009"
        assert await store.update_message_metadata(plain_id, {"session_id": moved_to})
        metadata, column = await db.fetchone(
            "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
            (plain_id,),
        )
        assert json.loads(metadata)["session_id"] == moved_to
        assert column == moved_to


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_batch_metadata_api_moves_every_row_it_touches(db_backend):
    """``update_messages_metadata`` is the same door, opened for many rows.

    It holds its guarantee by delegating per row to the method above rather
    than by writing SQL of its own, and that is precisely why it is pinned
    here: the obvious future optimization is one statement with an ``IN``
    clause, which would reintroduce the stale column for a whole batch while
    every single-row test stayed green. Asserted through the public batch API
    against a real engine, so the delegation is verified by behaviour and not
    by reading the call.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    agent_id = f"did:test:session-column-batch:{uuid4()}"
    moved_to = "3a6c1f0e-2b7d-4c8a-9e10-000000000002"
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        for _ in range(3):
            await db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                (agent_id, "user", "turn", json.dumps({"session_id": UUID_A})),
            )
        await db._migrate_conversation_session_id_column()
        store = AsyncConversationStore(db, agent_id=agent_id)
        rows = await db.fetchall(
            "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id",
            (agent_id,),
        )
        message_ids = [row[0] for row in rows]

        assert await store.update_messages_metadata(
            message_ids, {"session_id": moved_to}
        ) == len(message_ids)
        for message_id in message_ids:
            metadata, column = await db.fetchone(
                "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
                (message_id,),
            )
            assert json.loads(metadata)["session_id"] == moved_to
            assert column == moved_to

        # Outside the contract the whole batch blanks, rather than every row
        # keeping the session it has just been moved out of.
        assert await store.update_messages_metadata(
            message_ids, {"session_id": "did:x:1"}
        ) == len(message_ids)
        for message_id in message_ids:
            metadata, column = await db.fetchone(
                "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
                (message_id,),
            )
            assert json.loads(metadata)["session_id"] == "did:x:1"
            assert column is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_stamped_id_at_the_length_limit_is_indexable(db_backend):
    """The cap is chosen so the longest allowed id still fits a B-tree entry.

    A cap that let through an id the index could not hold would move the
    failure from the backfill to ``CREATE INDEX`` — still a failed boot, just
    later. Written after the index exists so the entry is inserted through it.
    """
    agent_id = f"did:test:session-column:{uuid4()}"
    longest = "b" * SESSION_ID_MAX_LENGTH
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        await db._migrate_conversation_session_id_column()
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, session_id) VALUES (?, ?, ?, ?, ?)",
            (agent_id, "user", "longest", json.dumps({"session_id": longest}), longest),
        )

        assert await db.fetchval(
            "SELECT session_id FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        ) == longest


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_migration_is_idempotent_on_both_backends(db_backend):
    """Re-running must not disturb a column later writes already own.

    The schema IS the marker for this backfill, so a second run that re-derived
    from metadata would be undetectable in production — assert the column a
    newer write set survives instead.
    """
    agent_id = f"did:test:session-column:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        await db.execute(
            "INSERT INTO conversation_history (agent_id, role, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            (agent_id, "user", "legacy", json.dumps({"session_id": "1314"})),
        )

        await db._migrate_conversation_session_id_column()
        # A later write stamps a column value metadata alone would not produce.
        await db.execute(
            "UPDATE conversation_history SET session_id = ? WHERE agent_id = ?",
            (UUID_A, agent_id),
        )
        await db._migrate_conversation_session_id_column()

        assert (
            await db.fetchval(
                "SELECT session_id FROM conversation_history WHERE agent_id = ?",
                (agent_id,),
            )
            == UUID_A
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_index_existence_probe_answers_truthfully_on_both_backends(db_backend):
    """``ensure_index``'s gate is dialect-specific, so both spellings must work.

    A probe that always answered "missing" would leave every boot re-issuing
    the DDL under the lock (degraded, silent); one that always answered
    "present" would mean the index is never built at all — and neither shows up
    as a failure anywhere else, because ``CREATE INDEX IF NOT EXISTS`` shrugs
    either way.
    """
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        name = f"idx_probe_{uuid4().hex[:12]}"

        assert await db._index_exists(name, "conversation_history") is False
        await db.ensure_index(name, "conversation_history", "agent_id")
        assert await db._index_exists(name, "conversation_history") is True
        # Second call is the every-boot-after-the-first path.
        await db.ensure_index(name, "conversation_history", "agent_id")
        assert await db._index_exists(name, "conversation_history") is True


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_session_index_lands_on_both_backends(db_backend):
    """The migration's own index, through whichever dialect actually ran."""
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        await db._migrate_conversation_session_id_column()

        assert await db._index_exists(
            "idx_conversation_agent_session", "conversation_history"
        ) is True


@pytest.mark.asyncio
async def test_a_decoy_index_in_an_earlier_schema_does_not_suppress_the_real_one(
    postgres_db,
):
    """#2958 Finding 3, in the only arrangement that can exhibit it.

    An index name is unique per schema, so ``to_regclass(index_name)`` answers
    about the FIRST index of that name on the search path. Put a decoy in the
    first schema and the target table in the second and a name-only probe
    reports "already indexed" — the real table is never indexed and nothing
    downstream notices, because ``CREATE INDEX IF NOT EXISTS`` would have
    shrugged too.

    A single-schema arrangement cannot show this: with one schema on the path,
    the first index of that name is necessarily the right one. That is exactly
    why the earlier attempt's test — which did ``SET LOCAL search_path TO
    "{schema}"`` and so made the target the only schema — passed against the
    broken probe.
    """
    db = postgres_db
    decoy_schema = f"decoy_{uuid4().hex[:10]}"
    target_schema = f"target_{uuid4().hex[:10]}"
    index = "idx_conversation_agent_session"

    await db.execute(f'CREATE SCHEMA "{decoy_schema}"')
    await db.execute(f'CREATE SCHEMA "{target_schema}"')
    try:
        async with db.transaction():
            # Both schemas on the path, decoy FIRST. The table resolves from the
            # second schema; the index name resolves from the first.
            await db.execute(
                f'SET LOCAL search_path TO "{decoy_schema}", "{target_schema}"'
            )
            await db.execute(
                f'CREATE TABLE "{decoy_schema}".unrelated (id BIGSERIAL PRIMARY KEY)'
            )
            await db.execute(
                f'CREATE INDEX {index} ON "{decoy_schema}".unrelated (id)'
            )
            await db.execute(
                f'CREATE TABLE "{target_schema}".conversation_history ('
                "id BIGSERIAL PRIMARY KEY, agent_id TEXT NOT NULL DEFAULT '', "
                "role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )

            # The unqualified name the broken probe asked about does resolve —
            # to the decoy, in the FIRST schema. This is the trap, asserted
            # rather than assumed.
            assert await db.fetchval(
                "SELECT n.nspname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.oid = to_regclass(?)",
                (index,),
            ) == decoy_schema

            assert await db._index_exists(index, "conversation_history") is False
            await db._migrate_conversation_session_id_column()
            assert await db._index_exists(index, "conversation_history") is True

            # ...and it was built beside the TARGET table, not the decoy.
            owner = await db.fetchval(
                "SELECT n.nspname || '.' || t.relname FROM pg_index i "
                "JOIN pg_class ix ON ix.oid = i.indexrelid "
                "JOIN pg_class t ON t.oid = i.indrelid "
                "JOIN pg_namespace n ON n.oid = t.relnamespace "
                "WHERE ix.relname = ? AND n.nspname = ?",
                (index, target_schema),
            )
            assert owner == f"{target_schema}.conversation_history"
    finally:
        await db.execute(f'DROP SCHEMA IF EXISTS "{decoy_schema}" CASCADE')
        await db.execute(f'DROP SCHEMA IF EXISTS "{target_schema}" CASCADE')


@pytest.mark.asyncio
async def test_a_same_schema_name_collision_is_reported_not_shrugged_off(postgres_db):
    """The other half of Finding 3, and the half ``IF NOT EXISTS`` hides.

    A decoy in a *different* schema is harmless — the previous test shows the
    index still gets built beside its own table. A decoy in the SAME schema is
    not: PostgreSQL keeps relation names unique per namespace, so
    ``CREATE INDEX IF NOT EXISTS`` sees the name taken and does nothing at all.
    The table-aware probe correctly reports "absent" both before and after, so
    without a post-DDL check ``ensure_index`` would log a creation that never
    happened and every boot would repeat it.
    """
    db = postgres_db
    schema = f"collide_{uuid4().hex[:10]}"
    index = "idx_conversation_agent_session"

    await db.execute(f'CREATE SCHEMA "{schema}"')
    try:
        async with db.transaction():
            await db.execute(f'SET LOCAL search_path TO "{schema}"')
            await db.execute("CREATE TABLE decoy (id BIGSERIAL PRIMARY KEY)")
            await db.execute(f"CREATE INDEX {index} ON decoy (id)")
            await _create_pre_migration_table(db)

            with pytest.raises(RuntimeError, match="decoy"):
                await db._migrate_conversation_session_id_column()

            # The engine really did nothing; the name still belongs to the decoy.
            assert await db._index_exists(index, "conversation_history") is False
            assert await db._index_exists(index, "decoy") is True
            # ...and the column half of the migration DID land, so the failure
            # is about the index alone and a rename is all the operator needs.
            assert await db._column_exists("conversation_history", "session_id")
    finally:
        await db.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.mark.asyncio
async def test_index_creation_waits_on_the_real_advisory_lock(postgres_db):
    """#2958 Finding 2, proved by blocking rather than by racing.

    ``CREATE INDEX IF NOT EXISTS`` is idempotent in sequence and unsafe in
    parallel: PostgreSQL evaluates the existence test before taking the lock
    that would exclude a peer, so two initializers that pass it together both
    build and one loses on ``pg_class``'s unique index. ``_init_schema`` runs on
    every ``from_pool()``, so a post-upgrade request burst IS the parallel case.

    Racing two builders would be a coin toss that usually passes either way.
    Instead: hold the migration lock that ``ensure_index`` should be taking, and
    assert the builder does not finish while it is held. If the DDL escaped the
    lock — the defect — the builder would complete immediately and the
    ``TimeoutError`` below would never be raised.
    """
    db = postgres_db
    table = f"lock_probe_{uuid4().hex[:10]}"
    index = f"idx_{table}"
    await db.execute(f"CREATE TABLE {table} (id BIGSERIAL PRIMARY KEY, agent_id TEXT)")

    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold_the_lock():
        # Same name ensure_index derives, so this is the same advisory key.
        async with db.migration_lock(f"index_{index}"):
            holder_entered.set()
            await release_holder.wait()

    holder = asyncio.create_task(hold_the_lock())
    builder = None
    try:
        await asyncio.wait_for(holder_entered.wait(), timeout=10)

        builder = asyncio.create_task(db.ensure_index(index, table, "agent_id"))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(builder), timeout=1.0)
        assert await db._index_exists(index, table) is False

        release_holder.set()
        await asyncio.wait_for(holder, timeout=10)
        await asyncio.wait_for(builder, timeout=10)
        assert await db._index_exists(index, table) is True
    finally:
        # A failing assertion above must not strand the holder: its open
        # transaction would keep the advisory lock and the DROP below would
        # wait on it forever, turning one failed test into a hung suite.
        release_holder.set()
        pending = [t for t in (holder, builder) if t is not None and not t.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(
            *[t for t in (holder, builder) if t is not None], return_exceptions=True
        )
        await db.execute(f"DROP TABLE IF EXISTS {table}")


@pytest.mark.asyncio
async def test_concurrent_initializers_leave_exactly_one_index(postgres_db):
    """The burst itself, against the real engine.

    Weaker than the blocking test above — a passing run does not prove the race
    is closed, only that this run did not lose it — but it is the only case that
    exercises the real ``pg_advisory_xact_lock`` under real contention, and it
    would catch an advisory-lock call that errors or deadlocks rather than
    serializes.
    """
    db = postgres_db
    table = f"burst_{uuid4().hex[:10]}"
    index = f"idx_{table}"
    await db.execute(f"CREATE TABLE {table} (id BIGSERIAL PRIMARY KEY, agent_id TEXT)")
    try:
        await asyncio.gather(
            *(db.ensure_index(index, table, "agent_id") for _ in range(4))
        )
        count = await db.fetchval(
            "SELECT COUNT(*) FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE i.indrelid = to_regclass(?) AND c.relname = ?",
            (table, index),
        )
        assert count == 1
    finally:
        await db.execute(f"DROP TABLE IF EXISTS {table}")


@pytest.mark.asyncio
async def test_postgres_refuses_a_nul_in_the_column_so_the_contract_must(postgres_db):
    """Why the NUL exclusion is a contract clause and not a nicety.

    The backfill guard keeps a NUL out of the column during upgrade; this pins
    the reason. If a later change relaxed the charset, PostgreSQL would start
    rejecting ordinary conversation writes — not one row, every turn in that
    session — and the failure would surface as a broken chat, not a bad value.
    """
    db = postgres_db
    table = f"nul_probe_{uuid4().hex[:10]}"
    await db.execute(f"CREATE TABLE {table} (id BIGSERIAL PRIMARY KEY, session_id TEXT)")
    try:
        with pytest.raises(Exception) as raised:
            await db.execute(f"INSERT INTO {table} (session_id) VALUES (?)", ("a\x00b",))
        assert "0x00" in str(raised.value), str(raised.value)

        assert column_session_id({"session_id": "a\x00b"}) is None
    finally:
        await db.execute(f"DROP TABLE IF EXISTS {table}")
