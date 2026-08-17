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
* **Refusal.** PostgreSQL — not SQLite — raises on ``metadata::jsonb`` for a
  document that is malformed, carries a NUL escape, or carries ill-formed
  UTF-16 such as a lone surrogate, and it refuses a B-tree entry over ~2704
  bytes. Inside a mandatory migration each of those is a failed boot, not a
  skipped row, which is why they are exercised against a server rather than
  reasoned about. The lone surrogate is in that list because it was *not*
  reasoned about: it passed a guard that enumerated the other two.

Run the PostgreSQL leg with::

    TEST_POSTGRES_URL=postgresql://u:p@127.0.0.1:5432/db uv run pytest \\
        tests/integration/test_session_id_column_backend_parity.py

A skipped PostgreSQL case is not a passing one. When a DSN is configured this
file must report zero skips.
"""

from __future__ import annotations

import asyncio
import contextlib
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

# The other half of that trap, and the one an enumerating guard missed. ``jsonb``
# rejects ill-formed UTF-16 as well as NUL, so a lone surrogate is valid JSON
# text that cannot become jsonb — it passes ``IS JSON OBJECT``, carries no NUL,
# and raises "Unicode low surrogate must follow a high surrogate" on the cast.
# Python's own encoder emits these (``json.dumps("\ud800")`` succeeds), so they
# are reachable from every write path this repository has.
LONE_HIGH_SURROGATE_METADATA = '{"session_id": "\\ud800"}'
LONE_LOW_SURROGATE_METADATA = '{"session_id": "\\udc00"}'
SURROGATE_INSIDE_AN_ID_METADATA = '{"session_id": "a\\ud800b"}'

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
    ("lone high surrogate", LONE_HIGH_SURROGATE_METADATA, None),
    ("lone low surrogate", LONE_LOW_SURROGATE_METADATA, None),
    ("surrogate inside an id", SURROGATE_INSIDE_AN_ID_METADATA, None),
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


# ── Nothing here may wait forever ────────────────────────────────────────
#
# The lock tests below park a transaction on the server on purpose, which makes
# them the tests in this repository most able to hang. One of them did: a run
# was found still executing after 23h26m, its backend ``idle in transaction``
# on ``SELECT pg_advisory_xact_lock($1)`` with the lock granted for the whole
# period. Nothing reported it. ``-x`` never returned, no assertion failed, and
# the only symptom was that unrelated work later could not take the same key.
#
# A test that can hang forever is indistinguishable from a slow one until it
# blocks a person, and in CI it is a timed-out job with no failing assertion —
# which reads as flake. So every await that can block on a lock is bounded, and
# blowing the budget is a FAILURE naming what was still waiting.
LOCK_WAIT_BUDGET = 20.0

# How long a deliberately-parked lock holder may live before releasing itself.
# This is the load-bearing half: the budget above only helps a test that is
# still running its own code. If the awaiting task dies — a failed assertion
# elsewhere, a cancelled loop, a killed worker — nothing runs the cleanup that
# would release the holder, and the holder's transaction outlives the process
# that created it. Parking on a *bounded* wait means the holder's lock has a
# maximum lifetime that does not depend on anyone else's cleanup running.
HOLDER_PARK_BUDGET = 60.0

# Reconstructs a bigint advisory key from the two halves ``pg_locks`` splits it
# into. ``_backfill_lock_id`` returns a SIGNED 64-bit key and ``pg_locks`` stores
# it as two UNSIGNED halves, so the join has to put the sign back.
#
# Bit concatenation is chosen for saying that plainly, not because the obvious
# arithmetic alternative is wrong: measured, ``(classid::bigint << 32) |
# objid::bigint`` yields the same value, because PostgreSQL's bigint shift wraps
# rather than erroring and lands on the signed result. That was written here as
# a difference before it was checked, and it is not one.
#
# Where the sign DOES bite is the Python side of the comparison: masking the key
# to unsigned before binding it — the natural thing to reach for, and what a
# first draft of this file did — makes the probe report "not held" for every
# negatively-hashing name. ``test_the_lock_probe_sees_a_negative_key`` is the
# only case in this file that catches it.
_ADVISORY_KEY = (
    "(classid::bigint::bit(32) || objid::bigint::bit(32))::bit(64)::bigint"
)


async def _advisory_lock_holders(db: AsyncDatabase, name: str) -> list:
    """Sessions holding the migration lock ``name``, with what they are doing.

    Returned rather than counted so a failure message can show the wedge's
    actual signature — ``idle in transaction`` on ``pg_advisory_xact_lock`` is
    a stranded client, while a waiter blocked in ``Lock`` is real contention.

    **What an empty result does and does not prove.** It proves the observable
    contract: after the call, nobody is holding the key, so the next initializer
    is not queued behind a ghost. It does NOT prove the lock is transaction
    -scoped. Measured: swapping ``migration_lock`` from ``pg_advisory_xact_lock``
    to the session-scoped ``pg_advisory_lock`` leaves every assertion in this
    file passing, because asyncpg runs ``SELECT pg_advisory_unlock_all()`` as
    part of the reset query when a connection goes back to the pool. Two
    independent mechanisms deliver the same observable, and no probe from
    outside can tell which one did. Said here so a reader does not credit these
    assertions with a guarantee they do not carry.
    """
    from kestrel_sovereign.storage.async_database import _backfill_lock_id

    return await db.fetchall(
        "SELECT l.pid, a.state, a.wait_event_type, left(a.query, 60) "
        "FROM pg_locks l JOIN pg_stat_activity a USING (pid) "
        f"WHERE l.locktype = 'advisory' AND l.granted AND {_ADVISORY_KEY} = ?",
        (_backfill_lock_id(name),),
    )


async def _before_the_budget(db: AsyncDatabase, what: str, awaitable, budget=None):
    """Await ``awaitable``, failing — never hanging — if it blocks too long."""
    try:
        async with asyncio.timeout(budget or LOCK_WAIT_BUDGET):
            return await awaitable
    except TimeoutError:
        activity = await db.fetchall(
            "SELECT pid, state, wait_event_type, wait_event, left(query, 60) "
            "FROM pg_stat_activity WHERE datname IS NOT NULL "
            "AND query NOT LIKE '%pg_stat_activity%'",
            (),
        )
        raise AssertionError(
            f"{what} did not finish within {budget or LOCK_WAIT_BUDGET}s. "
            f"Server activity: {activity}"
        ) from None


@asynccontextmanager
async def _parked_lock_holder(db: AsyncDatabase, name: str) -> AsyncIterator[None]:
    """Hold the migration lock ``name`` for the body, and never past it.

    Three obligations, each learned from the wedge:

    * The body does not begin until the lock is really held, so a test that
      asserts "the builder blocks" is not just observing a slow start.
    * The holder self-releases after :data:`HOLDER_PARK_BUDGET` even if this
      context manager's ``__aexit__`` never runs.
    * On the way out the holder is confirmed to have *finished*, and to have
      finished because the body ended rather than because it blew its park
      budget. Those are different outcomes and only one of them means the test
      observed what it claims to.

    What it deliberately does NOT assert is that the key is free afterwards.
    Releasing it is precisely what unblocks whoever was waiting, so a correct
    run routinely leaves the successor mid-``CREATE INDEX`` holding it. The
    release assertions belong in the tests, after the successor is awaited,
    where "who holds it now" has a single right answer.
    """
    entered = asyncio.Event()
    release = asyncio.Event()
    parked_out = False

    async def hold():
        nonlocal parked_out
        async with db.migration_lock(name):
            entered.set()
            # Bounded: see HOLDER_PARK_BUDGET. A timeout here is the holder
            # rescuing itself, not an error to propagate — the assertion that
            # something went wrong belongs to whoever was waiting on it.
            try:
                await asyncio.wait_for(release.wait(), HOLDER_PARK_BUDGET)
            except TimeoutError:
                parked_out = True

    holder = asyncio.create_task(hold())
    try:
        await _before_the_budget(db, f"taking the migration lock {name!r}",
                                 entered.wait())
        yield
    finally:
        release.set()
        await _before_the_budget(
            db, f"the parked holder of {name!r} finishing",
            asyncio.gather(holder, return_exceptions=True),
        )
        assert not parked_out, (
            f"the holder of {name!r} released itself on the {HOLDER_PARK_BUDGET}s "
            "park budget rather than on the test — the body blocked"
        )


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
    """The table as it stood before session_id was a column.

    "Pre-migration" means *this* migration, not every one before it: the
    soft-delete and archive columns are here because the store's own write
    paths read them, and a fixture missing those would fail for a reason that
    has nothing to do with what these tests measure. The #2959 projection table
    is created for the same reason — the store maintains it on every write, and
    the tests below drive real store methods.
    """
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
        "deleted_at TIMESTAMP DEFAULT NULL, "
        "archived_at TIMESTAMP DEFAULT NULL)"
    )
    await db.execute(
        "CREATE TABLE IF NOT EXISTS conversation_sessions ("
        "agent_id TEXT NOT NULL, session_id TEXT NOT NULL, "
        "started_at TIMESTAMP, last_message_at TIMESTAMP, "
        "message_count INTEGER NOT NULL DEFAULT 0, "
        "user_message_count INTEGER NOT NULL DEFAULT 0, "
        "first_user_message_id INTEGER, wake_source TEXT, "
        "PRIMARY KEY (agent_id, session_id))"
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
        ("lone high surrogate", LONE_HIGH_SURROGATE_METADATA, None),
        ("lone low surrogate", LONE_LOW_SURROGATE_METADATA, None),
        ("surrogate inside an id", SURROGATE_INSIDE_AN_ID_METADATA, None),
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
async def test_updating_a_root_that_is_not_an_object_stamps_nothing(db_backend):
    """A merge both engines decline while reporting success.

    ``metadata`` is a free-text column and legacy rows hold whatever a writer
    put there — ``42``, ``[]``, ``"text"``, JSON ``null``. Handed one of those,
    neither engine merges and neither complains:

    * SQLite's ``json_set`` returns the document's value UNCHANGED (it is
      reserialized compactly, but nothing is added to it).
    * PostgreSQL's ``||`` promotes the non-object operand to a one-element
      array and CONCATENATES, so ``42`` becomes ``[42, {"session_id": ...}]``.

    Either way the row ends up with no top-level ``session_id`` at all, while
    ``update_message_metadata`` returns True and — before this guard — stamped
    the incoming id into the indexed column anyway. That is the column naming a
    session no reader of the row can see, on BOTH backends, which is the single
    state it may never occupy.

    The rows start with a stale column value, planted directly, so the test
    cannot pass by the update merely leaving a NULL alone: the assertion is
    that the clause actively nulls it. The SQL-NULL row is the control — it is
    the case ``COALESCE`` exists for, it really does become an object, and it
    must still stamp normally, so a guard that simply refused everything fails
    here too.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    stale = "3a6c1f0e-2b7d-4c8a-9e10-00000000000f"
    # (label, stored metadata, does the merge produce an object?)
    roots = [
        ("json number", "42", False),
        ("json array", "[]", False),
        ("json string", '"text"', False),
        ("json null", "null", False),
        ("json true", "true", False),
        ("nested object in an array", '[{"session_id": "aaaa-1111"}]', False),
        # The control: SQL NULL is an ABSENCE, COALESCE turns it into {}.
        ("sql null", None, True),
        # ...as is an empty object, which is what the absence becomes.
        ("empty object", "{}", True),
    ]
    agent_id = f"did:test:session-column-root:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_pre_migration_table(db)
        for label, metadata, _ in roots:
            await db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                (agent_id, "user", label, metadata),
            )
        await db._migrate_conversation_session_id_column()
        # The backfill refuses all of these, so plant a stale value by hand.
        # Nothing legitimate produces this row state; it is here so "the column
        # was nulled" cannot be confused with "the column was never set".
        await db.execute(
            "UPDATE conversation_history SET session_id = ? WHERE agent_id = ?",
            (stale, agent_id),
        )
        store = AsyncConversationStore(db, agent_id=agent_id)

        rows = await db.fetchall(
            "SELECT id, content FROM conversation_history WHERE agent_id = ? "
            "ORDER BY id",
            (agent_id,),
        )
        assert [row[1] for row in rows] == [label for label, _, _ in roots]

        for (message_id, label), (_, stored, merges_to_object) in zip(rows, roots):
            assert await store.update_message_metadata(
                message_id, {"session_id": UUID_A}
            ), label
            metadata, column = await db.fetchone(
                "SELECT metadata, session_id FROM conversation_history "
                "WHERE id = ?",
                (message_id,),
            )

            # The invariant, stated the same way on both backends and both
            # branches: whatever the merge did, the column agrees with the
            # metadata that is actually stored.
            assert column == column_session_id(metadata), label
            assert column in (None, _grouped_session_id(metadata)), (
                f"{label}: column {column!r} names a session the grouper does "
                f"not: {_grouped_session_id(metadata)!r} from {metadata!r}"
            )

            if merges_to_object:
                assert json.loads(metadata) == {"session_id": UUID_A}, label
                assert column == UUID_A, label
                continue

            assert column is None, label
            assert not isinstance(json.loads(metadata), dict), label
            if db.backend_type == "sqlite":
                # Declined outright: the document still SAYS what it said. Not
                # byte-for-byte — ``json_set`` reserializes into SQLite's
                # compact form even on the path it refuses to apply, so
                # ``[{"session_id": "x"}]`` comes back without the space. That
                # is a rendering change, and the comparison is made at the
                # value level so it cannot be mistaken for a content one.
                assert json.loads(metadata) == json.loads(stored), label
            else:
                # Concatenated: the patch lands BESIDE the old root rather than
                # merging into it, so no top-level session_id exists. An array
                # root is appended to; anything else is promoted to a
                # one-element array first. Both measured on 16.14.
                old = json.loads(stored)
                expected = (old if isinstance(old, list) else [old])
                assert json.loads(metadata) == expected + [
                    {"session_id": UUID_A}
                ], label


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


@pytest.mark.timeout(120)
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

    This is the test that hung for 23 hours holding the key it parks. It no
    longer can: the holder self-releases (:func:`_parked_lock_holder`), every
    wait is bounded, and the lock is asserted gone at the end.
    """
    db = postgres_db
    table = f"lock_probe_{uuid4().hex[:10]}"
    index = f"idx_{table}"
    lock = f"index_{index}"  # the name ensure_index derives, so the same key
    await db.execute(f"CREATE TABLE {table} (id BIGSERIAL PRIMARY KEY, agent_id TEXT)")

    builder = None
    try:
        async with _parked_lock_holder(db, lock):
            assert len(await _advisory_lock_holders(db, lock)) == 1

            builder = asyncio.create_task(db.ensure_index(index, table, "agent_id"))
            # The claim: it does NOT finish while the key is held. Bounded from
            # below, so this one is a timeout by design.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(builder), timeout=1.0)
            assert await db._index_exists(index, table) is False

        await _before_the_budget(db, "the builder after the lock was released",
                                 builder)
        assert await db._index_exists(index, table) is True
        # Success path: ensure_index leaves nothing behind on the key either.
        assert await _advisory_lock_holders(db, lock) == []
    finally:
        if builder is not None and not builder.done():
            builder.cancel()
            await asyncio.gather(builder, return_exceptions=True)
        await db.execute(f"DROP TABLE IF EXISTS {table}")


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_concurrent_initializers_leave_exactly_one_index(postgres_db):
    """The burst itself, against the real engine.

    Weaker than the blocking test above — a passing run does not prove the race
    is closed, only that this run did not lose it — but it is the only case that
    exercises the real ``pg_advisory_xact_lock`` under real contention, and it
    would catch an advisory-lock call that errors or deadlocks rather than
    serializes.

    The gather is bounded because "deadlocks rather than serializes" is a claim
    about *blocking*, and an unbounded gather cannot report it — it would simply
    never return, which is the failure mode this file was found in.
    """
    db = postgres_db
    table = f"burst_{uuid4().hex[:10]}"
    index = f"idx_{table}"
    lock = f"index_{index}"
    await db.execute(f"CREATE TABLE {table} (id BIGSERIAL PRIMARY KEY, agent_id TEXT)")
    try:
        await _before_the_budget(
            db,
            "four concurrent initializers building the same index",
            asyncio.gather(
                *(db.ensure_index(index, table, "agent_id") for _ in range(4))
            ),
        )
        count = await db.fetchval(
            "SELECT COUNT(*) FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE i.indrelid = to_regclass(?) AND c.relname = ?",
            (table, index),
        )
        assert count == 1
        # Four takers, zero holders afterwards.
        assert await _advisory_lock_holders(db, lock) == []
    finally:
        await db.execute(f"DROP TABLE IF EXISTS {table}")


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_the_lock_probe_sees_a_negative_key(postgres_db):
    """Half of all lock names hash negative, and the probe must see those too.

    Every other lock name generated in this file happens to hash positive, so
    every release assertion in it is made against the easy half of the key
    space. This case picks a name that does not, which makes it the only place a
    sign error can be caught — and a sign error fails in the direction that
    reads as success, reporting nobody holding a lock that is held.

    Mutation-tested by masking the key to unsigned before binding it (``key &
    0xFFFFFFFFFFFFFFFF``, which is what a first draft of the probe did), and the
    result is the reason this case is written at all: the mutation IS caught
    elsewhere in the file — but only when a randomly generated table name
    happens to hash negative, which is a coin toss per run. Coverage that
    depends on a uuid4 is coverage that will one day not be there. This case
    searches for a negative-hashing name, so it fails every time.
    """
    from kestrel_sovereign.storage.async_database import _backfill_lock_id

    db = postgres_db
    tag = uuid4().hex[:8]
    name = next(
        candidate
        for candidate in (f"negative_key_{tag}_{n}" for n in range(100_000))
        if _backfill_lock_id(candidate) < 0
    )
    assert _backfill_lock_id(name) < 0

    async with _parked_lock_holder(db, name):
        assert len(await _advisory_lock_holders(db, name)) == 1
    assert await _advisory_lock_holders(db, name) == []


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_the_failure_path_releases_the_migration_lock(postgres_db):
    """A raise must not keep the key, or one bad boot wedges every later one.

    ``ensure_index`` raises when the index name is already taken on the same
    table's schema. That raise happens *outside* the ``migration_lock`` block on
    purpose — but "outside" is a claim about code, and the thing that matters is
    the server's answer. So: force the failure, then assert both that no session
    still holds the key and that it can be taken again.

    An "is it empty" probe is worth nothing unless it is also shown non-empty
    for the same key, so the first assertion below parks a holder and requires
    the probe to see it. Without that, a probe that had silently stopped
    matching — a renamed lock, a botched key reconstruction — would certify
    every release in this file.
    """
    db = postgres_db
    table = f"release_{uuid4().hex[:10]}"
    decoy = f"decoy_{uuid4().hex[:10]}"
    index = f"idx_{table}"
    lock = f"index_{index}"
    await db.execute(f"CREATE TABLE {decoy} (id BIGSERIAL PRIMARY KEY)")
    await db.execute(f"CREATE TABLE {table} (id BIGSERIAL PRIMARY KEY, agent_id TEXT)")
    await db.execute(f"CREATE INDEX {index} ON {decoy} (id)")
    try:
        async with _parked_lock_holder(db, lock):
            assert len(await _advisory_lock_holders(db, lock)) == 1

        with pytest.raises(RuntimeError, match=decoy):
            await _before_the_budget(
                db, "ensure_index against a taken name",
                db.ensure_index(index, table, "agent_id"),
            )

        assert await _advisory_lock_holders(db, lock) == []
        # Re-acquirable, which is the property an operator actually needs after
        # a failed boot: the next attempt must not queue behind a ghost.
        async def take_and_release():
            async with db.migration_lock(lock):
                return True

        assert await _before_the_budget(
            db, "re-taking the lock after the failure path", take_and_release()
        )
    finally:
        await db.execute(f"DROP TABLE IF EXISTS {table}")
        await db.execute(f"DROP TABLE IF EXISTS {decoy}")


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_ensure_index_under_an_open_migration_transaction_does_not_deadlock(
    postgres_db,
):
    """Whether the 23-hour hang was ``ensure_index`` itself. It was not.

    The suspicion was the classic ABBA shape: ``migration_lock`` holds one
    transaction for its whole block, so taking a second lock inside it would be
    a lock-ordering hazard, and if ``ensure_index`` were reachable while
    ``migrate_columns_once`` still held its transaction that would be a deadlock
    in the shipping path rather than a test defect.

    Measured, it is not, for two independent reasons — and both are asserted
    here rather than argued, because "we reasoned it cannot deadlock" is what
    every deadlock was before it happened:

    1. **A session never blocks on its own advisory lock.** ``transaction()``
       nests per task as a savepoint on that task's connection, so a nested
       ``migration_lock`` runs ``pg_advisory_xact_lock`` on the session that
       already holds the outer key. That is granted immediately.
    2. **A genuine ordering inversion is detected, not waited out** — see the
       next test.

    So an indefinite hang cannot come from these locks. The wedge's signature
    said the same thing: a granted lock with ZERO waiters and the backend in
    ``ClientRead`` is a client that stopped talking, not a server-side cycle.
    """
    db = postgres_db
    table = f"nested_{uuid4().hex[:10]}"
    index = f"idx_{table}"
    await db.execute(f"CREATE TABLE {table} (id BIGSERIAL PRIMARY KEY, agent_id TEXT)")
    try:
        async def migrate_then_index():
            async with db.migration_lock(f"{table}_columns"):
                await db.ensure_index(index, table, "agent_id")

        await _before_the_budget(
            db, "ensure_index inside an open migration transaction",
            migrate_then_index(),
        )
        assert await db._index_exists(index, table) is True
        assert await _advisory_lock_holders(db, f"index_{index}") == []
        assert await _advisory_lock_holders(db, f"{table}_columns") == []
    finally:
        await db.execute(f"DROP TABLE IF EXISTS {table}")


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_a_lock_order_inversion_is_detected_rather_than_waited_out(postgres_db):
    """The second leg of the answer above, and the one worth pinning.

    Two sessions taking two migration locks in opposite orders is the worst case
    the lock design can produce. PostgreSQL's deadlock detector breaks that
    cycle within ``deadlock_timeout`` and aborts one side with an error. That is
    what makes "a hang cannot be the locks" a *checkable* statement rather than
    a reassurance: if a future change moved these gates onto something without a
    detector — a client-side mutex, a session-scoped lock, a lock taken on a
    second connection inside the first's transaction — this test would stop
    passing and start hanging, and the bounded wait turns that into a failure.
    """
    db = postgres_db
    tag = uuid4().hex[:10]
    first, second = f"inv_a_{tag}", f"inv_b_{tag}"
    got_first = asyncio.Event()
    got_second = asyncio.Event()

    async def first_then_second():
        async with db.migration_lock(first):
            got_first.set()
            await asyncio.wait_for(got_second.wait(), LOCK_WAIT_BUDGET)
            async with db.migration_lock(second):
                pass

    async def second_then_first():
        async with db.migration_lock(second):
            got_second.set()
            await asyncio.wait_for(got_first.wait(), LOCK_WAIT_BUDGET)
            async with db.migration_lock(first):
                pass

    outcomes = await _before_the_budget(
        db,
        "a deliberate lock-order inversion",
        asyncio.gather(
            first_then_second(), second_then_first(), return_exceptions=True
        ),
    )

    failures = [o for o in outcomes if isinstance(o, BaseException)]
    assert len(failures) == 1, outcomes
    assert "deadlock detected" in str(failures[0]), str(failures[0])
    # And the survivor released everything, so the cycle left nothing behind.
    assert await _advisory_lock_holders(db, first) == []
    assert await _advisory_lock_holders(db, second) == []


# ── #2959: the projection, against a real engine ─────────────────────────
#
# The projection's SQL is dialect-sensitive in three places the SQLite suite
# cannot speak for: an ``ON CONFLICT ... DO UPDATE`` upsert, a TIMESTAMP column
# bound from the grouper's ISO text (asyncpg rejects a str where SQLite wants
# one), and a maintenance write that has to live inside the purge's real
# transaction. Each is exercised here through the real store.


def _without_embeddings(monkeypatch) -> None:
    """Take the embedding provider out of these tests, deliberately.

    ``add_conversation`` co-writes ``embedding_vec`` when a provider is
    reachable, and that column arrives by a migration these fixtures do not
    run — so whether the INSERT takes its primary path would depend on whether
    a local model happens to be up. The projection has nothing to do with
    embeddings; this is the documented opt-out, so the test measures the same
    thing on every machine.
    """
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")


async def _create_core_tables(db: AsyncDatabase, *tables: str) -> None:
    """Create named tables from the SHIPPED schema, not a copy of it.

    ``_create_pre_migration_table`` above deliberately builds a legacy shape by
    hand. These tests need the current one, and hand-copying it is how a
    fixture starts testing a table the product does not have.
    """
    from kestrel_sovereign.storage.async_database import CORE_SCHEMA
    from kestrel_sovereign.storage.db import normalize_schema

    statements = normalize_schema(CORE_SCHEMA, db.backend_type).split(";")
    for table in tables:
        needle = f"CREATE TABLE IF NOT EXISTS {table} ("
        matching = [s for s in statements if needle in s]
        assert len(matching) == 1, f"{table}: {len(matching)} declarations in CORE_SCHEMA"
        await db.execute(matching[0].strip())


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_projection_is_written_and_maintained_on_both_backends(
    db_backend, monkeypatch
):
    """Insert, soft-delete, restore, purge — each through the real dialect.

    The pointer assertions are the same claims the SQLite suite makes; what is
    new here is that every statement carrying them is the PostgreSQL spelling.
    A broken upsert would surface as a duplicate-key error on the second turn,
    and a timestamp bound as text would fail at the column.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-projection:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_core_tables(
            db, "conversation_history", "conversation_sessions",
            "conversation_lexical_tokens",
        )
        store = AsyncConversationStore(db, agent_id=agent_id)

        await store.add_conversation("user", "first turn", session_id=UUID_A)
        await store.add_conversation("assistant", "reply", session_id=UUID_A)
        await store.add_conversation("user", "second turn", session_id=UUID_A)

        ids = [
            row[0]
            for row in await db.fetchall(
                "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id",
                (agent_id,),
            )
        ]
        projected = await store.session_projection.get(UUID_A)
        assert projected["message_count"] == 3
        assert projected["user_message_count"] == 2
        assert projected["first_user_message_id"] == ids[0]

        # The pointer moves off a trashed row and back, on this engine.
        assert await store.delete_message(ids[0])
        assert (await store.session_projection.get(UUID_A))[
            "first_user_message_id"
        ] == ids[2]
        assert await store.restore_message(ids[0])
        assert (await store.session_projection.get(UUID_A))[
            "first_user_message_id"
        ] == ids[0]

        # ...and a purge takes the row with it, inside the destructive
        # transaction rather than after it.
        assert await store.purge_conversation_session(UUID_A) == 3
        assert await store.session_projection.get(UUID_A) is None
        assert await db.fetchval(
            "SELECT COUNT(*) FROM conversation_sessions WHERE agent_id = ?",
            (agent_id,),
        ) == 0


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_projection_agrees_with_the_grouper_on_both_backends(
    db_backend, monkeypatch
):
    """The differential claim, restated where the timestamps are native.

    PostgreSQL hands back ``datetime`` objects where SQLite hands back text,
    and the grouper's gap arithmetic and the projection's stored boundaries
    both run through that difference. Agreement on one engine is therefore not
    agreement on the other, which is why the differential is asked twice.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )
    from kestrel_sovereign.storage.session_grouping import (
        coalesce_sessions_by_session_id,
        coerce_session_timestamp,
    )

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-projection-diff:{uuid4()}"
    other = "3a6c1f0e-2b7d-4c8a-9e10-00000000000b"
    async with _isolated_schema(db_backend) as db:
        await _create_core_tables(
            db, "conversation_history", "conversation_sessions",
            "conversation_lexical_tokens",
        )
        store = AsyncConversationStore(db, agent_id=agent_id)

        await store.add_conversation("user", "A one", session_id=UUID_A)
        await store.add_conversation("assistant", "A two", session_id=UUID_A)
        await store.add_conversation("user", "B one", session_id=other)
        # An autonomous wake is a user row that may not become the pointer.
        await store.add_conversation(
            "user", "B wake", session_id=other,
            metadata={"signal_wake": {"source": "heartbeat"}},
        )

        history = [
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "created_at": row[4],
            }
            for row in await db.fetchall(
                "SELECT id, role, content, metadata, created_at "
                "FROM conversation_history WHERE agent_id = ? "
                "AND deleted_at IS NULL AND archived_at IS NULL ORDER BY id",
                (agent_id,),
            )
        ]
        reference = {
            str(session["session_id"]): session
            for session in coalesce_sessions_by_session_id(
                group_messages_into_sessions(history, keep_empty_markers=True)
            )
        }
        stored = {
            row["session_id"]: row for row in await store.session_projection.list()
        }

        assert set(stored) == set(reference)
        for session_id, row in stored.items():
            expected = reference[session_id]
            # As instants, not spellings: this column comes back as text from
            # SQLite and as a ``datetime`` from PostgreSQL, and the claim is
            # about the moment, not about which engine formatted it.
            assert coerce_session_timestamp(row["started_at"]) == (
                coerce_session_timestamp(expected["started_at"])
            ), session_id
            assert coerce_session_timestamp(row["last_message_at"]) == (
                coerce_session_timestamp(expected["last_message_at"])
            ), session_id
            assert row["message_count"] == expected["message_count"], session_id
            assert (
                row["user_message_count"] == expected["user_message_count"]
            ), session_id
            assert row["wake_source"] == expected["preview_wake_source"], session_id
        assert stored[other]["wake_source"] == "heartbeat"


# ── #2959: concurrent maintenance of one session ─────────────────────────
#
# Recomputing from the surviving rows is only truthful about the rows the
# refresh could SEE. Two refreshes of one session can each be describing a world
# the other has already changed, and the projection then records the loser's
# answer: a count that does not match the live rows, or — worse — a pointer at a
# row the other transaction deleted.
#
# PostgreSQL is where this bites, and it bites in a way SQLite cannot reproduce:
# separate connections mean separate snapshots, so two transactions really do
# each see the other's deleted row as live. The SQLite legs below assert the same
# invariant and pass either way (its single writer already serializes the
# destructive paths, which open their own transaction) — so credit them with
# covering the dialect, not with proving the boundary. The insert case
# discriminates on BOTH engines, because an insert's refresh is not inside a
# caller's transaction on either.
#
# Every wait here is bounded. A lock test that can hang cannot report the bug it
# exists to catch — see the 23h26m wedge at the top of this file.

#: How long the second writer is given to complete AHEAD of the parked one.
#: Under the boundary it cannot, so blowing this budget is the EXPECTED path and
#: must never be an assertion; without the boundary the second writer needs two
#: statements, which fit inside this window on any machine that can run the
#: suite at all. It is therefore the length of the window in which the bug would
#: have happened, not a guess at how slow the engine is.
INTERLEAVE_GRACE = 2.0


async def _within_budget(what: str, awaitable, budget: float = LOCK_WAIT_BUDGET):
    """Await ``awaitable``, failing — never hanging — if it blocks too long.

    Dialect-neutral counterpart to ``_before_the_budget``: these cases run on
    SQLite too, and the PostgreSQL activity dump that makes that helper's
    failure message useful cannot be asked of SQLite at all.
    """
    try:
        async with asyncio.timeout(budget):
            return await awaitable
    except TimeoutError:
        raise AssertionError(f"{what} did not finish within {budget}s") from None


@asynccontextmanager
async def _schema_every_task_can_see(db: AsyncDatabase) -> AsyncIterator[AsyncDatabase]:
    """Like ``_isolated_schema``, but the isolation survives into other TASKS.

    ``_isolated_schema`` pins its schema with ``SET LOCAL search_path`` inside
    one transaction — connection-local state on the one connection that
    transaction holds. A concurrent task's ``transaction()`` acquires a
    DIFFERENT pool connection (the per-task ContextVar of #1726), which never saw
    that ``SET``, so a concurrency test built on it would quietly write half its
    rows into ``public`` and measure nothing there. (Measured: the first draft of
    the index case below did exactly that — four gathered initializers built the
    index in ``public`` while the assertion looked in the isolated schema and
    found none.) Setting the search_path as a pool server setting makes it the
    default for every connection the pool opens, which is the only form of this
    isolation a multi-task test can use.

    ``db`` is the control connection — used to create and drop the schema, and
    yielded unchanged on SQLite, where a file is already the isolation.
    """
    if db.backend_type == "sqlite":
        yield db
        return

    import asyncpg

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    schema = f"session_projection_{uuid4().hex}"
    await db.execute(f'CREATE SCHEMA "{schema}"')
    pool = await asyncpg.create_pool(
        POSTGRES_URL,
        min_size=2,
        max_size=8,
        server_settings={"search_path": schema},
    )
    try:
        yield AsyncDatabase(PostgresBackend.from_pool(pool))
    finally:
        await pool.close()
        await db.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


async def _assert_the_projection_matches_the_rows(
    db: AsyncDatabase, store, agent_id: str, session_id: str
) -> None:
    """The invariant asked of the database: counts match, pointer is live."""
    live = await db.fetchall(
        "SELECT id, role FROM conversation_history WHERE agent_id = ? "
        "AND session_id = ? AND deleted_at IS NULL AND archived_at IS NULL "
        "ORDER BY id",
        (agent_id, session_id),
    )
    row = await store.session_projection.get(session_id)
    if not live:
        assert row is None, (
            f"session {session_id!r} has no live rows but still has {row}"
        )
        return
    assert row is not None, f"session {session_id!r} has {len(live)} live rows, no row"
    assert row["message_count"] == len(live), (
        f"projection counts {row['message_count']} for {len(live)} live rows: "
        f"{live}"
    )
    pointer = row["first_user_message_id"]
    if pointer is not None:
        assert pointer in {row_id for row_id, _ in live}, (
            f"pointer {pointer} is not among the live rows {live}"
        )


@asynccontextmanager
async def _parked_between_the_read_and_the_write(monkeypatch, *, park_when):
    """Hold one projection write open between its own read and its upsert.

    The window this ticket's boundary closes is invisible without help: it is
    the gap between a refresh reading the session's rows and writing what it
    read. So one write is parked inside that gap while a second mutation runs to
    completion, which is the interleaving that produced the stale projection.
    ``park_when`` picks which write parks, from the projection it is about to
    store.

    Yields ``(parked, other_write_landed, release)``. ``other_write_landed`` is
    what discriminates: it is set only if a SECOND write got through while the
    first was parked, which the boundary must prevent.
    """
    from kestrel_sovereign.storage.conversation_sessions import (
        ConversationSessionProjection,
    )

    original = ConversationSessionProjection._store
    parked = asyncio.Event()
    other_write_landed = asyncio.Event()
    release = asyncio.Event()

    async def parking_store(self, projection):
        if park_when(projection) and not parked.is_set():
            parked.set()
            # Bounded, and its timeout is the holder rescuing itself rather
            # than an error: whoever was waiting on it owns that assertion.
            try:
                await asyncio.wait_for(release.wait(), HOLDER_PARK_BUDGET)
            except TimeoutError:
                pass
        elif parked.is_set() and not release.is_set():
            other_write_landed.set()
        await original(self, projection)

    monkeypatch.setattr(ConversationSessionProjection, "_store", parking_store)
    try:
        yield parked, other_write_landed, release
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.timeout(120)
async def test_a_stale_refresh_cannot_overwrite_a_newer_one(db_backend, monkeypatch):
    """Two turns in one session, the first refresh parked mid-flight.

    Without the boundary the second turn's refresh reads two live rows and
    stores ``message_count = 2``, then the parked first refresh stores the ``1``
    it read before that row existed — leaving the projection claiming one
    message for a session that has two. This is the SQLite reproduction from
    review as well as the PostgreSQL one: an insert's refresh has no enclosing
    transaction on either engine, so nothing else serializes it.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-projection-race:{uuid4()}"
    async with _schema_every_task_can_see(AsyncDatabase(db_backend)) as db:
        await _create_core_tables(
            db, "conversation_history", "conversation_sessions",
            "conversation_lexical_tokens",
        )
        store = AsyncConversationStore(db, agent_id=agent_id)

        async with _parked_between_the_read_and_the_write(
            monkeypatch, park_when=lambda p: p.message_count == 1
        ) as (parked, second_write_landed, release):
            first = asyncio.create_task(
                store.add_conversation("user", "first turn", session_id=UUID_A)
            )
            await _within_budget(
                "the first turn's refresh reaching its write", parked.wait()
            )
            second = asyncio.create_task(
                store.add_conversation("user", "second turn", session_id=UUID_A)
            )
            # The grace is the window the bug needed. A timeout here is the
            # boundary doing its job, so it is deliberately not asserted on.
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(INTERLEAVE_GRACE):
                    await second_write_landed.wait()
            release.set()
            await _within_budget(
                "both turns finishing",
                asyncio.gather(first, second),
            )

        assert await db.fetchval(
            "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ? "
            "AND session_id = ?",
            (agent_id, UUID_A),
        ) == 2
        await _assert_the_projection_matches_the_rows(db, store, agent_id, UUID_A)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.timeout(120)
async def test_concurrent_deletions_cannot_leave_the_pointer_on_a_deleted_row(
    db_backend, monkeypatch
):
    """The MVCC case, and the reason the boundary is transaction-scoped.

    Two transactions soft-delete different rows of one session. On PostgreSQL
    each has its own snapshot, so each sees the OTHER's row as still live and
    picks it as the pointer — and the second write to land names a row that is
    deleted by the time both commit. Holding the boundary until the mutation
    commits is what forces the later refresh to read after the earlier one's
    delete rather than beside it.

    The SQLite leg asserts the same invariant but cannot exhibit the failure:
    its destructive paths already run inside one writer's transaction. Read the
    PostgreSQL leg as the one under test.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-projection-delete-race:{uuid4()}"
    async with _schema_every_task_can_see(AsyncDatabase(db_backend)) as db:
        await _create_core_tables(
            db, "conversation_history", "conversation_sessions",
            "conversation_lexical_tokens",
        )
        store = AsyncConversationStore(db, agent_id=agent_id)
        for turn in ("first", "second", "third"):
            await store.add_conversation("user", f"{turn} turn", session_id=UUID_A)
        ids = [
            row[0]
            for row in await db.fetchall(
                "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id",
                (agent_id,),
            )
        ]
        assert (await store.session_projection.get(UUID_A))[
            "first_user_message_id"
        ] == ids[0]

        # The refresh that follows the FIRST deletion parks: it has already
        # read, and has chosen the row the second deletion is about to remove.
        async with _parked_between_the_read_and_the_write(
            monkeypatch, park_when=lambda p: p.first_user_message_id == ids[1]
        ) as (parked, other_write_landed, release):
            deleting_first = asyncio.create_task(store.delete_message(ids[0]))
            await _within_budget(
                "the first deletion's refresh reaching its write", parked.wait()
            )
            deleting_second = asyncio.create_task(store.delete_message(ids[1]))
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(INTERLEAVE_GRACE):
                    await other_write_landed.wait()
            release.set()
            assert await _within_budget(
                "both deletions finishing",
                asyncio.gather(deleting_first, deleting_second),
            ) == [True, True]

        assert await db.fetchval(
            "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ? "
            "AND session_id = ? AND deleted_at IS NULL",
            (agent_id, UUID_A),
        ) == 1
        row = await store.session_projection.get(UUID_A)
        assert row["first_user_message_id"] == ids[2], (
            "the surviving turn is the only row a pointer may name"
        )
        await _assert_the_projection_matches_the_rows(db, store, agent_id, UUID_A)


async def _advisory_lock_waiters(db: AsyncDatabase, name: str) -> list:
    """Sessions BLOCKED on the migration lock ``name``.

    The counterpart to ``_advisory_lock_holders``, and it answers a question no
    other probe in this file can: whether a given piece of work actually goes
    through the serialized path. A waiter on the key is positive evidence — the
    code reached the creation and asked for the boundary. Its absence, when the
    work has finished anyway, is positive evidence of the opposite.
    """
    from kestrel_sovereign.storage.async_database import _backfill_lock_id

    return await db.fetchall(
        "SELECT l.pid, a.state, left(a.query, 60) "
        "FROM pg_locks l JOIN pg_stat_activity a USING (pid) "
        f"WHERE l.locktype = 'advisory' AND NOT l.granted AND {_ADVISORY_KEY} = ?",
        (_backfill_lock_id(name),),
    )


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_first_upgrade_creates_the_projection_index_through_the_lock(
    postgres_db,
):
    """The shipped initialization takes the boundary for this index.

    ``_init_schema`` runs on every ``from_pool()`` — frinz calls it per request —
    so the first boot after this table ships has several initializers arriving at
    once. A bare ``CREATE INDEX IF NOT EXISTS`` passes its existence test before
    taking any lock, so two of them build the index together and the loser dies
    on ``pg_class``' unique index, failing its whole request rather than merely
    skipping an index. ``ensure_index`` owns the probe → lock → re-probe that
    closes that, and the tests above already prove that helper serializes.

    What is left to prove is that THIS index goes through it, which is a claim
    about ``_init_schema`` and not about ``ensure_index``. So the real
    initialization is driven against a virgin schema while the key
    ``ensure_index`` derives for this index is held, and the assertion is that a
    waiter appears on that key: the initializer reached the creation and asked
    for the boundary. Declared in ``CORE_SCHEMA`` instead, it would create the
    index unserialized and run to completion without ever asking — which is the
    other branch below, reported as a failure that names what happened.
    """
    from kestrel_sovereign.storage.async_database import _SESSION_PROJECTION_INDEX

    name, table, _columns = _SESSION_PROJECTION_INDEX
    lock = f"index_{name}"  # the name ensure_index derives, so the same key
    # The pool-level schema, not ``_isolated_schema``: initialization runs in its
    # own task on its own connection, and only a pool default follows it there.
    async with _schema_every_task_can_see(postgres_db) as db:
        initializing = asyncio.create_task(db._init_schema())
        try:
            async with _parked_lock_holder(postgres_db, lock):
                waited = False
                async with asyncio.timeout(LOCK_WAIT_BUDGET):
                    while True:
                        if await _advisory_lock_waiters(postgres_db, lock):
                            waited = True
                            break
                        if initializing.done():
                            break
                        await asyncio.sleep(0.05)
                assert waited, (
                    f"initialization finished without ever waiting on {lock!r}: "
                    f"{name} was created outside the serialized path. Index "
                    f"present while the key was held: "
                    f"{await db._index_exists(name, table)}"
                )
                assert await db._index_exists(name, table) is False
            await _before_the_budget(
                postgres_db, "initialization after the lock was released",
                initializing,
            )
        finally:
            if not initializing.done():
                initializing.cancel()
                await asyncio.gather(initializing, return_exceptions=True)

        assert await db.fetchval(
            "SELECT COUNT(*) FROM pg_index i JOIN pg_class c "
            "ON c.oid = i.indexrelid WHERE c.relname = ? "
            "AND i.indrelid = to_regclass(?)",
            (name, table),
        ) == 1


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
