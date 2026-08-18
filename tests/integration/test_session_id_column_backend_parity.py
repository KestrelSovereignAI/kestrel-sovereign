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
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator
from uuid import uuid4

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.conversation_sessions import (
    PROJECTION_INPUT_COLUMNS,
    ConversationSessionProjection,
)
from kestrel_sovereign.storage.session_grouping import (
    group_messages_into_sessions,
    timestamp_query_param,
)
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
#: Sorts after ``UUID_A``, which the fence case depends on: a repair folds a
#: chunk's sessions in sorted order, so the pass parked on its first ``_store``
#: is holding session A's row.
UUID_B = "3a6c1f0e-2b7d-4c8a-9e10-000000000002"
#: A third, for the cases that need to change a value to something that is
#: neither of the first two.
UUID_C = "3a6c1f0e-2b7d-4c8a-9e10-000000000003"

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


def _accounting(watermark) -> tuple:
    """What a watermark claims, as a tuple a case can pin exactly.

    Spelled out rather than compared field by field, so a case that means "this
    is the whole state" cannot silently stop checking one of them. Pinning it in
    an assertion would make every case that adds one fail for a reason unrelated
    to what it tests. The four fields below are the promise.
    """
    return (
        watermark.valid,
        watermark.stamp,
        watermark.through,
        watermark.target,
    )


async def _assert_the_projection_agrees_with_the_grouper(
    db: AsyncDatabase, projection, agent_id: str
) -> None:
    """The differential, on whichever engine actually ran.

    Extracted rather than written twice: the concurrency case has to make the
    same claim as the direct one — a projection that has stopped disagreeing
    with the grouper only because it stopped projecting anything would satisfy a
    weaker one — and two spellings of one claim is how they drift apart.
    """
    from kestrel_sovereign.storage.conversation_sessions import canonical_order
    from kestrel_sovereign.storage.session_id_column import is_stampable_session_id
    from kestrel_sovereign.storage.session_grouping import (
        coalesce_sessions_by_session_id,
        coerce_session_timestamp,
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
            # `canonical_order()`, not `id` — this must be the order
            # /api/conversations hands the grouper, or the differential
            # compares the projection against a read path that does not exist
            # and the one case where the two orders differ cannot fail it.
            "SELECT id, role, content, metadata, created_at "
            "FROM conversation_history WHERE agent_id = ? "
            f"AND deleted_at IS NULL AND archived_at IS NULL {canonical_order()}",
            (agent_id,),
        )
    ]
    reference = {
        str(session["session_id"]): session
        for session in coalesce_sessions_by_session_id(
            group_messages_into_sessions(history, keep_empty_markers=True)
        )
    }
    stored = {row["session_id"]: row for row in await projection.list()}

    # Every session the grouper finds *whose id the column may hold*. A row
    # filed under nothing gets a synthetic key from the grouper — a bare row id
    # — which `session_id` cannot store, so the projection is legitimately
    # silent about it (Phase A's invariant: silent where it must be, never
    # wrong). Asserted through `is_stampable_session_id` rather than by listing
    # the shapes, which is the same rule the unit differential applies; stating
    # it twice is how the two drifted apart until round 6.
    assert set(stored) == {
        session_id for session_id in reference
        if is_stampable_session_id(session_id)
    }
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


async def _reached(event: asyncio.Event, budget: float) -> bool:
    """Whether ``event`` fires within ``budget``. Never raises, never hangs.

    For the cases whose claim is that something did NOT happen. ``wait_for``
    alone would turn "it correctly never happened" into a timeout error, and the
    absence is the assertion.
    """
    try:
        await asyncio.wait_for(event.wait(), budget)
        return True
    except TimeoutError:
        return False


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


@asynccontextmanager
async def _schema_every_task_can_see(db: AsyncDatabase) -> AsyncIterator[AsyncDatabase]:
    """Like ``_isolated_schema``, but the isolation survives into other TASKS.

    ``_isolated_schema`` pins its schema with ``SET LOCAL search_path`` inside
    one transaction — connection-local state on the one connection that
    transaction holds. A concurrent task's queries acquire a DIFFERENT pool
    connection, which never saw that ``SET``, so a concurrency test built on it
    would quietly write half its rows into ``public`` and measure nothing there.
    (Measured during Phase A: four gathered initializers built their index in
    ``public`` while the assertion looked in the isolated schema and found none.)
    Setting the search_path as a pool server setting makes it the default for
    every connection the pool opens, which is the only form of this isolation a
    multi-task test can use.

    ``db`` is the control connection — used to create and tear down the schema.
    """
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


# ── #2959: the projection and its change stamp, against a real engine ─────
#
# The projection's SQL is dialect-sensitive in four places the SQLite suite
# cannot speak for:
#
# * an ``ON CONFLICT ... DO UPDATE`` upsert, and a compare-and-swap ``UPDATE``
#   whose affected-row count decides whether a watermark may advance -- asyncpg
#   reports that count by parsing a status string, SQLite by ``cursor.rowcount``;
# * a TIMESTAMP column bound from the grouper's ISO text, which asyncpg rejects
#   where SQLite requires it;
# * the TRIGGERS that maintain the change stamp. SQLite writes three trigger
#   bodies; PostgreSQL writes one PL/pgSQL function and three triggers that call
#   it, in a language SQLite does not have. Nothing about one leg is evidence
#   about the other, and a stamp that does not move is a projection that reports
#   itself current forever;
# * the WIDTH of the watermark columns. SQLite has one integer type and stores
#   whatever fits in 64 bits, so an overflow of the declared width is not a
#   thing that can happen there. PostgreSQL raises, out of the compare-and-swap,
#   and the projection then never advances again.


def _without_embeddings(monkeypatch) -> None:
    """Take the embedding provider out of these tests, deliberately.

    ``add_conversation`` co-writes ``embedding_vec`` when a provider is
    reachable, and that column arrives by a migration these fixtures do not
    run -- so whether the INSERT takes its primary path would depend on whether
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


#: Every CORE_SCHEMA table the #2959 projection reads or writes beside. The
#: projection's OWN tables are deliberately absent: they are no longer declared
#: in ``CORE_SCHEMA`` at all, and are created through the shipped
#: ``ensure_session_projection_schema`` by :func:`_create_projection_schema`
#: below — so a case cannot exercise a hand-copied table the product does not
#: have, nor skip the triggers that come with them.
_PROJECTION_TABLES = (
    "conversation_history",
    "conversation_lexical_tokens",
)


async def _create_projection_schema(db: AsyncDatabase) -> None:
    """Everything the projection needs, through the path production uses."""
    await _create_core_tables(db, *_PROJECTION_TABLES)
    await db.ensure_session_projection_schema()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_whole_projection_schema_lands_on_both_backends(db_backend):
    """Tables, triggers and indexes, through whichever dialect actually ran.

    None of it is decoration. ``idx_conversation_sessions_recent`` is what makes
    a page of the newest sessions cost the page instead of the table, which is
    the only reason the projection exists. ``idx_conversation_agent_row_id`` is
    what makes reading the agent's highest id one backward index step — SQLite
    gets that
    plan free from its implicit rowid, so without a PostgreSQL leg an absent
    index would look fine and quietly be an O(history) scan on the engine that
    matters. And the triggers are the entire staleness contract: PostgreSQL
    needs a PL/pgSQL function SQLite has no counterpart for, so "it works on
    SQLite" is no evidence at all here.

    Every declaration is read from the shipped constants rather than respelled,
    so this cannot pass against objects the product does not create.
    """
    from kestrel_sovereign.storage.async_database import (
        _SESSION_FRONTIER_INDEX,
        _SESSION_PROJECTION_INDEX,
    )
    from kestrel_sovereign.storage.conversation_sessions import (
        mutation_triggers,
        projection_tables,
    )

    async with _isolated_schema(db_backend) as db:
        await _create_core_tables(db, *_PROJECTION_TABLES)
        for table, _ddl in projection_tables():
            assert await db.table_exists(table) is False, table
        for name, table, _columns in (
            _SESSION_PROJECTION_INDEX,
            _SESSION_FRONTIER_INDEX,
        ):
            assert await db._index_exists(name, table) is False

        await db.ensure_session_projection_schema()
        # Idempotent: ``_init_schema`` runs on every ``from_pool()``, so the
        # second call is the one that happens thousands of times.
        await db.ensure_session_projection_schema()

        for table, _ddl in projection_tables():
            assert await db.table_exists(table) is True, table
        for trigger, _ddl in mutation_triggers(db.backend_type):
            assert await db._trigger_exists(trigger, "conversation_history"), trigger
        for name, table, _columns in (
            _SESSION_PROJECTION_INDEX,
            _SESSION_FRONTIER_INDEX,
        ):
            assert await db._index_exists(name, table) is True


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_each_row_event_moves_the_change_stamp_by_exactly_one(db_backend):
    """The arithmetic the incremental branch rests on, on the real engine.

    A repair may skip everything it has already accounted for only when the
    stamp's movement equals the number of rows standing above its target, and
    that equality is
    only meaningful while every row event counts once. PostgreSQL routes all
    three triggers through one PL/pgSQL function with a ``TG_OP`` switch, so a
    mis-written branch could stamp an UPDATE twice — which would make the cheap
    branch unreachable, silently, with every test that only checks CONTENT still
    passing.

    Writes go through raw SQL rather than the store so that each statement's
    row-event count is exactly what this case says it is.
    """
    from kestrel_sovereign.storage.conversation_sessions import (
        ConversationSessionProjection as _Projection,
    )
    from kestrel_sovereign.storage.session_grouping import timestamp_query_param

    def _at(hour: int, minute: int):
        return timestamp_query_param(
            db.backend_type, f"2026-03-01T{hour:02d}:{minute:02d}:00"
        )

    agent_id = f"did:test:change-stamp:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_projection_schema(db)
        projection = _Projection(db, agent_id)
        assert await projection.observed_changes() == 0

        for index in range(3):
            await db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, session_id, created_at) "
                "VALUES (?, 'user', ?, ?, ?, ?)",
                (agent_id, f"turn {index}", json.dumps({"session_id": UUID_A}),
                 UUID_A, _at(9, index)),
            )
        assert await projection.observed_changes() == 3

        first = int(await db.fetchval(
            "SELECT MIN(id) FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        ))
        await db.execute(
            "UPDATE conversation_history SET deleted_at = ? WHERE id = ?",
            (_at(10, 0), first),
        )
        assert await projection.observed_changes() == 4, (
            "one UPDATE moved the stamp by more than one, so the incremental "
            "branch's arithmetic no longer holds"
        )

        # A write to a column no field of the projection is derived from must
        # not move it at all -- otherwise the encryption backfill and the
        # embedding co-write would each force a full rebuild.
        await db.execute(
            "UPDATE conversation_history SET content = ? WHERE id = ?",
            ("rewritten", first),
        )
        assert await projection.observed_changes() == 4

        await db.execute(
            "DELETE FROM conversation_history WHERE id = ?", (first,)
        )
        assert await projection.observed_changes() == 5


#: The columns an UPDATE trigger must watch, spelled out **independently** of
#: ``PROJECTION_INPUT_COLUMNS`` and then required to equal it by
#: :func:`test_every_watched_column_has_a_case`.
#:
#: Not derived from that constant, and the difference is the whole point:
#: parametrizing a case over the list it defends means deleting an entry deletes
#: the case with it, so the suite stays green while the protection goes away.
#: Measured — the first spelling of this file did exactly that, and four
#: mutations of the watched list survived it.
_WATCHED_COLUMNS = (
    "agent_id",
    "session_id",
    "role",
    "metadata",
    "created_at",
    "deleted_at",
    "archived_at",
)


def test_every_watched_column_has_a_case():
    """The two lists are one list, checked rather than assumed.

    Adding a column to ``PROJECTION_INPUT_COLUMNS`` without adding it here would
    leave the new entry undefended; removing one from there without removing it
    here leaves the case below standing, to fail — which is the direction that
    matters.
    """
    assert _WATCHED_COLUMNS == PROJECTION_INPUT_COLUMNS


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("column", _WATCHED_COLUMNS)
async def test_writing_any_watched_column_alone_moves_the_change_stamp(
    db_backend, column
):
    """Every watched column, defended one at a time.

    A column can be in that list and still have no case that fails when it is
    removed, because most mutations move several columns at once — the metadata
    merge that re-homes a row rewrites ``session_id`` *and* ``metadata``, so a
    trigger watching only the second still notices. Which means the first entry
    would read as protection while a mutation deleting it went unnoticed, and
    that is precisely the shape Phase A's review named as the worst kind.

    ``session_id`` alone really does move: Phase A's backfill is
    ``UPDATE conversation_history SET session_id = ...`` and touches nothing
    else, so it is the shape written here. Each column gets its own
    parametrization so removing any one of them fails exactly one case.
    """
    from kestrel_sovereign.storage.conversation_sessions import (
        ConversationSessionProjection as _Projection,
    )
    from kestrel_sovereign.storage.session_grouping import timestamp_query_param

    agent_id = f"did:test:watched-{column}:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_projection_schema(db)
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, session_id, created_at) "
            "VALUES (?, 'user', 'a turn', ?, ?, ?)",
            (agent_id, json.dumps({"session_id": UUID_A}), UUID_A,
             timestamp_query_param(db.backend_type, "2026-03-01T09:00:00")),
        )
        projection = _Projection(db, agent_id)
        assert await projection.observed_changes() == 1

        # A value genuinely different from the one the row holds, so the
        # trigger's IS DISTINCT FROM / IS NOT test really fires.
        replacement = {
            "agent_id": f"did:test:somewhere-else:{uuid4()}",
            "session_id": "3a6c1f0e-2b7d-4c8a-9e10-00000000000f",
            "role": "assistant",
            "metadata": json.dumps({"session_id": UUID_A, "operator_signal": True}),
            "created_at": timestamp_query_param(
                db.backend_type, "2026-03-01T11:00:00"
            ),
            "deleted_at": timestamp_query_param(
                db.backend_type, "2026-03-01T12:00:00"
            ),
            "archived_at": timestamp_query_param(
                db.backend_type, "2026-03-01T13:00:00"
            ),
        }[column]
        await db.execute(
            f"UPDATE conversation_history SET {column} = ? WHERE agent_id = ?",
            (replacement, agent_id),
        )

        # Asked of the ORIGINAL agent even when the row has just left it: a row
        # changing hands is a change for the agent that lost it too, and only
        # the trigger's second stamp says so.
        assert await projection.observed_changes() == 2, (
            f"writing {column} alone did not move the change stamp, so a "
            "projection derived from it would report itself current"
        )


async def _seed_one_projected_row(db: AsyncDatabase, agent_id: str, metadata) -> int:
    """One stamped history row, and its id."""
    await db.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, session_id, created_at) "
        "VALUES (?, 'user', 'a turn', ?, ?, ?)",
        (
            agent_id,
            metadata,
            column_session_id(metadata),
            timestamp_query_param(db.backend_type, "2026-03-01T09:00:00"),
        ),
    )
    return int(
        await db.fetchval(
            "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        )
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_metadata_bookkeeping_does_not_move_the_change_stamp(
    db_backend, monkeypatch
):
    """The read path's own write, on whichever engine renders the comparison.

    ``MemoryRetriever.update_access`` and ``update_applied`` bump counters and
    stamp timestamps inside the same document the session id lives in, on every
    retrieval. A trigger comparing ``metadata`` whole moved the stamp for those,
    and a stamp movement with no row appended is the movement ``_plan`` cannot
    attribute — so recall rebuilt the whole projection.

    This has to be asked of both engines because the two comparisons share no
    code at all: SQLite reads four paths out with ``json_extract`` under a
    ``json_valid`` guard, PostgreSQL builds a ``jsonb`` object under
    ``pg_input_is_valid``. And the statement being defended against differs too
    — the PostgreSQL form of ``atomic_increment_metadata_counter`` rewrites the
    document through ``jsonb_set``, which reserializes it, so the bytes differ
    there even where no value the projection reads did.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:bookkeeping:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_projection_schema(db)
        message_id = await _seed_one_projected_row(
            db, agent_id, json.dumps({"session_id": UUID_A, "access_count": 0})
        )
        projection = ConversationSessionProjection(db, agent_id)
        assert (await projection.repair()).current
        before = await projection.observed_changes()

        store = AsyncConversationStore(db, agent_id=agent_id)
        assert await store.atomic_increment_metadata_counter(
            message_id, "access_count", "last_accessed"
        )

        assert await projection.observed_changes() == before, (
            "bookkeeping on the read path moved the change stamp, so every "
            "memory retrieval invalidates the projection"
        )
        assert not await projection.is_stale()

        # The write really landed, so nothing above passed because nothing
        # happened — and the session id survived the merge, which is the value
        # the comparison is narrowed to watching.
        written = json.loads(
            await db.fetchval(
                "SELECT metadata FROM conversation_history WHERE id = ?",
                (message_id,),
            )
        )
        assert written["access_count"] == 1
        assert written["last_accessed"]
        assert written["session_id"] == UUID_A


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_metadata_the_engine_cannot_parse_neither_raises_nor_hides(
    db_backend,
):
    """The fallback, on both engines, in both directions it has to be right.

    ``metadata`` is free text and legacy rows hold documents that will not
    parse. So the comparison must never *raise* — it rides along with every
    UPDATE of every other column, including the encryption backfill and the
    #1402 canonical/transport split, and a trigger that raised would fail those
    writes on exactly the rows least able to afford it. And it must never
    *hide* a change: two unparseable documents are two documents.

    PostgreSQL is the engine that makes this urgent. ``metadata::jsonb`` raises
    on malformed text, on the NUL escape, and on a lone surrogate — the last of
    which Phase A had to reach for ``pg_input_is_valid`` to cover, after an
    enumeration of "ways JSON can fail jsonb" turned out to be incomplete.

    A duplicated key ends in the same place by two different routes, which is
    the parity claim worth making: PostgreSQL's ``jsonb`` resolves it the way
    ``json.loads`` does, so it can extract; SQLite's ``json_extract`` would read
    the *first* occurrence where the derivation reads the last, so it declines
    to extract and compares the document instead.
    """
    agent_id = f"did:test:unparseable:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_projection_schema(db)
        message_id = await _seed_one_projected_row(
            db, agent_id, json.dumps({"session_id": UUID_A})
        )
        projection = ConversationSessionProjection(db, agent_id)

        async def _set(metadata):
            await db.execute(
                "UPDATE conversation_history SET metadata = ? WHERE id = ?",
                (metadata, message_id),
            )

        stamp = await projection.observed_changes()
        for document, moves, why in (
            ("not json at all", 1, "a readable document became an unreadable one"),
            ("also not json, but different", 1, "two unreadable documents differ"),
            ("also not json, but different", 0, "the same document was rewritten"),
            ('{"session_id": "%s"}' % UUID_A, 1, "a session id came back"),
            ('{"session_id": "%s", "seen": 1}' % UUID_A, 0,
             "an unwatched key was added"),
            ('{"session_id": "%s", "session_id": "%s"}' % (UUID_A, UUID_B), 1,
             "a duplicated key changed what the derivation would read"),
            ('{"session_id": "%s", "session_id": "%s"}' % (UUID_A, UUID_C), 1,
             "the last occurrence of a duplicated key changed"),
            # Valid to Python's json, refused by jsonb: the case an enumeration
            # of "how JSON fails jsonb" missed during Phase A.
            ('{"session_id": "\\ud800"}', 1, "an unrepresentable document"),
        ):
            await _set(document)
            moved = await projection.observed_changes() - stamp
            assert moved == moves, f"{why}: stamp moved by {moved}, wanted {moves}"
            stamp += moved

        # ...and a write to a column the projection cannot see, on a row whose
        # metadata the engine cannot parse. The trigger runs. It must not raise.
        await _set("not json at all")
        stamp = await projection.observed_changes()
        await db.execute(
            "UPDATE conversation_history SET content = ? WHERE id = ?",
            ("rewritten while the metadata is unreadable", message_id),
        )
        assert await projection.observed_changes() == stamp, (
            "an unwatched write moved the stamp for a row whose metadata "
            "cannot be parsed"
        )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_concurrent_initializers_build_the_projection_schema_once(postgres_db):
    """A post-upgrade request burst, with none of the objects present yet.

    ``_init_schema`` runs on every ``from_pool()`` — frinz calls it per request —
    so the first boot after this ticket ships is genuinely parallel. A bare
    ``CREATE TABLE IF NOT EXISTS`` evaluates its existence test before taking the
    lock that would exclude a peer, so two initializers can both proceed and one
    dies on ``pg_class``' unique index: not a skipped table, a failed request.
    The same is true of ``CREATE OR REPLACE FUNCTION`` on ``pg_proc``.

    PostgreSQL-only because SQLite serializes writers on one file and cannot
    exhibit the catalog race at all, so a dual-backend spelling would contribute
    a leg that passes no matter what the code does.

    Bounded: the case carries a timeout, because a lock test that can hang is a
    timed-out job with no failing assertion.
    """
    from kestrel_sovereign.storage.conversation_sessions import (
        mutation_triggers,
        projection_tables,
    )

    async with _schema_every_task_can_see(postgres_db) as db:
        await _create_core_tables(db, *_PROJECTION_TABLES)
        for table, _ddl in projection_tables():
            assert await db.table_exists(table) is False, table

        outcomes = await _before_the_budget(
            db,
            "eight concurrent initializers building the projection schema",
            asyncio.gather(
                *(db.ensure_session_projection_schema() for _ in range(8)),
                return_exceptions=True,
            ),
            budget=60.0,
        )
        raised = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        assert not raised, raised

        for table, _ddl in projection_tables():
            assert await db.table_exists(table) is True, table
        for trigger, _ddl in mutation_triggers(db.backend_type):
            assert await db.fetchval(
                "SELECT COUNT(*) FROM pg_trigger WHERE tgname = ? "
                "AND tgrelid = to_regclass('conversation_history') "
                "AND NOT tgisinternal",
                (trigger,),
            ) == 1, trigger
        assert not await _advisory_lock_holders(db, "conversation_sessions_2959")


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_watermark_is_comparable_on_both_backends(db_backend, monkeypatch):
    """A repaired projection reports itself current on this engine too.

    Staleness is an equality between two values that arrive by different routes:
    one counted by a trigger, one read back out of a column. Nothing normalizes
    them -- deliberately, since a coercion there would be a guard no test could
    kill -- so this is where the two routes are required to meet. If they did
    not, the equality would be false forever: the projection rebuilt on every
    read, never once reporting itself current, while every assertion about its
    CONTENT still passed.

    ``accounted_valid`` is asserted here rather than inferred, because on
    PostgreSQL it is an ``INTEGER`` bound from a Python ``bool`` and asyncpg is
    strict about that pairing: getting it wrong raises out of the statement that
    records the watermark rather than storing a wrong flag.

    Width is the other half of that, and is a separate case:
    :func:`test_a_watermark_wider_than_int4_is_storable`.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )
    from kestrel_sovereign.storage.conversation_sessions import (
        CURRENT,
        REBUILT,
        SessionWatermark,
    )

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-watermark:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_projection_schema(db)
        store = AsyncConversationStore(db, agent_id=agent_id)
        projection = ConversationSessionProjection(db, agent_id)

        assert await projection.observed_changes() == 0
        assert _accounting(await projection.accounted()) == (False, 0, 0, 0)
        assert await projection.is_stale()
        assert (await projection.repair()).kind == REBUILT
        assert _accounting(await projection.accounted()) == (True, 0, 0, 0)
        assert not await projection.is_stale()

        await store.add_conversation("user", "first turn", session_id=UUID_A)
        await store.add_conversation("assistant", "reply", session_id=UUID_A)

        assert await projection.is_stale()
        # Compared as the ints they are declared to be, on the engine whose
        # aggregates are typed. A driver handing back any other numeric type
        # would make the equality below false forever.
        changes = await projection.observed_changes()
        assert isinstance(changes, int) and changes == 2

        assert (await projection.repair()).current
        accounted = await projection.accounted()
        assert accounted.valid is True
        assert accounted.stamp == changes
        assert accounted.through > 0
        assert not await projection.is_stale()
        assert (await projection.repair()).kind == CURRENT


#: PostgreSQL's ``int4`` ceiling. Named because the case below is about this
#: number and nothing else, and a bare ``2147483647`` in an assertion reads like
#: an arbitrary large integer.
_INT4_MAX = 2_147_483_647


@pytest.mark.asyncio
async def test_a_watermark_wider_than_int4_is_storable(postgres_db):
    """``accounted_stamp`` counts row events for the agent's whole life.

    Which is precisely the number this table exists to stop being bounded by: an
    agent expected to run indefinitely crosses 2,147,483,647 row events, and the
    symptom would not be a wrong count. PostgreSQL refuses the parameter, the
    statement that records the watermark raises out of :meth:`repair`, and the
    projection can never advance again — every read stale, every repair raising.

    PostgreSQL-only because SQLite has a single 64-bit integer type and cannot
    exhibit a declared-width overflow at all, so a dual-backend spelling would
    contribute a leg that passes no matter what the column says.

    The ledger is set directly rather than accumulated: reaching this count by
    performing two billion row events would test the same column and never
    finish. It is exercised in both places it lives — the ledger the triggers
    write and the watermark a chunk records — because a wide value has to
    survive being read out of one and written into the other.

    ``accounted_through`` is not pushed past ``int4`` beside it, and that is a
    fact about the schema rather than an omission: ``conversation_history.id`` is
    ``SERIAL``, so PostgreSQL refuses a row id in that range at the INSERT
    (measured — "value out of int32 range"). The column is declared ``BIGINT``
    anyway because it is part of the same watermark, not because a value could
    reach it today.
    """
    from kestrel_sovereign.storage.conversation_sessions import CURRENT

    agent_id = f"did:test:watermark-width:{uuid4()}"
    async with _schema_every_task_can_see(postgres_db) as db:
        await _create_projection_schema(db)
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, session_id, created_at) "
            "VALUES (?, 'user', 'x', ?, ?, ?)",
            (
                agent_id, json.dumps({"session_id": UUID_A}), UUID_A,
                datetime(2026, 3, 1, 9, 0, 0),
            ),
        )
        row_id = int(await db.fetchval(
            "SELECT MAX(id) FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        ))
        changes = _INT4_MAX + 7
        await db.execute(
            "UPDATE conversation_history_changes SET changes = ? WHERE agent_id = ?",
            (changes, agent_id),
        )

        projection = ConversationSessionProjection(db, agent_id)
        assert await projection.observed_changes() == changes > _INT4_MAX

        assert (await projection.repair()).current
        accounted = await projection.accounted()
        assert accounted.valid is True
        assert accounted.stamp == changes, (
            "the change stamp read back out of the table is not the one that "
            "was written, so a column truncated it"
        )
        assert accounted.through == row_id
        assert not await projection.is_stale()
        assert (await projection.repair()).kind == CURRENT


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_projection_is_repaired_after_each_mutation_on_both_backends(
    db_backend, monkeypatch
):
    """Insert, soft-delete, restore, archive, purge -- through the real dialect.

    The claims are the ones the SQLite suite makes; what is new here is that
    every statement carrying them is the PostgreSQL spelling. A broken upsert
    would surface as a duplicate-key error on the second repair, and a timestamp
    bound as text would fail at the column.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-projection:{uuid4()}"
    async with _isolated_schema(db_backend) as db:
        await _create_projection_schema(db)
        store = AsyncConversationStore(db, agent_id=agent_id)
        projection = ConversationSessionProjection(db, agent_id)

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

        assert await projection.is_stale()
        await projection.repair()
        projected = await projection.get(UUID_A)
        assert projected["message_count"] == 3
        assert projected["user_message_count"] == 2
        assert projected["first_user_message_id"] == ids[0]

        # The pointer moves off a trashed row and back, on this engine, with the
        # census noticing each way round.
        assert await store.delete_message(ids[0])
        assert await projection.is_stale()
        await projection.repair()
        assert (await projection.get(UUID_A))["first_user_message_id"] == ids[2]

        assert await store.restore_message(ids[0])
        assert await projection.is_stale()
        await projection.repair()
        assert (await projection.get(UUID_A))["first_user_message_id"] == ids[0]

        # An archive leaves through a different column and is seen the same way.
        assert await store.archive_conversation_session(UUID_A) == 3
        assert await projection.is_stale()
        await projection.repair()
        assert await projection.get(UUID_A) is None

        assert await store.unarchive_conversation_session(UUID_A) == 3
        assert await projection.is_stale()
        await projection.repair()
        assert (await projection.get(UUID_A))["message_count"] == 3

        # ...and a purge takes the projection row with it.
        assert await store.purge_conversation_session(UUID_A) == 3
        assert await projection.is_stale()
        await projection.repair()
        assert await projection.get(UUID_A) is None
        assert await db.fetchval(
            "SELECT COUNT(*) FROM conversation_sessions WHERE agent_id = ?",
            (agent_id,),
        ) == 0
        assert not await projection.is_stale()


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

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-projection-diff:{uuid4()}"
    other = "3a6c1f0e-2b7d-4c8a-9e10-00000000000b"
    async with _isolated_schema(db_backend) as db:
        await _create_projection_schema(db)
        store = AsyncConversationStore(db, agent_id=agent_id)
        projection = ConversationSessionProjection(db, agent_id)

        await store.add_conversation("user", "A one", session_id=UUID_A)
        await store.add_conversation("assistant", "A two", session_id=UUID_A)
        await store.add_conversation("user", "B one", session_id=other)
        # An autonomous wake is a user row that may not become the pointer.
        await store.add_conversation(
            "user", "B wake", session_id=other,
            metadata={"signal_wake": {"source": "heartbeat"}},
        )
        await projection.repair()

        await _assert_the_projection_agrees_with_the_grouper(db, projection, agent_id)
        stored = {row["session_id"]: row for row in await projection.list()}
        assert stored[other]["wake_source"] == "heartbeat"


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_rebuild_equals_the_repairs_on_both_backends(db_backend, monkeypatch):
    """Dropping the table and rebuilding it lands on the same rows.

    Asked of PostgreSQL as well because the rebuild path binds the same
    timestamps through a different code route than the incremental one, and
    because ``ORDER BY last_message_at`` over a TIMESTAMP column is the engine's
    collation rather than Python's.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-rebuild:{uuid4()}"
    other = "3a6c1f0e-2b7d-4c8a-9e10-00000000000c"
    async with _isolated_schema(db_backend) as db:
        await _create_projection_schema(db)
        store = AsyncConversationStore(db, agent_id=agent_id)
        projection = ConversationSessionProjection(db, agent_id)

        await store.add_conversation("user", "A one", session_id=UUID_A)
        await projection.repair()
        await store.add_conversation("user", "B one", session_id=other)
        await projection.repair()
        first = await db.fetchval(
            "SELECT MIN(id) FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        )
        assert await store.delete_message(int(first))
        await projection.repair()

        incremental = await projection.list()
        assert incremental

        await db.execute(
            "DELETE FROM conversation_sessions WHERE agent_id = ?", (agent_id,)
        )
        assert await projection.rebuild() == len(incremental)
        assert await projection.list() == incremental
        assert not await projection.is_stale()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_repair_that_loses_a_race_can_only_leave_the_pair_behind(
    postgres_db, monkeypatch
):
    """The one pass that can still derive from history somebody has moved past.

    PostgreSQL is where the concurrency claim has to be made: separate
    connections mean separate snapshots. Repair *steps* are serialized on the
    agent's watermark row, so a chunked pass cannot act on a conclusion another
    pass has invalidated — it replans under the lock. The transcript derivation
    is the exception, and deliberately so: attribution is a property of the whole
    sequence, so that pass must read every live row, and reading every live row
    is exactly the unbounded work nothing may hold a lock across. It therefore
    reaches the lock holding a stamp it read before the read.

    The claim is not that this loser is prevented. It is that it can only leave
    the projection **behind**, never falsely current: it writes its rows and its
    watermark in one transaction, so what stands afterwards is the pair it
    derived — an older stamp, and rows consistent with it. A pair that is behind
    is a detected stale projection, which this contract permits and the next
    repair fixes.

    So the loser is parked after it derives and before it takes the lock, a
    mutation lands, a second pass repairs to current, and then the loser is let
    go to overwrite the watermark with its older stamp. What must never happen is
    the projection calling itself current over rows nobody rechecked.

    The unstamped row is what puts this agent on the transcript path, and it is
    load-bearing rather than corpus decoration: without it every pass is chunked
    and there is no loser left to test.

    Bounded on both sides: the parked pass rescues itself on the wait budget and
    the case carries a timeout, because a concurrency test that can hang is a
    timed-out job with no failing assertion — which is how a 23-hour
    advisory-lock wedge went unnoticed during Phase A.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-race:{uuid4()}"
    async with _schema_every_task_can_see(postgres_db) as db:
        await _create_projection_schema(db)
        store = AsyncConversationStore(db, agent_id=agent_id)
        projection = ConversationSessionProjection(db, agent_id)

        await store.add_conversation("user", "first turn", session_id=UUID_A)
        await store.add_conversation("assistant", "reply", session_id=UUID_A)
        # A row filed under no session id at all, which is what forces the
        # transcript derivation this case is about. Written directly because
        # ``add_conversation`` resolves an implicit session for every row it
        # takes, so it cannot produce the legacy shape this needs.
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, session_id, created_at) "
            "VALUES (?, 'assistant', 'an unlabeled reply', NULL, NULL, ?)",
            (
                agent_id,
                timestamp_query_param(db.backend_type, "2026-03-01T09:05:00"),
            ),
        )
        assert await db.fetchval(
            "SELECT COUNT(*) FROM conversation_history "
            "WHERE agent_id = ? AND session_id IS NULL",
            (agent_id,),
        ) == 1, "the transcript derivation this case needs is not reachable"
        assert (await projection.repair()).current
        first = int(await db.fetchval(
            "SELECT MIN(id) FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        ))
        assert await store.delete_message(first)

        parked = asyncio.Event()
        resume = asyncio.Event()

        class _Parks(ConversationSessionProjection):
            """Derives, then waits — before it takes the agent's repair lock."""

            deriving = False
            held = False

            async def _rebuild_from_transcript(self):
                self.deriving = True
                return await super()._rebuild_from_transcript()

            async def _claim(self):
                if self.deriving and not self.held:
                    self.held = True
                    parked.set()
                    await asyncio.wait_for(resume.wait(), LOCK_WAIT_BUDGET)
                return await super()._claim()

        slow = asyncio.create_task(_Parks(db, agent_id, step_budget=1).repair())
        try:
            await _before_the_budget(
                db, "the parked repair reaching its pause", parked.wait()
            )
            # History moves under the parked pass, and a second pass accounts
            # for it. From here the parked pass's stamp is provably old.
            await store.add_conversation("user", "a later turn", session_id=UUID_A)
            assert (await ConversationSessionProjection(db, agent_id).repair()).current
        finally:
            resume.set()
        await _before_the_budget(
            db, "the parked repair finishing", asyncio.shield(slow)
        )

        # What UUID_A ends up holding: its surviving reply and the later turn.
        # A row filed under nothing belongs to the cluster it fell next to —
        # and which cluster that is, is decided by the order the transcript is
        # read in. This one is stamped 2026-03-01 while the rest are written by
        # the clock, so in `canonical_order()` it stands five months before them
        # and the gap rule makes it a session of its own. Read in id order it
        # would instead land beside UUID_A and count as a third message there,
        # which is what this asserted until round 6 — a merge the conversation
        # list, which orders by time, would never have shown.
        settled = 2

        accounted = await projection.accounted()
        observed = await projection.observed_changes()
        assert accounted.stamp <= observed
        if accounted.stamp != observed or not accounted.complete:
            assert await projection.is_stale(), (
                "the loser left a watermark that does not describe history, and "
                "the projection still reports itself current"
            )
        else:
            # The loser landed on the same accounting; then it must also hold
            # the same rows, which is what "idempotent" has to mean here.
            assert (await projection.get(UUID_A))["message_count"] == settled

        # ...and whatever it left, one more repair is all it costs.
        assert (await projection.repair()).current
        assert not await projection.is_stale()
        assert (await projection.get(UUID_A))["message_count"] == settled
        await _assert_the_projection_agrees_with_the_grouper(
            db, projection, agent_id
        )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("primed", [False, True], ids=["unbuilt", "already-built"])
async def test_a_newer_rebuild_cannot_publish_over_an_older_ones_orphan(
    postgres_db, monkeypatch, primed
):
    """The state MVCC makes reachable and the stamp cannot see.

    A rebuild clears the ground and then walks it. Under PostgreSQL's MVCC a
    ``DELETE`` cannot see rows another transaction has inserted but not
    committed, so with nothing serializing the two, this interleaving is
    available:

    1. the older rebuild deletes, derives, and inserts its rows — including a
       session that still existed when it read history;
    2. history moves: that session is purged;
    3. the newer rebuild deletes — and sees none of the older's uncommitted rows;
    4. the older commits, so its rows land *after* the newer's delete;
    5. the newer stores what it derived and commits a watermark that says
       **current**, standing over an orphan session no live row supports.

    Nothing downstream detects that. ``is_stale`` compares the stamp, and the
    stamp is right; it is the table that is wrong. It is also not covered by the
    atomic-pair argument that governs everything else here, because each
    transaction did write a consistent pair — the *table* is the union of two of
    them.

    So the interleaving is scripted exactly, and the assertion is that step 3
    cannot happen while step 1 is in flight: the newer rebuild is held at the
    agent's repair lock and never reaches its delete. Asserting only the end
    state would pass for a projection that got lucky, which is why the block
    itself is what is checked, and the end state checked beside it.

    **Both starting states, because ``_claim`` has two statements and they hold
    different halves of the same door.** The insert that makes the watermark row
    exist is what serializes a pair racing to *create* it, on the primary key;
    the row lock is what serializes everyone after that, because an insert that
    conflicts takes no lock on the row it declined to write. So each statement is
    the only mechanism in exactly one of these states, and a case that ran only
    one of them passes with the other statement deleted. Measured, in both
    directions, against this file.

    ``unbuilt`` is the first repair an agent ever runs, where there is no
    watermark row to lock. ``already-built`` repairs to current first, so the
    insert is a no-op for both passes; its soft-delete is what makes each pass
    plan a REBUILD rather than an incremental step.

    Both parks are bounded and the case carries a timeout, so a mechanism that
    stopped working reports a failed assertion rather than a wedged run.
    """
    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-orphan:{uuid4()}"
    async with _schema_every_task_can_see(postgres_db) as db:
        await _create_projection_schema(db)
        projection = ConversationSessionProjection(db, agent_id)

        for index, session in enumerate((UUID_A, UUID_A, UUID_B, UUID_B)):
            await db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, session_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    agent_id,
                    "user" if index % 2 == 0 else "assistant",
                    f"turn {index}",
                    json.dumps({"session_id": session}),
                    session,
                    timestamp_query_param(
                        db.backend_type, f"2026-03-01T09:0{index}:00"
                    ),
                ),
            )

        if primed:
            # Repair first, so the watermark row exists and ``_claim``'s insert
            # cannot be what serializes the two passes below.
            assert (await projection.repair()).current
            # A change no append can explain, so both passes plan a REBUILD —
            # which is the plan whose clearing DELETE this case is about.
            await db.execute(
                "UPDATE conversation_history SET deleted_at = ? "
                "WHERE agent_id = ? AND session_id = ? "
                "AND id = (SELECT MIN(id) FROM conversation_history "
                "          WHERE agent_id = ? AND session_id = ?)",
                (
                    timestamp_query_param(db.backend_type, "2026-03-01T10:00:00"),
                    agent_id, UUID_A, agent_id, UUID_A,
                ),
            )
        # The starting state each leg claims to be in, asserted rather than
        # assumed — the two legs differ ONLY in this, and a leg that had drifted
        # into the other's state would defend the other's statement twice.
        assert await db.fetchval(
            "SELECT COUNT(*) FROM conversation_session_watermarks "
            "WHERE agent_id = ?",
            (agent_id,),
        ) == (1 if primed else 0)
        assert await projection.is_stale()

        older_parked = asyncio.Event()
        older_may_commit = asyncio.Event()
        newer_reached_delete = asyncio.Event()
        newer_may_finish = asyncio.Event()

        class _OlderRebuild(ConversationSessionProjection):
            """Stores its rows, then waits — uncommitted, inside its step."""

            held = False

            async def _record(self, watermark):
                if not self.held:
                    self.held = True
                    older_parked.set()
                    await asyncio.wait_for(
                        older_may_commit.wait(), LOCK_WAIT_BUDGET
                    )
                return await super()._record(watermark)

        class _NewerRebuild(ConversationSessionProjection):
            """Announces reaching its clearing delete, then waits after it."""

            held = False

            async def _chunk(self, plan):
                if plan.discard:
                    newer_reached_delete.set()
                return await super()._chunk(plan)

            async def _store(self, session):
                if not self.held:
                    self.held = True
                    await asyncio.wait_for(
                        newer_may_finish.wait(), LOCK_WAIT_BUDGET
                    )
                return await super()._store(session)

        older = asyncio.create_task(
            _OlderRebuild(db, agent_id, step_budget=1).repair()
        )
        newer = None
        try:
            await _before_the_budget(
                db, "the older rebuild storing its rows", older_parked.wait()
            )
            # History moves: the session the older rebuild has already derived
            # and inserted no longer exists. Only its uncommitted row does.
            purged = await db.execute(
                "DELETE FROM conversation_history "
                "WHERE agent_id = ? AND session_id = ?",
                (agent_id, UUID_B),
            )
            assert purged == 2

            newer = asyncio.create_task(_NewerRebuild(db, agent_id).repair())
            raced = await _reached(newer_reached_delete, LOCK_WAIT_BUDGET / 4)
            assert not raced, (
                "a second rebuild reached its clearing DELETE while another was "
                "mid-flight — under MVCC that DELETE cannot see the rows in "
                "flight, so the projection can be left holding an orphan under "
                "a watermark that reports itself current"
            )
        finally:
            older_may_commit.set()
            newer_may_finish.set()
        await _before_the_budget(
            db, "the older rebuild finishing", asyncio.shield(older)
        )
        await _before_the_budget(
            db, "the newer rebuild finishing", asyncio.shield(newer)
        )

        assert (await projection.repair()).current
        assert not await projection.is_stale()
        assert await projection.get(UUID_B) is None, (
            "the projection holds a session whose every row was purged, while "
            "reporting itself current"
        )
        await _assert_the_projection_agrees_with_the_grouper(
            db, projection, agent_id
        )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_three_repairs_racing_never_leave_a_projection_that_lies(
    postgres_db, monkeypatch
):
    """Overlapping idempotent work is the whole concurrency mechanism.

    Three passes are started at once over one agent, at ``chunk_rows=1`` so
    their chunks interleave rather than each pass being one statement. Every
    chunk recomputes from the rows instead of incrementing, so duplicated work
    is harmless; every chunk writes its rows and its watermark together, so no
    pass can leave another's rows under its own claim.

    What is asserted is the invariant, not an outcome: the projection may end up
    behind — a pass that read an older state can land last and knock the
    accounting back, costing a redo — but it may never report itself current
    while disagreeing with the grouper. Asserting convergence directly would be
    asserting that racing passes cannot lose a step, which is neither true nor
    required.

    This also exercises the ordering the sessions are refreshed in: every pass
    walks a chunk's sessions sorted, and takes the watermark row last, so two
    passes cannot acquire the same two rows in opposite orders and deadlock. On
    PostgreSQL that would surface here as a deadlock detection rather than as a
    hang, and either way the timeout bounds it.
    """
    from kestrel_sovereign.storage.async_conversation_store import (
        AsyncConversationStore,
    )

    _without_embeddings(monkeypatch)
    agent_id = f"did:test:session-racers:{uuid4()}"
    async with _schema_every_task_can_see(postgres_db) as db:
        await _create_projection_schema(db)
        store = AsyncConversationStore(db, agent_id=agent_id)
        projection = ConversationSessionProjection(db, agent_id)

        for index in range(4):
            await store.add_conversation(
                "user", f"turn {index}", session_id=UUID_A if index % 2 else UUID_B
            )
        assert (await projection.repair()).current
        first = int(await db.fetchval(
            "SELECT MIN(id) FROM conversation_history WHERE agent_id = ?",
            (agent_id,),
        ))
        assert await store.delete_message(first)

        await _before_the_budget(
            db,
            "three concurrent repairs",
            asyncio.gather(
                *(
                    ConversationSessionProjection(
                        db, agent_id, chunk_rows=1
                    ).repair()
                    for _ in range(3)
                )
            ),
        )

        if not await projection.is_stale():
            await _assert_the_projection_agrees_with_the_grouper(
                db, projection, agent_id
            )

        assert (await projection.repair()).current
        assert not await projection.is_stale()
        await _assert_the_projection_agrees_with_the_grouper(db, projection, agent_id)
