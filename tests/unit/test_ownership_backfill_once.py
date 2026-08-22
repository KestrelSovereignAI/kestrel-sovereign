"""Regression: #2649 ownership backfills must run once, not on every from_pool.

Two independent bugs made ``_init_schema`` unusable on a populated Postgres
database (frinz companion creation 500'd / hung):

1. The ownership backfills ran on EVERY ``_init_schema()`` — and ``from_pool()``
   runs it per request — so concurrent inits re-scanned the ledger tables and
   contended on locks. They are one-time legacy migrations (new rows record
   ownership at write time), so they are now gated behind a persistent
   ``schema_backfills`` marker.

2. The document-chunk backfill grouped AFTER the chunk×owner join, exploding on
   a file owned by many agents (26k chunks × 1.4k owners) only to discard them
   with ``HAVING COUNT(DISTINCT) = 1``. It now resolves single-owner files
   first, then joins — same result, no explosion.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Iterator
from unittest.mock import patch

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture(autouse=True)
def _kestrel_data_key(monkeypatch) -> Iterator[None]:
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-32-bytes-fixed--")
    yield


@pytest_asyncio.fixture
async def db(tmp_path):
    database = await AsyncDatabase.sqlite(str(tmp_path / "ownership.db"))
    try:
        yield database
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_first_init_records_marker(db):
    row = await db.fetchone(
        "SELECT name FROM schema_backfills WHERE name = 'ownership_2649'"
    )
    assert row is not None, "ownership backfill marker not recorded on first init"


@pytest.mark.asyncio
async def test_backfills_skip_when_marker_present(db, monkeypatch):
    """A second _init_schema must NOT re-run the expensive backfills."""
    calls = {"n": 0}

    async def spy():
        calls["n"] += 1

    # Marker was set by the fixture's first init; a re-init must skip.
    monkeypatch.setattr(db, "_backfill_graph_ownership", spy)
    monkeypatch.setattr(db, "_backfill_file_ownership", spy)
    monkeypatch.setattr(db, "_backfill_document_chunk_ownership", spy)
    await db._init_schema()
    assert calls["n"] == 0, "backfills re-ran despite the completion marker"


@pytest.mark.asyncio
async def test_backfills_rerun_if_marker_absent(db, monkeypatch):
    """Clearing the marker makes the gated backfills run again (retry-safe)."""
    await db.execute("DELETE FROM schema_backfills")
    calls = {"n": 0}

    async def spy():
        calls["n"] += 1

    monkeypatch.setattr(db, "_backfill_graph_ownership", spy)
    monkeypatch.setattr(db, "_backfill_file_ownership", spy)
    monkeypatch.setattr(db, "_backfill_document_chunk_ownership", spy)
    await db._init_schema()
    assert calls["n"] == 3
    # ...and the marker is re-recorded so the next init skips again.
    row = await db.fetchone(
        "SELECT name FROM schema_backfills WHERE name = 'ownership_2649'"
    )
    assert row is not None


@pytest.mark.asyncio
async def test_document_chunk_backfill_assigns_only_single_owner_files(db):
    """The rewritten query assigns a chunk iff its file has exactly one owner."""
    # File A: exactly one owner -> its chunk should be assigned.
    await db.execute(
        "INSERT INTO file_owners (content_hash, agent_id, original_name) "
        "VALUES ('hashA', 'agentA', 'a.txt')"
    )
    # File B: two distinct owners -> its chunk must remain unassigned.
    await db.execute(
        "INSERT INTO file_owners (content_hash, agent_id, original_name) "
        "VALUES ('hashB', 'agentB1', 'b.txt')"
    )
    await db.execute(
        "INSERT INTO file_owners (content_hash, agent_id, original_name) "
        "VALUES ('hashB', 'agentB2', 'b.txt')"
    )
    await db.execute(
        "INSERT INTO document_chunks (file_hash, content) VALUES ('hashA', 'ca')"
    )
    await db.execute(
        "INSERT INTO document_chunks (file_hash, content) VALUES ('hashB', 'cb')"
    )
    ca = (await db.fetchone(
        "SELECT chunk_id FROM document_chunks WHERE file_hash='hashA'"))[0]

    await db._backfill_document_chunk_ownership()

    owners = await db.fetchall(
        "SELECT chunk_id, agent_id FROM document_chunk_owners ORDER BY chunk_id"
    )
    assert owners == [(ca, "agentA")], (
        "only the single-owner file's chunk should be assigned; got %r" % (owners,)
    )


# ---------------------------------------------------------------------------
# ``migration_lock`` — the serialization above, extracted so feature schemas
# can reuse it instead of reimplementing it. The restart coordinator's
# ``restart_requests`` migration (#2774) is the first other caller.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_lock_rolls_the_whole_body_back(db):
    """One transaction spans the entire migration, so an interruption leaves
    nothing behind.

    Callers rely on this to make the schema its own marker: a column and the
    data backfill that column implies land together or not at all, which
    removes "column added, backfill never ran" from the state space rather
    than adding a ledger to detect it afterwards.
    """
    with pytest.raises(Exception):
        async with db.migration_lock("rollback_probe"):
            await db.execute(
                "INSERT INTO schema_backfills (name) VALUES (?)",
                ("half-applied",),
            )
            raise RuntimeError("interrupted mid-migration")

    assert await db._backfill_completed("half-applied") is False, (
        "a partial migration must not survive the failure that stopped it"
    )


@pytest.mark.asyncio
async def test_migration_lock_serializes_concurrent_holders(db):
    """Exactly one holder runs at a time, so a post-upgrade request burst does
    the migration once instead of stampeding it (``_init_schema`` runs on
    every ``from_pool()``, which frinz calls per request)."""
    order: list[str] = []

    async def _hold(tag: str) -> None:
        async with db.migration_lock("contended"):
            order.append(f"enter-{tag}")
            await asyncio.sleep(0)
            order.append(f"exit-{tag}")

    await asyncio.gather(_hold("a"), _hold("b"))

    assert order in (
        ["enter-a", "exit-a", "enter-b", "exit-b"],
        ["enter-b", "exit-b", "enter-a", "exit-a"],
    ), f"holders interleaved: {order}"


@pytest.mark.asyncio
async def test_migration_lock_takes_the_sqlite_writer_slot_up_front(db):
    """SQLite must BEGIN IMMEDIATE, not deferred.

    A deferred transaction that has already read fails outright when it later
    tries to upgrade to the writer slot — it does not wait out
    ``busy_timeout`` — so a second initializer racing the first raises
    "database is locked" mid-migration instead of waiting and then finding the
    work done.
    """
    from kestrel_sovereign.storage.db import SQLiteBackend

    real = SQLiteBackend.transaction
    seen: list = []

    def _record(self, *, immediate: bool = False):
        seen.append(immediate)
        return real(self, immediate=immediate)

    with patch.object(SQLiteBackend, "transaction", _record):
        async with db.migration_lock("writer_slot"):
            pass

    assert seen == [True], f"expected one IMMEDIATE transaction, got {seen}"


@pytest.mark.asyncio
async def test_postgres_column_probe_follows_the_resolved_relation():
    """The column probe must ask about the relation the search path resolves.

    Name-based probes are wrong in both directions, and both failures are
    silent — no exception, no log:

    - Unscoped, ``information_schema.columns`` unions columns from EVERY
      schema on the search path, so a same-named table elsewhere reports
      columns this one lacks and the migration is skipped as already-applied.
    - Scoped to ``current_schema()``, it names only the FIRST schema on the
      path, so a table that resolves from a later one reports every column
      missing and the migration never converges. With callers failing closed
      on an incomplete migration, that disables a feature on a database that
      is perfectly fine.

    ``to_regclass`` asks the question ``ALTER TABLE`` answers. Same pattern as
    ``PostgresBackend.table_exists`` and the conversation-store probe.
    """
    seen: list = []

    class _FakePostgresBackend:
        backend_type = "postgres"

        async def fetch_one(self, sql, params=()):
            seen.append((sql, params))
            return (1,)

    db = AsyncDatabase(_FakePostgresBackend())

    assert await db._column_exists("restart_requests", "wake_dispatched_at")

    sql, params = seen[-1]
    assert "to_regclass" in sql, sql
    assert "current_schema" not in sql, (
        "current_schema() names the first schema on the path, not the one "
        "holding the table"
    )
    assert "information_schema" not in sql, (
        "an unscoped information_schema query unions every schema on the path"
    )
    assert params == ("restart_requests", "wake_dispatched_at")


# ---------------------------------------------------------------------------
# ``migrate_columns_once`` — the column-migration mechanism, owned by the
# platform so callers declare only what to migrate (#2791). Its correctness
# properties were established for restart_requests in #2774; these pin the
# generic contract now that any schema can reach for it.
# ---------------------------------------------------------------------------


async def _legacy_table(db, *, with_flag: bool = False) -> None:
    flag = ", flag INTEGER DEFAULT 0" if with_flag else ""
    await db.execute(
        f"CREATE TABLE widgets (id TEXT PRIMARY KEY, status TEXT{flag})"
    )


@pytest.mark.asyncio
async def test_migrate_columns_once_backfills_only_columns_it_adds(db):
    """The schema is the marker — the whole basis for skipping a backfill.

    A backfill rewrites live rows the running system owns. Running one against
    a column that was already present is silent damage, so the gate must be
    "am I the one adding this column", not a marker that can disagree with the
    schema.
    """
    await _legacy_table(db, with_flag=True)
    await db.execute("INSERT INTO widgets (id, status, flag) VALUES ('a','done',0)")

    await db.migrate_columns_once(
        "widgets",
        (("flag", "INTEGER DEFAULT 0"), ("note", "TEXT DEFAULT ''")),
        {
            "flag": ("UPDATE widgets SET flag = 1 WHERE status = 'done'", ()),
            "note": ("UPDATE widgets SET note = ? WHERE status = 'done'", ("seen",)),
        },
    )

    row = await db.fetchone("SELECT flag, note FROM widgets WHERE id = 'a'")
    assert row[0] == 0, (
        "flag already existed, so its backfill must not have run — the live "
        "0 is a value the running system owns"
    )
    assert row[1] == "seen", "note was added by this call, so its backfill ran"


@pytest.mark.asyncio
async def test_migrate_columns_once_needs_no_backfills(db):
    """Backfills are optional — a purely additive schema change passes none.

    ``a2a_outbound_tasks`` is the real caller of this shape.
    """
    await _legacy_table(db)
    await db.execute("INSERT INTO widgets (id, status) VALUES ('a','done')")

    await db.migrate_columns_once(
        "widgets", (("note", "TEXT DEFAULT ''"),),
    )

    assert await db._column_exists("widgets", "note")
    assert (await db.fetchone("SELECT note FROM widgets WHERE id='a'"))[0] == ""


@pytest.mark.asyncio
async def test_migrate_columns_once_honours_declared_column_order(db):
    """Declared order is load-bearing: a backfill may read what an earlier
    entry's backfill wrote. ``restart_requests`` depends on exactly this — its
    dispatch sentinel is keyed on the ``wake_delivered`` the entry above it
    backfills."""
    await _legacy_table(db)
    await db.execute("INSERT INTO widgets (id, status) VALUES ('a','done')")

    await db.migrate_columns_once(
        "widgets",
        (("first", "INTEGER DEFAULT 0"), ("second", "INTEGER DEFAULT 0")),
        {
            "first": ("UPDATE widgets SET first = 1 WHERE status = 'done'", ()),
            # Reads what the entry above just wrote.
            "second": ("UPDATE widgets SET second = 1 WHERE first = 1", ()),
        },
    )

    assert (await db.fetchone("SELECT first, second FROM widgets"))[:2] == (1, 1)


@pytest.mark.asyncio
async def test_migrate_columns_once_rolls_back_a_failed_backfill(db):
    """Column and backfill land together or not at all, for any table."""
    await _legacy_table(db)

    with pytest.raises(Exception):
        await db.migrate_columns_once(
            "widgets",
            (("note", "TEXT DEFAULT ''"),),
            {"note": ("UPDATE widgets SET no_such_column = 1", ())},
        )

    assert not await db._column_exists("widgets", "note"), (
        "committing the column while its backfill failed strands the backfill "
        "forever — every later call sees the column and skips"
    )


@pytest.mark.asyncio
async def test_migrate_columns_once_materializes_a_one_shot_columns_iterable(db):
    """A generator must not silently no-op AND then report success.

    ``columns`` is probed three times — before the lock, again under it, and
    once more to verify. Un-materialized, a one-shot iterable drains on the
    first probe: no ALTER runs, and the verification then sees nothing missing
    and returns cleanly. The check that exists to guarantee "the table is
    ready" is defeated by the same exhausted iterator that skipped the work.

    Unreachable while this was module-private and always read a tuple; making
    it a public platform method is what opened it.
    """
    await _legacy_table(db)

    await db.migrate_columns_once(
        "widgets",
        ((name, "TEXT DEFAULT ''") for name in ("note",)),
    )

    assert await db._column_exists("widgets", "note"), (
        "the column must be added, not skipped by a drained iterator"
    )


@pytest.mark.asyncio
async def test_migrate_columns_once_verifies_the_schema_before_reporting_ready(db):
    """The post-migration check is what makes a clean return mean something.

    Modelled on an ALTER that neither raises nor adds the column — the shape a
    mis-scoped Postgres catalog lookup produces, where the migration is skipped
    as already-applied and any verification asking the same wrong question
    agrees. Callers project their full column list, so one silently-skipped
    ALTER breaks every later read for the rest of the boot.
    """
    await _legacy_table(db)
    real = AsyncDatabase._migrate_add_column

    async def _skip_one(self, table, column, col_def):
        if column == "note":
            return
        await real(self, table, column, col_def)

    with patch.object(AsyncDatabase, "_migrate_add_column", _skip_one):
        with pytest.raises(Exception, match="note"):
            await db.migrate_columns_once(
                "widgets",
                (("flag", "INTEGER DEFAULT 0"), ("note", "TEXT DEFAULT ''")),
            )

    assert not await db._column_exists("widgets", "flag"), (
        "a migration that cannot complete must leave nothing behind"
    )


@pytest.mark.asyncio
async def test_migrate_columns_once_skips_the_writer_slot_when_nothing_is_missing(db):
    """Callers run this on every init, so the steady state must stay cheap.

    Entering takes SQLite's single writer slot (or a Postgres advisory lock) —
    the same every-boot cost #2649 gated its ownership backfills behind.
    """
    await _legacy_table(db, with_flag=True)

    with patch.object(
        AsyncDatabase, "migration_lock", side_effect=AssertionError(
            "a table with nothing missing must not take the writer slot"
        ),
    ):
        await db.migrate_columns_once("widgets", (("flag", "INTEGER DEFAULT 0"),))


@pytest.mark.asyncio
async def test_migrate_columns_once_rechecks_the_schema_under_the_lock(db, monkeypatch):
    """Obligation 1 of ``migration_lock``: the pre-lock probe is stale.

    A rival initializer can finish the whole migration while this caller waits
    for the lock, so the list that decided to enter no longer describes the
    database. Acting on it re-runs a backfill against rows the rival already
    backfilled — the double-application that the schema-as-marker gate exists
    to prevent.

    SQLite cannot surface this by concurrency alone (both callers serialize on
    one write lock and the columns get re-probed anyway), so the rival is
    injected into the window directly.
    """
    await _legacy_table(db)
    await db.execute("INSERT INTO widgets (id, status) VALUES ('a', 'done')")

    real_lock = AsyncDatabase.migration_lock

    @asynccontextmanager
    async def _rival_wins_the_race(self, name):
        # Completes the migration while this caller is "waiting" for the lock.
        await self.execute("ALTER TABLE widgets ADD COLUMN hits INTEGER DEFAULT 0")
        await self.execute("UPDATE widgets SET hits = hits + 1")
        async with real_lock(self, name):
            yield

    monkeypatch.setattr(AsyncDatabase, "migration_lock", _rival_wins_the_race)

    await db.migrate_columns_once(
        "widgets",
        (("hits", "INTEGER DEFAULT 0"),),
        {"hits": ("UPDATE widgets SET hits = hits + 1", ())},
    )

    assert (await db.fetchone("SELECT hits FROM widgets"))[0] == 1, (
        "the rival already backfilled; re-checking under the lock is what "
        "stops this call applying it a second time"
    )
