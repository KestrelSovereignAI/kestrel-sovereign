"""Commit-ordered sequence numbers for append-only receipt feeds (#3159 R6).

A keyset feed is lossless only if a row committed AFTER a page was served can
never sort BEHIND the cursor that page issued. A timestamp cannot promise
that: two transactions read the clock, the one with the EARLIER reading can
commit LATER, and a consumer that already paged past the later reading skips
it forever. A clock can also step backwards.

``feed_seq`` is the answer: a dense integer allocated as ``MAX + 1`` while the
inserting transaction holds the one lock every writer of that table takes
until commit — SQLite's ``BEGIN IMMEDIATE`` writer slot, or a PostgreSQL
transaction-scoped advisory lock. Allocation and commit are then one
serialized step, so every later commit carries a larger number than every
earlier one. ``occurred_at`` stays exactly what the database clock said; it is
the displayed time, not the paging key.

**The database allocates it, not the writer.** An insert trigger numbers every
row that arrives without a ``feed_seq`` — which is every row, because no
writer names the column. That is deliberate: during a rolling upgrade an
older binary keeps appending receipts after a newer one has added the column.
Its inserts cannot know the column exists, and a row it wrote with
``feed_seq = NULL`` would be excluded from every feed page for good — the
schema backfill only runs at initialization, and nothing guarantees another
one. A trigger is the one allocator both binaries pass through, so there is
exactly one numbering rule and no writer can bypass it. On PostgreSQL the
trigger itself takes the table's advisory lock, so an older writer that never
heard of that lock is serialized exactly like a current one.
"""

from __future__ import annotations

import re
from typing import Any

FEED_SEQUENCE_COLUMN = "feed_seq"

_TABLE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*")
# Embedded as a SQL string literal in the PostgreSQL trigger function, so it
# must be a plain key with no quote or placeholder character.
_LOCK_KEY = re.compile(r"[a-z0-9_][a-z0-9_:.\-]*")
# A trigger changes meaning only by changing its definition; rotate the suffix
# with the body so an upgraded binary never mistakes an old one for its own.
_TRIGGER_VERSION = "v1"


def _checked_table(table: str) -> str:
    if not isinstance(table, str) or not _TABLE_IDENTIFIER.fullmatch(table):
        raise ValueError("feed table must be a plain SQL identifier")
    return table


def _checked_lock_key(lock_key: str) -> str:
    if not isinstance(lock_key, str) or not _LOCK_KEY.fullmatch(lock_key):
        raise ValueError("feed lock key must be a plain advisory-lock key")
    return lock_key


def feed_sequence_index_name(table: str) -> str:
    return f"idx_{_checked_table(table)}_feed_seq"


def feed_sequence_trigger_name(table: str) -> str:
    return f"trg_{_checked_table(table)}_feed_seq_{_TRIGGER_VERSION}"


def _feed_sequence_function_name(table: str) -> str:
    return f"{_checked_table(table)}_assign_feed_seq_{_TRIGGER_VERSION}"


async def ensure_feed_sequence(db: Any, *, table: str, lock_key: str) -> None:
    """Add, backfill, uniquely index, and database-allocate ``table.feed_seq``.

    Idempotent. Rows without a sequence — every row written before this column
    existed — are numbered AFTER the current maximum in
    ``(occurred_at, receipt_id)`` order. Appending them rather than
    interleaving them is what keeps the numbering append-only: a consumer
    holding a cursor still finds a late-numbered row, because it sorts after
    everything that consumer has already seen.

    ``lock_key`` is the PostgreSQL advisory key every writer of ``table``
    holds until commit; the trigger takes it before reading the maximum. The
    caller must hold that same serialization and run this inside one
    transaction, so the column, its backfill, its index, and its allocator
    land together.
    """

    table = _checked_table(table)
    lock_key = _checked_lock_key(lock_key)
    if not await db.column_exists(table, FEED_SEQUENCE_COLUMN):
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN {FEED_SEQUENCE_COLUMN} INTEGER"
        )
    unnumbered = await db.fetchall(
        f"SELECT receipt_id FROM {table} "
        f"WHERE {FEED_SEQUENCE_COLUMN} IS NULL "
        "ORDER BY occurred_at, receipt_id"
    )
    if unnumbered:
        # Numbered in Python, one row at a time, rather than by one UPDATE
        # with a correlated subquery: SQLite re-evaluates such a subquery
        # against rows the same statement already changed, PostgreSQL against
        # its statement snapshot, so the two engines would number differently.
        sequence = await _next_feed_sequence(db, table=table)
        for row in unnumbered:
            await db.execute(
                f"UPDATE {table} SET {FEED_SEQUENCE_COLUMN} = ? "
                f"WHERE receipt_id = ? AND {FEED_SEQUENCE_COLUMN} IS NULL",
                (sequence, row[0]),
            )
            sequence += 1
    await db.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS {feed_sequence_index_name(table)} "
        f"ON {table}({FEED_SEQUENCE_COLUMN})"
    )
    if getattr(db, "backend_type", "") == "postgres":
        await _ensure_postgres_allocator(db, table=table, lock_key=lock_key)
    else:
        await _ensure_sqlite_allocator(db, table=table)


async def _ensure_sqlite_allocator(db: Any, *, table: str) -> None:
    # SQLite cannot assign ``NEW`` in a BEFORE trigger, so the row is numbered
    # immediately after it lands, inside the same statement. Its writers are
    # already serialized by the database's single writer slot, so ``MAX + 1``
    # read here is the largest number any committed or in-flight row holds.
    await db.execute(
        f"CREATE TRIGGER IF NOT EXISTS {feed_sequence_trigger_name(table)} "
        f"AFTER INSERT ON {table} FOR EACH ROW "
        f"WHEN NEW.{FEED_SEQUENCE_COLUMN} IS NULL "
        "BEGIN "
        f"UPDATE {table} SET {FEED_SEQUENCE_COLUMN} = ("
        f"SELECT COALESCE(MAX({FEED_SEQUENCE_COLUMN}), 0) + 1 FROM {table}"
        ") WHERE receipt_id = NEW.receipt_id; "
        "END"
    )


async def _ensure_postgres_allocator(
    db: Any, *, table: str, lock_key: str
) -> None:
    function_name = _feed_sequence_function_name(table)
    trigger_name = feed_sequence_trigger_name(table)
    # The lock is taken INSIDE the trigger so a writer that predates it is
    # serialized too; for a current writer that already holds it, the
    # re-acquisition is reentrant. PL/pgSQL gives the following SELECT a fresh
    # READ COMMITTED snapshot, taken after the lock was granted, so it sees
    # every number the previous holder committed.
    await db.execute(
        f"CREATE OR REPLACE FUNCTION {function_name}() "
        "RETURNS trigger AS $kestrel_feed_seq$ "
        "BEGIN "
        "PERFORM pg_advisory_xact_lock("
        f"hashtextextended('{lock_key}', 0)); "
        f"SELECT COALESCE(MAX({FEED_SEQUENCE_COLUMN}), 0) + 1 "
        f"INTO NEW.{FEED_SEQUENCE_COLUMN} FROM {table}; "
        "RETURN NEW; "
        "END; "
        "$kestrel_feed_seq$ LANGUAGE plpgsql"
    )
    existing = await db.fetchone(
        "SELECT 1 FROM pg_trigger "
        "WHERE tgrelid = to_regclass(?) AND tgname = ? AND NOT tgisinternal",
        (table, trigger_name),
    )
    if existing is None:
        await db.execute(
            f"CREATE TRIGGER {trigger_name} "
            f"BEFORE INSERT ON {table} FOR EACH ROW "
            f"WHEN (NEW.{FEED_SEQUENCE_COLUMN} IS NULL) "
            f"EXECUTE FUNCTION {function_name}()"
        )


async def _next_feed_sequence(db: Any, *, table: str) -> int:
    row = await db.fetchone(
        f"SELECT COALESCE(MAX({FEED_SEQUENCE_COLUMN}), 0) FROM {table}"
    )
    if row is None or len(row) != 1:
        raise RuntimeError(f"{table} feed sequence could not be read")
    current = row[0]
    if isinstance(current, bool) or not isinstance(current, int) or current < 0:
        raise RuntimeError(f"{table} holds an invalid feed sequence")
    return current + 1


__all__ = [
    "FEED_SEQUENCE_COLUMN",
    "ensure_feed_sequence",
    "feed_sequence_index_name",
    "feed_sequence_trigger_name",
]
