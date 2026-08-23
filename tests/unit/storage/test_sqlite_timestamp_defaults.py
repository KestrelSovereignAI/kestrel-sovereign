"""#3048: ``DEFAULT CURRENT_TIMESTAMP`` has to survive the trip to SQLite.

``CORE_SCHEMA`` is authored in SQLite dialect and passed through
``normalize_schema``, whose sqlite branch is written as a *PostgreSQL → SQLite*
converter. Among its removals were two clauses that should not have been
removed:

* ``DEFAULT NOW()`` — the PostgreSQL spelling of something SQLite HAS. Dropping
  it leaves no default at all; translating it keeps the meaning.
* ``DEFAULT CURRENT_TIMESTAMP`` — not PostgreSQL-specific in the first place.
  It is standard SQL and SQLite supports it natively.

The consequence was a silent divergence in a column readers date: an INSERT
that omitted one stored NULL on SQLite and a real timestamp on PostgreSQL.
Measured on a live agent database before the fix — 55 of 70 TIMESTAMP columns
carried no default, where the authored schema gives 28 of them one.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase, CORE_SCHEMA
from kestrel_sovereign.storage.db.placeholder import normalize_schema


def test_sqlite_keeps_a_default_sqlite_supports():
    normalized = normalize_schema(
        "CREATE TABLE t (a TIMESTAMP DEFAULT CURRENT_TIMESTAMP)", "sqlite"
    )
    assert "DEFAULT CURRENT_TIMESTAMP" in normalized


def test_the_postgres_spelling_is_translated_not_dropped():
    """``NOW()`` has no SQLite equivalent by that name; it has one by another."""
    normalized = normalize_schema(
        "CREATE TABLE t (a TIMESTAMP DEFAULT NOW())", "sqlite"
    )
    assert "DEFAULT CURRENT_TIMESTAMP" in normalized
    assert "NOW()" not in normalized


def test_postgres_is_left_alone():
    """The other direction is untouched: this was only ever a sqlite bug."""
    schema = "CREATE TABLE t (a TIMESTAMP DEFAULT NOW(), b TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    normalized = normalize_schema(schema, "postgres")
    assert "DEFAULT NOW()" in normalized
    assert "DEFAULT CURRENT_TIMESTAMP" in normalized


def _authored_defaults() -> set:
    """Every ``<table>.<column>`` the authored schema gives a stamp default."""
    found, table = set(), None
    for line in CORE_SCHEMA.splitlines():
        seen = re.search(r"CREATE TABLE (?:IF NOT EXISTS )?(\w+)", line, re.IGNORECASE)
        if seen:
            table = seen.group(1)
        column = re.match(
            r"\s*(\w+)\s+TIMESTAMP\s+DEFAULT\s+(?:CURRENT_TIMESTAMP|NOW\(\))",
            line, re.IGNORECASE,
        )
        if column and table:
            found.add(f"{table}.{column.group(1)}")
    return found


@pytest.mark.asyncio
async def test_every_authored_default_reaches_a_real_sqlite_database(tmp_path):
    """Asked of the DATABASE, not of the regex.

    The regex test above passes on a string. This one creates the schema and
    reads the defaults back out of ``pragma_table_info``, which is what a writer
    that omits the column actually meets.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "defaults.db"))
    try:
        authored = _authored_defaults()
        assert len(authored) >= 25, (
            f"the authored schema should carry ~28 stamp defaults, found "
            f"{len(authored)} — this test would prove nothing"
        )
        missing = []
        for qualified in sorted(authored):
            table, column = qualified.split(".")
            row = await db.fetchone(
                f"SELECT \"dflt_value\" FROM pragma_table_info('{table}') "
                f"WHERE name = '{column}'"
            )
            if row is None:
                continue  # table not created by this path; not this test's claim
            if not row[0] or "CURRENT_TIMESTAMP" not in str(row[0]).upper():
                missing.append(f"{qualified} -> {row[0]!r}")
        assert not missing, (
            "these columns reached SQLite with no CURRENT_TIMESTAMP default, so "
            "an INSERT omitting them stores NULL here and a timestamp on "
            f"PostgreSQL:\n  " + "\n  ".join(missing)
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_writer_stamps_the_column_without_help_from_the_schema(tmp_path):
    """The fix has to reach databases that already exist, and the schema cannot.

    SQLite has no ``ALTER TABLE ... SET DEFAULT``: every database created before
    #3048 keeps its defaultless columns for ever, and rebuilding 26 tables to
    add a default is a migration out of all proportion to the fault. So the
    writers that used to lean on the default now supply the value themselves,
    which repairs old and new databases alike and leaves the schema default as
    belt and braces for whatever is written next.

    This recreates the pre-fix shape — the table with its default stripped, as
    the old ``normalize_schema`` would have produced — and then drives the real
    store.
    """
    from kestrel_sovereign.storage.async_pending_a2a_question_store import (
        PendingA2AQuestionStore,
    )

    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy-shape.db"))
    try:
        created = await db.fetchone(
            "SELECT sql FROM sqlite_master WHERE name = 'pending_a2a_questions'"
        )
        assert created and "CURRENT_TIMESTAMP" in created[0], (
            "the fixture needs the fixed schema to strip, or it proves nothing"
        )
        # Exactly what the old rule produced: the clause removed entirely.
        await db.execute("DROP TABLE pending_a2a_questions")
        await db.execute(
            re.sub(
                r"\s+DEFAULT\s+CURRENT_TIMESTAMP", "", created[0],
                flags=re.IGNORECASE,
            )
        )
        assert "CURRENT_TIMESTAMP" not in (await db.fetchone(
            "SELECT sql FROM sqlite_master WHERE name = 'pending_a2a_questions'"
        ))[0]

        store = PendingA2AQuestionStore(db, "did:test:stamps")
        await store.insert(
            task_id="t-1",
            recipient="someone",
            original_question="are you there",
            origin_turn_id="turn-1",
            origin_session_id="sess-1",
            deadline=datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc),
        )

        stamped = await db.fetchval(
            "SELECT created_at FROM pending_a2a_questions WHERE task_id = 't-1'"
        )
        assert stamped is not None, (
            "the writer left created_at NULL on a database whose schema has no "
            "default — which is every database created before this fix"
        )
        # ...and the shape the NULL used to take on the way out: the row is read
        # back through `str(r[7])` into a `str` field, so NULL surfaced as the
        # literal "None" rather than as an absent value.
        assert str(stamped) != "None"
    finally:
        await db.close()
