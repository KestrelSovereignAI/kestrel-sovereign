"""``route_state`` must be enforced by the schema, not merely intended (#2804).

A database created fresh got ``CHECK (route_state IN (...))`` from the
``CREATE TABLE``. One that gained the column by ``ALTER`` did not, and SQLite
has no ``ADD CONSTRAINT`` to retrofit it — so fresh and upgraded databases
diverged permanently and nothing detected it.

``route_state`` is routing *authorization* state: ``get_outbound_task`` fails
closed on ``ambiguous`` and treats the other values as usable. On an
unconstrained database an arbitrary string lands in that column and the
fail-closed value is only one of several the code branches on. The constraint
is what makes "holds one of three known values" true rather than assumed —
the same shape as ``wake_delivered`` in #2774.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.a2a.outbound_store import (
    ROUTE_STATES,
    ensure_a2a_outbound_tasks_table,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


# The table as it existed BEFORE route_state — what a real upgraded database
# still has on disk. Deliberately spelled out rather than derived from the
# canonical DDL: deriving it would track future edits and stop reproducing the
# historical shape this migration exists for.
LEGACY_DDL = """
    CREATE TABLE a2a_outbound_tasks (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        recipient TEXT NOT NULL,
        verb TEXT NOT NULL,
        session_id TEXT NOT NULL,
        skill_id TEXT,
        dispatch_tool TEXT NOT NULL,
        message_summary TEXT,
        created_at TEXT NOT NULL,
        terminal_state TEXT,
        terminal_at TEXT,
        error TEXT
    )
"""

_INSERT = (
    "INSERT INTO a2a_outbound_tasks "
    "(id, agent_id, task_id, recipient, verb, session_id, dispatch_tool, "
    "created_at{extra_cols}) "
    "VALUES (?, ?, ?, 'rcp', 'ask', 'sess', 'tool', 'now'{extra_vals})"
)


async def _legacy_db() -> AsyncDatabase:
    """A database whose a2a table predates ``route_state`` entirely."""
    db = await AsyncDatabase.sqlite(":memory:")
    await db.execute("DROP TABLE IF EXISTS a2a_outbound_tasks")
    await db.execute(LEGACY_DDL)
    return db


async def _table_sql(db: AsyncDatabase) -> str:
    row = await db.fetchone(
        "SELECT sql FROM sqlite_master WHERE type='table' "
        "AND name='a2a_outbound_tasks'"
    )
    return row[0] if row else ""


@pytest.mark.asyncio
async def test_upgraded_database_gains_the_constraint():
    """The whole point: an ALTERed table ends up enforcing the CHECK."""
    db = await _legacy_db()
    try:
        assert "CHECK" not in await _table_sql(db)

        await ensure_a2a_outbound_tasks_table(db)

        sql = await _table_sql(db)
        assert "CHECK" in sql
        assert "route_state IN" in " ".join(sql.split())

        with pytest.raises(Exception):
            await db.execute(
                _INSERT.format(
                    extra_cols=", route_state", extra_vals=", 'nonsense'"
                ),
                ("bad", "agent", "task-bad"),
            )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_rebuild_preserves_rows_and_indexes():
    """Data and indexes survive the SQLite table rebuild.

    ``DROP TABLE`` takes a table's indexes with it. A silently-missing index
    degrades queries without failing them, so its loss would not surface until
    something got slow in production.
    """
    db = await _legacy_db()
    try:
        await db.execute(
            "CREATE INDEX idx_probe_outbound_session "
            "ON a2a_outbound_tasks(session_id)"
        )
        for i in range(3):
            await db.execute(
                _INSERT.format(extra_cols="", extra_vals=""),
                (f"row-{i}", "agent", f"task-{i}"),
            )

        await ensure_a2a_outbound_tasks_table(db)

        assert await db.fetchval("SELECT COUNT(*) FROM a2a_outbound_tasks") == 3
        # Legacy rows take the column default rather than NULL, which the
        # NOT NULL in the canonical shape would reject.
        assert await db.fetchval(
            "SELECT COUNT(*) FROM a2a_outbound_tasks WHERE route_state='routable'"
        ) == 3
        surviving = await db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='a2a_outbound_tasks' AND sql IS NOT NULL"
        )
        assert "idx_probe_outbound_session" in {r[0] for r in surviving}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_rows_violating_the_vocabulary_are_quarantined_not_dropped():
    """A value outside the vocabulary fails CLOSED and the row is kept.

    Both backends refuse to add a constraint rows already violate, so these
    rows have to be dealt with. Deleting them would destroy audit history;
    guessing ``routable`` would grant routing authorization to a state nobody
    recognises. ``ambiguous`` is the value ``get_outbound_task`` fails closed
    on, so it is the only safe landing place.
    """
    db = await _legacy_db()
    try:
        await db.execute(
            "ALTER TABLE a2a_outbound_tasks ADD COLUMN route_state TEXT "
            "NOT NULL DEFAULT 'routable'"
        )
        await db.execute(
            _INSERT.format(extra_cols=", route_state", extra_vals=", 'nonsense'"),
            ("weird", "agent", "task-weird"),
        )
        await db.execute(
            _INSERT.format(extra_cols=", route_state", extra_vals=", 'routable'"),
            ("fine", "agent", "task-fine"),
        )

        await ensure_a2a_outbound_tasks_table(db)

        assert await db.fetchval("SELECT COUNT(*) FROM a2a_outbound_tasks") == 2
        assert await db.fetchval(
            "SELECT route_state FROM a2a_outbound_tasks WHERE id='weird'"
        ) == "ambiguous"
        assert await db.fetchval(
            "SELECT route_state FROM a2a_outbound_tasks WHERE id='fine'"
        ) == "routable"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_second_call_does_not_rebuild_again():
    """Idempotent: init runs on every boot and must rebuild at most once.

    Asserting on the table's SQL cannot show this — a redundant rebuild
    reproduces byte-identical DDL and tidies its staging table away, so the
    obvious version of this test passes even with both fast paths removed
    (verified by mutation). ``rootpage`` is the signal a rebuild cannot fake:
    it names the b-tree root, and a rebuilt table is a different b-tree.
    """
    db = await _legacy_db()
    try:
        await ensure_a2a_outbound_tasks_table(db)
        first_sql = await _table_sql(db)
        first_root = await db.fetchval(
            "SELECT rootpage FROM sqlite_master WHERE type='table' "
            "AND name='a2a_outbound_tasks'"
        )

        await ensure_a2a_outbound_tasks_table(db)

        assert await _table_sql(db) == first_sql
        assert not await db.table_exists("a2a_outbound_tasks__rebuild")
        assert await db.fetchval(
            "SELECT rootpage FROM sqlite_master WHERE type='table' "
            "AND name='a2a_outbound_tasks'"
        ) == first_root, "the table was rebuilt a second time"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fresh_database_needs_no_rebuild():
    """A table created from the canonical DDL already satisfies the check."""
    db = await AsyncDatabase.sqlite(":memory:")
    try:
        await ensure_a2a_outbound_tasks_table(db)
        assert "CHECK" in await _table_sql(db)
        for state in ROUTE_STATES:
            await db.execute(
                _INSERT.format(extra_cols=", route_state", extra_vals=", ?"),
                (f"ok-{state}", "agent", f"task-{state}", state),
            )
        assert await db.fetchval(
            "SELECT COUNT(*) FROM a2a_outbound_tasks"
        ) == len(ROUTE_STATES)
    finally:
        await db.close()
