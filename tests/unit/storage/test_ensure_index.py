"""``ensure_index`` must serialize concurrent initializers, not just repeats.

``CREATE INDEX IF NOT EXISTS`` is idempotent in SEQUENCE and unsafe in
PARALLEL: Postgres evaluates the existence test before taking the lock that
would exclude a peer, so two builders that pass it together both proceed and
one loses on ``pg_class``' unique index. ``_init_schema`` runs on every
``from_pool()`` — frinz calls it per request — so a post-upgrade request burst
is exactly the parallel case, and the loser does not skip the index: its whole
initialization raises and the request fails.

These drive the real ``AsyncDatabase.ensure_index`` against a backend double
whose ``transaction(immediate=True)`` is a single writer slot, which is what
``BEGIN IMMEDIATE`` and ``pg_advisory_xact_lock`` both are. The double yields
control inside every statement, so the interleaving the race needs really does
happen — the counting test below fails with 4 CREATEs if either the lock or the
re-probe inside it is removed.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase


class _RecordingBackend:
    """A backend that reports index existence truthfully and records order.

    Faithful in the two dimensions that matter:

    * A statement is a suspension point (``await asyncio.sleep(0)``), so a
      caller that probes and then creates without holding the writer slot
      really can be overtaken between the two.
    * An index NAME is unique database-wide, so ``CREATE INDEX IF NOT EXISTS``
      no-ops when the name is taken even by an index on another table. An
      earlier version of this double let the same name exist on two tables at
      once — an impossible state, and one that made a silent no-op look like a
      successful build. Modelling the impossible is how a double starts
      certifying behaviour no engine has.
    """

    backend_type = "sqlite"

    def __init__(self) -> None:
        self.events: list[str] = []
        # ``index_name -> table`` — the probe is table-aware, because an index
        # name alone is not unique across schemas (#2958 Finding 3), while the
        # mapping is single-valued because the name IS unique per database.
        self.indexes: dict[str, str] = {}
        self._writer = asyncio.Lock()

    @asynccontextmanager
    async def transaction(self, immediate: bool = False):
        assert immediate, "migration_lock must take the writer slot up front"
        async with self._writer:
            self.events.append("lock:enter")
            try:
                yield
            finally:
                self.events.append("lock:exit")

    async def execute(self, sql: str, params: tuple = ()) -> int:
        await asyncio.sleep(0)
        self.events.append(sql)
        if sql.startswith("CREATE INDEX"):
            # CREATE INDEX IF NOT EXISTS <name> ON <table>(<cols>)
            name = sql.split()[5]
            table = sql.split()[7].split("(")[0]
            self.indexes.setdefault(name, table)  # IF NOT EXISTS, by name
        return 0

    async def fetch_one(self, sql: str, params: tuple = ()):
        await asyncio.sleep(0)
        if "type = 'index'" in sql:
            assert "sqlite_master" in sql, sql
            assert "tbl_name = ?" in sql, f"probe must be table-aware: {sql}"
            return (1 if self.indexes.get(params[0]) == params[1] else 0,)
        # The collision diagnostic: who holds this name?
        assert "sqlite_master" in sql and "tbl_name" in sql, sql
        return (self.indexes.get(params[0]),)

    @property
    def creates(self) -> list[str]:
        return [e for e in self.events if e.startswith("CREATE INDEX")]


@pytest.mark.asyncio
async def test_concurrent_initializers_build_the_index_once():
    """Four boots racing the same fresh database, one CREATE between them."""
    backend = _RecordingBackend()
    db = AsyncDatabase(backend)

    await asyncio.gather(
        *(
            db.ensure_index("idx_conversation_agent_session",
                            "conversation_history", "agent_id, session_id")
            for _ in range(4)
        )
    )

    assert backend.creates == [
        "CREATE INDEX IF NOT EXISTS idx_conversation_agent_session "
        "ON conversation_history(agent_id, session_id)"
    ]


@pytest.mark.asyncio
async def test_the_create_runs_inside_the_lock():
    """Not merely serialized — the statement itself is under the lock.

    Probing under the lock and then creating after releasing it would pass the
    count assertion above by luck of scheduling while still leaving the
    unprotected window this exists to close.
    """
    backend = _RecordingBackend()
    db = AsyncDatabase(backend)

    await db.ensure_index("idx_probe", "conversation_history", "agent_id")

    create = next(i for i, e in enumerate(backend.events) if e.startswith("CREATE"))
    assert backend.events.index("lock:enter") < create
    assert create < backend.events.index("lock:exit")


@pytest.mark.asyncio
async def test_an_existing_index_is_not_relocked():
    """The common path — every boot after the first — takes no write lock."""
    backend = _RecordingBackend()
    backend.indexes["idx_probe"] = "conversation_history"
    db = AsyncDatabase(backend)

    await db.ensure_index("idx_probe", "conversation_history", "agent_id")

    assert backend.events == []


@pytest.mark.asyncio
async def test_a_name_taken_by_another_table_is_reported_not_shrugged_off():
    """The probe and ``IF NOT EXISTS`` ask different questions (#2958).

    ``_index_exists`` asks about the (name, table) PAIR — it must, since a name
    alone is not unique across schemas. ``CREATE INDEX IF NOT EXISTS`` asks
    only about the NAME. Given a decoy on another table the two disagree: the
    probe says "absent", the DDL says "present" and does nothing, and the
    caller is told the index was built. Only a query plan would ever show it.

    The real-engine counterpart is below; this one pins that the failure is
    raised from ``ensure_index`` itself and names the colliding table.
    """
    backend = _RecordingBackend()
    backend.indexes["idx_probe"] = "some_other_table"
    db = AsyncDatabase(backend)

    with pytest.raises(RuntimeError, match="some_other_table"):
        await db.ensure_index("idx_probe", "conversation_history", "agent_id")

    # It did try — the raise is about the outcome, not a refusal to attempt.
    assert backend.creates == [
        "CREATE INDEX IF NOT EXISTS idx_probe ON conversation_history(agent_id)"
    ]


@pytest.mark.asyncio
async def test_a_name_collision_really_no_ops_on_a_live_sqlite_database(tmp_path):
    """The engine's own behaviour, not the double's model of it.

    SQLite keeps index names in one database-wide namespace. This asserts both
    halves of the trap against a real file: the DDL silently does nothing, and
    ``ensure_index`` refuses to report success for it.
    """
    db = await AsyncDatabase.sqlite(str(tmp_path / "collision.db"))
    try:
        await db.execute("CREATE TABLE decoy (agent_id TEXT)")
        await db.execute("CREATE TABLE target (agent_id TEXT)")
        await db.execute("CREATE INDEX idx_collide ON decoy(agent_id)")

        with pytest.raises(Exception) as raised:
            await db.ensure_index("idx_collide", "target", "agent_id")
        assert "decoy" in str(raised.value), str(raised.value)

        # The engine really did nothing: the name still belongs to the decoy.
        assert await db.fetchall(
            "SELECT name, tbl_name FROM sqlite_master WHERE name = 'idx_collide'",
            (),
        ) == [("idx_collide", "decoy")]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_session_index_survives_a_second_boot_of_a_real_database(tmp_path):
    """The double above asserts the mechanism; this asserts the real DDL runs."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "boots.db"))
    await db.close()
    db = await AsyncDatabase.sqlite(str(tmp_path / "boots.db"))
    try:
        names = {
            row[0]
            for row in await db.fetchall(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='conversation_history'",
                (),
            )
        }
        assert "idx_conversation_agent_session" in names
        assert "idx_conversation_deleted_at" in names
        assert "idx_conversation_archived_at" in names
    finally:
        await db.close()
