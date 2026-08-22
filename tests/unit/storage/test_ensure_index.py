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
        elif sql.startswith("DROP INDEX"):
            # DROP INDEX IF EXISTS "<name>" — the retirement of a superseded
            # family member. Modelled so a case can tell a retirement from a
            # no-op; a double that swallowed it would let the old index appear
            # to have gone while it was still there.
            self.indexes.pop(sql.split()[4].strip('"'), None)
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

    async def fetch_all(self, sql: str, params: tuple = ()):
        """The family probe: every index on this table whose name matches.

        Modelled on the same single-valued mapping as ``fetch_one`` rather than
        returning a canned list — a double that reports a family the ``indexes``
        dict does not contain would let the retirement DROP look correct while
        naming an index that never existed.
        """
        await asyncio.sleep(0)
        assert "sqlite_master" in sql and "type = 'index'" in sql, sql
        table, like = params
        prefix = like.rstrip("%")
        return [
            (name,)
            for name, owner in sorted(self.indexes.items())
            if owner == table and name.startswith(prefix)
        ]

    @property
    def creates(self) -> list[str]:
        return [e for e in self.events if e.startswith("CREATE INDEX")]

    @property
    def drops(self) -> list[str]:
        return [e for e in self.events if e.startswith("DROP INDEX")]


def _named(name: str, columns: str, where: str = "", backend: str = "sqlite") -> str:
    """The name ``ensure_index`` will actually use for this definition.

    Computed the same way the method computes it rather than pasted, so a case
    cannot drift from the mechanism it is about (#3009 step 5).
    """
    from kestrel_sovereign.storage.async_database import _definition_fingerprint

    return f"{name}_{_definition_fingerprint(backend, columns, where)}"


def _family(name: str, names) -> bool:
    """Whether any index in ``names`` belongs to the ``name`` family."""
    return any(str(n).startswith(f"{name}_") for n in names)


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
        f"CREATE INDEX IF NOT EXISTS {_named('idx_conversation_agent_session', 'agent_id, session_id')} "
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
    backend.indexes[_named("idx_probe", "agent_id")] = "conversation_history"
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
    backend.indexes[_named("idx_probe", "agent_id")] = "some_other_table"
    db = AsyncDatabase(backend)

    with pytest.raises(RuntimeError, match="some_other_table"):
        await db.ensure_index("idx_probe", "conversation_history", "agent_id")

    # It did try — the raise is about the outcome, not a refusal to attempt.
    assert backend.creates == [
        f"CREATE INDEX IF NOT EXISTS {_named('idx_probe', 'agent_id')} "
        "ON conversation_history(agent_id)"
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
        await db.execute(
            f"CREATE INDEX {_named('idx_collide', 'agent_id')} ON decoy(agent_id)"
        )

        with pytest.raises(Exception) as raised:
            await db.ensure_index("idx_collide", "target", "agent_id")
        assert "decoy" in str(raised.value), str(raised.value)

        # The engine really did nothing: the name still belongs to the decoy.
        assert await db.fetchall(
            "SELECT name, tbl_name FROM sqlite_master WHERE name = ?",
            (_named("idx_collide", "agent_id"),),
        ) == [(_named("idx_collide", "agent_id"), "decoy")]
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
        assert _family("idx_conversation_agent_session", names)
        assert _family("idx_conversation_deleted_at", names)
        assert _family("idx_conversation_archived_at", names)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_changed_definition_retires_the_index_it_replaces():
    """The reason the name carries a fingerprint at all (#3009 step 5).

    ``ensure_index`` used to answer "is there an index called X on this table",
    and nothing answered "is it the index we meant". Several of these
    definitions are COMPUTED — ``canonical_order_index_columns()`` renders the
    ordering key per backend — so changing the ordering left the old index in
    place, matching nothing the new ``ORDER BY`` asked for. No error, no
    missing index, just the O(history) scan back, visible only in a query plan.

    This is the #2998 shape a second time: an object identified by NAME when
    what matters is its SHAPE.
    """
    backend = _RecordingBackend()
    db = AsyncDatabase(backend)

    await db.ensure_index("idx_order", "conversation_history", "julianday(created_at), id")
    first = _named("idx_order", "julianday(created_at), id")
    assert backend.indexes == {first: "conversation_history"}

    await db.ensure_index("idx_order", "conversation_history", "created_at, id")
    second = _named("idx_order", "created_at, id")

    assert second in backend.indexes, "the new definition was never built"
    assert first not in backend.indexes, (
        "the superseded index survived; the ORDER BY it was built for no "
        "longer exists, so it matches nothing and nothing says so"
    )
    # Built BEFORE the old one is dropped. A window with two indexes costs a
    # little redundant work; a window with none is an unindexed scan taken
    # during a boot.
    assert backend.events.index(f"DROP INDEX IF EXISTS \"{first}\"") > next(
        i for i, e in enumerate(backend.events) if e.startswith("CREATE INDEX") and second in e
    ), "the old index was dropped before its replacement existed"


@pytest.mark.asyncio
async def test_an_unchanged_definition_is_not_rebuilt_on_the_next_boot():
    """The fingerprint must be stable, or every boot rebuilds every index."""
    backend = _RecordingBackend()
    db = AsyncDatabase(backend)

    await db.ensure_index("idx_stable", "conversation_history", "agent_id, id")
    await db.ensure_index("idx_stable", "conversation_history", "agent_id, id")

    assert len(backend.creates) == 1, backend.creates
    assert backend.drops == [], backend.drops
