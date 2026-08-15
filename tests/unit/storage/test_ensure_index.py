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

    Faithful in the one dimension that matters: a statement is a suspension
    point (``await asyncio.sleep(0)``), so a caller that probes and then creates
    without holding the writer slot really can be overtaken between the two.
    """

    backend_type = "sqlite"

    def __init__(self) -> None:
        self.events: list[str] = []
        self.indexes: set[str] = set()
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
            self.indexes.add(sql.split()[5])
        return 0

    async def fetch_one(self, sql: str, params: tuple = ()):
        await asyncio.sleep(0)
        assert "sqlite_master" in sql and "type = 'index'" in sql, sql
        return (1 if params[0] in self.indexes else 0,)

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
    backend.indexes.add("idx_probe")
    db = AsyncDatabase(backend)

    await db.ensure_index("idx_probe", "conversation_history", "agent_id")

    assert backend.events == []


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
