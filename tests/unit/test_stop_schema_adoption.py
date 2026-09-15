"""A generation ledger created before ``request_generation`` is adopted (#3292).

``cb5154e2b`` added ``request_generation INTEGER NOT NULL`` to the
``CREATE TABLE IF NOT EXISTS`` for ``stop_active_invocations`` and nothing
else: a table that already existed never gained the column, and on such a
host every ``register()`` — the admission every cognition turn passes through
— fails at its INSERT. The prod host ran that way for four days while every
action-mode scheduled tool kept reporting success.

The adoption follows #3289: an empty legacy ledger is carried to the canonical
shape in place; one that still holds rows is refused with the reason named,
because a registration that never recorded its exact generation has no honest
value for the column.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from kestrel_sovereign.stop import (
    DistributedInvocationStore,
    StopLegacyRegistrationsError,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase

# The table exactly as the prod host had it on 2026-09-15 (``sqlite_master``):
# created by the c8c6282ba shape, ``public_turn_digest`` added later by ALTER.
# Spelled out rather than derived from the canonical DDL so it keeps
# reproducing the historical shape this adoption exists for.
LEGACY_ACTIVE_DDL = (
    "CREATE TABLE stop_active_invocations ("
    "generation_id TEXT NOT NULL PRIMARY KEY, "
    "agent_id TEXT NOT NULL, "
    "turn_digest TEXT NOT NULL, "
    "owner_id TEXT NOT NULL, "
    "stop_requested INTEGER NOT NULL DEFAULT 0, "
    "registered_at TEXT NOT NULL, "
    "heartbeat_at TEXT NOT NULL, "
    "public_turn_digest TEXT, "
    "CHECK (stop_requested IN (0, 1)))"
)
LEGACY_ACTIVE_INDEXES = (
    (
        "CREATE INDEX idx_stop_active_agent_turn "
        "ON stop_active_invocations(agent_id, turn_digest)"
    ),
    (
        "CREATE INDEX idx_stop_active_owner "
        "ON stop_active_invocations(owner_id, stop_requested)"
    ),
)
# 01208c6d9's first shape of the unresolved ledger: neither column.
LEGACY_UNRESOLVED_DDL = (
    "CREATE TABLE stop_unresolved_invocations ("
    "generation_id TEXT NOT NULL PRIMARY KEY, "
    "agent_id TEXT NOT NULL, "
    "turn_digest TEXT NOT NULL, "
    "owner_id TEXT NOT NULL, "
    "expired_at TEXT NOT NULL)"
)
_LEGACY_ACTIVE_ROW = (
    "INSERT INTO stop_active_invocations (generation_id, agent_id, "
    "turn_digest, owner_id, registered_at, heartbeat_at) "
    "VALUES (?, 'did:test:legacy', 'digest', 'owner-old', 'then', 'then')"
)
EXACT_GENERATION_CHECK = "CHECK (request_generation > 0)"


async def _columns(db: AsyncDatabase, table: str) -> dict[str, int]:
    """``{name: notnull}`` for ``table`` on SQLite."""
    rows = await db.fetchall(f"PRAGMA table_info('{table}')")
    return {str(r[1]): int(r[3]) for r in rows}


async def _table_sql(db: AsyncDatabase, table: str) -> str:
    return await db.fetchval(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )


async def _rootpage(db: AsyncDatabase, table: str) -> int:
    return await db.fetchval(
        "SELECT rootpage FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )


async def _index_names(db: AsyncDatabase, table: str) -> set[str]:
    rows = await db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table,),
    )
    return {str(r[0]) for r in rows}


async def _legacy_db(tmp_path, *, with_row: bool = False) -> AsyncDatabase:
    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy-host.db"))
    await db.execute(LEGACY_ACTIVE_DDL)
    for ddl in LEGACY_ACTIVE_INDEXES:
        await db.execute(ddl)
    if with_row:
        await db.execute(_LEGACY_ACTIVE_ROW, (f"generation-{uuid4().hex}",))
    return db


async def _register_once(store: DistributedInvocationStore) -> bool:
    suffix = uuid4().hex
    return await store.register(
        generation_id=f"generation-{suffix}",
        agent_id=f"did:test:agent-{suffix}",
        turn_id=f"turn-{suffix}",
        owner_id=f"owner-{suffix}",
        request_generation=1,
    )


@pytest.mark.asyncio
async def test_empty_legacy_active_ledger_is_adopted_and_registers(tmp_path):
    """The prod shape: no ``request_generation``, no rows → canonical, in place."""
    db = await _legacy_db(tmp_path)
    try:
        before = await _rootpage(db, "stop_active_invocations")
        assert "request_generation" not in await _columns(
            db, "stop_active_invocations"
        )

        store = DistributedInvocationStore(db)
        await store.ensure_schema()

        columns = await _columns(db, "stop_active_invocations")
        assert columns["request_generation"] == 1, "column present and NOT NULL"
        assert EXACT_GENERATION_CHECK in await _table_sql(
            db, "stop_active_invocations"
        )
        assert await _rootpage(db, "stop_active_invocations") != before, (
            "the legacy table was not rebuilt"
        )
        # DROP TABLE takes the indexes with it; the rebuild must replay the
        # legacy ones and ensure_schema must still add the newer partial one.
        names = await _index_names(db, "stop_active_invocations")
        assert {"idx_stop_active_agent_turn", "idx_stop_active_owner"} <= names
        # ``ensure_index`` suffixes its names with a digest of the definition.
        assert any(
            n.startswith("idx_stop_active_agent_public_turn") for n in names
        ), names
        assert not await db.table_exists("stop_active_invocations__rebuild")

        # The failure the host actually hit: INSERT naming the column.
        assert await _register_once(store) is True
        assert await db.fetchval(
            "SELECT request_generation FROM stop_active_invocations"
        ) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_adoption_runs_once(tmp_path):
    """A second boot must not rebuild again; ``rootpage`` cannot be faked."""
    db = await _legacy_db(tmp_path)
    try:
        store = DistributedInvocationStore(db)
        await store.ensure_schema()
        adopted_root = await _rootpage(db, "stop_active_invocations")
        adopted_sql = await _table_sql(db, "stop_active_invocations")

        await DistributedInvocationStore(db).ensure_schema()

        assert await _rootpage(db, "stop_active_invocations") == adopted_root
        assert await _table_sql(db, "stop_active_invocations") == adopted_sql
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_legacy_ledger_with_registrations_is_refused_until_empty(
    tmp_path,
):
    """Rows with no recorded generation are refused, named, and never invented.

    The refusal must hold on every boot (no accidental convergence on the
    second attempt) and must leave the ledger's rows and its lack of the
    CHECK untouched, so an operator sees the same state the error describes.
    Once the rows are gone the same boot path adopts the table.
    """
    db = await _legacy_db(tmp_path, with_row=True)
    try:
        store = DistributedInvocationStore(db)
        for _attempt in range(2):
            with pytest.raises(StopLegacyRegistrationsError) as excinfo:
                await store.ensure_schema()
            assert excinfo.value.table == "stop_active_invocations"
            assert excinfo.value.count == 1
            assert "stop_active_invocations" in str(excinfo.value)
            assert "1 registration" in str(excinfo.value)
            assert EXACT_GENERATION_CHECK not in await _table_sql(
                db, "stop_active_invocations"
            )
            assert await db.fetchval(
                "SELECT COUNT(*) FROM stop_active_invocations"
            ) == 1
            assert await db.fetchval(
                "SELECT COUNT(*) FROM stop_active_invocations "
                "WHERE request_generation IS NOT NULL"
            ) == 0, "a generation was invented for a legacy row"

        await db.execute(
            "DELETE FROM stop_active_invocations "
            "WHERE request_generation IS NULL"
        )
        await store.ensure_schema()

        assert (await _columns(db, "stop_active_invocations"))[
            "request_generation"
        ] == 1
        assert EXACT_GENERATION_CHECK in await _table_sql(
            db, "stop_active_invocations"
        )
        assert await _register_once(store) is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_first_shape_unresolved_ledger_is_adopted(tmp_path):
    """01208c6d9's unresolved ledger lacked both later columns."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy-unresolved.db"))
    try:
        await db.execute(LEGACY_UNRESOLVED_DDL)
        before = await _rootpage(db, "stop_unresolved_invocations")

        await DistributedInvocationStore(db).ensure_schema()

        columns = await _columns(db, "stop_unresolved_invocations")
        assert columns["request_generation"] == 1
        assert "public_turn_digest" in columns
        assert EXACT_GENERATION_CHECK in await _table_sql(
            db, "stop_unresolved_invocations"
        )
        assert await _rootpage(db, "stop_unresolved_invocations") != before
        assert any(
            n.startswith("idx_stop_unresolved_agent_public_turn")
            for n in await _index_names(db, "stop_unresolved_invocations")
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_canonical_ledgers_are_never_rebuilt(tmp_path):
    """A fresh host pays the probes and nothing else, boot after boot."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "fresh.db"))
    try:
        await DistributedInvocationStore(db).ensure_schema()
        roots = {
            table: await _rootpage(db, table)
            for table in (
                "stop_active_invocations",
                "stop_unresolved_invocations",
            )
        }

        await DistributedInvocationStore(db).ensure_schema()

        for table, root in roots.items():
            assert await _rootpage(db, table) == root, f"{table} was rebuilt"
            assert not await db.table_exists(f"{table}__rebuild")
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_adoption_has_sqlite_postgres_parity(db_backend):
    """The same boot converges both engines; PostgreSQL in place, not by copy."""
    db = AsyncDatabase(db_backend)
    for table in (
        "stop_unresolved_invocations",
        "stop_invocation_fences",
        "stop_active_invocations",
        "stop_receipt_outcomes",
        "stop_receipts",
        "stop_operation_claims",
    ):
        await db.execute(f"DROP TABLE IF EXISTS {table}")
    await db.execute(LEGACY_ACTIVE_DDL)
    await db.execute(_LEGACY_ACTIVE_ROW, (f"generation-{uuid4().hex}",))

    store = DistributedInvocationStore(db)
    with pytest.raises(StopLegacyRegistrationsError):
        await store.ensure_schema()
    assert await db.fetchval(
        "SELECT COUNT(*) FROM stop_active_invocations"
    ) == 1

    await db.execute("DELETE FROM stop_active_invocations")
    await store.ensure_schema()

    if db.backend_type == "postgres":
        assert await db.fetchval(
            "SELECT attnotnull FROM pg_attribute "
            "WHERE attrelid = to_regclass('stop_active_invocations') "
            "AND attname = 'request_generation'"
        ) is True
        assert await db.fetchval(
            "SELECT convalidated FROM pg_constraint "
            "WHERE conrelid = to_regclass('stop_active_invocations') "
            "AND conname = 'stop_active_invocations_request_generation_check'"
        ) is True
    else:
        assert (await _columns(db, "stop_active_invocations"))[
            "request_generation"
        ] == 1
        assert EXACT_GENERATION_CHECK in await _table_sql(
            db, "stop_active_invocations"
        )
    assert await _register_once(store) is True
