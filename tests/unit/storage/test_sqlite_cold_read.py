"""Cold-read semantics for the SQLite backend (#2920).

A cold read exists so an inspection tool can report on an agent's database
without disturbing it. "Without disturbing it" has two failure modes that pull
in opposite directions, and these tests pin both:

  * ``mode=ro`` alone CREATES the ``-wal``/``-shm`` sidecars on a WAL database
    and cannot remove them again, and the cold identity lookup reads leftover
    sidecars as a live agent — so a read-only inspection would plant evidence
    that the agent it inspected was running;
  * ``immutable=1`` creates no sidecars but makes SQLite ignore WAL content,
    so it would report pre-WAL state as current.
"""
import sqlite3

import pytest

from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


def _wal_db(path, *, rows=(("committed",),)):
    """A WAL-mode database with `rows`, cleanly checkpointed and closed."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?)", rows)
    conn.commit()
    conn.close()


def _sidecars(path):
    return sorted(p.name for p in path.parent.glob(f"{path.name}-*"))


@pytest.mark.asyncio
async def test_cold_read_of_quiescent_db_leaves_no_sidecars(tmp_path):
    db = tmp_path / "quiet.db"
    _wal_db(db)
    assert _sidecars(db) == []

    backend = SQLiteBackend(str(db), cold_read=True)
    await backend.connect()
    try:
        assert await backend.fetch_one("SELECT v FROM t") is not None
    finally:
        await backend.close()

    assert _sidecars(db) == [], (
        "a cold read of a checkpointed database created sidecars it cannot "
        "remove; the cold identity lookup reads those as a live agent"
    )


@pytest.mark.asyncio
async def test_cold_read_sees_data_that_is_still_only_in_the_wal(tmp_path):
    """A WAL already exists, so the read must not be blind to it.

    ``immutable=1`` would return the pre-WAL row set and report stale
    governance state as current — the reason the mode is chosen per-database
    rather than fixed.
    """
    db = tmp_path / "busy.db"
    _wal_db(db)

    holder = sqlite3.connect(db)
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("INSERT INTO t VALUES ('only-in-wal')")
    holder.commit()
    assert _sidecars(db), "setup must leave a live WAL"

    backend = SQLiteBackend(str(db), cold_read=True)
    await backend.connect()
    try:
        rows = await backend.fetch_all("SELECT v FROM t ORDER BY v")
    finally:
        await backend.close()
        holder.close()

    values = {tuple(r)[0] for r in rows}
    assert "only-in-wal" in values, (
        f"cold read ignored committed WAL content: {values}"
    )


@pytest.mark.asyncio
async def test_cold_read_does_not_migrate_a_database_it_only_inspects(tmp_path):
    """Opening a cold read must not run schema DDL against the target.

    With a WAL present the connection is a plain read-only one, so any DDL
    ``_init_schema`` attempted would fail outright with "attempt to write a
    readonly database". The point is not that it fails gracefully — it is that
    an inspection does not migrate the database it was asked to report on.
    """
    db = tmp_path / "old-schema.db"
    _wal_db(db)
    holder = sqlite3.connect(db)
    holder.execute("INSERT INTO t VALUES ('pending')")
    holder.commit()
    assert _sidecars(db), "setup must leave a live WAL"

    try:
        storage = AsyncStorage(str(db), backend="sqlite", agent_id="did:test", cold_read=True)
        async with storage:
            assert await storage.db.fetchone("SELECT v FROM t") is not None
            # An inspection records nothing, so it has no destructive
            # operation to audit and must not create the audit database.
            assert storage.destructive_audit is None
    finally:
        holder.close()

    assert not (tmp_path / "kestrel_audit.db").exists(), (
        "a cold read created the destructive-audit database beside the one "
        "it was inspecting"
    )


@pytest.mark.asyncio
async def test_cold_read_follows_a_symlink_to_where_the_sidecars_really_are(tmp_path):
    """Sidecars sit beside the RESOLVED database, not beside an alias.

    Checking one location while opening another picks ``immutable=1`` for a
    database that has a live WAL, and then serves pre-WAL state as current —
    a governance decision made from a stale hash.
    """
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    db = real_dir / "actual.db"
    _wal_db(db)

    holder = sqlite3.connect(db)
    holder.execute("INSERT INTO t VALUES ('only-in-wal')")
    holder.commit()
    assert _sidecars(db), "setup must leave a live WAL beside the target"

    link = tmp_path / "alias.db"
    link.symlink_to(db)
    assert not list(tmp_path.glob("alias.db-*")), (
        "setup: no sidecars beside the alias, which is the whole trap"
    )

    backend = SQLiteBackend(str(link), cold_read=True)
    await backend.connect()
    try:
        rows = await backend.fetch_all("SELECT v FROM t ORDER BY v")
    finally:
        await backend.close()
        holder.close()

    values = {tuple(r)[0] for r in rows}
    assert "only-in-wal" in values, (
        f"cold read through a symlink ignored committed WAL content: {values}"
    )


@pytest.mark.asyncio
async def test_cold_read_refuses_to_report_after_a_writer_came_and_went(tmp_path):
    """A whole write cycle can complete inside an immutable read.

    The writer's final close checkpoints and REMOVES the sidecars, so looking
    only for sidecars afterwards finds none and concludes nothing happened —
    while this connection served the pre-write state throughout.
    """
    db = tmp_path / "raced.db"
    _wal_db(db)
    assert _sidecars(db) == []

    backend = SQLiteBackend(str(db), cold_read=True)
    await backend.connect()
    try:
        before = await backend.fetch_all("SELECT v FROM t")
        assert len(before) == 1

        writer = sqlite3.connect(db)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO t VALUES ('arrived-during-the-read')")
        writer.commit()
        writer.close()  # checkpoints, and takes the sidecars with it

        assert _sidecars(db) == [], (
            "setup: the writer must leave no trace for the naive check to find"
        )
        with pytest.raises(Exception) as excinfo:
            backend.assert_cold_read_still_valid()
        assert "changed while it was being read" in str(excinfo.value)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_cold_read_via_config_does_not_migrate_the_database(tmp_path):
    """The facade must not think it is writable while its backend is not.

    ``create_backend`` reads ``cold_read`` out of the config dict while the
    keyword argument keeps its default; a facade that disagreed would run
    schema migrations against an immutable connection.
    """
    db = tmp_path / "configured.db"
    _wal_db(db)

    storage = AsyncStorage(
        config={"backend": "sqlite", "db_path": str(db), "cold_read": True},
        agent_id="did:test",
    )
    assert storage.cold_read is True
    async with storage:
        assert await storage.db.fetchone("SELECT v FROM t") is not None
        assert storage.destructive_audit is None
    assert _sidecars(db) == []


@pytest.mark.asyncio
async def test_cold_read_of_a_missing_default_store_creates_nothing(tmp_path, monkeypatch):
    """"There is nothing here" must be answerable without making somewhere.

    The backend already refuses to mkdir; the facade was creating the default
    agent data directory first, which made that refusal moot.
    """
    import kestrel_sovereign.storage.async_storage as async_storage

    absent = tmp_path / "never-created"
    monkeypatch.setattr(
        async_storage, "get_default_agent_data_dir", lambda: str(absent)
    )

    AsyncStorage(backend="sqlite", agent_id="did:test", cold_read=True)

    assert not absent.exists(), (
        "a cold read brought the store's directory into existence just by "
        "asking whether it was there"
    )


def test_config_requested_cold_read_survives_a_backend_that_cannot_carry_it():
    """PostgreSQL has no file lock, so its backend does not carry the flag.

    Falling back to the keyword default would silently upgrade a requested
    inspection into a writer: `initialize()` would run migrations.
    """
    storage = AsyncStorage(
        config={
            "backend": "postgres",
            "dsn": "postgresql://user@localhost/nowhere",
            "cold_read": True,
        },
        agent_id="did:test",
    )
    assert storage.cold_read is True
