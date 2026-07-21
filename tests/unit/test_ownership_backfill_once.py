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

from typing import Iterator

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
