"""#2871 — the PostgreSQL-only half of birth-record replication.

Two defects here cannot be reproduced on SQLite, so the SQLite suite in
``tests/unit/test_birth_record_replication.py`` is blind to them:

* ``document_chunks.embedding_vec`` is created unconditionally on SQLite
  (``sqla/migrations.py`` ``_migrate_sqlite_table``) but **deferred** on
  PostgreSQL until the table has an embedded row — which is exactly the state
  of a fresh runtime database at first boot.
* PostgreSQL aborts an entire transaction on any failed statement. SQLite does
  not. So a "best-effort, errors are non-fatal" write that fails inside a
  caller's transaction is genuinely non-fatal on SQLite and silently
  catastrophic on PostgreSQL.

Together they produced: 47 chunks reported written, 0 committed, with the real
cause visible only at debug level. Measured on a live PostgreSQL 16 before the
fix.

Run against any throwaway PostgreSQL:

    TEST_POSTGRES_URL=postgresql://u:p@127.0.0.1:5432/db pytest \
        tests/integration/test_birth_record_replication_postgres.py

Skipped when that is not set, so CI (which has no PostgreSQL) stays green.
"""

import contextlib
import os
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

POSTGRES_URL = (
    os.environ.get("TEST_POSTGRES_URL")
    or os.environ.get("KESTREL_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
)

if not POSTGRES_URL:  # pragma: no cover - environment gate
    pytest.skip(
        "TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL required",
        allow_module_level=True,
    )


@pytest.fixture
async def pg():
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.postgres(POSTGRES_URL)
    try:
        yield db
    finally:
        await db.close()


async def _purge_agent(db, agent_did: str) -> None:
    """Remove every row a run left behind, including the shared ones.

    The constitution is content-addressed, so its ``files`` and ``graph_nodes``
    rows are shared by every agent under it. Deleting only the *owner* rows
    leaves them ownerless, and the next run's ``store_file`` then raises
    "Cannot claim an unowned legacy file" — the suite would fail in setup
    instead of reporting a regression, having itself created the shape that
    bricks a real agent.
    """
    owned_files = [
        row[0]
        for row in await db.fetchall(
            "SELECT content_hash FROM file_owners WHERE agent_id = $1",
            (agent_did,),
        )
    ]
    governed = [
        row[0]
        for row in await db.fetchall(
            "SELECT target_id FROM graph_edges WHERE source_id = $1", (agent_did,),
        )
    ]
    await db.execute(
        "DELETE FROM document_chunk_owners WHERE agent_id = $1", (agent_did,)
    )
    # Only chunks nobody else still claims. The constitution is shared, so an
    # unconditional delete by file_hash would destroy a co-owner's chunks and
    # leave their owner rows pointing at nothing — a cleanup manufacturing the
    # exact damage this suite exists to catch, against a database whose DSN
    # comes from KESTREL_DATABASE_URL, i.e. possibly a live one.
    await db.execute(
        "DELETE FROM document_chunks WHERE NOT EXISTS ("
        "  SELECT 1 FROM document_chunk_owners o WHERE o.chunk_id = "
        "document_chunks.chunk_id)",
        (),
    )
    await db.execute("DELETE FROM file_owners WHERE agent_id = $1", (agent_did,))
    for content_hash in owned_files:
        # Only if nobody else still claims it — a co-owner's data must survive.
        remaining = await db.fetchone(
            "SELECT COUNT(*) FROM file_owners WHERE content_hash = $1",
            (content_hash,),
        )
        if not int(remaining[0]):
            await db.execute(
                "DELETE FROM files WHERE content_hash = $1", (content_hash,)
            )
    await db.execute("DELETE FROM graph_edge_owners WHERE agent_id = $1", (agent_did,))
    await db.execute("DELETE FROM graph_edges WHERE source_id = $1", (agent_did,))
    await db.execute("DELETE FROM graph_node_owners WHERE agent_id = $1", (agent_did,))
    for node_id in [agent_did, *governed]:
        remaining = await db.fetchone(
            "SELECT COUNT(*) FROM graph_node_owners WHERE node_id = $1", (node_id,)
        )
        if not int(remaining[0]):
            await db.execute(
                "DELETE FROM graph_nodes WHERE node_id = $1", (node_id,)
            )


@contextlib.asynccontextmanager
async def _without_embedding_vec(db):
    """Put document_chunks in the state a fresh PostgreSQL runtime is in.

    The Phase-2 migration skips the column while the table has no embedded
    rows, so a first boot always finds it missing.

    Restored on the way out. Left dropped, the next `_migrate_pg_table` run
    recreates it by sniffing the dimension from the first non-NULL `embedding`
    row — so an interrupted run leaving this test's 3-float vectors behind
    would pin the whole database's column to `vector(3)`.
    """
    had_column = await db._column_exists("document_chunks", "embedding_vec")
    await db.execute(
        "ALTER TABLE document_chunks DROP COLUMN IF EXISTS embedding_vec", ()
    )
    try:
        yield
    finally:
        if had_column:
            await db.execute(
                "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS "
                "embedding_vec vector",
                (),
            )


async def test_precomputed_chunks_commit_when_the_vector_column_is_missing(pg):
    """The caller's transaction must survive a failed parallel-column write.

    Before the fix the first failed UPDATE aborted the enclosing transaction,
    every later statement raised, both fallbacks were swallowed at debug level,
    and store_precomputed_chunks returned len(chunks) having committed nothing.
    """
    from kestrel_sovereign.storage.async_file_store import AsyncFileStore
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore, IndexedChunk

    agent = f"did:web:test.invalid:vec-{uuid.uuid4().hex[:8]}"
    try:
        async with _without_embedding_vec(pg):
            files = AsyncFileStore(pg, agent_id=agent)
            file_hash = await files.store_file(
                f"vector column probe {agent}".encode(), "probe.md",
            )
            rag = AsyncRAGStore(pg, agent_id=agent)
            payload = [
                IndexedChunk("one", [0.5, 0.25, 0.125], "profile-x"),
                IndexedChunk("two", [1.0, 0.0, -1.0], "profile-x"),
            ]

            # The shape replicate_birth_record uses: one transaction on the
            # target.
            async with pg.transaction():
                written = await rag.store_precomputed_chunks(file_hash, payload)
            assert written == 2

            row = await pg.fetchone(
                "SELECT COUNT(*) FROM document_chunks WHERE file_hash = $1",
                (file_hash,),
            )
            assert int(row[0]) == 2, (
                "the copy reported success but committed nothing"
            )

            # The legacy embedding column carries the vectors even with the
            # parallel column absent, so retrieval still has something to work
            # with.
            read_back = await rag.read_indexed_chunks(file_hash)
            assert [c.content for c in read_back] == ["one", "two"]
            assert read_back[0].embedding == pytest.approx([0.5, 0.25, 0.125])
            # The profile-id-only fallback ran rather than being swallowed by
            # an aborted transaction.
            assert [c.profile_id for c in read_back] == ["profile-x", "profile-x"]
    finally:
        await _purge_agent(pg, agent)


async def test_embedding_profile_rows_cross_from_a_sqlite_anchor(pg, tmp_path):
    """The registry row must survive the SQLite -> PostgreSQL type boundary.

    ``embedding_profiles.normalized`` is INTEGER on SQLite and BOOLEAN on
    PostgreSQL, and asyncpg's encoder rejects an int outright. Binding the
    anchor's row through unchanged makes this function a silent no-op on the
    only backend combination it exists for — every copied vector then reports
    as an unknown profile in the embeddings audit.
    """
    from kestrel_sovereign.identity.birth_record import (
        carry_embedding_profiles,
        read_embedding_profiles,
    )
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    profile_id = f"prof-{uuid.uuid4().hex[:8]}"
    anchor = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    try:
        await anchor.execute(
            "INSERT INTO embedding_profiles "
            "(id, provider, model, dim, space_id, normalized) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (profile_id, "ollama", "nomic-embed-text", 768, "space-a", 1),
        )
        rows = await read_embedding_profiles(
            anchor_db=anchor, profile_ids={profile_id},
        )
        assert len(rows) == 1
        assert isinstance(rows[0][5], int), "SQLite really does yield an int here"

        assert await carry_embedding_profiles(runtime_db=pg, rows=rows) == 1

        landed = await pg.fetchone(
            "SELECT provider, model, dim, space_id, normalized "
            "FROM embedding_profiles WHERE id = $1",
            (profile_id,),
        )
        assert landed is not None, "the registry row did not cross"
        assert landed[0] == "ollama"
        assert landed[2] == 768
        assert landed[4] is True

        # Idempotent, and never overwrites a description that is already there.
        assert await carry_embedding_profiles(runtime_db=pg, rows=rows) == 0
    finally:
        await anchor.close()
        await pg.execute(
            "DELETE FROM embedding_profiles WHERE id = $1", (profile_id,)
        )


async def test_birth_record_replicates_into_postgres_with_vectors(pg, tmp_path):
    """End to end on the real backend: a SQLite anchor's record — including
    chunk vectors — lands in PostgreSQL, and a second pass is a no-op."""
    from kestrel_sovereign.identity.birth_record import (
        diagnose_birth_record,
        replicate_birth_record,
    )
    from kestrel_sovereign.inception_service import (
        DID_WEB_DOMAIN_ENV,
        create_kestrel_identity_async,
    )
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore, IndexedChunk

    os.environ.setdefault("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    os.environ.setdefault(DID_WEB_DOMAIN_ENV, "agents.kestrel-sovereign.test")

    creds = await create_kestrel_identity_async(
        str(tmp_path), None, agent_name="Postgres Bird",
    )
    anchor = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    try:
        constitution_hash = (
            await AsyncGraphStore(anchor).get_node(creds.agent_did)
        ).properties["constitution_hash"]
        # The unit environment has no embedding service, so seed vectors the
        # way a real host's inception would produce them.
        seeded = [
            IndexedChunk("alpha", [0.5, -0.25, 0.125], "profile-x"),
            IndexedChunk("beta", [-1.0, 0.75, 0.0], "profile-x"),
        ]
        await AsyncRAGStore(
            anchor, agent_id=creds.agent_did,
        ).store_precomputed_chunks(constitution_hash, seeded)

        result = await replicate_birth_record(
            runtime_db=pg, anchor_db=anchor, agent_did=creds.agent_did,
        )
        assert result.chunks == 2

        node = await AsyncGraphStore(pg, agent_id=creds.agent_did).get_node(
            creds.agent_did,
        )
        assert node is not None and node.label == "Postgres Bird"
        assert node.properties["bootstrap_state"] == "pending"

        copied = await AsyncRAGStore(
            pg, agent_id=creds.agent_did,
        ).read_indexed_chunks(constitution_hash)
        assert [c.content for c in copied] == ["alpha", "beta"]
        assert [c.embedding for c in copied] == [
            pytest.approx(c.embedding) for c in seeded
        ]

        assert not await diagnose_birth_record(
            runtime_db=pg, anchor_db=anchor, agent_did=creds.agent_did,
        )
    finally:
        await anchor.close()
        await _purge_agent(pg, creds.agent_did)
