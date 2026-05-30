#!/usr/bin/env python3
"""End-to-end validation of the vector-lift architecture on real PG.

What this script proves: the chain from a fresh ``AsyncDatabase`` →
the Phase-2 migrations adding ``embedding_vec`` columns → ``Async*Store``
dual-write → ``get_vector_backend`` factory picking ``PgVectorBackend``
on PG → pgvector kNN actually serving the search — all works end-to-end
against a real Postgres with the pgvector extension installed.

Most of this is already covered by unit tests (mocked + SQLite). The
gap this fills: nothing in CI has actually run the PG-specific paths
(BYTEA → vector(N) backfill, ``CREATE EXTENSION``, HNSW index, ``<=>``
operator, asyncpg vector return shape) against a real Postgres.

Runs against the local ``pgvector/pgvector:pg15`` container on :5433
(``frinz-postgres``). Creates a dedicated ``vector_lift_validation``
database so it never touches the live ``frinz`` or ``kestrel``
databases. Drops it at the end whether the run succeeds or fails.

Uses Ollama's ``nomic-embed-text`` (768-dim) for real embeddings —
verified to be running locally at ``localhost:11434`` before this was
written.

Usage:
    # Defaults assume a local ``pgvector/pgvector`` container on :5433
    # with ``POSTGRES_USER=frinz_user`` (the local frinz dev pattern).
    # Override via env vars or CLI flags:
    python scripts/validate_vector_lift_e2e.py
    PGVECTOR_HOST=localhost PGVECTOR_PORT=5433 \\
        PGVECTOR_USER=postgres PGVECTOR_PASSWORD=secret \\
        python scripts/validate_vector_lift_e2e.py
    python scripts/validate_vector_lift_e2e.py --host db --port 5432 \\
        --user myuser

Connection settings are read from (in order of precedence):
    1. CLI flags (``--host``, ``--port``, ``--user``, ``--password``)
    2. Env vars (``PGVECTOR_HOST`` etc.)
    3. Built-in defaults that match the local frinz-postgres container

Expected output:
    Each phase prints ✓ on success; the final summary line confirms
    pgvector kNN returned results in the expected ranking order.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from urllib.parse import quote as _urlquote

import asyncpg

# Suppress sovereign's verbose info logging so the script output stays
# readable. Errors and warnings still surface.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Connection settings — read from env / CLI in ``_resolve_config`` and
# threaded through the rest of the script. Module-level globals stay
# mutable so ``_drop_test_db`` etc. can reach them without dragging a
# config object through every helper.
PG_HOST = "localhost"
PG_PORT = 5433
PG_USER = "frinz_user"
PG_PASSWORD = "frinz_password_2024"
PG_ADMIN_DB = "postgres"
TEST_DB = "vector_lift_validation"


def _resolve_config() -> None:
    """Pull PG connection settings from env vars + CLI args, with
    sensible defaults for the local frinz-postgres container. Mutates
    the module-level globals so the rest of the script picks them up.
    """
    parser = argparse.ArgumentParser(
        description="Validate the vector-lift architecture end-to-end "
                    "against a real Postgres + pgvector instance.",
    )
    parser.add_argument("--host", default=os.environ.get("PGVECTOR_HOST", "localhost"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PGVECTOR_PORT", "5433")))
    parser.add_argument("--user", default=os.environ.get("PGVECTOR_USER", "frinz_user"))
    parser.add_argument("--password",
                        default=os.environ.get("PGVECTOR_PASSWORD", "frinz_password_2024"))
    parser.add_argument("--admin-db",
                        default=os.environ.get("PGVECTOR_ADMIN_DB", "postgres"),
                        help="DB to connect to for CREATE / DROP DATABASE.")
    parser.add_argument("--test-db",
                        default=os.environ.get("PGVECTOR_TEST_DB", "vector_lift_validation"))
    args = parser.parse_args()

    global PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_ADMIN_DB, TEST_DB
    PG_HOST = args.host
    PG_PORT = args.port
    PG_USER = args.user
    PG_PASSWORD = args.password
    PG_ADMIN_DB = args.admin_db
    TEST_DB = args.test_db

    # Safety guard against accidentally dropping a live database: the
    # test DB name MUST contain ``validation`` (or be the default
    # ``vector_lift_validation``). Anyone setting
    # ``PGVECTOR_TEST_DB=frinz`` via a typo or inherited env var would
    # otherwise get their data wiped. Known-live names are extra-
    # explicitly blocked. (Caught by codex review.)
    _LIVE_DB_BLOCKLIST = {"frinz", "kestrel", "postgres", "template0", "template1"}
    if TEST_DB.lower() in _LIVE_DB_BLOCKLIST:
        raise SystemExit(
            f"refusing to use --test-db={TEST_DB!r}: that's a real database "
            "this script will DROP. Pick a name with 'validation' in it."
        )
    if "validation" not in TEST_DB.lower():
        raise SystemExit(
            f"refusing to use --test-db={TEST_DB!r}: this script does "
            "DROP DATABASE on the target, so the name must contain "
            "'validation' as a safety guard. Default is "
            "'vector_lift_validation'."
        )
    # Identifier-shape guard: even with the ``validation`` substring,
    # an attacker-controlled env var could carry a quote or semicolon
    # that would terminate the f-string'd ``DROP DATABASE`` and inject
    # arbitrary SQL. Restrict to a plain pg_identifier shape.
    # (Caught by codex review.)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", TEST_DB):
        raise SystemExit(
            f"refusing to use --test-db={TEST_DB!r}: must match "
            "[A-Za-z_][A-Za-z0-9_]* (no quotes, semicolons, etc.)."
        )


def _build_dsn() -> str:
    """Build a DSN with URL-encoded user + password components so
    credentials containing ``@``, ``/``, ``:``, ``#``, ``%`` etc. don't
    misparse on either the asyncpg or the SQLAlchemy side. (Caught by
    codex review.)
    """
    user = _urlquote(PG_USER, safe="")
    password = _urlquote(PG_PASSWORD, safe="")
    return f"postgresql://{user}:{password}@{PG_HOST}:{PG_PORT}/{TEST_DB}"


def _build_dsn_redacted() -> str:
    """Same DSN with the password masked — used for the banner log line
    so passwords aren't echoed into terminal scrollback or CI logs.
    """
    user = _urlquote(PG_USER, safe="")
    return f"postgresql://{user}:***@{PG_HOST}:{PG_PORT}/{TEST_DB}"


async def _drop_test_db() -> None:
    """DROP DATABASE outside any transaction so leftover state from a
    prior partial run doesn't break this one.

    Identifier interpolation is safe here because ``_resolve_config``'s
    regex guard already restricted ``TEST_DB`` to a plain
    ``[A-Za-z_][A-Za-z0-9_]*`` identifier shape. ``DROP DATABASE`` and
    ``pg_terminate_backend`` don't accept bind parameters for the
    target name, so direct interpolation is the only path — the
    pre-validation is what makes it safe.
    """
    conn = await asyncpg.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, database=PG_ADMIN_DB,
    )
    try:
        # ``datname`` IS bindable here (it's a value, not an
        # identifier), so prefer the safe form.
        await conn.execute(
            """SELECT pg_terminate_backend(pid)
               FROM pg_stat_activity
               WHERE datname = $1 AND pid <> pg_backend_pid()""",
            TEST_DB,
        )
        await conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}"')
    finally:
        await conn.close()


async def _create_test_db() -> None:
    """Same identifier-quoting rules as :func:`_drop_test_db`."""
    conn = await asyncpg.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, database=PG_ADMIN_DB,
    )
    try:
        await conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    finally:
        await conn.close()


async def _embed_via_ollama(text: str) -> list[float]:
    """Pull a real 768-dim embedding from local Ollama
    ``nomic-embed-text`` so the validation exercises the production
    embedding shape, not synthetic vectors."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": text},
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        assert embeddings and len(embeddings) == 1
        return embeddings[0]


async def main() -> int:
    print("Vector-lift end-to-end validation against pgvector PG")
    print(f"  DSN: {_build_dsn_redacted()}")
    print()

    print("→ Reset test database")
    await _drop_test_db()
    await _create_test_db()
    print("  ✓ vector_lift_validation database created fresh")
    print()

    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
    from kestrel_sovereign.storage.saved_items_store import SavedItemsStore

    print("→ Phase A: fresh DB boot. Phase-2 migrations should defer "
          "column creation (no rows to sniff a dimension from).")
    db = await AsyncDatabase.postgres(_build_dsn())
    try:
        # Verify both target tables exist but embedding_vec does not.
        async with db.backend._pool.acquire() as conn:
            for tbl in ("saved_items", "document_chunks"):
                has_vec = await conn.fetchval(
                    f"""SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{tbl}' AND column_name = 'embedding_vec'
                    )"""
                )
                assert not has_vec, (
                    f"  ✗ {tbl}.embedding_vec exists on fresh DB — migration "
                    f"should have deferred"
                )
                print(f"  ✓ {tbl}.embedding_vec correctly absent (deferred)")
    finally:
        await db.close()
    print()

    print("→ Phase B: write rows with real Ollama embeddings, restart "
          "AsyncDatabase, watch the migration sniff the dim + create "
          "the column + backfill + index.")
    db = await AsyncDatabase.postgres(_build_dsn())
    try:
        rag = AsyncRAGStore(db)
        # ``chunk_document`` computes embeddings itself if the embedding
        # service is available. We pre-embed manually so we don't have
        # to wire the global lazy ``_get_embedding_service`` to point at
        # Ollama from inside this script. Instead, insert rows via raw
        # SQL with the legacy ``embedding`` BYTEA column populated —
        # that's the exact state the migration is designed to convert.
        import struct
        texts = [
            "Postgres pgvector enables fast HNSW-indexed cosine search.",
            "Cats are mammals that purr to communicate contentment.",
            "Cosine similarity measures angle between two vectors.",
        ]
        embeddings = [await _embed_via_ollama(t) for t in texts]
        for text, emb in zip(texts, embeddings):
            packed = struct.pack(f"<{len(emb)}f", *emb)
            await db.execute(
                "INSERT INTO document_chunks (file_hash, content, embedding) "
                "VALUES (?, ?, ?)",
                ("validation-doc", text, packed),
            )
        await db.commit()
        print(f"  ✓ Wrote {len(texts)} chunks with real Ollama embeddings "
              f"(dim {len(embeddings[0])}, all to BYTEA ``embedding`` column)")
    finally:
        await db.close()

    # Second boot — this is where the migration fires.
    db = await AsyncDatabase.postgres(_build_dsn())
    try:
        async with db.backend._pool.acquire() as conn:
            udt = await conn.fetchval(
                """SELECT udt_name FROM information_schema.columns
                   WHERE table_name = 'document_chunks'
                     AND column_name = 'embedding_vec'"""
            )
            assert udt == "vector", (
                f"  ✗ document_chunks.embedding_vec is {udt!r}; expected "
                f"'vector' after Phase-2 migration"
            )
            print(f"  ✓ document_chunks.embedding_vec created as type {udt!r}")

            # Confirm HNSW index exists.
            idx = await conn.fetchval(
                """SELECT indexname FROM pg_indexes
                   WHERE tablename = 'document_chunks'
                     AND indexname = 'idx_document_chunks_embedding_vec_hnsw'"""
            )
            assert idx, "  ✗ HNSW index missing"
            print(f"  ✓ HNSW cosine index created: {idx}")

            # Confirm backfill: all 3 rows have embedding_vec populated.
            filled = await conn.fetchval(
                "SELECT COUNT(*) FROM document_chunks WHERE embedding_vec IS NOT NULL"
            )
            assert filled == len(texts), (
                f"  ✗ {filled} rows backfilled; expected {len(texts)}"
            )
            print(f"  ✓ Backfill: {filled}/{len(texts)} rows have embedding_vec")
    finally:
        await db.close()
    print()

    print("→ Phase C: kNN through AsyncRAGStore._search_via_vector_backend "
          "actually returns ranked rows from PgVectorBackend.")
    db = await AsyncDatabase.postgres(_build_dsn())
    try:
        rag = AsyncRAGStore(db)

        # Stub the global embedding service in the rag module so
        # ``_search_by_embedding`` picks the vector path. We embed the
        # query via Ollama here and feed it through the stub.
        import kestrel_sovereign.storage.async_rag_store as rag_mod
        query = "fast similarity search in the database"
        query_emb = await _embed_via_ollama(query)

        class _StubEmbed:
            async def aembed(self, _q):
                return query_emb

        rag_mod._embedding_service = _StubEmbed()

        # Drive PgVectorBackend DIRECTLY, not via
        # ``rag._search_by_embedding`` — that wrapper has a designed-in
        # fallback to the legacy in-Python loop on any backend error,
        # which would silently mask a pgvector failure (the BYTEA
        # embeddings are still populated, so the legacy path would
        # return the same ranked results). Calling
        # ``PgVectorBackend.knn`` directly exercises the actual code
        # path this validation is meant to prove. (Caught by codex
        # review.)
        sf = rag._get_vector_session_factory()
        from kestrel_sovereign.storage.sqla import build_document_chunk_spec
        from kestrel_sovereign.storage.vector import (
            PgVectorBackend,
            get_vector_backend,
        )
        backend = get_vector_backend(sf, build_document_chunk_spec(768))
        assert isinstance(backend, PgVectorBackend), (
            f"  ✗ factory returned {type(backend).__name__}; expected "
            f"PgVectorBackend on PG"
        )
        print(f"  ✓ Factory dispatched to {type(backend).__name__} "
              f"(NOT the legacy in-Python fallback)")

        # Run kNN through the actual PgVectorBackend instance.
        import struct as _struct
        packed_query = _struct.pack(f"<{len(query_emb)}f", *query_emb)
        knn_rows = await backend.knn(packed_query, k=3, filter=None)
        assert knn_rows, "  ✗ PgVectorBackend.knn returned no rows"

        # Materialize: re-read the rows by chunk_id so we can sanity-
        # check ordering by content.
        async with db.backend._pool.acquire() as conn:
            id_to_content = {}
            for row_id_str, score in knn_rows:
                row = await conn.fetchrow(
                    "SELECT content FROM document_chunks WHERE chunk_id = $1",
                    int(row_id_str),
                )
                id_to_content[row_id_str] = row["content"] if row else None

        print(f"  ✓ PgVectorBackend.knn returned {len(knn_rows)} rows:")
        for row_id_str, score in knn_rows:
            preview = (id_to_content.get(row_id_str) or "")[:60]
            print(f"     score={score:.4f}  {preview}…")

        top_content = (id_to_content.get(knn_rows[0][0]) or "").lower()
        assert "pgvector" in top_content or "cosine" in top_content, (
            "  ✗ top pgvector result didn't match the query's semantic "
            "intent — expected the pgvector / cosine sentence to rank first"
        )
        print("  ✓ Top result matches semantic intent (pgvector / cosine "
              "sentence ranks first)")
    finally:
        await db.close()
    print()

    print()
    print("→ Phase D: store-API dual-writes. ``AsyncRAGStore.chunk_document`` "
          "and ``SavedItemsStore.save_item`` should populate ``embedding_vec`` "
          "immediately on insert (no second-boot backfill needed). Earlier "
          "phases tested the backfill via raw SQL; this phase tests the dual-"
          "write path the stores are actually advertised to support.")
    db = await AsyncDatabase.postgres(_build_dsn())
    try:
        # Wire the global embedding service in BOTH stores so their
        # internal ``compute_embeddings`` branches actually fire and
        # produce real vectors. The stubs return one fixed embedding —
        # we don't care about semantic ranking here, only about
        # whether embedding_vec gets populated.
        sample_emb = await _embed_via_ollama("warmup sample for store-api phase")

        class _StubEmbed:
            async def aembed(self, _q):
                return sample_emb
            async def aembed_batch(self, items):
                return [sample_emb] * len(items)

        import kestrel_sovereign.storage.async_rag_store as rag_mod
        import kestrel_sovereign.storage.saved_items_store as si_mod
        rag_mod._embedding_service = _StubEmbed()

        # SavedItemsStore: per-instance lazy load.
        si_store = si_mod.SavedItemsStore(db, agent_id="agent-validation")
        si_store._embedding_service = _StubEmbed()

        # 1. AsyncRAGStore.chunk_document path.
        rag = AsyncRAGStore(db)
        new_chunks_count = await rag.chunk_document(
            file_hash="store-api-doc",
            content="The store-API phase D content " * 30,  # ensures >1 chunk
            chunk_size=200,
            compute_embeddings=True,
        )
        assert new_chunks_count > 0, "  ✗ chunk_document produced no chunks"
        print(f"  ✓ chunk_document inserted {new_chunks_count} chunks via "
              "the store API")

        # Verify ``embedding_vec`` is populated for those new rows —
        # the dual-write path must have fired.
        async with db.backend._pool.acquire() as conn:
            filled = await conn.fetchval(
                """SELECT COUNT(*) FROM document_chunks
                   WHERE file_hash = $1 AND embedding_vec IS NOT NULL""",
                "store-api-doc",
            )
            assert filled == new_chunks_count, (
                f"  ✗ dual-write regression: {filled}/{new_chunks_count} "
                f"new chunks have embedding_vec populated"
            )
            print(f"  ✓ All {filled} new chunks have embedding_vec "
                  "populated by the dual-write (no second-boot needed)")

        # 2. SavedItemsStore.save_item path. The migration deferred
        # this table's embedding_vec column on the fresh DB (Phase A);
        # Phase D's first write should sniff-and-create on the
        # NEXT-boot logic. So we save an item, restart the DB so the
        # migration runs against the now-populated rows, and verify
        # everything backfilled correctly.
        item = await si_store.save_item(
            item_type="stash",
            name="store-api validation item",
            content="content for the store-api phase D saved-item write",
            compute_embedding=True,
        )
        assert item.embedding is not None, (
            "  ✗ save_item did not compute / store an embedding"
        )
        print("  ✓ save_item wrote one saved-item with an embedding via "
              "the store API")
    finally:
        await db.close()

    # Restart to fire the saved_items migration now that there's a
    # sniff-able row.
    db = await AsyncDatabase.postgres(_build_dsn())
    try:
        async with db.backend._pool.acquire() as conn:
            udt = await conn.fetchval(
                """SELECT udt_name FROM information_schema.columns
                   WHERE table_name = 'saved_items'
                     AND column_name = 'embedding_vec'"""
            )
            assert udt == "vector", (
                f"  ✗ saved_items.embedding_vec is {udt!r}; expected "
                "'vector' after Phase-D restart"
            )
            print(f"  ✓ Restart created saved_items.embedding_vec as {udt!r}")

            # The save_item path computed an embedding before
            # embedding_vec existed → ``_write_embedding_vec`` logged
            # and degraded gracefully. After the migration, the row
            # has been backfilled from the legacy ``embedding`` BYTEA.
            filled = await conn.fetchval(
                "SELECT COUNT(*) FROM saved_items "
                "WHERE embedding_vec IS NOT NULL"
            )
            assert filled >= 1, (
                f"  ✗ saved_items backfill: {filled} rows have embedding_vec; "
                "expected at least 1 (the one written via save_item)"
            )
            print(f"  ✓ {filled} saved_items row(s) backfilled with embedding_vec")
    finally:
        await db.close()

    # Test DB drop happens in ``_run_with_cleanup``'s finally so it
    # also fires on the failure paths.
    print()
    print("All checks passed. The vector-lift architecture is live on "
          "real Postgres + pgvector.")
    return 0


async def _run_with_cleanup() -> int:
    """Run ``main()`` with guaranteed cleanup of the test DB no matter
    how the body exits — assertion failure, Ollama HTTP error, asyncpg
    timeout, KeyboardInterrupt, anything. The earlier
    ``except AssertionError`` only path leaked the validation DB on
    every other failure mode. (Caught by codex review.)
    """
    try:
        return await main()
    finally:
        try:
            await _drop_test_db()
        except Exception as e:
            print(f"WARNING: failed to drop test DB during cleanup: {e}",
                  file=sys.stderr)


if __name__ == "__main__":
    # Resolve config BEFORE any asyncio.run so ``--help`` etc. exit
    # quickly and the connection settings are pinned for the run.
    _resolve_config()
    try:
        sys.exit(asyncio.run(_run_with_cleanup()))
    except AssertionError as e:
        # Make assertion failures stand out from the success-path ✓ output.
        print()
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print()
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
