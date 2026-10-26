"""Actual corpus surveys, localhost SDK embeddings and native PG reindex control.

The model-preference projection supplies the real callback to the real service;
it is not an initialized/READY agent or authenticated HTTP acceptance.
"""

import os
from contextlib import asynccontextmanager

import asyncpg
import pytest

from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.storage import embedding_reindex
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.postgres import PostgresBackend, PostgresPoolAdapter
from kestrel_sovereign.storage.embedding_column import ensure_embedding_vec_column
from kestrel_sovereign.storage.sqla import embedding_profile
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door",
    [
        "dominant-scan",
        "dominant-registry",
        "stored",
        "count",
        "classify",
        "embed",
        "write",
        "registry",
        "registry-write",
        "registry-describe",
        "parity-insert",
        "parity-update",
    ],
)
@pytest.mark.parametrize("kind", ["native-control", "ordinary", "healthy"])
async def test_actual_embedding_corpus_and_reindex_preserve_original_control(
    native_memory_owners,
    monkeypatch,
    door,
    kind,
):
    n = native_memory_owners
    pool = await asyncpg.create_pool(
        os.environ["TEST_POSTGRES_URL"],
        min_size=0,
        max_size=2,
        server_settings={"search_path": f'"{n.schema}",public'},
    )
    backend = PostgresBackend.from_pool(pool)
    db = AsyncDatabase(backend)
    actual_service = n.service.get_embedding_service()
    target = actual_service.describe().profile_id
    original = OSError("original reindex native release acknowledgement lost")
    ordinary = OSError("ordinary callback unavailable after acknowledged native work")
    reached, sources, fired = [], [], []
    armed = False

    @asynccontextmanager
    async def lose_ack(native):
        async with native as connection:
            yield connection
        if armed and not fired:
            fired.append(True)
            raise original

    async def at_native(operation, *args, **kwargs):
        nonlocal armed
        reached.append(True)
        if kind == "native-control":
            armed = True
        try:
            result = await operation(*args, **kwargs)
        except BaseException as error:
            sources.append(error)
            raise
        if kind == "ordinary":
            raise ordinary
        return result

    def at_callback():
        reached.append(True)
        if kind == "native-control":
            try:
                n.lease.require_live()
            except BaseException as error:
                sources.append(error)
                raise
        if kind == "ordinary":
            raise ordinary

    try:
        assert await ensure_embedding_vec_column(db, "saved_items", 3)
        # Explicit component schema: the base native fixture does not promise
        # complete startup migrations (some legacy PG probes are cross-schema).
        # These are real DDL operations, not invented SQL/commit results.
        for table in embedding_reindex.REINDEX_TABLES:
            await db.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT"
            )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS embedding_profiles ("
            "id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, "
            "dim INTEGER NOT NULL, space_id TEXT NOT NULL, normalized BOOLEAN NOT NULL)"
        )
        await db.execute(
            "ALTER TABLE embedding_profiles ADD COLUMN IF NOT EXISTS parity_cosine DOUBLE PRECISION"
        )
        for row_id in ("corpus-a", "corpus-b"):
            await db.execute(
                "INSERT INTO saved_items (id,agent_id,item_type,name,content,content_hash,embedding_profile_id) VALUES (?,?,?,?,?,?,?)",
                (
                    row_id,
                    n.did,
                    "excerpt",
                    row_id,
                    "local native text",
                    row_id,
                    "old-profile",
                ),
            )
        await db.execute(
            "INSERT INTO embedding_profiles (id,provider,model,dim,space_id,normalized) VALUES (?,?,?,?,?,?)",
            ("old-profile", "openai", "old-model", 3, "old-space", False),
        )
        # Real vector required by the stored-vectors survey, not a fabricated
        # profile count or replacement SQL result.
        await db.execute(
            "UPDATE saved_items SET embedding_vec=?::vector WHERE agent_id=?",
            ("[1,0,0]", n.did),
        )
        actual_fetch = db.fetchall

        async def fetch(sql, *args, **kwargs):
            selected = (
                door == "dominant-scan"
                and "GROUP BY embedding_profile_id" in sql
                and "FROM saved_items" in sql
                or door == "dominant-registry"
                and "FROM embedding_profiles WHERE" in sql
                or door == "stored"
                and "GROUP BY embedding_profile_id" in sql
                or door == "count"
                and "SELECT COUNT(*) FROM saved_items" in sql
                or door == "classify"
                and "SELECT id, summary, content FROM saved_items" in sql
            )
            if selected:
                return await at_native(actual_fetch, sql, *args, **kwargs)
            return await actual_fetch(sql, *args, **kwargs)

        monkeypatch.setattr(db, "fetchall", fetch)
        actual_commit = db.execute_commit
        actual_execute = db.execute

        async def commit(sql, *args, **kwargs):
            if door == "write" and "UPDATE saved_items SET embedding_vec" in sql:
                return await at_native(actual_commit, sql, *args, **kwargs)
            return await actual_commit(sql, *args, **kwargs)

        async def execute(sql, *args, **kwargs):
            if door == "registry-write" and "INSERT INTO embedding_profiles" in sql:
                return await at_native(actual_execute, sql, *args, **kwargs)
            if (
                door == "parity-insert"
                and "INSERT INTO embedding_profiles" in sql
                or door == "parity-update"
                and "UPDATE embedding_profiles SET parity_cosine" in sql
            ):
                return await at_native(actual_execute, sql, *args, **kwargs)
            return await actual_execute(sql, *args, **kwargs)

        monkeypatch.setattr(db, "execute_commit", commit)
        monkeypatch.setattr(db, "execute", execute)

        class Service:
            def describe(self):
                result = actual_service.describe()
                if door == "registry-describe":
                    at_callback()
                return result

            async def aembed_batch(self, texts):
                # The local fixture SDK returns one native embedding per call.
                # Compose real SDK outputs, not fake vectors or success receipts.
                result = []
                for text in texts:
                    result.extend(await actual_service.aembed_batch([text]))
                if door == "embed" and not reached:
                    at_callback()
                return result

        reindexer = embedding_reindex.EmbeddingReindexer(
            db,
            Service(),
            target,
            column_dim=3,
            retry_backoff_s=0,
            max_split_retries=1,
        )
        actual_upsert = embedding_reindex.upsert_embedding_profile

        async def upsert(*args, **kwargs):
            result = await actual_upsert(*args, **kwargs)
            if door == "registry":
                at_callback()
            return result

        monkeypatch.setattr(embedding_reindex, "upsert_embedding_profile", upsert)

        class PreferenceOwner(ModelPreferenceMixin):
            agent_id = n.did
            _raw_storage = type("StorageProjection", (), {"db": db})()

        n.service.set_corpus_embedding_profile_provider(
            PreferenceOwner()._dominant_embedding_profile,
        )
        backend._execution_custody = ExecutionCustody(GenerationFence())
        if kind == "native-control":
            backend._pool = PostgresPoolAdapter(pool, lose_ack)
        result = failure = None
        try:
            if door.startswith("dominant-"):
                result = await n.service._get_corpus_embedding_profile()
            elif door == "stored":
                result = await embedding_reindex.stored_vectors_by_profile(
                    db,
                    agent_id=n.did,
                    tables=["saved_items"],
                )
            elif door == "count":
                result = await reindexer.count_all_stale(n.did, ["saved_items"])
            elif door == "classify":
                result = await reindexer.classify_all_stale(n.did, ["saved_items"])
            elif door.startswith("parity-"):
                result = await embedding_profile.record_space_parity(
                    db,
                    space_id="native-parity",
                    model="local-test-embedding",
                    dim=3,
                    normalized=False,
                    parity_cosine=0.99,
                )
            else:
                result = await reindexer.reindex_table("saved_items", n.did)
        except BaseException as error:
            failure = error
        assert reached, f"selected actual producer was not reached: {door} {failure!r}"
        if kind == "native-control":
            assert sources and is_execution_control_error(sources[0])
            if door in {"embed", "registry", "registry-describe"}:
                assert not fired
            else:
                assert fired and execution_commit_outcome(sources[0]) == "committed"
                assert original in tuple(_explicit_control_chain(sources[0]))
            assert failure is not None and is_execution_control_error(failure), (
                f"actual embedding producer control became fallback/retry/success: "
                f"door={door} result={result!r} failure={failure!r}"
            )
            assert sources[0] in tuple(_explicit_control_chain(failure))
            if door == "embed":
                assert len(n.requests) == 2, "control must not enter split retry"
            if door == "write":
                assert (
                    await n.observer.fetchval(
                        f'SELECT count(*) FROM "{n.schema}".saved_items WHERE embedding_profile_id=$1',
                        target,
                    )
                    == 1
                ), "the first COMMIT happened; no second row may be written"
            if door == "registry-write":
                assert (
                    await n.observer.fetchval(
                        f'SELECT count(*) FROM "{n.schema}".embedding_profiles WHERE id=$1',
                        target,
                    )
                    == 1
                )
            if door.startswith("parity-"):
                assert await n.observer.fetchval(
                    f'SELECT parity_cosine FROM "{n.schema}".embedding_profiles WHERE space_id=$1',
                    "native-parity",
                ) == pytest.approx(0.99)
        else:
            assert failure is None, f"availability/healthy behavior broke: {failure!r}"
            if kind == "healthy":
                if door.startswith("dominant-"):
                    assert (
                        result["profile_id"] == "old-profile"
                        and result["row_count"] == 2
                    )
                elif door == "stored":
                    assert result["saved_items"] == {"old-profile": 2}
                elif door == "count":
                    assert result["saved_items"] == 2
                elif door == "classify":
                    assert result == {"stale": 2, "actionable": 2, "unembeddable": 0}
                elif door.startswith("parity-"):
                    assert result is None
                    assert await n.observer.fetchval(
                        f'SELECT parity_cosine FROM "{n.schema}".embedding_profiles WHERE space_id=$1',
                        "native-parity",
                    ) == pytest.approx(0.99)
                else:
                    assert result.reembedded == 2 and result.failed == 0
            elif door.startswith("dominant-") or door.startswith("parity-"):
                assert result is None
            elif door == "stored":
                assert result == {"saved_items": None}
            elif door == "count":
                assert result == {"saved_items": 0}
            elif door == "classify":
                assert result == {"stale": 0, "actionable": 0, "unembeddable": 0}
            elif door == "write":
                assert result.reembedded == 0 and result.failed == 2
            else:
                assert result.reembedded == 2 and result.failed == 0
                if door == "embed":
                    assert len(n.requests) == 4
        if door.startswith(("dominant-", "parity-")) or door in {
            "stored",
            "count",
            "classify",
        }:
            assert not n.requests, "survey must not call any provider"
    finally:
        backend._pool = pool
        backend._execution_custody = None
        await db.close()
        await pool.close()
