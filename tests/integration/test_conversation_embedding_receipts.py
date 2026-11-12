"""Canonical receipts through actual SQL shapes and localhost SDK enrichment.

The fixed local provider tests transport/persistence, not model quality or an
initialized agent. Every database and provider connection belongs to the test.
"""

import json
import logging
import os
import struct
from uuid import uuid4

import pytest
import asyncpg

from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_add_embedding_profile_id,
    migrate_conversation_history_add_embedding_vec,
)
from tests.utils.postgres_schema import pgvector_schema, quoted_search_path, with_search_path
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_conversation_write_receipts import (
    canonical_store as canonical_store,
    canonical_receipt_database,
    prepare_owned_vector_column,
    observe_actual_insert_attempts,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.utils.native_a2a import native_a2a_backend_factory as native_a2a_backend_factory


@pytest.fixture(params=["sqlite", "postgres"])
async def provider_receipt_store(native_memory_owners, request):
    n = native_memory_owners
    if request.param == "sqlite":
        yield AsyncConversationStore(n.sqlite_saved.db, agent_id=n.did, llm_service=n.service)
    else:
        # Borrow no pool from the shared owner: this fixture owns its recipe,
        # including the actual extension namespace and physical retirement.
        backend = PostgresBackend(
            with_search_path(os.environ["TEST_POSTGRES_URL"], n.schema),
            min_pool_size=1, max_pool_size=2,
        )
        async with canonical_receipt_database(backend) as database:
            yield AsyncConversationStore(database, agent_id=n.did, llm_service=n.service)


@pytest.fixture
async def native_fenced_receipt_storage(native_memory_owners):
    n = native_memory_owners
    extension = await pgvector_schema(n.pg_storage.db)
    dsn = with_search_path(os.environ["TEST_POSTGRES_URL"], quoted_search_path(n.schema, extension))
    raw = AsyncStorage(
        backend=PostgresBackend(dsn, min_pool_size=1, max_pool_size=2),
        agent_id=n.did, llm_service=n.service,
        _assertion_tenant_capability=n.pg_storage._assertion_tenant_capability,
    )
    observer = PostgresBackend(dsn, min_pool_size=1, max_pool_size=2)
    try:
        await raw.initialize()
        await prepare_owned_vector_column(raw.conversation, 3)
        await observer.connect()
        fact = await PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL).save_explicit_fact(
            subject="user", predicate="preferred_deploy_region", value="owned-receipt-region",
            confidence=0.9, invocation_id=uuid4().hex,
        )
        yield raw, observer, {
            "semantic_recall_dependencies": [{
                "assertion_id": fact.assertion_id, "revision_id": fact.revision_id,
            }],
        }
    finally:
        await observer.close()
        await raw.close()


async def _stored_vector(store, receipt):
    expression = "embedding_vec::text" if store.db.backend_type == "postgres" else "embedding_vec"
    row = await store.db.fetchone(
        f"SELECT {expression} FROM conversation_history WHERE id = ? AND agent_id = ?",
        (receipt, store.agent_id),
    )
    assert row is not None, "Receipt must identify the actual canonical row"
    if row[0] is None:
        return None
    if isinstance(row[0], str):
        return json.loads(row[0])
    return list(struct.unpack(f"<{len(row[0]) // 4}f", row[0]))


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [(), ("lexical_index_id", "lexical_index_version"),
    ("embedding_profile_id",), ("embedding_vec",),
    ("lexical_index_id", "lexical_index_version", "embedding_vec", "embedding_profile_id")])
async def test_native_schema_shapes_keep_the_exact_inserted_id(canonical_store, missing):
    store = canonical_store
    vector = [0.1, 0.2, 0.3]
    await prepare_owned_vector_column(store, len(vector))
    if missing and store.db.backend_type == "sqlite":
        indexes = await store.db.fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = 'conversation_history' AND sql IS NOT NULL"
        )
        for (name,) in indexes:
            await store.db.execute_commit('DROP INDEX "' + name.replace('"', '""') + '"')
    for column in missing:
        # Only the parametrized literal column names above enter this DDL.
        await store.db.execute_commit(f"ALTER TABLE conversation_history DROP COLUMN {column}")
    marker = uuid4().hex
    async with store.db.transaction():
        inserted = await store._insert_message(
            role="assistant", content=marker, rendered_content=None, metadata=None,
            embedding=vector, embedding_profile_id="owned-profile", model="owned-model",
            provider="owned-provider", session_id=str(uuid4()),
        )
        # A failed optional shape must not poison the caller's PG transaction.
        assert await store.db.fetchone("SELECT 1") == (1,)
    assert type(inserted.row_id) is int and inserted.row_id > 0
    assert await store.db.fetchone(
        "SELECT content, model, provider FROM conversation_history WHERE id = ? AND agent_id = ?",
        (inserted.row_id, store.agent_id),
    ) == (marker, "owned-model", "owned-provider")
    assert await store.db.fetchone(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?", (store.agent_id,)
    ) == (1,), "Schema retries must not duplicate the canonical row"
    assert inserted.lexical_columns_written == (not any(column in missing for column in ("lexical_index_id", "embedding_vec")))
    if "embedding_vec" not in missing:
        assert await _stored_vector(store, inserted.row_id) == pytest.approx(vector)
        if "embedding_profile_id" not in missing:
            assert await store.db.fetchone(
                "SELECT embedding_profile_id FROM conversation_history WHERE id = ?",
                (inserted.row_id,),
            ) == ("owned-profile",)
    else:
        assert inserted.embedding_drop is not None


@pytest.mark.asyncio
async def test_native_non_schema_insert_error_does_not_retry(canonical_store, monkeypatch):
    store = canonical_store
    attempts = observe_actual_insert_attempts(store, monkeypatch)
    # A real SQL CHECK violation replaces the former fake disk-error response.
    # SQLite's native trigger and PostgreSQL's constraint both reject the row.
    if store.db.backend_type == "postgres":
        await store.db.execute_commit(
            "ALTER TABLE conversation_history ADD CONSTRAINT owned_refusal CHECK (role <> 'assistant')"
        )
    else:
        await store.db.execute_commit(
            "CREATE TRIGGER owned_refusal BEFORE INSERT ON conversation_history "
            "WHEN NEW.role = 'assistant' BEGIN SELECT RAISE(ABORT, 'owned refusal'); END"
        )
    with pytest.raises(Exception, match="owned_refusal|owned refusal"):
        await store.add_conversation("assistant", "rejected write")
    assert len(attempts) == 1, "Non-schema native refusal must not retry its INSERT"
    assert await store.db.fetchone(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?", (store.agent_id,)
    ) == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["healthy", "route-none", "opt-out", "empty-content", "mismatch"])
async def test_actual_provider_write_receipt(native_memory_owners, provider_receipt_store, monkeypatch, caplog, scenario):
    n = native_memory_owners
    store = provider_receipt_store
    await prepare_owned_vector_column(store, 4 if scenario == "mismatch" else 3)
    if scenario == "route-none":
        n.service._embedding_route = "none"
        assert store.embedding_service is None
    elif scenario == "opt-out":
        monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
        assert store.embedding_service is None
    else:
        assert store.embedding_service.client is n.service.providers[0]["client"]
        assert store.embedding_service.current_profile_id() == n.service.get_embedding_service().current_profile_id()
    before = len(n.requests)
    receipts = []
    with caplog.at_level(logging.WARNING):
        for index in range(3 if scenario == "mismatch" else 1):
            receipts.append(await store.add_conversation(
                "assistant", "" if scenario == "empty-content" else f"owned content {index}",
                model="owned-model", provider="openai:local-fixture", session_id=str(uuid4()),
            ))
    assert len(set(receipts)) == len(receipts)
    assert all(type(receipt) is int and receipt > 0 for receipt in receipts)
    for receipt in receipts:
        assert await store.db.fetchone(
            "SELECT model, provider FROM conversation_history WHERE agent_id = ? AND id = ?",
            (store.agent_id, receipt),
        ) == ("owned-model", "openai:local-fixture")
        vector = await _stored_vector(store, receipt)
        if scenario == "healthy":
            assert vector == pytest.approx([1.0, 0.0, 0.0])
            assert await store.db.fetchone(
                "SELECT embedding_profile_id FROM conversation_history WHERE id = ?", (receipt,),
            ) == (store.embedding_service.current_profile_id(),)
        else:
            assert vector is None
    assert len(n.requests) - before == (3 if scenario == "mismatch" else 1 if scenario == "healthy" else 0)
    if scenario == "mismatch":
        assert store.dropped_embedding_count == 3
        assert len([r for r in caplog.records if "dimension mismatch" in r.message.lower()]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("native_memory_owners", ["empty", "unavailable"], indirect=True)
async def test_actual_provider_failure_keeps_the_native_receipt(native_memory_owners, provider_receipt_store):
    n = native_memory_owners
    store = provider_receipt_store
    await prepare_owned_vector_column(store, 3)
    before = len(n.requests)
    receipt = await store.add_conversation("user", "owned enrichment failure", session_id=str(uuid4()))
    assert type(receipt) is int and receipt > 0
    assert len(n.requests) > before, "The actual SDK must reach the owned embedding peer"
    assert all(request["input"] == "owned enrichment failure" for request in n.requests[before:])
    assert await _stored_vector(store, receipt) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("column", ["embedding_vec", "embedding_profile_id"])
async def test_embedding_migrations_inspect_the_original_resolved_relation(native_pg, column):
    backend, _, schema = native_pg
    shadow = "owned_receipt_shadow_" + uuid4().hex
    await backend.execute(f'CREATE SCHEMA "{shadow}"')
    try:
        # This real foreign relation deliberately supplies both column names.
        # Its existence must not satisfy migration of the resolved agent table.
        await backend.execute(
            f'CREATE TABLE "{shadow}".conversation_history '
            "(embedding_vec INTEGER, embedding_profile_id TEXT)"
        )
        owned = PostgresBackend(
            with_search_path(os.environ["TEST_POSTGRES_URL"], schema),
            min_pool_size=1, max_pool_size=2,
        )
        async with canonical_receipt_database(owned) as database:
            assert await database.column_exists("conversation_history", column), (
                f"Foreign schema must not suppress the canonical {column} migration")
    finally:
        await backend.execute(f'DROP SCHEMA "{shadow}" CASCADE')


@pytest.mark.asyncio
async def test_native_migration_reentry_keeps_the_original_row(canonical_store):
    store = canonical_store
    receipt = await store.add_conversation("user", "owned idempotent migration", session_id=str(uuid4()))
    for _ in range(2):
        await migrate_conversation_history_add_embedding_vec(store.db)
        await migrate_add_embedding_profile_id(store.db, table="conversation_history")
    assert await store.db.fetchone(
        "SELECT id FROM conversation_history WHERE agent_id = ?", (store.agent_id,),
    ) == (receipt,)
    assert await store.db.column_exists("conversation_history", "embedding_vec")
    assert await store.db.column_exists("conversation_history", "embedding_profile_id")


class OwnedMigrationFailure(Exception):
    """Failure injected only after the original native ALTER has completed."""


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_after_alter", [True, False])
async def test_native_greenfield_migration_has_atomic_ddl_and_configured_width(
    canonical_store, monkeypatch, fail_after_alter,
):
    store = canonical_store
    suffix = " CASCADE" if store.db.backend_type == "postgres" else ""
    await store.db.execute_commit("ALTER TABLE conversation_history DROP COLUMN embedding_vec" + suffix)
    monkeypatch.setattr(
        "kestrel_sovereign.storage.sqla.conversation_message.CONVERSATION_MESSAGE_EMBEDDING_DIM", 4,
    )
    original = store.db.execute
    injected = OwnedMigrationFailure("owned failure after native ALTER")
    reached = []

    async def fail_after_original_alter(sql, params=()):
        result = await original(sql, params)
        if "ADD COLUMN embedding_vec" in sql:
            reached.append(sql)
            raise injected
        return result

    if fail_after_alter:
        monkeypatch.setattr(store.db, "execute", fail_after_original_alter)
        with pytest.raises(TransactionError) as captured:
            await migrate_conversation_history_add_embedding_vec(store.db)
        assert captured.value.__cause__ is injected and len(reached) == 1
        assert not await store.db.column_exists("conversation_history", "embedding_vec"), (
            "Failed native migration must roll back its completed ALTER")
        monkeypatch.setattr(store.db, "execute", original)
    await migrate_conversation_history_add_embedding_vec(store.db)
    assert await store.db.column_exists("conversation_history", "embedding_vec")
    if store.db.backend_type == "postgres":
        assert await store.db.fetchone(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = to_regclass('conversation_history') "
            "AND attname = 'embedding_vec' AND NOT attisdropped"
        ) == ("vector(4)",)
        row = await store.db.fetchone(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() "
            "AND indexname = 'idx_conversation_history_embedding_vec_hnsw'"
        )
        assert row is not None and "USING hnsw" in row[0] and "vector_cosine_ops" in row[0]
    else:
        assert await store.db.fetchone(
            "SELECT type FROM pragma_table_info('conversation_history') WHERE name = 'embedding_vec'"
        ) == ("BLOB",)


@pytest.mark.asyncio
async def test_native_embedding_migrations_leave_an_absent_relation_absent(canonical_store):
    database = canonical_store.db
    async with database.transaction():
        if database.backend_type == "postgres":
            schema = await database.fetchval("SELECT current_schema()")
            await database.execute('SET LOCAL search_path = "' + schema.replace('"', '""') + '"')
        suffix = " CASCADE" if database.backend_type == "postgres" else ""
        await database.execute("DROP TABLE conversation_history" + suffix)
        await migrate_conversation_history_add_embedding_vec(database)
        await migrate_add_embedding_profile_id(database, table="conversation_history")
        assert not await database.table_exists("conversation_history")


@pytest.mark.asyncio
async def test_registry_outage_cannot_return_a_rolled_back_fenced_row(native_fenced_receipt_storage):
    raw, observer, metadata = native_fenced_receipt_storage
    await raw.db.execute_commit("DROP TABLE embedding_profiles")
    receipt, write_failure = None, None
    try:
        receipt = await raw.add_conversation(
            "assistant", "owned healthy embedding with unavailable registry", metadata=metadata,
            session_id=str(uuid4()),
        )
    except BaseException as error:
        write_failure = error
    assert write_failure is None, "Registry outage must not abort the canonical message's transaction"
    assert type(receipt) is int and receipt > 0
    row = await observer.fetch_one(
        "SELECT id FROM conversation_history WHERE agent_id = ? AND id = ?", (raw.agent_id, receipt),
    )
    assert row == (receipt,), "Fenced receipt must identify an independently committed row despite registry outage"


@pytest.mark.asyncio
async def test_a_swallowed_sql_failure_cannot_publish_a_successful_commit(native_fenced_receipt_storage):
    raw, observer, _ = native_fenced_receipt_storage
    raw.conversation._llm_service = None
    observed = None
    try:
        async with raw.db.transaction():
            await raw.add_conversation("user", "owned poisoned transaction", session_id=str(uuid4()))
            assert not raw.db.backend._current_after_commit().pending
            try:
                await raw.db.execute("SELECT 1 / 0")
            except QueryError:
                pass
    except TransactionError as error:
        observed = error
    assert observed is not None, "Aborted native transaction must never publish a successful commit"
    assert await observer.fetch_one(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?", (raw.agent_id,),
    ) == (0,)


@pytest.mark.asyncio
async def test_lost_fence_commit_ack_keeps_the_committed_rows_lexical_tokens(
    native_fenced_receipt_storage, monkeypatch,
):
    raw, observer, metadata = native_fenced_receipt_storage
    raw.conversation._llm_service = None
    armed, fired = [], []
    original_fetch = raw.db.fetchone
    original_exit = asyncpg.transaction.Transaction.__aexit__

    async def observe_original_insert(sql, params=()):
        result = await original_fetch(sql, params)
        if "INSERT INTO conversation_history" in sql:
            owner = raw.db.backend._current_txn_conn()
            native = owner._con if isinstance(owner, asyncpg.pool.PoolConnectionProxy) else owner
            assert isinstance(native, asyncpg.Connection)
            armed.append(native)
        return result

    async def lose_original_commit_ack(transaction, *args):
        result = await original_exit(transaction, *args)
        if (armed and not fired and transaction._connection is armed[0]
                and not transaction._nested and args[0] is None):
            fired.append(True)
            raise OSError("owned acknowledgement lost after original native COMMIT")
        return result

    monkeypatch.setattr(raw.db, "fetchone", observe_original_insert)
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", lose_original_commit_ack)
    observed = None
    try:
        await raw.add_conversation(
            "assistant", "owned lexical commit acknowledgement witness", metadata=metadata,
            session_id=str(uuid4()),
        )
    except BaseException as error:
        observed = error
    assert len(fired) == 1 and observed is not None and is_execution_control_error(observed)
    row = await observer.fetch_one(
        "SELECT id, lexical_index_id, lexical_index_version FROM conversation_history WHERE agent_id = ?",
        (raw.agent_id,),
    )
    assert row is not None and row[1] and row[2], "The original native COMMIT must remain independently visible"
    tokens = await observer.fetch_one(
        "SELECT COUNT(*) FROM conversation_lexical_tokens WHERE agent_id = ? AND lexical_index_id = ?",
        (raw.agent_id, row[1]),
    )
    assert tokens is not None and tokens[0] > 0, "Lost commit acknowledgement must not delete a committed row's lexical tokens"
