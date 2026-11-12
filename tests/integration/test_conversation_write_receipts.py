"""Exact canonical write identity on native SQLite and PostgreSQL.

These are storage component contracts, not MCP server/client acceptance.
The shared backend factory supplies independent connections to one private
database scope, with no operator database or embedding/model service.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_conversation_store import (
    AsyncConversationStore, ConversationWriteIdentityError,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.features.privacy.feature import PrivacyAgent
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.db.interface import TransactionError
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from tests.utils.postgres_schema import pgvector_schema, quoted_search_path, with_search_path
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage, PrivacyViolationError
from tests.integration.test_storage_backend_parity import _incepted_assertion_identity
from tests.utils.native_a2a import native_a2a_backend_factory as native_a2a_backend_factory


@asynccontextmanager
async def canonical_receipt_database(backend):
    """Own a native database whose pool can resolve its actual vector type."""
    database = None
    try:
        await backend.connect()
        if backend.backend_type == "postgres":
            # Journal fixtures intentionally hide other schemas. Resolve the
            # real extension namespace and include it on this owned recipe.
            probe = AsyncDatabase(backend)
            extension = await pgvector_schema(probe)
            schema = await probe.fetchval("SELECT current_schema()")
            dsn = with_search_path(backend._dsn, quoted_search_path(schema, extension))
            await backend.close()
            backend = PostgresBackend(dsn, min_pool_size=1, max_pool_size=2)
            await backend.connect()
        database = await AsyncDatabase.from_connected_backend(backend)
        yield database
    finally:
        await (database.close() if database is not None else backend.close())


async def prepare_owned_vector_column(store, dimension):
    """Declare the real vector width on this empty test-owned table."""
    assert type(dimension) is int and dimension > 0
    if store.db.backend_type == "postgres":
        assert await store.db.fetchone("SELECT COUNT(*) FROM conversation_history") == (0,)
        await store.db.execute_commit(
            "ALTER TABLE conversation_history ALTER COLUMN embedding_vec "
            f"TYPE vector({dimension}) USING embedding_vec::vector({dimension})"
        )
    store._embedding_column_dim = dimension


@pytest_asyncio.fixture
async def canonical_store(native_a2a_backend_factory):
    async with canonical_receipt_database(native_a2a_backend_factory()) as database:
        yield AsyncConversationStore(database, agent_id="did:test:receipt:" + uuid4().hex)


@pytest_asyncio.fixture
async def receipt_storage(native_a2a_backend_factory, tmp_path):
    did, identity = await _incepted_assertion_identity(tmp_path, "conversation-receipts")
    storage = AsyncStorage(
        backend=native_a2a_backend_factory(), agent_id=did,
        _assertion_tenant_capability=_resolve_authenticated_agent_assertion_capability(did, identity),
    )
    try:
        await storage.initialize()
        yield storage
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_native_transaction_insert_returns_its_original_row_id(canonical_store):
    """Prove the existing backend API before changing the canonical writer."""
    store = canonical_store
    async with store.db.transaction(savepoint=True):
        row = await store.db.fetchone(
            "INSERT INTO conversation_history (agent_id, role, content, created_at) "
            f"VALUES (?, ?, ?, {store._now_sql()}) RETURNING id",
            (store.agent_id, "user", "owned primitive probe"),
        )
        assert row is not None and type(row[0]) is int and row[0] > 0
    assert store.db.owns_open_transaction is False
    assert await store.db.fetchone(
        "SELECT id FROM conversation_history WHERE agent_id = ? AND id = ?",
        (store.agent_id, row[0]),
    ) == (row[0],)


@pytest.mark.asyncio
async def test_canonical_writer_returns_the_exact_persisted_message_id(
    canonical_store, native_a2a_backend_factory,
):
    store = canonical_store
    marker = uuid4().hex
    receipt = await store.add_conversation(
        "user", "same content is not a row identity", metadata={"write_marker": marker},
        session_id=str(uuid4()),
    )
    assert type(receipt) is int and receipt > 0, "Canonical write must return its own inserted message ID"
    observer = native_a2a_backend_factory()
    await observer.connect()
    try:
        row = await observer.fetch_one(
            "SELECT id, agent_id, role, metadata FROM conversation_history WHERE id = ?",
            (receipt,),
        )
        assert row is not None and row[:3] == (receipt, store.agent_id, "user")
        assert json.loads(row[3])["write_marker"] == marker
    finally:
        await observer.close()


@pytest.mark.asyncio
async def test_concurrent_identical_messages_keep_their_original_write_receipts(canonical_store):
    store = canonical_store
    markers = [uuid4().hex for _ in range(3)]
    receipts = await asyncio.gather(*[
        store.add_conversation(
            "user", "identical concurrent message", metadata={"write_marker": marker},
            session_id=str(uuid4()),
        ) for marker in markers
    ])
    assert all(type(value) is int and value > 0 for value in receipts), (
        "Every canonical write must retain its own inserted message ID")
    assert len(set(receipts)) == len(markers), "Concurrent writes must not share a latest-row lookup"
    for receipt, marker in zip(receipts, markers, strict=True):
        row = await store.db.fetchone(
            "SELECT metadata FROM conversation_history WHERE agent_id = ? AND id = ?",
            (store.agent_id, receipt),
        )
        assert row is not None and json.loads(row[0])["write_marker"] == marker


@pytest.mark.asyncio
@pytest.mark.parametrize("layer", ["raw", "normal", "public", "anonymous", "agent"])
@pytest.mark.parametrize("lineage", ["plain", "empty", "malformed", "current", "stale"])
async def test_facades_return_their_exact_canonical_write(receipt_storage, layer, lineage):
    raw = receipt_storage
    if layer == "agent":
        store = PrivacyAgent(PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL))
    else:
        store = raw if layer == "raw" else PrivacyEnforcingStorage(raw, PrivacyMode(layer))
    metadata = {"write_marker": uuid4().hex}
    if lineage in {"current", "stale"}:
        fact = await PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL).save_explicit_fact(
            subject="user", predicate="preferred_deploy_region", value="owned-receipt-region",
            confidence=0.9, invocation_id=uuid4().hex,
        )
        metadata["semantic_recall_dependencies"] = [{
            "assertion_id": fact.assertion_id,
            "revision_id": fact.revision_id if lineage == "current" else uuid4().hex,
        }]
    elif lineage != "plain":
        metadata["semantic_recall_dependencies"] = [] if lineage == "empty" else "invalid"
    receipt = await store.add_conversation(
        "assistant", "owned facade receipt", metadata=metadata, session_id=str(uuid4()),
    )
    assert type(receipt) is int and receipt > 0, "Every persistent facade must return the canonical row ID"
    row = await raw.db.fetchone(
        "SELECT metadata FROM conversation_history WHERE agent_id = ? AND id = ?",
        (raw.agent_id, receipt),
    )
    assert row is not None and json.loads(row[0])["write_marker"] == metadata["write_marker"]
    assert bool(json.loads(row[0]).get("excluded_from_context")) == (lineage in {"malformed", "stale"})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.ISOLATED])
async def test_volatile_facade_does_not_invent_a_persistent_receipt(receipt_storage, mode):
    store = PrivacyEnforcingStorage(receipt_storage, mode)
    assert await store.add_conversation("user", "owned volatile content") is None
    assert await receipt_storage.db.fetchone(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?", (receipt_storage.agent_id,),
    ) == (0,)
    assert len(store._session_conversations) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.EPHEMERAL, PrivacyMode.DEIDENTIFIED])
async def test_volatile_facade_refuses_a_canonical_write(receipt_storage, mode):
    store = PrivacyEnforcingStorage(receipt_storage, mode)
    with pytest.raises(PrivacyViolationError, match="ephemeral|deidentified"):
        await store.add_conversation("user", "owned refused content")
    assert await receipt_storage.db.fetchone(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?", (receipt_storage.agent_id,),
    ) == (0,)


class OwnedRollback(Exception):
    """Explicit rollback requested by this test's transaction owner."""


@pytest.mark.asyncio
@pytest.mark.parametrize("commit", [True, False])
async def test_receipt_remains_provisional_until_the_caller_commits(
    canonical_store, native_a2a_backend_factory, commit,
):
    store = canonical_store
    observer = native_a2a_backend_factory()
    await observer.connect()
    try:
        try:
            async with store.db.transaction():
                receipt = await store.add_conversation("user", "owned outer transaction", session_id=str(uuid4()))
                assert type(receipt) is int and receipt > 0
                assert await observer.fetch_one(
                    "SELECT id FROM conversation_history WHERE id = ? AND agent_id = ?",
                    (receipt, store.agent_id),
                ) is None, "Returning an ID must not commit the caller's transaction"
                if not commit:
                    raise OwnedRollback()
        except TransactionError as error:
            assert not commit and isinstance(error.__cause__, OwnedRollback)
        row = await observer.fetch_one(
            "SELECT id FROM conversation_history WHERE id = ? AND agent_id = ?",
            (receipt, store.agent_id),
        )
        assert row == ((receipt,) if commit else None)
    finally:
        await observer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [False, True])
async def test_native_insert_without_a_row_cannot_return_a_receipt(canonical_store, monkeypatch, vector):
    store = canonical_store
    if store.db.backend_type == "postgres":
        if vector:
            await prepare_owned_vector_column(store, 1)
        await store.db.execute_commit(
            "CREATE FUNCTION owned_suppress_row() RETURNS trigger LANGUAGE plpgsql "
            "AS $$ BEGIN RETURN NULL; END $$"
        )
        await store.db.execute_commit(
            "CREATE TRIGGER owned_suppress BEFORE INSERT ON conversation_history "
            "FOR EACH ROW EXECUTE FUNCTION owned_suppress_row()"
        )
    else:
        await store.db.execute_commit(
            "CREATE TRIGGER owned_suppress BEFORE INSERT ON conversation_history "
            "BEGIN SELECT RAISE(IGNORE); END"
        )
    inserts = []
    original_fetch = store.db.fetchone

    async def observe_original_insert(sql, params=()):
        if "INSERT INTO conversation_history" in sql:
            inserts.append(sql)
        return await original_fetch(sql, params)

    monkeypatch.setattr(store.db, "fetchone", observe_original_insert)
    observed = None
    try:
        if vector:
            await store._insert_message(
                role="user", content="owned suppressed vector insert", rendered_content=None,
                metadata=None, embedding=[0.1], embedding_profile_id="owned-profile",
            )
        else:
            await store.add_conversation("user", "owned suppressed insert", session_id=str(uuid4()))
    except TransactionError as error:
        observed = error.__cause__
    assert isinstance(observed, ConversationWriteIdentityError), "Missing native row must fail with the canonical identity error"
    assert len(inserts) == 1, "Missing row identity must never authorize schema retry"
    assert await store.db.fetchone(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?", (store.agent_id,),
    ) == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_id", [0, -1])
async def test_native_nonpositive_id_is_rolled_back_before_return(canonical_store, invalid_id):
    store = canonical_store
    observed = None
    try:
        await store._execute_confined(
            "INSERT INTO conversation_history (id, agent_id, role, content, created_at) "
            f"VALUES (?, ?, ?, ?, {store._now_sql()})",
            (invalid_id, store.agent_id, "user", "owned invalid identity"), returning_id=True,
        )
    except TransactionError as error:
        observed = error.__cause__
    assert isinstance(observed, ConversationWriteIdentityError), "Nonpositive native ID must be rejected before commit"
    assert await store.db.fetchone(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?", (store.agent_id,),
    ) == (0,)


@pytest.mark.asyncio
async def test_unbound_derivative_still_returns_its_excluded_row_id(
    receipt_storage, native_a2a_backend_factory,
):
    raw = AsyncStorage(backend=native_a2a_backend_factory(), agent_id=receipt_storage.agent_id)
    try:
        await raw.initialize()
        receipt = await raw.add_conversation(
            "assistant", "owned unbound derivative", session_id=str(uuid4()),
            metadata={"semantic_recall_dependencies": [{
                "assertion_id": uuid4().hex, "revision_id": uuid4().hex,
            }]},
        )
        assert type(receipt) is int and receipt > 0, "Excluded unbound write must retain its canonical row receipt"
        row = await raw.db.fetchone(
            "SELECT metadata FROM conversation_history WHERE agent_id = ? AND id = ?",
            (raw.agent_id, receipt),
        )
        assert row is not None and json.loads(row[0])["excluded_from_context"] is True
    finally:
        await raw.close()
