"""Real storage proofs for native context ownership and concurrent admission."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, replace

import pytest
import pytest_asyncio

from kestrel_sovereign.a2a.native_context import NativeRecipientBinding
from kestrel_sovereign.a2a.stores.unified.context_store import (
    ContextOwnership,
    ContextOwnershipError,
    ContextPersistenceError,
    ContextStore,
)
from kestrel_sovereign.storage.db.interface import (
    ConnectionError as DatabaseConnectionError,
)
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError


@pytest_asyncio.fixture
async def store(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    context_store = ContextStore(backend)
    try:
        await context_store.initialize()
        yield context_store
    finally:
        await backend.close()


@pytest.fixture
def binding():
    return NativeRecipientBinding(
        "recipient", "did:kestrel:recipient", "tenant-one", "JSONRPC"
    )


async def test_reserve_is_durable_and_idempotent_for_the_same_owner(store, binding):
    ownership = await store.reserve(
        "context", binding.scope, creator_agent_id="did:kestrel:alice"
    )
    assert ownership == ContextOwnership(
        "context", "tenant-one", "did:kestrel:recipient", "did:kestrel:alice"
    )
    await store.backend.execute(
        "UPDATE a2a_contexts SET created_at = '2020-01-01 00:00:00+00:00'"
    )
    original_row = await store.backend.fetch_one(
        "SELECT creator_agent_id, created_at FROM a2a_contexts"
    )
    assert (
        await store.reserve(
            "context", binding.scope, creator_agent_id="did:kestrel:alice"
        )
        == ownership
    )
    assert (
        await store.require_owned(
            "context", binding.scope, creator_agent_id="did:kestrel:alice"
        )
        == ownership
    )
    assert await store.backend.fetch_all(
        "SELECT creator_agent_id, created_at FROM a2a_contexts"
    ) == [original_row]


async def test_a_different_sender_cannot_claim_an_existing_context(store, binding):
    await store.reserve("context", binding.scope, creator_agent_id="did:kestrel:alice")
    with pytest.raises(ContextOwnershipError, match="not owned") as denied:
        await store.reserve(
            "context", binding.scope, creator_agent_id="did:kestrel:bob"
        )
    assert "alice" not in str(denied.value)
    with pytest.raises(ContextOwnershipError, match="not owned"):
        await store.require_owned(
            "context", binding.scope, creator_agent_id="did:kestrel:bob"
        )
    assert await store.backend.fetch_all(
        "SELECT creator_agent_id FROM a2a_contexts"
    ) == [("did:kestrel:alice",)]


@pytest.mark.parametrize(
    "change",
    [{"tenant": "tenant-two"}, {"tenant": ""}, {"recipient_id": "did:kestrel:other"}],
)
async def test_context_identity_is_scoped_to_the_selected_recipient_and_tenant(
    store, binding, change
):
    await store.reserve(
        "same-context", binding.scope, creator_agent_id="did:kestrel:alice"
    )
    other = replace(binding, **change)
    with pytest.raises(ContextOwnershipError):
        await store.require_owned(
            "same-context", other.scope, creator_agent_id="did:kestrel:alice"
        )
    await store.reserve("same-context", other.scope, creator_agent_id="did:kestrel:bob")
    assert (
        await store.require_owned(
            "same-context", binding.scope, creator_agent_id="did:kestrel:alice"
        )
    ).creator_agent_id == "did:kestrel:alice"
    assert (
        await store.require_owned(
            "same-context", other.scope, creator_agent_id="did:kestrel:bob"
        )
    ).creator_agent_id == "did:kestrel:bob"
    assert (await store.backend.fetch_one("SELECT COUNT(*) FROM a2a_contexts"))[0] == 2


async def test_http_bindings_share_the_same_logical_context(store, binding):
    await store.reserve("context", binding.scope, creator_agent_id="did:kestrel:alice")
    rest = replace(binding, protocol_binding="HTTP+JSON")
    assert await store.reserve(
        "context", rest.scope, creator_agent_id="did:kestrel:alice"
    ) == await store.require_owned(
        "context",
        binding.scope,
        creator_agent_id="did:kestrel:alice",
    )
    with pytest.raises(ContextOwnershipError):
        await store.reserve("context", rest.scope, creator_agent_id="did:kestrel:bob")


async def test_checking_a_missing_context_does_not_create_it(store, binding):
    with pytest.raises(ContextOwnershipError, match="not owned"):
        await store.require_owned(
            "missing", binding.scope, creator_agent_id="did:kestrel:alice"
        )
    assert await store.backend.fetch_all("SELECT * FROM a2a_contexts") == []


async def test_two_independent_database_connections_have_one_winner(
    store, binding, native_a2a_backend_factory
):
    other_backend = native_a2a_backend_factory()
    await other_backend.connect()
    other_store = ContextStore(other_backend)
    try:
        results = await asyncio.gather(
            store.reserve(
                "raced-context", binding.scope, creator_agent_id="did:kestrel:alice"
            ),
            other_store.reserve(
                "raced-context", binding.scope, creator_agent_id="did:kestrel:bob"
            ),
            return_exceptions=True,
        )
        winners = [result for result in results if isinstance(result, ContextOwnership)]
        losers = [
            result for result in results if isinstance(result, ContextOwnershipError)
        ]
        assert len(winners) == len(losers) == 1
        assert await store.backend.fetch_all(
            "SELECT creator_agent_id FROM a2a_contexts"
        ) == [(winners[0].creator_agent_id,)]
    finally:
        await other_backend.close()


async def test_an_independent_connection_sees_the_committed_owner(
    store, binding, native_a2a_backend_factory
):
    await store.reserve(
        "durable-context", binding.scope, creator_agent_id="did:kestrel:alice"
    )
    other_backend = native_a2a_backend_factory()
    await other_backend.connect()
    try:
        assert (
            await ContextStore(other_backend).require_owned(
                "durable-context",
                binding.scope,
                creator_agent_id="did:kestrel:alice",
            )
        ).creator_agent_id == "did:kestrel:alice"
    finally:
        await other_backend.close()


async def test_a_failed_admission_transaction_does_not_claim_the_context(
    store, binding
):
    refusal = ValueError("admission refused")
    with pytest.raises(TransactionError, match="admission refused") as failed:
        async with store.backend.transaction():
            await store.reserve(
                "rolled-back-context",
                binding.scope,
                creator_agent_id="did:kestrel:alice",
            )
            raise refusal

    assert failed.value.__cause__ is refusal
    assert await store.backend.fetch_all("SELECT * FROM a2a_contexts") == []
    assert (
        await store.reserve(
            "rolled-back-context",
            binding.scope,
            creator_agent_id="did:kestrel:bob",
        )
    ).creator_agent_id == "did:kestrel:bob"


async def test_a_rolled_back_transaction_preserves_a_preexisting_owner(store, binding):
    original = await store.reserve(
        "owned-context", binding.scope, creator_agent_id="did:kestrel:alice"
    )
    with pytest.raises(TransactionError) as failed:
        async with store.backend.transaction():
            await store.reserve(
                "new-context", binding.scope, creator_agent_id="did:kestrel:bob"
            )
            await store.reserve(
                "owned-context", binding.scope, creator_agent_id="did:kestrel:bob"
            )

    assert isinstance(failed.value.__cause__, ContextOwnershipError)
    assert (
        await store.require_owned(
            "owned-context", binding.scope, creator_agent_id="did:kestrel:alice"
        )
        == original
    )
    assert await store.backend.fetch_all("SELECT context_id FROM a2a_contexts") == [
        ("owned-context",)
    ]


async def test_opaque_context_ids_are_parameterized_without_normalization(
    store, binding
):
    value = "context'; DROP TABLE a2a_contexts; --"
    assert (
        await store.reserve(value, binding.scope, creator_agent_id="did:kestrel:alice")
    ).context_id == value
    assert await store.backend.fetch_all("SELECT context_id FROM a2a_contexts") == [
        (value,)
    ]


@pytest.mark.parametrize("method", ["reserve", "require_owned"])
@pytest.mark.parametrize("context_id", [None, False, "", 4])
async def test_invalid_context_identity_does_not_write(
    store, binding, context_id, method
):
    with pytest.raises(ValueError, match="context_id"):
        await getattr(store, method)(
            context_id, binding.scope, creator_agent_id="did:kestrel:alice"
        )
    assert await store.backend.fetch_all("SELECT * FROM a2a_contexts") == []


@pytest.mark.parametrize("method", ["reserve", "require_owned"])
@pytest.mark.parametrize("creator", [None, False, "", " ", 4])
async def test_invalid_sender_identity_does_not_write(store, binding, creator, method):
    with pytest.raises(ValueError, match="creator_agent_id"):
        await getattr(store, method)("context", binding.scope, creator_agent_id=creator)
    assert await store.backend.fetch_all("SELECT * FROM a2a_contexts") == []


async def test_a_silently_ignored_write_cannot_acknowledge_a_reservation(
    store, binding
):
    if store.is_postgres:
        await store.backend.execute_script("""
            CREATE FUNCTION ignore_native_context() RETURNS trigger
            LANGUAGE plpgsql AS $$ BEGIN RETURN NULL; END; $$;
            CREATE TRIGGER ignore_native_context BEFORE INSERT ON a2a_contexts
            FOR EACH ROW EXECUTE FUNCTION ignore_native_context();
        """)
    else:
        await store.backend.execute_script("""
            CREATE TRIGGER ignore_native_context BEFORE INSERT ON a2a_contexts
            BEGIN SELECT RAISE(IGNORE); END;
        """)
    with pytest.raises(ContextPersistenceError, match="not persisted"):
        await store.reserve(
            "context", binding.scope, creator_agent_id="did:kestrel:alice"
        )
    assert await store.backend.fetch_all("SELECT * FROM a2a_contexts") == []


@pytest.mark.parametrize(
    "method,error",
    [("reserve", DatabaseConnectionError), ("require_owned", QueryError)],
)
async def test_storage_failure_propagates_without_an_in_memory_reservation(
    store, binding, method, error
):
    await store.reserve("context", binding.scope, creator_agent_id="did:kestrel:alice")
    await store.backend.close()
    expected = DatabaseConnectionError if store.is_postgres else error
    with pytest.raises(expected, match="Not connected") as failed:
        await getattr(store, method)(
            "context", binding.scope, creator_agent_id="did:kestrel:alice"
        )
    if method == "require_owned" and not store.is_postgres:
        assert isinstance(failed.value.__cause__, DatabaseConnectionError)


async def test_initializing_the_schema_again_preserves_ownership(store, binding):
    expected = await store.reserve(
        "context", binding.scope, creator_agent_id="did:kestrel:alice"
    )
    await store.initialize()
    assert (
        await store.require_owned(
            "context", binding.scope, creator_agent_id="did:kestrel:alice"
        )
        == expected
    )
    with pytest.raises(ContextOwnershipError):
        await store.reserve(
            "context", binding.scope, creator_agent_id="did:kestrel:bob"
        )


async def test_an_ownership_receipt_cannot_be_reassigned_by_a_caller(store, binding):
    ownership = await store.reserve(
        "context", binding.scope, creator_agent_id="did:kestrel:alice"
    )
    with pytest.raises(FrozenInstanceError):
        ownership.creator_agent_id = "did:kestrel:bob"


async def test_default_tenant_has_the_same_ownership_guarantee(store, binding):
    default = replace(binding, tenant="")
    assert (
        await store.reserve(
            "context", default.scope, creator_agent_id="did:kestrel:alice"
        )
    ).tenant == ""
    with pytest.raises(ContextOwnershipError):
        await store.reserve(
            "context", default.scope, creator_agent_id="did:kestrel:bob"
        )


@pytest.mark.parametrize("method", ["reserve", "require_owned"])
@pytest.mark.parametrize("scope", [None, {}, "tenant"])
async def test_an_unbound_scope_cannot_create_or_read_contexts(store, method, scope):
    with pytest.raises(TypeError, match="server-owned native recipient scope"):
        await getattr(store, method)(
            "context", scope, creator_agent_id="did:kestrel:alice"
        )
    assert await store.backend.fetch_all("SELECT * FROM a2a_contexts") == []
