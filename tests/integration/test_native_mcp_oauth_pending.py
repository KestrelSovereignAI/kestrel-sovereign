"""Original SQL transactions bound pending consent; this is not an OAuth server acceptance claim."""

import asyncio
import json
from contextlib import AsyncExitStack

import pytest
from kestrel_sdk.security.exceptions import DecryptionError
from mcp.server.auth.provider import AuthorizationParams

from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.mcp_server.pending import (
    PendingMCPAuthorizations,
    PendingMCPQuotaExceeded,
)
from kestrel_sovereign.mcp_server.policy import MCPServerPolicy
from kestrel_sovereign.storage.db.interface import TransactionError
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.fixture(params=["sqlite", "postgres"])
async def pending_store(request, native_pg, tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "o" * 32)
    policy = MCPServerPolicy.model_validate_json(json.dumps({
        "console_origin": "https://kestrel.example", "max_pending_per_agent": 1,
        "clients": [{"client_id": "owned-provider", "client_name": "Owned provider",
            "redirect_uris": ["https://provider.example/callback"],
            "client_secret": "owned-fixture-secret"}]}))
    async with AsyncExitStack() as owners:
        if request.param == "sqlite":
            backend = SQLiteBackend(str(tmp_path / "pending.db"))
            owners.push_async_callback(backend.close)
            await backend.connect()
            peer_backend = SQLiteBackend(str(tmp_path / "pending.db"))
            owners.push_async_callback(peer_backend.close)
            await peer_backend.connect()
        else:
            backend = native_pg[0]
            peer_backend = PostgresBackend.from_pool(backend._pool, advisory_max_pool_size=1)
            owners.push_async_callback(peer_backend.close)
        clock = [1000.0]
        store = PendingMCPAuthorizations(backend, "did:owned:mcp-agent",
            "https://kestrel.example/mcp/owned", policy, clock=lambda: clock[0])
        peer = PendingMCPAuthorizations(peer_backend, store.agent_id, store.resource_url,
            policy, clock=store.clock)
        await store.initialize()
        await peer.initialize()
        yield store, clock, peer


def parameters():
    return AuthorizationParams(state="owned-client-state", scopes=["memory:read"],
        code_challenge="a" * 43, redirect_uri="https://provider.example/callback",
        redirect_uri_provided_explicitly=True, resource="https://kestrel.example/mcp/owned")


async def test_pending_native_request_is_encrypted_once_and_scoped_to_its_agent(pending_store):
    store, _clock, _peer = pending_store
    original = parameters()
    handle = await store.create("owned-provider", original)
    original.state = "later-caller-mutation"
    row = await store.backend.fetch_one("SELECT handle_hash, ciphertext FROM mcp_server_oauth_pending")
    assert handle not in row and "owned-client-state" not in row[1]
    pending = await store.read(handle)
    assert pending.parameters.state == "owned-client-state"
    other = PendingMCPAuthorizations(store.backend, "did:owned:other-agent", store.resource_url,
        store.policy, clock=store.clock)
    await other.initialize()
    assert await other.read(handle) is None and await other.claim(handle) is None
    claimed = await store.claim(handle)
    assert claimed.client_id == "owned-provider" and claimed.parameters.state == "owned-client-state"
    assert await store.claim(handle) is None and await store.read(handle) is None


async def test_two_native_creators_cannot_overrun_the_original_agent_quota(pending_store):
    store, _clock, peer = pending_store
    results = await asyncio.gather(store.create("owned-provider", parameters()),
        peer.create("owned-provider", parameters()), return_exceptions=True)
    assert sum(isinstance(value, str) for value in results) == 1
    assert sum(isinstance(value, PendingMCPQuotaExceeded) for value in results) == 1
    assert await store.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_pending") == (1,)


async def test_missing_native_capacity_record_cannot_admit_authorization(pending_store):
    store, _clock, _peer = pending_store
    await store.backend.execute("DELETE FROM mcp_server_oauth_capacity WHERE agent_id = ?", (store.agent_id,))
    caught = None
    try:
        await store.create("owned-provider", parameters())
    except TransactionError as error:
        caught = error
    assert caught is not None and "capacity record" in str(caught), (
        "Missing native capacity ownership admitted an OAuth request without its quota lock"
    )
    assert await store.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_pending") == (0,)


async def test_two_native_consent_owners_receive_only_one_original_request(pending_store):
    store, _clock, peer = pending_store
    handle = await store.create("owned-provider", parameters())
    results = await asyncio.gather(store.claim(handle), peer.claim(handle))
    assert sum(value is not None for value in results) == 1, (
        "The original native pending OAuth request was claimed more than once"
    )
    claimed, = [value for value in results if value is not None]
    assert claimed.parameters.state == "owned-client-state"


@pytest.mark.parametrize("tampering", ["agent", "expiry"])
async def test_native_pending_cipher_binds_original_agent_and_expiry(pending_store, tampering):
    store, _clock, _peer = pending_store
    handle = await store.create("owned-provider", parameters())
    selected = store
    if tampering == "agent":
        selected = PendingMCPAuthorizations(store.backend, "did:owned:other-agent", store.resource_url,
            store.policy, clock=store.clock)
        await selected.initialize()
        await store.backend.execute("UPDATE mcp_server_oauth_pending SET agent_id = ? WHERE agent_id = ?",
            (selected.agent_id, store.agent_id))
    else:
        await store.backend.execute("UPDATE mcp_server_oauth_pending SET expires_at = expires_at + 1 WHERE agent_id = ?",
            (store.agent_id,))
    with pytest.raises(DecryptionError):
        await selected.read(handle)


async def test_native_expiry_and_rollback_never_claim_an_old_consent_request(pending_store):
    store, clock, _peer = pending_store
    handle = await store.create("owned-provider", parameters())
    original = RuntimeError("owned-consent-rollback")
    with pytest.raises(TransactionError, match="owned-consent-rollback") as failure:
        async with store.backend.transaction():
            assert await store.claim(handle) is not None
            raise original
    assert original in tuple(_explicit_control_chain(failure.value))
    assert await store.read(handle) is not None
    clock[0] = 1600.0
    assert await store.read(handle) is None and await store.claim(handle) is None
    replacement = await store.create("owned-provider", parameters())
    assert replacement != handle and await store.read(replacement) is not None


@pytest.mark.parametrize("field,value", [("resource", "https://other.example/mcp"),
    ("scopes", ["memory:write"]), ("scopes", None)])
async def test_native_pending_store_refuses_unregistered_scope_or_resource(pending_store, field, value):
    store, _clock, _peer = pending_store
    request = parameters().model_copy(update={field: value})
    with pytest.raises(ValueError):
        await store.create("owned-provider", request)
    assert await store.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_pending") == (0,)
