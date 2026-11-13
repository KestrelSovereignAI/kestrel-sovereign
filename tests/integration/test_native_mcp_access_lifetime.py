"""Original SQL authorization ownership spans private delivery and revocation."""

import asyncio

import pytest
from mcp.server.auth.provider import TokenError

from kestrel_sovereign.mcp_server.records import MCPRecordKind
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.storage.db.interface import TransactionError
from kestrel_sovereign.storage.db.postgres import PostgresBackend
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_native_mcp_grants import RESOURCE, pair, store_with_policy
from tests.integration.test_native_mcp_grants import consent_http as _consent_http
from tests.integration.test_native_mcp_grants import native_pg as _native_pg

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
consent_http = _consent_http
native_pg = _native_pg


async def test_independent_sql_owner_cannot_acknowledge_revocation_during_access(consent_http, native_pg):
    pending, keys, grants = consent_http[1], consent_http[4], consent_http[6]
    code, token = await pair(consent_http)
    backend = grants.backend
    if isinstance(backend, SQLiteBackend):
        peer_backend = SQLiteBackend(backend.db_path)
        await peer_backend.connect()
    else:
        peer_backend = PostgresBackend.from_pool(backend._pool, advisory_max_pool_size=1)
    peer_pending = type(pending)(peer_backend, pending.agent_id, pending.resource_url,
        pending.policy, clock=pending.clock)
    peer = store_with_policy(peer_pending, keys, grants, pending.policy)
    revocation = None
    started = asyncio.Event()

    async def revoke():
        started.set()
        await peer.revoke(MCPRecordKind.ACCESS, token.access_token,
            resource=RESOURCE, client_id=code.client_id)

    try:
        async with grants.access_lifetime(token.access_token, resource=RESOURCE, client_id=code.client_id) as current:
            assert current().token == token.access_token
            pid = (await backend.fetch_one("SELECT pg_backend_pid()"))[0] if not isinstance(backend, SQLiteBackend) else None
            revocation = asyncio.create_task(revoke())
            await started.wait()
            if pid is not None:
                control = native_pg[1]
                async with asyncio.timeout(5):
                    while True:
                        await control.execute("SELECT pg_stat_clear_snapshot()")
                        if await control.fetchval("SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                            "WHERE wait_event_type='Lock' AND $1=ANY(pg_blocking_pids(pid)))", pid):
                            break
                        await asyncio.sleep(0.01)
            else:
                await asyncio.wait({revocation}, timeout=0.1)
            assert not revocation.done(), "Independent SQL revocation acknowledged before original delivery retired"
            assert current().token == token.access_token
        await asyncio.wait_for(revocation, timeout=5)
        assert await grants.load(MCPRecordKind.ACCESS, token.access_token, resource=RESOURCE) is None
        with pytest.raises(HostCapabilityAccessDenied, match="lifetime"):
            current()
    finally:
        if revocation is not None and not revocation.done():
            revocation.cancel()
            await asyncio.gather(revocation, return_exceptions=True)
        await peer_backend.close()


async def test_access_reader_cannot_escape_its_original_task_or_reactivate_in_later_transaction(consent_http):
    grants = consent_http[6]
    code, token = await pair(consent_http)
    async with grants.access_lifetime(token.access_token, resource=RESOURCE, client_id=code.client_id) as current:
        async def copied_child():
            with pytest.raises(HostCapabilityAccessDenied, match="lifetime"):
                current()
        await asyncio.create_task(copied_child())
        assert current().token == token.access_token
    async with grants.backend.transaction():
        with pytest.raises(HostCapabilityAccessDenied, match="lifetime"):
            current()


@pytest.mark.parametrize("change", ["expiry", "sovereign-key"])
async def test_access_reader_rechecks_current_authority_before_release(consent_http, change):
    grants, clock, keys = consent_http[6], consent_http[5], consent_http[4]
    code, token = await pair(consent_http)
    with pytest.raises(TransactionError, match="grant") as refusal:
        async with grants.access_lifetime(token.access_token, resource=RESOURCE, client_id=code.client_id) as current:
            assert current().token == token.access_token
            if change == "expiry":
                clock[0] = current().expires_at
            else:
                keys[0] = "owned-retired-sovereign-key"
            with pytest.raises(HostCapabilityAccessDenied, match="grant"):
                current()
    assert isinstance(refusal.value.__cause__, HostCapabilityAccessDenied)


async def test_access_lifetime_refuses_foreign_resource_client_token_and_outer_transaction(consent_http):
    grants = consent_http[6]
    code, token = await pair(consent_http)
    with pytest.raises(TokenError) as target:
        async with grants.access_lifetime(token.access_token, resource=RESOURCE + "/other", client_id=code.client_id):
            pytest.fail("Foreign resource acquired original access lifetime")
    assert target.value.error == "invalid_target"
    for secret, client in ((token.access_token, "foreign-client"), ("malformed-token", code.client_id)):
        with pytest.raises(TransactionError, match="grant") as refusal:
            async with grants.access_lifetime(secret, resource=RESOURCE, client_id=client):
                pytest.fail("Foreign credential acquired original access lifetime")
        assert isinstance(refusal.value.__cause__, HostCapabilityAccessDenied)
    async with grants.backend.transaction():
        with pytest.raises(RuntimeError, match="top-level"):
            async with grants.access_lifetime(token.access_token, resource=RESOURCE, client_id=code.client_id):
                pytest.fail("Access delivery joined an enclosing uncommitted transaction")
