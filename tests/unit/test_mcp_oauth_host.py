"""Actual Core feature activation and HTTP credentials govern original OAuth leases."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx2
import pytest
import pytest_asyncio
from kestrel_sdk.features import Feature
from kestrel_sdk.payer_policy import PayerKind, PayerPolicy, PayerSpec

from kestrel_sovereign import server
from kestrel_sovereign.auth import caller_context_scope
from kestrel_sovereign.endpoints.auth_oauth import _create_jwt
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.features.privacy import PrivacyAgent
from kestrel_sovereign.mcp_oauth import (
    MCPOAuthAction, MCPOAuthConnectionLease, MCPOAuthHost, MCPOAuthHostPolicy,
)
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.request_auth import (
    AuthenticatedSessionMiddleware, InteractiveAuthenticationLifetimeMiddleware, RequestAuthenticationService,
)
from kestrel_sovereign.security.private_delivery import PrivateDeliveryMiddleware


API_KEY = "owned-oauth-host-api-key"
ALICE = "alice@oauth-host.test"


class OwnedOAuthFeature(Feature):
    @property
    def tool_description(self):
        return "Owned Core OAuth host contract consumer"

    async def initialize(self):
        return None


@pytest_asyncio.fixture
async def oauth_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)
    monkeypatch.setenv("JWT_SECRET_KEY", "owned-oauth-host-jwt-key-with-enough-entropy")
    monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "1")
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", ALICE)
    host_policy = PayerPolicy.host_env_default()
    payer_policy = PayerPolicy(llm=PayerSpec(vendor="openrouter", kind=PayerKind.NONE),
        storage=host_policy.storage, compute=host_policy.compute, tools=host_policy.tools,
        comms=host_policy.comms)
    agent = KestrelAgent(did="did:test:owned-oauth-host", storage_path=str(tmp_path / "agent.db"),
                         allowed_features=set(), sync_enabled=False, payer_policy=payer_policy)
    agent.privacy_agent = PrivacyAgent(agent._raw_storage, agent.privacy_mode)
    feature = OwnedOAuthFeature(agent)
    current = [True]
    await agent._register_feature(feature)
    host = agent.create_mcp_oauth_host(feature, is_current=lambda: current[0])
    clock = [datetime.now(UTC)]
    host.authority.clock = lambda: clock[0]
    app = FastAPI()
    app.state.agent = agent
    app.state.mcp_oauth_policy = MCPOAuthHostPolicy(console_origin="https://console.example")
    app.state.request_authentication = RequestAuthenticationService(
        api_key=lambda: server.get_api_key(generate_if_missing=False), clock=lambda: clock[0])
    app.add_middleware(InteractiveAuthenticationLifetimeMiddleware)
    app.middleware("http")(server.auth_middleware)
    app.add_middleware(AuthenticatedSessionMiddleware, secret_key="owned-oauth-session-key",
                       session_cookie="kestrel_session", max_age=600)
    app.add_middleware(PrivateDeliveryMiddleware)
    leases, contexts, issue_errors, writers = [], [], [], []

    @app.get("/api/owned-identity")
    async def identity(request: Request):
        try:
            async with host._resolve_context(request) as context:
                async with host._operation(context, MCPOAuthAction.READ):
                    if request.query_params.get('mode') == 'copied-task':
                        return await asyncio.create_task(async_identity(context))
                    value = host.client_identity(context)
                    value['client_document']['redirect_uris'].append('https://unrelated.example/callback')
                    return host.client_identity(context)
        except HostCapabilityAccessDenied:
            return JSONResponse({'error': 'access_denied'}, status_code=403)

    async def async_identity(context):
        return host.client_identity(context)

    @app.get("/api/owned-issue")
    async def issue(request: Request):
        action = MCPOAuthAction.READ if request.query_params.get("action") == "read" else MCPOAuthAction.CONNECT
        try:
            async with host._resolve_context(request) as context:
                async with host._operation(context, action):
                    if request.query_params.get("mode") == "copied-task":
                        async def copied_issuer():
                            return host.issue_lease(context, connection_id="owned-connection", slot_id="owned-slot")
                        lease = await asyncio.create_task(copied_issuer())
                    elif request.query_params.get("mode") == "unregistered":
                        lease = MCPOAuthConnectionLease(host, host._reader.get(),
                            connection_id="owned-connection", slot_id="owned-slot")
                        try:
                            lease.require_background()
                        finally:
                            lease.revoke()
                    else:
                        lease = host.issue_lease(context, connection_id="owned-connection", slot_id="owned-slot")
                    leases.append(lease)
                    contexts.append(context)
                    if request.query_params.get("mode") == "nested-credential":
                        queued = asyncio.Event()
                        async def authority_writer():
                            queued.set()
                            async with host._gate:
                                current[0] = False
                        writers.append(asyncio.create_task(authority_writer()))
                        await queued.wait()
                        async with asyncio.timeout(1):
                            async with lease.credential_operation() as check:
                                check()
                    return {"issued": True}
        except HostCapabilityAccessDenied as error:
            issue_errors.append(str(error))
            return JSONResponse({"error": "access_denied"}, status_code=403)

    @app.get("/api/owned-use")
    async def use(request: Request):
        try:
            with caller_context_scope(request.state.caller):
                if request.query_params.get("mode") == "credential":
                    async with leases[0].credential_operation() as check:
                        check()
                else:
                    leases[0].require_principal()
            return {"admitted": True}
        except HostCapabilityAccessDenied:
            return JSONResponse({"error": "access_denied"}, status_code=403)

    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app),
                                 base_url="https://console.example") as client:
        try:
            yield SimpleNamespace(app=app, client=client, agent=agent, feature=feature,
                host=host, leases=leases, contexts=contexts, clock=clock, current=current, issue_errors=issue_errors)
        finally:
            host.withdraw()
            if writers:
                await asyncio.gather(*writers)
            if agent.feature_contribution_runtime.is_active(feature):
                await agent._unregister_feature_runtime(feature, unload=True)
            await agent.llm_service.close()


async def issue(boundary):
    response = await boundary.client.get("/api/owned-issue", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200, (response.text, boundary.issue_errors)
    return boundary.leases[0]


@pytest.mark.asyncio
async def test_connection_lease_survives_request_end_with_separate_current_credential_proof(oauth_boundary):
    b = oauth_boundary
    lease = await issue(b)
    assert lease.agent_id == b.agent.did and lease.connection_id == "owned-connection"
    assert lease.slot_id == "owned-slot" and lease.principal_id == b.contexts[0].principal_id
    lease.require_background()
    with pytest.raises(HostCapabilityAccessDenied):
        lease.require_principal()
    with pytest.raises(HostCapabilityAccessDenied):
        b.host.issue_lease(b.contexts[0], connection_id="forged", slot_id="forged")


@pytest.mark.asyncio
async def test_current_principal_renews_the_original_connection_lease(oauth_boundary):
    b = oauth_boundary
    lease = await issue(b)
    original = lease.expires_at
    b.clock[0] += timedelta(seconds=30)
    response = await b.client.get("/api/owned-use", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200 and response.json() == {"admitted": True}
    assert lease.expires_at > original
    lease.require_background()


@pytest.mark.asyncio
async def test_other_native_principal_cannot_renew_or_use_the_lease(oauth_boundary):
    b = oauth_boundary
    lease = await issue(b)
    original = lease.expires_at
    response = await b.client.get("/api/owned-use", headers={"Authorization": "Bearer " + _create_jwt(ALICE)})
    assert response.status_code == 403 and lease.expires_at == original
    lease.require_background()


@pytest.mark.asyncio
async def test_background_credential_use_is_fenced_immediately_by_native_key_rotation(oauth_boundary, monkeypatch):
    lease = await issue(oauth_boundary)
    monkeypatch.setenv("KESTREL_API_KEY", "rotated-owned-oauth-key")
    with pytest.raises(HostCapabilityAccessDenied):
        lease.require_background()


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawal", ["host", "incarnation", "expiry", "feature", "privacy", "publication"])
async def test_original_owner_lifecycle_and_expiry_fence_background_credentials(oauth_boundary, withdrawal):
    b = oauth_boundary
    lease = await issue(b)
    if withdrawal == "host":
        b.host.withdraw()
    elif withdrawal == "incarnation":
        b.current[0] = False
    elif withdrawal == "expiry":
        b.clock[0] = lease.expires_at
    elif withdrawal == "feature":
        await b.agent._unregister_feature_runtime(b.feature, unload=True)
    elif withdrawal == "privacy":
        from kestrel_sovereign.privacy import PrivacyMode
        b.agent._privacy_mode = PrivacyMode.EPHEMERAL
    else:
        b.agent._agent_manager_published = False
    with pytest.raises(HostCapabilityAccessDenied):
        lease.require_background()


@pytest.mark.asyncio
async def test_read_operation_cannot_issue_a_connection_credential_lease(oauth_boundary):
    b = oauth_boundary
    response = await b.client.get("/api/owned-issue", params={"action": "read"}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 403 and b.leases == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["copied-task", "unregistered"])
async def test_copied_context_or_unregistered_description_cannot_issue_host_authority(oauth_boundary, mode):
    b = oauth_boundary
    response = await b.client.get("/api/owned-issue", params={"mode": mode}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 403 and b.leases == [] and b.host._leases == set()


@pytest.mark.asyncio
async def test_host_withdrawal_releases_every_original_credential_checker(oauth_boundary):
    b = oauth_boundary
    await issue(b)
    assert (await b.client.get("/api/owned-issue", headers={"X-API-Key": API_KEY})).status_code == 200
    assert len(b.leases) == len(b.host._leases) == 2
    b.host.withdraw()
    assert b.host._leases == set()
    for lease in b.leases:
        with pytest.raises(HostCapabilityAccessDenied):
            lease.require_background()


@pytest.mark.asyncio
async def test_fresh_restore_request_reads_one_immutable_host_identity_and_budget(oauth_boundary):
    b = oauth_boundary
    result = await b.client.get('/api/owned-identity', headers={'X-API-Key': API_KEY})
    assert result.status_code == 200, result.text
    assert result.json() == b.app.state.mcp_oauth_policy.model_dump(mode='json') | {
        'client_document': b.app.state.mcp_oauth_policy.client_document()}
    assert b.host._pending == {} and b.host._leases == set()


@pytest.mark.asyncio
async def test_copied_task_cannot_read_original_host_operation_identity(oauth_boundary):
    b = oauth_boundary
    result = await b.client.get('/api/owned-identity?mode=copied-task', headers={'X-API-Key': API_KEY})
    assert result.status_code == 403


@pytest.mark.asyncio
async def test_ended_context_cannot_reconstruct_host_identity(oauth_boundary):
    b = oauth_boundary
    await issue(b)
    with pytest.raises(HostCapabilityAccessDenied):
        b.host.client_identity(b.contexts[0])


@pytest.mark.asyncio
async def test_nested_credential_operation_reuses_the_original_lease_before_a_queued_writer(oauth_boundary):
    b = oauth_boundary
    response = await b.client.get("/api/owned-issue", params={"mode": "nested-credential"},
                                  headers={"X-API-Key": API_KEY})
    assert response.status_code == 200, (response.text, b.issue_errors)
    with pytest.raises(HostCapabilityAccessDenied):
        b.leases[0].require_background()


@pytest.mark.asyncio
async def test_fresh_current_principal_can_enter_a_credential_storage_operation(oauth_boundary):
    b = oauth_boundary
    await issue(b)
    response = await b.client.get("/api/owned-use", params={"mode": "credential"},
                                  headers={"X-API-Key": API_KEY})
    assert response.status_code == 200, response.text
