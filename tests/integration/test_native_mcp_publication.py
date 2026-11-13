"""Publish an actual initialized Core agent through the original native SDK."""

import asyncio
import json
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign.auth import AuthMethod, CallerContext
from kestrel_sovereign.endpoints.mcp_server import router
from kestrel_sovereign.mcp_server.http import NativeMCPHostMiddleware
from kestrel_sovereign.mcp_server.policy import MCPServerPolicy
from kestrel_sovereign.mcp_server.publication import NativeMCPPublication
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.private_delivery import PrivateDeliveryMiddleware
from kestrel_sovereign.security.request_auth import (
    InteractiveAuthenticationLifetimeMiddleware,
    RequestAuthenticationService,
    current_request_authentication,
)
from tests.integration.test_native_assistant_write_receipts import (
    assistant_agent as _assistant_agent,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
assistant_agent = _assistant_agent
KEY = "owned-publication-durable-key"


@pytest.fixture
async def publication(assistant_agent, monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", KEY)
    agent = assistant_agent
    assert agent._runtime_publication_ready.is_set()
    assert agent.llm_service.disabled
    app = FastAPI()
    app.state.agent = agent
    app.state.request_authentication = RequestAuthenticationService(api_key=lambda: KEY)
    app.include_router(router)
    app.add_middleware(InteractiveAuthenticationLifetimeMiddleware)
    app.middleware("http")(server.auth_middleware)
    app.add_middleware(NativeMCPHostMiddleware)
    app.add_middleware(PrivateDeliveryMiddleware)
    policy = MCPServerPolicy.model_validate_json(json.dumps({"console_origin": "https://kestrel.example",
        "clients": [{"client_id": "owned-provider", "client_name": "Owned provider",
            "redirect_uris": ["https://provider.example/callback"], "client_secret": "owned-secret"}]}))
    p = await NativeMCPPublication.prepare(app, agent, "OwnedNativeMCP", policy, sovereign_key=lambda: KEY)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app),
            base_url=policy.console_origin, follow_redirects=False) as client:
        try:
            yield p, client
        finally:
            await p.close()


async def test_prepared_mcp_routes_are_private_until_the_original_host_commit(publication):
    p, client = publication
    path = urlsplit(p.resource).path
    assert not p.registry.is_current(p.binding)
    with pytest.raises(HostCapabilityAccessDenied, match="publication"):
        p.require_current()
    response = await client.get(path + "/oauth/.missing")
    assert response.status_code in (401, 404)
    p.publish()
    p.require_current()
    response = await client.get("/.well-known/oauth-authorization-server" + path + "/oauth")
    assert response.status_code == 200, response.text
    assert response.json()["issuer"] == p.resource + "/oauth"


async def test_actual_mcp_lane_does_not_promote_an_added_owner_key(publication, monkeypatch):
    p, client = publication
    p.publish()
    path = urlsplit(p.resource).path
    original = p.runtime.authorization.verify_token
    interactive = []

    async def verify(token):
        interactive.append(current_request_authentication())
        return await original(token)

    monkeypatch.setattr(p.runtime.authorization, "verify_token", verify)
    response = await client.post(path, headers={"x-api-key": KEY, "authorization": "Bearer " + KEY,
        "content-type": "application/json",
        "accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert response.status_code == 401, "A Console key bypassed the official MCP Bearer verifier"
    assert "resource_metadata" in response.headers["www-authenticate"]
    assert interactive == [None], "The original native MCP request acquired interactive owner credentials"
    caller = CallerContext.mcp_transport()
    assert caller.auth_method is AuthMethod.MCP_TRANSPORT and not caller.is_sovereign


async def test_published_authorization_and_console_use_the_same_original_sql_grants(publication):
    p, client = publication
    p.publish()
    path = urlsplit(p.resource).path
    response = await client.get(path + "/oauth/authorize", params={"response_type": "code",
        "client_id": "owned-provider", "redirect_uri": "https://provider.example/callback",
        "scope": "memory:read", "resource": p.resource, "state": "original-state",
        "code_challenge": "a" * 43, "code_challenge_method": "S256"})
    assert response.status_code == 302, response.text
    location = response.headers["location"]
    handle = parse_qs(urlsplit(location).query)["request"][0]
    shell = await client.get(location)
    assert shell.status_code == 200
    assert 'data-api-prefix=""' in shell.text
    assert handle not in shell.text, "Opaque consent hint was rendered into HTML"
    description = await client.get("/api/mcp/server/consent/requests/" + handle, headers={"x-api-key": KEY})
    assert description.status_code == 200, description.text
    assert description.json()["resource"] == p.resource
    response = await client.post("/api/mcp/server/consent/requests/" + handle + "/decision",
        headers={"x-api-key": KEY}, json={"decision": "approve", "scopes": ["memory:read"]})
    assert response.status_code == 200, response.text
    assert parse_qs(urlsplit(response.json()["redirect_url"]).query)["state"] == ["original-state"]
    assert await p.runtime.authorization.pending.read(handle) is None


async def test_withdrawal_precedes_sdk_retirement_and_keeps_borrowed_primary_backend(publication):
    p, _client = publication
    p.publish()
    original = p._resident
    assert p.agent._resource_custody().is_held("native_mcp_server")
    p.withdraw()
    assert not p.registry.is_current(p.binding)
    with pytest.raises(HostCapabilityAccessDenied):
        p.require_current()
    await p.close()
    assert original.done() and original.exception() is None, "Original SDK lifetime was not physically joined"
    assert not p.agent._resource_custody().is_held("native_mcp_server")
    assert await p.backend.fetch_val("SELECT 1") == 1, "MCP retirement closed its borrowed primary backend"


async def test_retirement_waits_for_the_original_native_http_reader(publication):
    p, _client = publication
    p.publish()
    entered, release = asyncio.Event(), asyncio.Event()

    async def request():
        async with p.request_lifetime():
            entered.set()
            await release.wait()

    reader = asyncio.create_task(request())
    await entered.wait()
    closer = asyncio.create_task(p.close())
    await asyncio.sleep(0)
    assert not closer.done(), "Native MCP retired while its original HTTP reader was still active"
    release.set()
    await reader
    await closer


async def test_normal_agent_shutdown_physically_joins_the_original_mcp_owner(publication):
    p, _client = publication
    p.publish()
    original = p._resident
    await p.agent.shutdown()
    assert original.done() and original.exception() is None
    assert not p.agent._resource_custody().is_held("native_mcp_server")
    assert not p.registry.is_current(p.binding)


async def test_actual_single_agent_onboarding_is_independent_of_a2a_configuration(publication):
    p, _client = publication
    await p.close()
    environment = {"KESTREL_MCP_SERVER": json.dumps({
        **p.policy.model_dump(mode="json"), "clients": [{**p.policy.clients[0].model_dump(mode="json"),
            "client_secret": "owned-secret"}]})}
    assert "KESTREL_BASE_URL" not in environment
    replacement = await server._onboard_single_agent_mcp(p.app, p.agent, environment)
    try:
        replacement.require_current()
        assert replacement.registry.is_current(replacement.binding)
        assert replacement.runtime.authorization.grants.backend is p.backend
        assert replacement._resident is not p._resident
    finally:
        await replacement.close()


async def test_unconfigured_core_does_not_prepare_native_mcp_or_require_server_keys(publication):
    p, _client = publication
    await p.close()
    assert await server._onboard_single_agent_mcp(p.app, p.agent, {}) is None
    assert vars(p.agent).get("_native_mcp_publication") is None


async def test_actual_manager_commit_and_withdrawal_own_the_same_mcp_generation(publication, tmp_path):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    p, _client = publication
    await p.close()
    manager = AgentManager(base_data_dir=tmp_path, startup_runtime_env={}, startup_roster_enabled=False)
    p.app.state.agent = None
    p.app.state.agent_manager = manager
    assert "_agent_manager_publication_event" not in vars(p.agent)
    prepared = await NativeMCPPublication.prepare(p.app, p.agent, "OwnedNativeMCP", p.policy,
        sovereign_key=lambda: KEY, manager=manager)
    try:
        async with manager.a2a_lifecycle_lease():
            manager._register_agent("OwnedNativeMCP", p.agent)
        prepared.require_current()
        assert vars(p.agent)["_agent_manager_published"] is True
        assert prepared.registry.is_current(prepared.binding)
        async with manager.a2a_lifecycle_lease():
            manager._withdraw_initialized_agent("OwnedNativeMCP", p.agent)
        assert vars(p.agent)["_agent_manager_published"] is False
        with pytest.raises(HostCapabilityAccessDenied):
            prepared.require_current()
        assert manager.get_agent("OwnedNativeMCP") is None
        assert not prepared.registry.is_current(prepared.binding)
    finally:
        await prepared.close()


async def test_actual_host_onboarding_publishes_and_routes_original_console_tenant(publication, tmp_path, monkeypatch):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.multi_agent.route_name import encode_agent_route_name

    p, client = publication
    await p.close()
    policy = {**p.policy.model_dump(mode="json"), "clients": [{**p.policy.clients[0].model_dump(mode="json"),
        "client_secret": "owned-secret"}]}
    monkeypatch.setenv("KESTREL_MCP_SERVER", json.dumps(policy))
    monkeypatch.delenv("KESTREL_BASE_URL", raising=False)
    manager = AgentManager(base_data_dir=tmp_path, startup_runtime_env={}, startup_roster_enabled=False)
    app = p.app
    app.state.agent = None
    app.state.agent_manager = manager
    app.add_middleware(server.MultiAgentAgentRoutingMiddleware)
    route_name = "Owned native / MCP agent"
    async with manager.a2a_lifecycle_lease():
        manager._onboarding_agents[route_name] = p.agent
        rollback = await server._onboard_host_registered_agent(app, manager, route_name, p.agent)
        prepared = p.agent._native_mcp_publication
        assert not prepared.registry.is_current(prepared.binding)
        assert manager.get_agent(route_name) is None
        manager._register_agent(route_name, p.agent)
        manager._onboarding_agents.pop(route_name)
    try:
        prepared.require_current()
        response = await client.get(prepared.resource + "/oauth/authorize", params={"response_type": "code",
            "client_id": "owned-provider", "redirect_uri": "https://provider.example/callback",
            "scope": "memory:read", "resource": prepared.resource, "state": "original-host-state",
            "code_challenge": "a" * 43, "code_challenge_method": "S256"})
        assert response.status_code == 302, response.text
        location = response.headers["location"]
        handle = parse_qs(urlsplit(location).query)["request"][0]
        prefix = "/api/agent-routes/" + encode_agent_route_name(route_name)
        shell = await client.get(location)
        assert shell.status_code == 200
        assert f'data-api-prefix="{prefix}"' in shell.text
        description = await client.get(prefix + "/api/mcp/server/consent/requests/" + handle,
            headers={"x-api-key": KEY})
        assert description.status_code == 200, description.text
        assert description.json()["resource"] == prepared.resource
        assert description.json()["agent_name"] == route_name
        denied = await client.post(prefix + "/api/mcp/server/consent/requests/" + handle + "/decision",
            headers={"x-api-key": KEY}, json={"decision": "deny", "scopes": []})
        assert denied.status_code == 200, denied.text
        assert parse_qs(urlsplit(denied.json()["redirect_url"]).query)["state"] == ["original-host-state"]
    finally:
        async with manager.a2a_lifecycle_lease():
            manager._withdraw_initialized_agent(route_name, p.agent)
            await rollback()
        assert prepared._resident.done()
        assert await prepared.backend.fetch_val("SELECT 1") == 1


async def test_failed_mcp_owner_check_still_revokes_a2a_authority(publication, tmp_path, monkeypatch):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    p, _client = publication
    await p.close()
    policy = {**p.policy.model_dump(mode="json"), "clients": [{**p.policy.clients[0].model_dump(mode="json"),
        "client_secret": "owned-secret"}]}
    monkeypatch.setenv("KESTREL_MCP_SERVER", json.dumps(policy))
    monkeypatch.delenv("KESTREL_BASE_URL", raising=False)
    manager = AgentManager(base_data_dir=tmp_path, startup_runtime_env={}, startup_roster_enabled=False)
    p.app.state.agent = None
    p.app.state.agent_manager = manager
    async with manager.a2a_lifecycle_lease():
        manager._onboarding_agents["OwnedNativeMCP"] = p.agent
        rollback = await server._onboard_host_registered_agent(p.app, manager, "OwnedNativeMCP", p.agent)
        prepared = p.agent._native_mcp_publication
        manager._register_agent("OwnedNativeMCP", p.agent)
        manager._onboarding_agents.pop("OwnedNativeMCP")
    assert manager.a2a_hosted_policy_for(p.agent) is not None
    try:
        p.agent._native_mcp_publication = object()
        async with manager.a2a_lifecycle_lease():
            with pytest.raises(TypeError, match="original host publication"):
                manager._withdraw_initialized_agent("OwnedNativeMCP", p.agent)
        assert vars(p.agent)["_agent_manager_published"] is False
        assert manager.a2a_hosted_policy_for(p.agent) is None, "A failed MCP owner check left original A2A authority active"
    finally:
        p.agent._native_mcp_publication = prepared
        async with manager.a2a_lifecycle_lease():
            manager._withdraw_initialized_agent("OwnedNativeMCP", p.agent)
            await rollback()
