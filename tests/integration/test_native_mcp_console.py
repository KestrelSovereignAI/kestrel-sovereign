"""Native consent controller with actual HTTP credentials, privacy and SQL."""

import asyncio
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from fastapi import FastAPI
from mcp.server.auth.provider import AuthorizationParams

from kestrel_sovereign import server
from kestrel_sovereign.endpoints.auth_oauth import _create_jwt
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.mcp_server.consent import MCPConsentAuthority
from kestrel_sovereign.mcp_server.console import MCPConsentController, consent_page
from kestrel_sovereign.mcp_server.grants import MCPAuthorizationGrants
from kestrel_sovereign.mcp_server.pending import PendingMCPAuthorizations
from kestrel_sovereign.mcp_server.policy import MCPServerPolicy
from kestrel_sovereign.mcp_server.transport import NativeMCPServer
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.private_delivery import PrivateDeliveryMiddleware
from kestrel_sovereign.security.request_auth import (
    InteractiveAuthenticationLifetimeMiddleware,
    RequestAuthenticationService,
)
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
API_KEY = "owned-console-sovereign-key"
RESOURCE = "https://kestrel.example/mcp/owned"
CALLBACK = "https://provider.example/callback?keep=&tag=one&tag=two"
PATH = "/api/mcp/server/consent/requests/"
ALICE = "alice@owned-consent.test"
native_pg = _native_pg


@pytest.fixture(params=["sqlite", "postgres"])
async def console(request, native_pg, native_tool_effect_agent, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "o" * 32)
    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)
    monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "1")
    monkeypatch.setenv("JWT_SECRET_KEY", "owned-console-jwt-signing-key-with-entropy")
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", ALICE)
    agent = await native_tool_effect_agent(did="did:owned:mcp-console",
        connected_backend=native_pg[0] if request.param == "postgres" else None)
    policy = MCPServerPolicy.model_validate_json(json.dumps({
        "console_origin": "https://kestrel.example", "clients": [{
            "client_id": "owned-provider", "client_name": "Owned provider",
            "redirect_uris": [CALLBACK], "scopes": ["memory:read", "memory:write"],
            "client_secret": "owned-secret"}]}))
    pending = PendingMCPAuthorizations(resolve_feature_database(agent).backend, agent.did, RESOURCE, policy)
    await pending.initialize()
    keys, live = [API_KEY], [True]

    def current():
        if not live[0]:
            raise HostCapabilityAccessDenied("Owned publication retired")

    authority = MCPConsentAuthority(agent.did, RESOURCE, policy, require_current=current,
        sovereign_key=lambda: keys[0])
    grants = MCPAuthorizationGrants(pending, RESOURCE + "/oauth", consent_authority=authority)
    await grants.initialize()
    runtime = NativeMCPServer(grants, name=agent.agent_name, consent_url=RESOURCE + "/consent")
    controller = MCPConsentController(agent, runtime.authorization, require_current=current)
    app = FastAPI()
    app.state.agent = agent
    app.state.request_authentication = RequestAuthenticationService(api_key=lambda: keys[0])
    app.include_router(controller.router)
    app.add_middleware(InteractiveAuthenticationLifetimeMiddleware)
    app.middleware("http")(server.auth_middleware)
    app.add_middleware(PrivateDeliveryMiddleware)

    async def create():
        return await pending.create("owned-provider", AuthorizationParams(state="original-state",
            scopes=["memory:read", "memory:write"], code_challenge="a" * 43,
            redirect_uri=policy.clients[0].oauth_information().redirect_uris[0],
            redirect_uri_provided_explicitly=True, resource=RESOURCE))

    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app),
            base_url="https://kestrel.example", follow_redirects=False) as client:
        yield SimpleNamespace(agent=agent, app=app, client=client, create=create,
            grants=grants, pending=pending, keys=keys, live=live, controller=controller)


async def test_original_console_description_is_private_and_does_not_mint_or_consume(console):
    handle = await console.create()
    response = await console.client.get(PATH + handle, headers={"x-api-key": API_KEY})
    assert response.status_code == 200, response.text
    assert response.json() == {"client_name": "Owned provider", "agent_name": console.agent.agent_name,
        "resource": RESOURCE, "redirect_uri": CALLBACK, "scopes": ["memory:read", "memory:write"],
        "expires_at": (await console.pending.read(handle)).expires_at}
    assert response.headers["cache-control"] == "private, no-store, max-age=0"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert await console.grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_grants") == (0,)


@pytest.mark.parametrize("decision", ["approve", "deny"])
async def test_console_decision_commits_once_and_preserves_original_callback(console, decision):
    handle = await console.create()
    response = await console.client.post(PATH + handle + "/decision", headers={"x-api-key": API_KEY},
        json={"decision": decision, "scopes": ["memory:read"] if decision == "approve" else []})
    assert response.status_code == 200, response.text
    redirect = response.json()["redirect_url"]
    assert redirect.startswith(CALLBACK + "&"), "The original registered callback was rewritten"
    values = parse_qs(urlsplit(redirect).query, keep_blank_values=True)
    assert values["state"] == ["original-state"]
    assert values["iss"] == [RESOURCE + "/oauth"]
    if decision == "approve":
        code = await console.grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_credentials")
        assert code == (1,)
        assert len(values["code"][0]) == 43
    else:
        assert values["error"] == ["access_denied"]
        assert await console.grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_credentials") == (0,)
    assert await console.pending.read(handle) is None
    repeated = await console.client.post(PATH + handle + "/decision", headers={"x-api-key": API_KEY},
        json={"decision": decision, "scopes": ["memory:read"] if decision == "approve" else []})
    assert repeated.status_code == 404, "The consumed consent request produced another decision"


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("credential", ["none", "jwt"])
async def test_console_rejects_non_owner_credentials_before_reading_body_or_pending(console, method, credential):
    handle = await console.create()
    headers = {} if credential == "none" else {"authorization": "Bearer " + _create_jwt(ALICE)}
    path = PATH + handle + ("/decision" if method == "POST" else "")
    response = await console.client.request(method, path, headers=headers, content=b"invalid-json")
    assert response.status_code == (401 if credential == "none" else 403), response.text
    assert "Owned provider" not in response.text
    assert await console.pending.read(handle) is not None


@pytest.mark.parametrize("body", [
    {"decision": "approve", "scopes": []},
    {"decision": "deny", "scopes": ["memory:read"]},
    {"decision": "approve", "scopes": ["memory:read", "memory:read"]},
    {"decision": "approve", "scopes": ["unknown"]},
    {"decision": "approve", "scopes": ["memory:read"], "resource": "other-agent"},
])
async def test_console_invalid_selection_never_consumes_or_mints(console, body):
    handle = await console.create()
    response = await console.client.post(PATH + handle + "/decision", headers={"x-api-key": API_KEY}, json=body)
    assert response.status_code == 400, response.text
    assert await console.pending.read(handle) is not None
    assert await console.grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_grants") == (0,)


@pytest.mark.parametrize("change", ["rotation", "publication", "agent"])
async def test_original_console_authority_is_checked_after_read(console, change, monkeypatch):
    handle = await console.create()
    original = console.grants.read_authorization_request

    async def read(value):
        pending = await original(value)
        if change == "rotation":
            console.keys[0] = "owned-rotated-key"
        elif change == "publication":
            console.live[0] = False
        else:
            console.app.state.agent = object()
        return pending

    monkeypatch.setattr(console.grants, "read_authorization_request", read)
    response = await console.client.get(PATH + handle, headers={"x-api-key": API_KEY})
    assert response.status_code == 403, "Changed original authority delivered private pending consent"
    assert "Owned provider" not in response.text


async def test_concurrent_console_approval_and_denial_have_one_committed_winner(console):
    handle = await console.create()
    responses = await asyncio.gather(*(
        console.client.post(PATH + handle + "/decision", headers={"x-api-key": API_KEY},
            json={"decision": decision, "scopes": ["memory:read"] if decision == "approve" else []})
        for decision in ("approve", "deny")))
    assert sorted(r.status_code for r in responses) == [200, 404]
    assert await console.pending.read(handle) is None
    assert (await console.grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_grants"))[0] <= 1


@pytest.mark.parametrize("content,media", [(b"not-json", "application/json"),
    (b"private-body", "text/plain"), (b"x" * 65537, "application/json")], ids=["malformed-json", "wrong-media", "over-budget"])
async def test_invalid_bounded_console_body_never_consumes_pending(console, content, media):
    handle = await console.create()
    response = await console.client.post(PATH + handle + "/decision",
        headers={"x-api-key": API_KEY, "content-type": media}, content=content)
    assert response.status_code == 400, response.text
    assert await console.pending.read(handle) is not None
    assert "private-body" not in response.text


async def test_unknown_postcommit_console_outcome_exposes_no_code_and_must_not_be_replayed(console, monkeypatch):
    handle = await console.create()
    original = console.grants.issue_code

    async def interrupted(*args):
        await original(*args)
        raise OSError("private-transport-or-native-boundary-error")

    monkeypatch.setattr(console.grants, "issue_code", interrupted)
    response = await console.client.post(PATH + handle + "/decision", headers={"x-api-key": API_KEY},
        json={"decision": "approve", "scopes": ["memory:read"]})
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "mcp_consent_outcome_unknown_do_not_replay"
    assert "redirect_url" not in response.json()
    assert "private-transport" not in response.text
    assert await console.pending.read(handle) is None
    assert await console.grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_grants") == (1,)


@pytest.mark.parametrize("prefix", ["", "/api/agent-routes/b3duZWQ"])
async def test_consent_shell_contains_only_canonical_local_prefix_and_external_assets(prefix):
    response = consent_page(prefix)
    assert f'data-api-prefix="{prefix}"' in response.body.decode()
    assert "__KESTREL_MCP_API_PREFIX__" not in response.body.decode()
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "form-action 'none'; frame-ancestors 'none'" in response.headers["content-security-policy"]


@pytest.mark.parametrize("prefix", ["https://evil.example", "/api/agent-routes/../evil", "/api/agent-routes/b3duZWQ\" onclick=\"evil", "/api/agent-routes/!"])
async def test_consent_shell_rejects_noncanonical_or_external_prefix(prefix):
    with pytest.raises(ValueError):
        consent_page(prefix)
