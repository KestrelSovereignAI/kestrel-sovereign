"""Real first-party HTTP credentials approve only the original encrypted SQL request."""

import asyncio
import json
import time
from contextvars import copy_context
from dataclasses import replace

import httpx2
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.auth.provider import AuthorizationParams

from kestrel_sovereign import server
from kestrel_sovereign.endpoints.auth_oauth import _create_jwt
from kestrel_sovereign.mcp_server.consent import MCPConsentAuthority
from kestrel_sovereign.mcp_server.grants import MCPAuthorizationGrants
from kestrel_sovereign.mcp_server.pending import PendingMCPAuthorizations
from kestrel_sovereign.mcp_server.policy import MCPServerPolicy
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.request_auth import (
    InteractiveAuthenticationLifetimeMiddleware,
    RequestAuthenticationService,
)
from kestrel_sovereign.security.sovereign_key import sovereign_key_fingerprint
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
API_KEY = "owned-mcp-consent-sovereign-key"
RESOURCE = "https://kestrel.example/mcp/owned"
ALICE = "alice@mcp-consent.test"
native_pg = _native_pg


@pytest.fixture(params=["sqlite", "postgres"])
async def consent_http(request, native_pg, tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "o" * 32)
    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)
    monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "1")
    monkeypatch.setenv("JWT_SECRET_KEY", "owned-mcp-consent-jwt-signing-key-with-entropy")
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", ALICE)
    settings = request.param if isinstance(request.param, dict) else {"backend": request.param}
    backend_kind = settings["backend"]
    policy = MCPServerPolicy.model_validate_json(json.dumps({
        "console_origin": "https://kestrel.example", "clients": [{
            "client_id": "owned-provider", "client_name": "Owned provider",
            "redirect_uris": ["https://provider.example/callback"],
            "scopes": ["memory:read", "memory:write"], "client_secret": "owned-secret"}],
        **{name: value for name, value in settings.items() if name != "backend"}}))
    backend = SQLiteBackend(str(tmp_path / "consent.db")) if backend_kind == "sqlite" else native_pg[0]
    try:
        if backend_kind == "sqlite":
            await backend.connect()
        keys, live, clock, retained = [API_KEY], [True], [time.time()], []
        pending = PendingMCPAuthorizations(backend, "did:owned:mcp-consent", RESOURCE, policy,
            clock=lambda: clock[0])
        await pending.initialize()
        handle = await pending.create("owned-provider", AuthorizationParams(
            state="owned-original-state", scopes=["memory:read"], code_challenge="a" * 43,
            redirect_uri="https://provider.example/callback", redirect_uri_provided_explicitly=True,
            resource=RESOURCE))
        app = FastAPI()
        app.state.request_authentication = RequestAuthenticationService(api_key=lambda: keys[0])
        app.add_middleware(InteractiveAuthenticationLifetimeMiddleware)
        app.middleware("http")(server.auth_middleware)

        def current():
            if not live[0]:
                raise HostCapabilityAccessDenied("Original target authority retired")

        authority = MCPConsentAuthority(pending.agent_id, RESOURCE, policy, require_current=current,
            sovereign_key=lambda: keys[0], clock=lambda: clock[0])
        grants = MCPAuthorizationGrants(pending, RESOURCE + "/oauth", consent_authority=authority)
        await grants.initialize()

        @app.post("/api/owned-mcp-consent")
        async def consent(request: Request):
            selected_handle = request.query_params.get("handle", handle)
            value = await pending.read(selected_handle)
            if value is None:
                return JSONResponse({"error": "Original pending request unavailable"}, status_code=404)
            requested = (request.query_params.get("scope", "memory:read"),)
            target = request.query_params.get("resource", RESOURCE)
            selected_authority = authority if target == RESOURCE else MCPConsentAuthority(
                pending.agent_id, target, policy, require_current=current,
                sovereign_key=lambda: keys[0], clock=lambda: clock[0])
            try:
                with selected_authority.approve(selected_handle, value, requested) as approval:
                    retained.append((approval, copy_context()))
                    mode = request.query_params.get("mode")
                    if mode == "substitute-approval":
                        changes = {
                            "scopes": {"scopes": ("memory:write",)},
                            "client": {"client_id": "other-provider"},
                            "resource": {"resource": RESOURCE + "/other"},
                            "subject": {"subject": "forged-subject"},
                            "credential": {"credential_fingerprint": "f" * 64},
                            "copy": {},
                        }
                        altered = replace(approval, **changes[request.query_params["change"]])
                        code = await grants.issue_code(selected_handle, altered)
                        return code.model_dump(mode="json")
                    if mode in ("issue-code", "nested-issue-code"):
                        if mode == "nested-issue-code":
                            async with backend.transaction():
                                try:
                                    code = await grants.issue_code(selected_handle, approval)
                                except RuntimeError as error:
                                    return JSONResponse({"error": str(error)}, status_code=403)
                        else:
                            code = await grants.issue_code(selected_handle, approval)
                        return code.model_dump(mode="json")
                    if mode == "rotate":
                        keys[0] = "owned-rotated-key"
                    elif mode == "retire":
                        live[0] = False
                    elif mode == "expire":
                        clock[0] = value.expires_at
                    elif mode == "changed-request":
                        value.parameters.state = "changed-after-human-approval"
                    elif mode == "foreign-handle":
                        approval.require_pending("x" * 43, value)
                    await asyncio.sleep(0)
                    approval.require_pending(selected_handle, value)
                    return {"approved": True, "scopes": approval.scopes, "resource": approval.resource}
            except HostCapabilityAccessDenied as error:
                return JSONResponse({"error": str(error)}, status_code=403)

        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app),
                base_url="https://console.test") as client:
            yield client, pending, handle, retained, keys, clock, grants, app
    finally:
        if backend_kind == "sqlite":
            await backend.close()


async def test_actual_sovereign_request_approves_without_consuming_or_minting_tokens(consent_http):
    client, pending, handle, retained, _keys, _clock, _grants, _app = consent_http
    response = await client.post("/api/owned-mcp-consent", headers={"x-api-key": API_KEY})
    assert response.status_code == 200 and response.json() == {
        "approved": True, "scopes": ["memory:read"], "resource": RESOURCE}
    assert await pending.read(handle) is not None
    approval, context = retained[0]
    with pytest.raises(HostCapabilityAccessDenied, match="ended"):
        context.run(approval.require_current)


async def test_native_middleware_rejects_unauthenticated_consent(consent_http):
    client, _pending, _handle, retained, _keys, _clock, _grants, _app = consent_http
    response = await client.post("/api/owned-mcp-consent")
    assert response.status_code == 401 and not retained


async def test_actual_signed_in_user_is_not_promoted_to_sovereign(consent_http):
    client, _pending, _handle, retained, _keys, _clock, _grants, _app = consent_http
    response = await client.post("/api/owned-mcp-consent",
        headers={"authorization": "Bearer " + _create_jwt(ALICE)})
    assert response.status_code == 403 and "Sovereign" in response.json()["error"] and not retained


@pytest.mark.parametrize("query", ["scope=memory:write", "resource=https://kestrel.example/mcp/other"])
async def test_actual_consent_cannot_expand_requested_scope_or_target_agent(consent_http, query):
    client, _pending, _handle, retained, _keys, _clock, _grants, _app = consent_http
    response = await client.post("/api/owned-mcp-consent?" + query, headers={"x-api-key": API_KEY})
    assert response.status_code == 403 and not retained, "Consent expanded the original request"


@pytest.mark.parametrize("mode", ["rotate", "retire", "expire", "changed-request", "foreign-handle"])
async def test_original_consent_authority_is_rechecked_after_native_await(consent_http, mode):
    client, pending, handle, _retained, _keys, _clock, _grants, _app = consent_http
    response = await client.post("/api/owned-mcp-consent?mode=" + mode, headers={"x-api-key": API_KEY})
    assert response.status_code == 403, "Changed original consent authority survived an await"
    if mode != "expire":
        assert await pending.read(handle) is not None


async def test_ephemeral_boot_key_cannot_authorize_a_durable_mcp_grant(consent_http, monkeypatch):
    client, _pending, _handle, retained, _keys, _clock, _grants, _app = consent_http
    monkeypatch.setenv("KESTREL_INTERNAL_EPHEMERAL_SOVEREIGN_KEY_FINGERPRINT",
        sovereign_key_fingerprint(API_KEY))
    response = await client.post("/api/owned-mcp-consent", headers={"x-api-key": API_KEY})
    assert response.status_code == 403 and "Durable" in response.json()["error"] and not retained
