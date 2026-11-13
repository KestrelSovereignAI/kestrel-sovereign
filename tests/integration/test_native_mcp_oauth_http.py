"""Native HTTP OAuth handlers use original sovereign consent and SQL token custody."""

import asyncio
import base64
import hashlib
import logging
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx2
import pytest
from mcp.server.auth.provider import AuthorizationParams
from starlette.applications import Starlette
from starlette.routing import Route

from kestrel_sovereign.mcp_server.oauth import NativeMCPAuthorizationServer
from tests.integration.test_native_mcp_consent import API_KEY, RESOURCE
from tests.integration.test_native_mcp_consent import consent_http as _consent_http
from tests.integration.test_native_mcp_consent import native_pg as _native_pg
from tests.integration.test_native_mcp_oauth_pending import (
    wait_for_native_pending_sql_lock,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
consent_http = _consent_http
native_pg = _native_pg
VERIFIER = "owned-native-pkce-verifier-" + "v" * 43
CHALLENGE = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b"=").decode()
CALLBACK = "https://provider.example/callback"


def provider(boundary):
    return NativeMCPAuthorizationServer(boundary[6], "https://kestrel.example/console/mcp/consent")


def oauth_app(native):
    return Starlette(routes=[
        Route("/authorize", native.authorization_endpoint, methods=["GET", "POST"]),
        Route("/token", native.token_endpoint, methods=["POST"]),
        Route("/revoke", native.revocation_endpoint, methods=["POST"]),
    ])


async def approved_code(boundary):
    client, pending = boundary[:2]
    handle = await pending.create("owned-provider", AuthorizationParams(
        state="owned-http-state", scopes=["memory:read"], code_challenge=CHALLENGE,
        redirect_uri=pending.policy.clients[0].oauth_information().redirect_uris[0],
        redirect_uri_provided_explicitly=True, resource=RESOURCE))
    response = await client.post("/api/owned-mcp-consent", params={"mode": "issue-code", "handle": handle},
        headers={"x-api-key": API_KEY})
    assert response.status_code == 200, response.text
    return response.json()["code"]


def authorization_request():
    return {"response_type": "code", "client_id": "owned-provider", "redirect_uri": CALLBACK,
        "scope": "memory:read", "state": "original-http-state", "code_challenge": CHALLENGE,
        "code_challenge_method": "S256", "resource": RESOURCE}


def code_request(code):
    return {"grant_type": "authorization_code", "code": code, "redirect_uri": CALLBACK,
        "client_id": "owned-provider", "client_secret": "owned-secret", "code_verifier": VERIFIER, "resource": RESOURCE}


async def test_authorization_redirect_only_creates_original_pending_consent(consent_http):
    native = provider(consent_http)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)), base_url="https://kestrel.example") as client:
        response = await client.get("/authorize", params=authorization_request())
    assert response.status_code == 302, response.text
    location = urlsplit(response.headers["location"])
    assert location.scheme == "https" and location.netloc == "kestrel.example"
    assert location.path == "/console/mcp/consent"
    handle, = parse_qs(location.query)["request"]
    pending = await consent_http[1].read(handle)
    assert pending.parameters.code_challenge == CHALLENGE
    assert pending.parameters.state == "original-http-state"
    assert not await consent_http[1].backend.fetch_all("SELECT secret_hash FROM mcp_server_oauth_credentials")


async def test_native_http_code_pkce_refresh_rotation_and_revocation(consent_http):
    native = provider(consent_http)
    code = await approved_code(consent_http)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)), base_url="https://kestrel.example") as client:
        response = await client.post("/token", data=code_request(code))
        assert response.status_code == 200, response.text
        first = response.json()
        assert response.headers["cache-control"] == "no-store"
        access = await native.verify_token(first["access_token"])
        assert access.resource == RESOURCE and access.scopes == ["memory:read"]
        refreshed = await client.post("/token", data={"grant_type": "refresh_token", "resource": RESOURCE,
            "refresh_token": first["refresh_token"], "client_id": "owned-provider", "client_secret": "owned-secret"})
        assert refreshed.status_code == 200, refreshed.text
        second = refreshed.json()
        assert first["refresh_token"] != second["refresh_token"]
        response = await client.post("/revoke", data={"token": second["access_token"],
            "client_id": "owned-provider", "client_secret": "owned-secret"})
        assert response.status_code == 200, response.text
        assert await native.verify_token(second["access_token"]) is None


async def test_wrong_resource_cannot_consume_original_http_code(consent_http):
    native = provider(consent_http)
    code = await approved_code(consent_http)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)), base_url="https://kestrel.example") as client:
        request = code_request(code)
        request["resource"] = RESOURCE + "/other"
        response = await client.post("/token", data=request)
        assert response.status_code == 400 and response.json()["error"] == "invalid_target"
        response = await client.post("/token", data=code_request(code))
        assert response.status_code == 200, response.text


async def test_wrong_pkce_cannot_consume_original_http_code(consent_http):
    native = provider(consent_http)
    code = await approved_code(consent_http)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)), base_url="https://kestrel.example") as client:
        request = code_request(code)
        request["code_verifier"] = "x" * 43
        response = await client.post("/token", data=request)
        assert response.status_code == 400 and response.json()["error"] == "invalid_grant"
        response = await client.post("/token", data=code_request(code))
        assert response.status_code == 200, response.text


def client_settings(backend, method):
    client = {"client_id": "owned-provider", "client_name": "Owned provider",
        "redirect_uris": [CALLBACK], "scopes": ["memory:read", "memory:write"],
        "token_endpoint_auth_method": method}
    if method != "none":
        client["client_secret"] = "owned-secret"
    return {"backend": backend, "clients": [client]}


@pytest.mark.parametrize("consent_http", [client_settings(backend, "client_secret_basic")
    for backend in ("sqlite", "postgres")], indirect=True)
async def test_basic_authentication_does_not_require_body_client_identity(consent_http):
    native = provider(consent_http)
    code = await approved_code(consent_http)
    request = code_request(code)
    request.pop("client_id")
    request.pop("client_secret")
    encoded = base64.b64encode(b"owned-provider:owned-secret").decode()
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)),
            base_url="https://kestrel.example") as client:
        response = await client.post("/token", data=request, headers={"authorization": "Basic " + encoded})
    assert response.status_code == 200, response.text
    assert await native.verify_token(response.json()["access_token"]) is not None


@pytest.mark.parametrize("consent_http", [client_settings(backend, "none")
    for backend in ("sqlite", "postgres")], indirect=True)
async def test_public_client_revocation_does_not_require_a_client_secret(consent_http):
    native = provider(consent_http)
    code = await approved_code(consent_http)
    request = code_request(code)
    request.pop("client_secret")
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)),
            base_url="https://kestrel.example") as client:
        response = await client.post("/token", data=request)
        assert response.status_code == 200, response.text
        token = response.json()["access_token"]
        revoked = await client.post("/revoke", data={"token": token, "client_id": "owned-provider"})
    assert revoked.status_code == 200, revoked.text
    assert await native.verify_token(token) is None


@pytest.mark.parametrize("change", ["secret", "client", "redirect", "method", "duplicate-method"])
async def test_refused_client_request_preserves_original_code(consent_http, change):
    native = provider(consent_http)
    code = await approved_code(consent_http)
    request = code_request(code)
    headers = {}
    if change == "secret":
        request["client_secret"] = "wrong-secret"
    elif change == "client":
        request["client_id"] = "unregistered-client"
    elif change == "redirect":
        request["redirect_uri"] = CALLBACK + "/other"
    else:
        encoded = base64.b64encode(b"owned-provider:owned-secret").decode()
        headers["authorization"] = "Basic " + encoded
        if change == "method":
            request.pop("client_secret")
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)),
            base_url="https://kestrel.example") as client:
        response = await client.post("/token", data=request, headers=headers)
        assert response.status_code in (400, 401), response.text
        assert response.json()["error"] in {"invalid_client", "invalid_request", "invalid_grant"}
        response = await client.post("/token", data=code_request(code))
    assert response.status_code == 200, response.text


async def test_revocation_survives_a_temporary_sovereign_key_change(consent_http):
    native = provider(consent_http)
    code = await approved_code(consent_http)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)),
            base_url="https://kestrel.example") as client:
        response = await client.post("/token", data=code_request(code))
        assert response.status_code == 200, response.text
        token = response.json()["access_token"]
        consent_http[4][0] = "owned-rotated-sovereign-key"
        assert await native.verify_token(token) is None
        revoked = await client.post("/revoke", data={"token": token,
            "client_id": "owned-provider", "client_secret": "owned-secret"})
        assert revoked.status_code == 200, revoked.text
        consent_http[4][0] = API_KEY
        assert await native.verify_token(token) is None


@pytest.mark.parametrize("field", ["resource", "code", "code_verifier", "client_id"])
async def test_duplicate_token_parameters_do_not_consume_original_code(consent_http, field):
    native = provider(consent_http)
    code = await approved_code(consent_http)
    request = code_request(code)
    body = urlencode([*request.items(), (field, request[field])])
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)),
            base_url="https://kestrel.example") as client:
        response = await client.post("/token", content=body,
            headers={"content-type": "application/x-www-form-urlencoded"})
        assert response.status_code == 400 and response.json()["error"] == "invalid_request"
        response = await client.post("/token", data=request)
    assert response.status_code == 200, response.text


async def test_authorization_preserves_original_host_lease_control_after_body_receive(consent_http):
    from kestrel_sovereign.execution_custody import (
        ConfirmedLeaseDeadline,
        ExecutionAuthorityError,
    )
    from kestrel_sovereign.mcp_server.consent import MCPConsentAuthority
    from kestrel_sovereign.mcp_server.grants import MCPAuthorizationGrants

    pending = consent_http[1]
    lease, originals = ConfirmedLeaseDeadline(), []

    def require_current():
        try:
            lease.require_work()
        except ExecutionAuthorityError as error:
            originals.append(error)
            raise

    authority = MCPConsentAuthority(pending.agent_id, RESOURCE, pending.policy,
        require_current=require_current, sovereign_key=lambda: API_KEY, clock=pending.clock)
    grants = MCPAuthorizationGrants(pending, consent_http[6].issuer, consent_authority=authority)
    native = NativeMCPAuthorizationServer(grants, "https://kestrel.example/console/mcp/consent")
    app = oauth_app(native)

    async def retire_during_body(scope, receive, send):
        async def original_receive():
            message = await receive()
            if message["type"] == "http.request":
                lease.mark_lost()
            return message
        await app(scope, original_receive, send)

    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=retire_during_body),
            base_url="https://kestrel.example") as client:
        with pytest.raises(ExecutionAuthorityError) as escaped:
            await client.post("/authorize", data=authorization_request())
    assert originals and escaped.value is originals[0], "The original host callback control was replaced"


@pytest.mark.parametrize("redirect", ["HTTPS://provider.example/callback", "https://provider.example/a/../callback"])
async def test_authorization_compares_the_original_redirect_bytes(consent_http, redirect):
    native = provider(consent_http)
    request = authorization_request()
    request["redirect_uri"] = redirect
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)),
            base_url="https://kestrel.example") as client:
        response = await client.get("/authorize", params=request)
    assert response.status_code == 400, "A normalized redirect replaced the exact registered identity"
    assert "location" not in response.headers


@pytest.mark.parametrize("field", ["client_id", "redirect_uri", "code_challenge"])
async def test_authorization_refuses_repeated_parameters_before_consent(consent_http, field):
    native = provider(consent_http)
    request = authorization_request()
    body = urlencode([*request.items(), (field, request[field])])
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)),
            base_url="https://kestrel.example") as client:
        response = await client.post("/authorize", content=body,
            headers={"content-type": "application/x-www-form-urlencoded"})
    assert response.status_code == 400, "Repeated OAuth parameters created consent"
    assert "location" not in response.headers



@pytest.mark.parametrize("consent_http", [
    {"backend": backend, "clients": [{"client_id": "owned-provider", "client_name": "Owned provider",
        "redirect_uris": ["https://provider.example/callback?keep=&tag=one&tag=two"],
        "scopes": ["memory:read", "memory:write"], "client_secret": "owned-secret"}]}
    for backend in ("sqlite", "postgres")], indirect=True)
async def test_authorization_error_preserves_registered_query_and_state(consent_http):
    native = provider(consent_http)
    callback = consent_http[1].policy.clients[0].redirect_uris[0]
    request = authorization_request()
    request.update(redirect_uri=callback, scope="unregistered-scope", state="original + state & value")
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)),
            base_url="https://kestrel.example") as client:
        response = await client.get("/authorize", params=request)
    assert response.status_code == 302, response.text
    location = response.headers["location"]
    assert location.startswith(callback + "&"), "The original registered callback query was rewritten"
    values = parse_qs(urlsplit(location).query, keep_blank_values=True)
    assert values["keep"] == [""] and values["tag"] == ["one", "two"]
    assert values["state"] == [request["state"]]
    assert values["iss"] == [consent_http[6].issuer]
    assert values["error"] == ["invalid_scope"]



@pytest.mark.parametrize("content_type", ["application/json", "text/plain"])
async def test_token_endpoint_requires_the_standard_form_media_type(consent_http, content_type):
    native = provider(consent_http)
    code = await approved_code(consent_http)
    request = code_request(code)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)),
            base_url="https://kestrel.example") as client:
        response = await client.post("/token", content=urlencode(request), headers={"content-type": content_type})
        assert response.status_code == 400 and response.json()["error"] == "invalid_request"
        response = await client.post("/token", data=request)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("consent_http", [client_settings(backend, "client_secret_basic")
    for backend in ("sqlite", "postgres")], indirect=True)
@pytest.mark.parametrize("endpoint", ["token", "revoke"])
@pytest.mark.parametrize("authorization", [b"Basic /w==", b"Basic ", b"Basic \xff"])
async def test_malformed_basic_authentication_refuses_without_consuming_credentials(
        consent_http, endpoint, authorization):
    native = provider(consent_http)
    code = await approved_code(consent_http)
    request = code_request(code)
    request.pop("client_id")
    request.pop("client_secret")
    encoded = base64.b64encode(b"owned-provider:owned-secret")
    good = [(b"authorization", b"Basic " + encoded)]
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native),
            raise_app_exceptions=False), base_url="https://kestrel.example") as client:
        if endpoint == "revoke":
            issued = await client.post("/token", data=request, headers=good)
            assert issued.status_code == 200, issued.text
            access = issued.json()["access_token"]
            request = {"token": access}
        refused = await client.post("/" + endpoint, data=request,
            headers=[(b"authorization", authorization)])
        assert refused.status_code == 401, "Malformed Basic credentials escaped client authentication"
        assert refused.json()["error"] == "invalid_client"
        assert refused.headers["www-authenticate"].startswith("Basic")
        if endpoint == "revoke":
            assert await native.verify_token(access) is not None, "Malformed authentication revoked the original grant"
        accepted = await client.post("/" + endpoint, data=request, headers=good)
        assert accepted.status_code == 200, accepted.text
        if endpoint == "token":
            assert await native.verify_token(accepted.json()["access_token"]) is not None
        else:
            assert await native.verify_token(access) is None


@pytest.mark.parametrize("consent_http", [{"backend": backend, "max_request_bytes": 4096}
    for backend in ("sqlite", "postgres")], indirect=True)
@pytest.mark.parametrize("state", ["s" * 4096, "\U0001f985" * 1024], ids=["ascii", "unicode"])
async def test_authorization_state_budget_refuses_without_creating_pending_consent(consent_http, state, caplog):
    caplog.set_level(logging.WARNING, logger="httpx2")
    native = provider(consent_http)
    pending = consent_http[1]
    count_sql = "SELECT COUNT(*) FROM mcp_server_oauth_pending WHERE agent_id = ?"
    before = await pending.backend.fetch_one(count_sql, (pending.agent_id,))
    request = authorization_request()
    request["state"] = state
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native),
            raise_app_exceptions=False), base_url="https://kestrel.example") as client:
        refused = await client.get("/authorize", params=request)
        assert refused.status_code == 302, "Oversized state escaped the standard OAuth refusal"
        location = refused.headers["location"]
        assert location.startswith(CALLBACK + "?")
        values = parse_qs(urlsplit(location).query)
        assert values["error"] == ["invalid_request"]
        assert values["state"] == [state] and values["iss"] == [consent_http[6].issuer]
        after = await pending.backend.fetch_one(count_sql, (pending.agent_id,))
        assert before == after, "Oversized state consumed pending-consent capacity"
        assert await pending.read(consent_http[2]) is not None
        request["state"] = "bounded-original-state"
        accepted = await client.get("/authorize", params=request)
        assert accepted.status_code == 302, accepted.text
        handle, = parse_qs(urlsplit(accepted.headers["location"]).query)["request"]
        stored = await pending.read(handle)
        assert stored is not None and stored.parameters.state == "bounded-original-state"


@pytest.mark.parametrize("consent_http", ["postgres"], indirect=True)
async def test_native_authorization_expiry_uses_standard_oauth_refusal(consent_http, native_pg):
    native = provider(consent_http)
    _backend, control, schema = native_pg
    pending, clock = consent_http[1], consent_http[5]
    task = None
    try:
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native),
                raise_app_exceptions=False), base_url="https://kestrel.example") as client:
            async with control.transaction():
                await control.execute(f'LOCK TABLE "{schema}".mcp_server_oauth_capacity IN ACCESS EXCLUSIVE MODE')
                task = asyncio.create_task(client.get("/authorize", params=authorization_request()))
                await wait_for_native_pending_sql_lock(control)
                clock[0] += pending.policy.authorization_seconds + 1
            refused = await task
            assert refused.status_code == 302, "Native pending expiry escaped the standard OAuth refusal"
            values = parse_qs(urlsplit(refused.headers["location"]).query)
            assert values["error"] == ["temporarily_unavailable"]
            assert values["state"] == [authorization_request()["state"]]
            rows = await pending.backend.fetch_all("SELECT expires_at FROM mcp_server_oauth_pending WHERE agent_id = ?",
                (pending.agent_id,))
            assert rows == [], "Admission already expired at its native SQL lock persisted consent"
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
