"""Native HTTP OAuth handlers use original sovereign consent and SQL token custody."""

import base64
import hashlib
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from mcp.server.auth.provider import AuthorizationParams
from starlette.applications import Starlette
from starlette.routing import Route

from kestrel_sovereign.mcp_server.oauth import NativeMCPOAuthProvider
from tests.integration.test_native_mcp_consent import API_KEY, RESOURCE
from tests.integration.test_native_mcp_consent import consent_http as _consent_http
from tests.integration.test_native_mcp_consent import native_pg as _native_pg

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
consent_http = _consent_http
native_pg = _native_pg
VERIFIER = "owned-native-pkce-verifier-" + "v" * 43
CHALLENGE = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b"=").decode()
CALLBACK = "https://provider.example/callback"


def provider(boundary):
    return NativeMCPOAuthProvider(boundary[6], "https://kestrel.example/console/mcp/consent")


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
        redirect_uri=CALLBACK, redirect_uri_provided_explicitly=True, resource=RESOURCE))
    response = await client.post("/api/owned-mcp-consent", params={"mode": "issue-code", "handle": handle},
        headers={"x-api-key": API_KEY})
    assert response.status_code == 200, response.text
    return response.json()["code"]


def code_request(code):
    return {"grant_type": "authorization_code", "code": code, "redirect_uri": CALLBACK,
        "client_id": "owned-provider", "client_secret": "owned-secret", "code_verifier": VERIFIER, "resource": RESOURCE}


async def test_authorization_redirect_only_creates_original_pending_consent(consent_http):
    native = provider(consent_http)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=oauth_app(native)), base_url="https://kestrel.example") as client:
        response = await client.get("/authorize", params={"response_type": "code", "client_id": "owned-provider",
            "redirect_uri": CALLBACK, "scope": "memory:read", "state": "original-http-state",
            "code_challenge": CHALLENGE, "code_challenge_method": "S256", "resource": RESOURCE})
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
    from urllib.parse import urlencode

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
