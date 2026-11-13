"""Official MCP client and HTTP transport retain standard discovery and audience."""

from urllib.parse import urlsplit

import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from kestrel_sovereign.mcp_server.transport import NativeMCPServer
from tests.integration.test_native_mcp_consent import RESOURCE
from tests.integration.test_native_mcp_consent import consent_http as _consent_http
from tests.integration.test_native_mcp_consent import native_pg as _native_pg
from tests.integration.test_native_mcp_oauth_http import approved_code, code_request

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
consent_http = _consent_http
native_pg = _native_pg


def runtime(boundary):
    return NativeMCPServer(boundary[6], name="Owned MCP agent",
        consent_url="https://kestrel.example/console/mcp/consent")


async def test_native_resource_challenge_and_pathful_issuer_discovery(consent_http):
    native = runtime(consent_http)
    async with (native.app.router.lifespan_context(native.app),
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=native.app),
            base_url="https://kestrel.example") as client):
        denied = await client.post(RESOURCE, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert denied.status_code == 401
        resource_metadata = "https://kestrel.example/.well-known/oauth-protected-resource/mcp/owned"
        assert resource_metadata in denied.headers["www-authenticate"]
        response = await client.get(resource_metadata)
        assert response.status_code == 200, response.text
        metadata = response.json()
        assert metadata["resource"] == RESOURCE
        assert metadata["authorization_servers"] == [RESOURCE + "/oauth"]
        issuer = await client.get("/.well-known/oauth-authorization-server/mcp/owned/oauth")
        assert issuer.status_code == 200, issuer.text
        value = issuer.json()
        assert value["issuer"] == RESOURCE + "/oauth"
        assert value["token_endpoint"] == RESOURCE + "/oauth/token"
        assert value["authorization_endpoint"] == RESOURCE + "/oauth/authorize"
        assert value["code_challenge_methods_supported"] == ["S256"]
        assert "registration_endpoint" not in value
        assert urlsplit(value["revocation_endpoint"]).path == "/mcp/owned/oauth/revoke"


async def test_actual_official_client_uses_native_authenticated_streamable_http(consent_http):
    native = runtime(consent_http)
    code = await approved_code(consent_http)
    async with (native.app.router.lifespan_context(native.app),
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=native.app),
            base_url="https://kestrel.example") as http):
        response = await http.post(RESOURCE + "/oauth/token", data=code_request(code))
        assert response.status_code == 200, response.text
        http.headers["authorization"] = "Bearer " + response.json()["access_token"]
        async with Client(streamable_http_client(RESOURCE, http_client=http), mode="2026-07-28") as client:
            assert (await client.list_tools()).tools == []
        foreign = await http.post(RESOURCE + "/another-agent", json={"jsonrpc": "2.0", "id": 2, "method": "ping"})
        assert foreign.status_code == 404


async def test_native_oauth_body_limit_refuses_before_code_consumption(consent_http):
    native = runtime(consent_http)
    code = await approved_code(consent_http)
    async with (native.app.router.lifespan_context(native.app),
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=native.app),
            base_url="https://kestrel.example") as client):
        response = await client.post(RESOURCE + "/oauth/token", content=b"x" * (consent_http[1].policy.max_request_bytes + 1),
            headers={"content-type": "application/x-www-form-urlencoded"})
        assert response.status_code == 413
        response = await client.post(RESOURCE + "/oauth/token", data=code_request(code))
        assert response.status_code == 200, response.text
