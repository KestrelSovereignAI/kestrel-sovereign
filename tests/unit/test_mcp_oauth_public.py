"""Public callback shell carries no authority or reflected callback credentials."""

from fastapi import FastAPI
from fastapi.routing import iter_route_contexts
import httpx2
import pytest

from kestrel_sovereign import server
from kestrel_sovereign.endpoints.mcp_oauth import router
from kestrel_sovereign.mcp_oauth import MCPOAuthHostPolicy
from kestrel_sovereign.security.access_log import (
    PRIVATE_CAPABILITY_URL_PATTERN, SensitiveQueryStringRedactionMiddleware,
)
from uvicorn.protocols.utils import get_path_with_query_string


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False])
async def test_native_public_identity_and_callback_shell(configured, monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "owned-public-oauth-key")
    monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "1")
    app = FastAPI()
    app.state.mcp_oauth_policy = (MCPOAuthHostPolicy(console_origin="https://console.example")
                                 if configured else None)
    app.middleware("http")(server.auth_middleware)
    app.include_router(router)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app),
                                 base_url="https://console.example") as client:
        metadata = await client.get("/oauth/mcp/client-metadata.json")
        callback = await client.get("/oauth/mcp/callback?code=private-code&state=private-state&iss=private-issuer")
        assert metadata.status_code == callback.status_code == (200 if configured else 404)
        if configured:
            assert metadata.json() == app.state.mcp_oauth_policy.client_document()
            assert metadata.json()["application_type"] == "web"
            assert callback.headers["referrer-policy"] == "no-referrer"
            assert callback.headers["cache-control"] == "no-store"
            assert "frame-ancestors 'none'" in callback.headers["content-security-policy"]
            assert '<script type="module" src="/js/mcp-oauth-callback.mjs">' in callback.text
            assert "private-code" not in callback.text and "private-state" not in callback.text
            assert "private-issuer" not in callback.text
            assert callback.headers.get("set-cookie") is None


def test_native_server_wires_public_identity_and_callback_routes():
    paths = {context.path for context in iter_route_contexts(server.app.routes)}
    assert {"/oauth/mcp/callback", "/oauth/mcp/client-metadata.json"} <= paths


@pytest.mark.asyncio
async def test_callback_access_log_redacts_all_parameters_at_actual_send_seam():
    import re
    path = "/oauth/mcp/callback"
    query = b"code=private-owned-code&state=private-owned-state&iss=https%3A%2F%2Fissuer.example"
    scope = dict(type="http", path=path, raw_path=path.encode(), query_string=query)
    observed = []
    async def app(inner, receive, send):
        assert inner["query_string"] == query
        await send(dict(type="http.response.start", status=200, headers=[]))
        assert inner["query_string"] == query
    async def send(message):
        observed.append(get_path_with_query_string(scope))
    await SensitiveQueryStringRedactionMiddleware(app)(scope, None, send)
    assert observed == [path]
    assert scope["query_string"] == query
    assert re.search(PRIVATE_CAPABILITY_URL_PATTERN, path + "?" + query.decode())


def test_native_sdk_oauth_logger_drops_secrets_and_exception_traceback(caplog):
    import logging
    from mcp.client.auth import oauth2
    with caplog.at_level(logging.WARNING, logger=oauth2.logger.name):
        try:
            raise ValueError("private-state-value private-code-value private-token-value")
        except ValueError:
            oauth2.logger.exception("private-state-value %s", "private-code-value", stack_info=True)
    assert "private-state-value" not in caplog.text
    assert "private-code-value" not in caplog.text and "private-token-value" not in caplog.text
    assert "MCP OAuth diagnostic redacted" in caplog.text
    assert caplog.records[-1].exc_info is None and caplog.records[-1].stack_info is None


def test_configured_web_client_refuses_to_start_without_callback_assets():
    import os
    import subprocess
    import sys
    env = dict(os.environ, KESTREL_SERVE_UI="false",
               KESTREL_MCP_OAUTH_CONSOLE_ORIGIN="https://console.example")
    result = subprocess.run([sys.executable, "-B", "-c", "import kestrel_sovereign.server"],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "MCP OAuth web client identity requires KESTREL_SERVE_UI=true" in result.stderr
