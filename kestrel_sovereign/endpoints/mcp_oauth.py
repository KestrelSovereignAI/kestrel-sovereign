"""Public web OAuth client identity and callback shell; neither grants authority."""

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import HTMLResponse, JSONResponse

from kestrel_sovereign.mcp_oauth import (
    MCP_OAUTH_CALLBACK_PATH,
    MCP_OAUTH_CLIENT_METADATA_PATH,
    MCPOAuthHostPolicy,
)
from kestrel_sovereign.security.browser_shell import OAUTH_BROWSER_HEADERS

router = APIRouter(tags=["mcp-oauth"])
_CALLBACK_HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Kestrel MCP authorization</title>
<main><h1>MCP authorization</h1><p id="oauth-status">Completing authorization…</p>
<a id="oauth-return" hidden>Return to Console</a></main>
<script type="module" src="/js/mcp-oauth-callback.mjs"></script></html>"""


def _policy(request: Request) -> MCPOAuthHostPolicy:
    value = getattr(request.app.state, "mcp_oauth_policy", None)
    if not isinstance(value, MCPOAuthHostPolicy):
        raise HTTPException(status_code=404, detail="mcp_oauth_unconfigured")
    return value


@router.get(MCP_OAUTH_CLIENT_METADATA_PATH)
async def client_metadata(request: Request):
    return JSONResponse(_policy(request).client_document(), headers=OAUTH_BROWSER_HEADERS)


@router.get(MCP_OAUTH_CALLBACK_PATH, response_class=HTMLResponse)
async def callback_page(request: Request):
    _policy(request)
    return HTMLResponse(_CALLBACK_HTML, headers=OAUTH_BROWSER_HEADERS)
