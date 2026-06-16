"""Unit tests for the Claude subscription (OAuth/plan route) request shaping
and OAuth token-refresh lifecycle.

The plan route's ``sk-ant-oat`` token is rejected by Anthropic's subscription
endpoint unless the request is shaped like Claude Code (first system block ==
the identity string). These tests pin that shaping and prove it applies ONLY
to the OAuth route, never the metered API-key route.
"""
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from kestrel_sovereign.llm.anthropic_adapter import (
    CACHE_CONTROL_EPHEMERAL,
    CLAUDE_CODE_IDENTITY,
    AnthropicAdapter,
    _CLAUDE_CODE_BETA,
    _OAUTH_BETA,
)
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter
from kestrel_sovereign.llm.anthropic_oauth import (
    ClaudeOAuthTokenManager,
    OAuthCredentials,
    _coerce_expires_at,
    parse_credentials,
    refresh_anthropic_token,
)


async def _capture(adapter: AnthropicAdapter, messages: List[Dict[str, Any]], **kw) -> Dict:
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(type="text", text="ok")],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=1),
        )
    )
    await adapter.get_response(
        client=fake_client, model="claude-sonnet-4-5-20250929", messages=messages, **kw
    )
    return fake_client.messages.create.call_args.kwargs


# ---------------------------------------------------------------------------
# Request shaping: gated to the OAuth route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_route_does_not_inject_identity():
    """The metered API-key route (base AnthropicAdapter) must NOT add the
    Claude Code identity block or the OAuth betas."""
    messages = [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "hi"}]
    captured = await _capture(AnthropicAdapter(), messages)
    system = captured["system"]
    assert system[0]["text"] == "Be helpful."
    assert all(b["text"] != CLAUDE_CODE_IDENTITY for b in system)
    beta = (captured.get("extra_headers") or {}).get("anthropic-beta", "")
    assert _OAUTH_BETA not in beta and _CLAUDE_CODE_BETA not in beta


@pytest.mark.asyncio
async def test_plan_route_prepends_identity_as_first_block():
    """ClaudeMaxAdapter (OAuth/plan) prepends the identity as the FIRST system
    block, with the real system following — the shape Anthropic requires."""
    messages = [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "hi"}]
    captured = await _capture(ClaudeMaxAdapter(), messages)
    system = captured["system"]
    assert system[0]["text"] == CLAUDE_CODE_IDENTITY
    assert system[1]["text"] == "Be helpful."
    # Cache breakpoint stays on the trailing real-system block (covers identity).
    assert system[-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL
    beta = captured["extra_headers"]["anthropic-beta"]
    assert _CLAUDE_CODE_BETA in beta and _OAUTH_BETA in beta


@pytest.mark.asyncio
async def test_plan_route_identity_when_no_system_prompt():
    """With no system prompt, the identity block is still injected (and
    cache-marked as a stable anchor) so the request is accepted."""
    messages = [{"role": "user", "content": "hi"}]
    captured = await _capture(ClaudeMaxAdapter(), messages)
    system = captured["system"]
    assert system[0]["text"] == CLAUDE_CODE_IDENTITY
    assert system[0]["cache_control"] == CACHE_CONTROL_EPHEMERAL


@pytest.mark.asyncio
async def test_plan_route_identity_first_with_tools_and_history():
    """Identity stays first even with tools + multi-turn history; history cache
    markers are unaffected."""
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    captured = await _capture(ClaudeMaxAdapter(), messages, tools=tools)
    assert captured["system"][0]["text"] == CLAUDE_CODE_IDENTITY
    # Penultimate (assistant a1) still cache-marked.
    assert captured["messages"][-2]["content"][-1]["cache_control"] == CACHE_CONTROL_EPHEMERAL


# ---------------------------------------------------------------------------
# Credentials parsing
# ---------------------------------------------------------------------------


def test_coerce_expires_at_ms_vs_seconds():
    assert _coerce_expires_at(1_700_000_000_000) == 1_700_000_000.0  # ms → s
    assert _coerce_expires_at(1_700_000_000) == 1_700_000_000.0      # already s
    assert _coerce_expires_at(None) is None
    assert _coerce_expires_at(0) is None


def test_parse_credentials_claude_code_wrapper():
    creds = parse_credentials(
        {"claudeAiOauth": {"accessToken": "sk-ant-oat-a", "refreshToken": "r", "expiresAt": 1_700_000_000_000}}
    )
    assert creds.access == "sk-ant-oat-a"
    assert creds.refresh == "r"
    assert creds.expires_at == 1_700_000_000.0


def test_parse_credentials_snake_case_and_missing_access():
    assert parse_credentials({"access_token": "x", "refresh_token": "y"}).refresh == "y"
    assert parse_credentials({"refresh_token": "only"}) is None


def test_needs_refresh_requires_refresh_and_expiry():
    # No refresh token → never proactively refresh (static setup-token case).
    assert OAuthCredentials(access="a", refresh=None, expires_at=1.0).needs_refresh(now=10.0) is False
    # No expiry → unknown lifetime → don't refresh.
    assert OAuthCredentials(access="a", refresh="r", expires_at=None).needs_refresh(now=10.0) is False
    # Within skew → refresh.
    assert OAuthCredentials(access="a", refresh="r", expires_at=1000.0).needs_refresh(now=800.0, skew=300) is True
    # Comfortably fresh → no refresh.
    assert OAuthCredentials(access="a", refresh="r", expires_at=1000.0).needs_refresh(now=100.0, skew=300) is False


# ---------------------------------------------------------------------------
# Token manager
# ---------------------------------------------------------------------------


def test_from_sources_static_only_and_empty():
    assert ClaudeOAuthTokenManager.from_sources(static_token="sk-ant-oat-s", credentials_path=None) is not None
    assert ClaudeOAuthTokenManager.from_sources(static_token=None, credentials_path=None) is None


@pytest.mark.asyncio
async def test_access_token_static_returns_unchanged():
    mgr = ClaudeOAuthTokenManager(OAuthCredentials(access="sk-ant-oat-static"))
    assert await mgr.access_token() == "sk-ant-oat-static"


@pytest.mark.asyncio
async def test_access_token_refreshes_when_near_expiry(monkeypatch):
    # Credentials that are within the skew window → must refresh.
    mgr = ClaudeOAuthTokenManager(
        OAuthCredentials(access="old", refresh="refresh-1", expires_at=1000.0)
    )
    monkeypatch.setattr("kestrel_sovereign.llm.anthropic_oauth.time.time", lambda: 900.0)

    async def fake_refresh(refresh_token, **kw):
        assert refresh_token == "refresh-1"
        return OAuthCredentials(access="new", refresh="refresh-2", expires_at=99_999.0)

    monkeypatch.setattr(
        "kestrel_sovereign.llm.anthropic_oauth.refresh_anthropic_token", fake_refresh
    )
    assert await mgr.access_token() == "new"
    # Rotated refresh token retained; a second call does not refresh again.
    assert await mgr.access_token() == "new"


def test_from_sources_bootstraps_from_credentials_file(tmp_path):
    """A credentials file ALONE (no static token) must build a manager — the
    OAuth route can be configured by file only."""
    import json

    f = tmp_path / "creds.json"
    f.write_text(json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat-file", "refreshToken": "r"}}))
    mgr = ClaudeOAuthTokenManager.from_sources(static_token=None, credentials_path=str(f))
    assert mgr is not None
    assert mgr.initial_access_token == "sk-ant-oat-file"


@pytest.mark.asyncio
async def test_persist_preserves_claude_code_wrapper_shape(tmp_path, monkeypatch):
    """Refreshing a Claude Code ``{"claudeAiOauth": ...}`` file must keep the
    wrapper, camelCase keys, ms expiry, and unrelated fields — not flatten it."""
    import json

    f = tmp_path / "creds.json"
    f.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "old",
                    "refreshToken": "refresh-1",
                    "expiresAt": 1000_000,
                    "scopes": ["user:inference"],
                },
                "otherTool": {"keep": "me"},
            }
        )
    )
    mgr = ClaudeOAuthTokenManager.from_sources(static_token=None, credentials_path=str(f))
    monkeypatch.setattr("kestrel_sovereign.llm.anthropic_oauth.time.time", lambda: 2_000_000.0)

    async def fake_refresh(refresh_token, **kw):
        return OAuthCredentials(access="new", refresh="refresh-2", expires_at=1_700_000_000.0)

    monkeypatch.setattr("kestrel_sovereign.llm.anthropic_oauth.refresh_anthropic_token", fake_refresh)
    assert await mgr.access_token() == "new"

    written = json.loads(f.read_text())
    assert "claudeAiOauth" in written  # wrapper preserved
    block = written["claudeAiOauth"]
    assert block["accessToken"] == "new"  # camelCase kept
    assert block["refreshToken"] == "refresh-2"
    assert block["expiresAt"] == 1_700_000_000_000  # written back in ms
    assert block["scopes"] == ["user:inference"]  # unrelated field kept
    assert written["otherTool"] == {"keep": "me"}  # sibling field kept


@pytest.mark.asyncio
async def test_refresh_anthropic_token_posts_grant(monkeypatch):
    calls = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        calls["body"] = json.loads(request.content)
        return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "rot", "expires_in": 3600})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        creds = await refresh_anthropic_token("rt", http_client=client)
    assert creds.access == "fresh"
    assert creds.refresh == "rot"
    assert calls["body"]["grant_type"] == "refresh_token"
    assert calls["body"]["refresh_token"] == "rt"


@pytest.mark.asyncio
async def test_refresh_anthropic_token_raises_on_http_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "bad"}))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="HTTP 401"):
            await refresh_anthropic_token("rt", http_client=client)
