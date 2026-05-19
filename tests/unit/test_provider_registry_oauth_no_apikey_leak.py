"""Regression guard: the Anthropic OAuth / Claude-Max ``plan`` route must
never authenticate with ``ANTHROPIC_API_KEY``.

Bug: ``_build_client_and_adapter`` built the OAuth client as
``anthropic.AsyncAnthropic(auth_token=token)`` without suppressing the
API key. The Anthropic SDK (>=0.75) back-fills ``api_key`` from
``ANTHROPIC_API_KEY`` in the environment when the constructor arg is
``None``, and ``auth_headers`` then emits **both** ``X-Api-Key`` and
``Authorization: Bearer``. An agent mandated to ``anthropic:plan``
(subscription OAuth) was therefore silently billing/authenticating
against the metered API key, and broke with an "api key" error the
moment that key was disabled — even though its OAuth token was valid.

The fix: a client built for the ``auth_token`` path must send the
Bearer header ONLY, regardless of whether ``ANTHROPIC_API_KEY`` is set
in the process environment.
"""
from __future__ import annotations

import anthropic
import pytest

from kestrel_sovereign.llm.provider_registry import ProviderRegistry
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter


def _empty_registry() -> ProviderRegistry:
    reg = ProviderRegistry.__new__(ProviderRegistry)
    reg.config = {}
    return reg


class TestOAuthRouteDoesNotLeakApiKey:
    @pytest.mark.parametrize(
        "adapter_cls, route",
        [(ClaudeMaxAdapter, "plan"), (AnthropicAdapter, "api")],
    )
    def test_auth_token_client_sends_bearer_only(
        self, adapter_cls, route, monkeypatch
    ):
        """With ``ANTHROPIC_API_KEY`` present in the env, an auth_token
        (OAuth) route must still send ONLY ``Authorization: Bearer`` —
        never ``X-Api-Key``."""
        # Hermetic: pin BOTH credentials to fakes so the test never reads
        # (or logs) the developer's real ~/.env tokens, which conftest
        # loads into the process environment.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-MUST-NOT-BE-USED")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-oat01-FAKE-OAUTH-TOKEN")

        reg = _empty_registry()
        route_cfg = {
            "auth_token_env": "ANTHROPIC_AUTH_TOKEN",
        }
        client, adapter = reg._build_client_and_adapter(
            vendor="anthropic",
            route=route,
            adapter_cls=adapter_cls,
            vendor_cfg={},
            route_cfg=route_cfg,
        )
        assert isinstance(client, anthropic.AsyncAnthropic)

        headers = client.auth_headers
        assert "X-Api-Key" not in headers, (
            "OAuth/plan route is leaking ANTHROPIC_API_KEY as an X-Api-Key "
            "header — the subscription route is authenticating/billing "
            "against the metered API key. Disabling that key then breaks "
            "the agent with a spurious 'api key' error."
        )
        assert headers.get("Authorization") == (
            "Bearer sk-ant-oat01-FAKE-OAUTH-TOKEN"
        ), "expected Bearer OAuth auth from the fake token"

        # The SDK's own header validation must still accept the request.
        client._validate_headers(headers, {})

    def test_api_key_route_still_uses_x_api_key(self, monkeypatch):
        """Sanity: the genuine API-key route is unaffected — it must
        still authenticate with X-Api-Key and no Bearer."""
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-REAL-KEY")
        reg = _empty_registry()
        route_cfg = {
            "api_key_env": "ANTHROPIC_API_KEY",
        }
        client, _ = reg._build_client_and_adapter(
            vendor="anthropic",
            route="api",
            adapter_cls=AnthropicAdapter,
            vendor_cfg={},
            route_cfg=route_cfg,
        )
        headers = client.auth_headers
        assert headers.get("X-Api-Key") == "sk-ant-api03-REAL-KEY"
        assert "Authorization" not in headers
