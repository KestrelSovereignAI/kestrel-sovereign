"""Tests for the OpenAI plan adapter and registry integration."""

import json
from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.adapter import LLMAdapter
from kestrel_sovereign.llm.codex_adapter import (
    CodexAdapter,
    _build_headers,
    _convert_tools_to_responses_format,
    _extract_account_id,
    _extract_instructions_and_input,
)
from kestrel_sovereign.llm.provider_registry import ProviderRegistry


class TestOpenAIPlanAdapterClass:
    def test_is_llm_adapter_subclass(self):
        adapter = CodexAdapter()
        assert isinstance(adapter, LLMAdapter)

    def test_name_is_openai_plan(self):
        adapter = CodexAdapter()
        assert adapter.name == "openai_plan"


class TestOpenAIPlanListModels:
    @pytest.mark.asyncio
    async def test_list_models_is_not_supported_directly(self):
        adapter = CodexAdapter()
        with pytest.raises(NotImplementedError, match="canonical openai provider"):
            await adapter.list_models()


class TestMessageHelpers:
    def test_extract_system_prompt(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        instructions, inputs = _extract_instructions_and_input(messages)
        assert instructions == "You are helpful"
        assert len(inputs) == 1
        assert inputs[0]["role"] == "user"

    def test_extract_no_system_prompt(self):
        instructions, inputs = _extract_instructions_and_input([{"role": "user", "content": "Hello"}])
        assert instructions is None
        assert len(inputs) == 1

    def test_extract_structured_system_content(self):
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]},
        ]
        instructions, _ = _extract_instructions_and_input(messages)
        assert "Part 1" in instructions
        assert "Part 2" in instructions

    def test_convert_tools_to_responses_format(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        result = _convert_tools_to_responses_format(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "get_weather"

    def test_convert_tools_none(self):
        assert _convert_tools_to_responses_format(None) is None

    def test_convert_tools_empty(self):
        assert _convert_tools_to_responses_format([]) is None


class TestAccountIdExtraction:
    def _make_jwt(self, claims: dict) -> str:
        import base64

        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
        sig = base64.urlsafe_b64encode(b"fake-sig").rstrip(b"=").decode()
        return f"{header}.{payload}.{sig}"

    def test_extracts_account_id(self):
        token = self._make_jwt({
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "test-account-123",
            }
        })
        assert _extract_account_id(token) == "test-account-123"

    def test_raises_on_missing_claim(self):
        token = self._make_jwt({"sub": "user123"})
        with pytest.raises(ValueError, match="No chatgpt_account_id"):
            _extract_account_id(token)

    def test_raises_on_invalid_token(self):
        with pytest.raises(ValueError, match="Failed to extract"):
            _extract_account_id("not-a-jwt")


class TestBuildHeaders:
    def test_includes_required_headers(self):
        headers = _build_headers("tok", "acct-123")
        assert headers["Authorization"] == "Bearer tok"
        assert headers["chatgpt-account-id"] == "acct-123"
        assert headers["OpenAI-Beta"] == "responses=experimental"
        assert "kestrel" in headers["User-Agent"]
        assert headers["Accept"] == "text/event-stream"
        assert headers["Content-Type"] == "application/json"


class TestOpenAIPlanProviderRegistry:
    """Vendor/route registry initializes the openai:plan route correctly."""

    def _plan_config(self, *, auth_token_env="CODEX_AUTH_TOKEN", inline_auth=None):
        route_cfg = {"adapter": "CodexAdapter"}
        if auth_token_env:
            route_cfg["auth_token_env"] = auth_token_env
        if inline_auth is not None:
            route_cfg["auth_token"] = inline_auth
        return {
            "route_priority": ["openai:plan"],
            "vendors": {
                "openai": {
                    "is_cloud": True,
                    "routes": {"plan": route_cfg},
                },
            },
        }

    def test_openai_plan_requires_auth(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                ProviderRegistry, "_read_codex_auth_file", return_value=(None, None)
            ):
                registry = ProviderRegistry(self._plan_config())
                # No providers initialize → raises.
                with pytest.raises(Exception, match="No routes could be initialized"):
                    registry.initialize_providers()

    def test_openai_plan_uses_env_token(self):
        with patch.dict("os.environ", {"CODEX_AUTH_TOKEN": "test-oauth-token"}):
            with patch.object(
                ProviderRegistry, "_read_codex_auth_file", return_value=(None, None)
            ):
                registry = ProviderRegistry(self._plan_config())
                providers = registry.initialize_providers()
                assert len(providers) == 1
                info = providers[0]
                assert info.name == "openai:plan"
                assert info.vendor == "openai"
                assert info.route == "plan"
                assert isinstance(info.adapter, CodexAdapter)
                assert info.client == "test-oauth-token"

    def test_openai_plan_reads_auth_file(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                ProviderRegistry,
                "_read_codex_auth_file",
                return_value=("file-oauth-token", "chatgpt"),
            ):
                registry = ProviderRegistry(self._plan_config())
                providers = registry.initialize_providers()
                assert len(providers) == 1
                info = providers[0]
                assert info.name == "openai:plan"
                assert info.client == "file-oauth-token"

    def test_openai_plan_config_auth_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                ProviderRegistry, "_read_codex_auth_file", return_value=(None, None)
            ):
                registry = ProviderRegistry(
                    self._plan_config(auth_token_env=None, inline_auth="config-token")
                )
                providers = registry.initialize_providers()
                assert len(providers) == 1
                info = providers[0]
                assert info.name == "openai:plan"
                assert info.client == "config-token"

    def test_openai_plan_model_defaults_to_auto(self):
        with patch.dict("os.environ", {"CODEX_AUTH_TOKEN": "tok"}):
            with patch.object(
                ProviderRegistry, "_read_codex_auth_file", return_value=(None, None)
            ):
                registry = ProviderRegistry(self._plan_config())
                providers = registry.initialize_providers()
                assert providers[0].model == "auto"


class TestReadCodexAuthFile:
    def test_reads_access_token(self, tmp_path):
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir()
        auth_file.write_text(json.dumps({
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": "my-oauth-token",
                "refresh_token": "my-refresh",
            },
        }))
        with patch("pathlib.Path.home", return_value=tmp_path):
            token, mode = ProviderRegistry._read_codex_auth_file()
        assert token == "my-oauth-token"
        assert mode == "chatgpt"

    def test_returns_none_when_no_file(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            token, mode = ProviderRegistry._read_codex_auth_file()
        assert token is None
        assert mode is None

    def test_returns_none_on_invalid_json(self, tmp_path):
        auth_file = tmp_path / ".codex" / "auth.json"
        auth_file.parent.mkdir()
        auth_file.write_text("not json")
        with patch("pathlib.Path.home", return_value=tmp_path):
            token, mode = ProviderRegistry._read_codex_auth_file()
        assert token is None
        assert mode is None
