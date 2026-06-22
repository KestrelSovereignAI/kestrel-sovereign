"""
Unit tests for LLMService core methods.

Tests the unified LLM provider manager with mocked dependencies.
No real API calls are made.
"""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from pydantic import BaseModel

from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.llm.service import (
    BackendType,
    LLMService,
    LLMServiceError,
    RemoteGPUConfig,
)
from kestrel_sovereign.llm.error_handling import LLMAllProvidersFailedError, LLMProviderError


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Mock configuration for LLMService."""
    return {
        "provider_priority": ["openai", "anthropic"],
        "openai": {
            "api_key": "sk-test-key",
            "model": "gpt-5-mini",
            "base_url": "https://api.openai.com/v1",
        },
        "anthropic": {
            "api_key": "sk-ant-test-key",
            "model": "claude-sonnet-4-5",
            "base_url": "https://api.anthropic.com/v1",
        },
    }


@pytest.fixture
def mock_mandate_config():
    """Mock model mandate configuration."""
    return {
        "defaults": {
            "preferred": "",
            "cheap_model": "auto",
            "cheap_model_hints": ["haiku", "mini", "flash"],
            "banned": ["gpt-3"],
        },
        "mandates": {
            "vision": "openai",
            "code": "anthropic",
        },
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI async client."""
    client = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_adapter():
    """Mock LLM adapter."""
    adapter = Mock()
    adapter.create_messages = Mock(return_value=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ])
    adapter.get_response = AsyncMock(return_value=LLMResponse(
        content="Hello! How can I help you?",
        input_tokens=10,
        output_tokens=8,
        total_tokens=18,
    ))
    return adapter


@pytest.fixture
def mock_provider_registry(mock_openai_client, mock_adapter):
    """Mock ProviderRegistry."""
    registry = Mock()

    # Mock ProviderInfo objects. Under the vendor/route architecture each
    # entry represents a route; `name` is the composite "vendor:route" key,
    # `vendor` carries the grouping dimension, `route` the per-route identity.
    provider_info_openai = Mock(spec=[
        "name", "vendor", "route", "client", "adapter", "model",
        "is_cloud", "is_local", "base_url", "selection_hints",
    ])
    provider_info_openai.name = "openai:api"
    provider_info_openai.vendor = "openai"
    provider_info_openai.route = "api"
    provider_info_openai.client = mock_openai_client
    provider_info_openai.adapter = mock_adapter
    provider_info_openai.model = "gpt-5-mini"
    provider_info_openai.is_cloud = True
    provider_info_openai.is_local = False
    provider_info_openai.base_url = None
    provider_info_openai.selection_hints = []

    provider_info_anthropic = Mock(spec=[
        "name", "vendor", "route", "client", "adapter", "model",
        "is_cloud", "is_local", "base_url", "selection_hints",
    ])
    provider_info_anthropic.name = "anthropic:api"
    provider_info_anthropic.vendor = "anthropic"
    provider_info_anthropic.route = "api"
    provider_info_anthropic.client = AsyncMock()
    provider_info_anthropic.adapter = mock_adapter
    provider_info_anthropic.model = "claude-sonnet-4-5"
    provider_info_anthropic.is_cloud = True
    provider_info_anthropic.is_local = False
    provider_info_anthropic.base_url = None
    provider_info_anthropic.selection_hints = []

    provider_info_cheap = Mock(spec=[
        "name", "vendor", "route", "client", "adapter", "model",
        "is_cloud", "is_local", "base_url", "selection_hints",
    ])
    provider_info_cheap.name = "anthropic:api"
    provider_info_cheap.vendor = "anthropic"
    provider_info_cheap.route = "api"
    provider_info_cheap.client = AsyncMock()
    provider_info_cheap.adapter = mock_adapter
    provider_info_cheap.model = "claude-haiku-4-5"
    provider_info_cheap.is_cloud = True
    provider_info_cheap.is_local = False
    provider_info_cheap.base_url = None
    provider_info_cheap.selection_hints = []

    registry.initialize_providers = Mock(return_value=[
        provider_info_openai,
        provider_info_anthropic,
    ])
    registry.get_provider_by_name = Mock(return_value=provider_info_anthropic)
    registry.get_providers_with_pattern = Mock(return_value=[provider_info_cheap])
    registry.update_provider_client = Mock(return_value=True)

    return registry


@pytest_asyncio.fixture
async def llm_service(mock_config, mock_mandate_config, mock_provider_registry):
    """Create LLMService with mocked dependencies."""
    with patch("kestrel_sovereign.llm.service.load_config") as mock_load_config, \
         patch("kestrel_sovereign.llm.service.ProviderRegistry") as mock_registry_class:

        # Setup config mocking
        mock_load_config.side_effect = lambda path: (
            mock_config if "llm_config" in path else mock_mandate_config
        )

        # Setup registry mocking
        mock_registry_class.return_value = mock_provider_registry

        service = LLMService()

        # Mock the usage tracking and constitutional profiles (which set attributes)
        service._usage_db = None
        service._db_initialized = False
        service._usage_database_url = None
        service._db_backend = "sqlite"

        yield service

        # Cleanup
        await service.close()


# =============================================================================
# Priority 1 — Model Preference Tests
# =============================================================================


class TestModelPreference:
    """Tests for model preference methods."""

    @pytest.mark.asyncio
    async def test_set_model_preference_with_vendor(self, llm_service):
        """Setting with an explicit vendor stores it in the mandate."""
        llm_service.set_model_preference("gpt-5", vendor="openai")

        pref = llm_service.get_model_preference()
        assert pref["model"] == "gpt-5"
        assert pref["vendor"] == "openai"
        assert pref["route"] is None

    @pytest.mark.asyncio
    async def test_set_model_preference_with_vendor_and_route(self, llm_service):
        """Route narrows routing to the exact <vendor>:<route> entry."""
        llm_service.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        pref = llm_service.get_model_preference()
        assert pref["vendor"] == "anthropic"
        assert pref["route"] == "plan"
        assert pref["model"] == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_set_model_preference_without_vendor_resolves_or_refuses(self, llm_service):
        """Bare model with no vendor must auto-resolve from catalog or raise.

        A bare mandate (``vendor=None``) used to persist as-is and broadcast
        to every provider on the next request — the cascade that caused a
        "switch to gpt-5-mini" to end up served by OpenRouter as a Gemini
        model. Now, set_model_preference either resolves the vendor via
        discovery or raises ValueError. The mandate must name a vendor.
        """
        from unittest.mock import MagicMock, patch

        # Empty discovery cache → refusal, mandate untouched.
        cache = MagicMock()
        cache.get_any = MagicMock(return_value=None)
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError):
                llm_service.set_model_preference("claude-sonnet-4-5")

        pref = llm_service.get_model_preference()
        assert pref == {"vendor": None, "model": None, "route": None}

    @pytest.mark.asyncio
    async def test_clear_model_preference(self, llm_service):
        """Clearing returns all three slots to None."""
        llm_service.set_model_preference("gpt-5", vendor="openai")
        llm_service.clear_model_preference()

        pref = llm_service.get_model_preference()
        assert pref["model"] is None
        assert pref["vendor"] is None
        assert pref["route"] is None

    @pytest.mark.asyncio
    async def test_model_preference_persistence_tasks_are_owned(self, llm_service):
        """Preference persistence is scheduled through owned service tasks."""
        calls = []

        async def persist(model, vendor, route):
            calls.append((model, vendor, route))

        llm_service.set_preference_persistence_callback(persist)
        llm_service.set_model_preference("gpt-5", vendor="openai")

        assert len(llm_service._preference_persistence_tasks) == 1

        await llm_service.drain_preference_persistence()

        assert calls == [("gpt-5", "openai", None)]
        assert llm_service._preference_persistence_tasks == set()

    @pytest.mark.asyncio
    async def test_close_waits_for_preference_persistence(self, llm_service):
        """close() waits for pending preference persistence before cleanup returns."""
        calls = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def persist(model, vendor, route):
            calls.append(("start", model, vendor, route))
            started.set()
            await release.wait()
            calls.append(("done", model, vendor, route))

        llm_service.providers = []
        llm_service.set_preference_persistence_callback(persist)
        llm_service.set_model_preference("gpt-5", vendor="openai")
        await started.wait()

        close_task = asyncio.create_task(llm_service.close())
        await asyncio.sleep(0)

        assert not close_task.done()

        release.set()
        await close_task

        assert calls == [
            ("start", "gpt-5", "openai", None),
            ("done", "gpt-5", "openai", None),
        ]
        assert llm_service._preference_persistence_tasks == set()

    @pytest.mark.asyncio
    async def test_get_model_preference_default(self, llm_service):
        """Default preference has None for all three slots (vendor, model, route)."""
        pref = llm_service.get_model_preference()
        assert isinstance(pref, dict)
        assert set(pref.keys()) == {"vendor", "model", "route"}
        assert pref["vendor"] is None
        assert pref["model"] is None
        assert pref["route"] is None

    @pytest.mark.asyncio
    async def test_get_cheap_model_from_config(self, llm_service):
        """Test get_cheap_model resolves configured selector via discovery-backed hints."""
        cheap_model = llm_service.get_cheap_model()
        assert cheap_model == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_get_cheap_model_returns_none_without_config_hints(self, llm_service, mock_provider_registry):
        """Test get_cheap_model requires config hints instead of hidden code defaults."""
        # Clear the config
        llm_service.mandate_config = {"defaults": {}}

        cheap_model = llm_service.get_cheap_model()
        assert cheap_model is None


# =============================================================================
# Priority 2 — Core Generation Tests
# =============================================================================


class TestCoreGeneration:
    """Tests for core generation methods."""

    @pytest.mark.asyncio
    async def test_get_response_basic(self, llm_service):
        """Test basic get_response with mocked provider."""
        response = await llm_service.get_response(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello, world!",
        )

        assert isinstance(response, str)
        assert "Hello" in response or "help" in response

    @pytest.mark.asyncio
    async def test_get_response_with_tools(self, llm_service, mock_adapter):
        """Test get_response with tool calling."""
        # Mock response with tool calls
        mock_adapter.get_response = AsyncMock(return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"})
            ],
            input_tokens=20,
            output_tokens=15,
        ))

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        response = await llm_service.get_response(
            system_prompt="You have access to tools.",
            user_prompt="What's the weather in NYC?",
            tools=tools,
        )

        assert isinstance(response, LLMResponse)
        assert response.has_tool_calls

    @pytest.mark.asyncio
    async def test_usage_log_line_includes_cache_tokens(self, llm_service, mock_adapter, caplog):
        """Issue #819 — every successful call emits one ``llm.usage:`` JSON
        line so callers that downcast to a string don't lose token / cache
        telemetry. Verifies cache_creation_input_tokens and
        cache_read_input_tokens are surfaced when the adapter reports them."""
        import json as _json
        mock_adapter.get_response = AsyncMock(return_value=LLMResponse(
            content="hi",
            input_tokens=2103,
            output_tokens=42,
            total_tokens=2145,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=66482,
        ))

        with caplog.at_level("INFO", logger="kestrel_sovereign.llm.service"):
            await llm_service.get_response(
                system_prompt="big stable system block",
                user_prompt="trigger a warm cache hit",
            )

        usage_lines = [r for r in caplog.records if r.message.startswith("llm.usage: ")]
        assert usage_lines, "expected at least one llm.usage: log line"
        payload = _json.loads(usage_lines[-1].message.removeprefix("llm.usage: "))
        assert payload["input_tokens"] == 2103
        assert payload["output_tokens"] == 42
        assert payload["cache_creation_input_tokens"] == 0
        assert payload["cache_read_input_tokens"] == 66482
        assert "duration_ms" in payload
        assert "model" in payload
        assert payload["tools"] is False
        assert payload["structured_output"] is False

    @pytest.mark.asyncio
    async def test_get_response_provider_fallback(self, llm_service, mock_adapter):
        """Test provider fallback when first provider fails."""
        # First call fails, second succeeds
        mock_adapter.get_response = AsyncMock(
            side_effect=[
                LLMProviderError("openai", "OpenAI failed"),
                LLMResponse(content="Success from fallback", input_tokens=10, output_tokens=8),
            ]
        )

        response = await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Test prompt",
        )

        assert isinstance(response, str)
        assert "Success" in response

    @pytest.mark.asyncio
    async def test_get_response_all_providers_fail(self, llm_service, mock_adapter):
        """Test that exception is raised when all providers fail."""
        mock_adapter.get_response = AsyncMock(
            side_effect=LLMProviderError("openai", "Provider failed")
        )

        with pytest.raises(LLMAllProvidersFailedError):
            await llm_service.get_response(
                system_prompt="Test",
                user_prompt="Test prompt",
            )

    @pytest.mark.asyncio
    async def test_get_response_with_model_override(self, llm_service):
        """Test get_response with explicit model override."""
        response = await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Test prompt",
            model_override="gpt-5",
        )

        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_get_response_force_local_only(self, llm_service):
        """Test get_response with force_local_only flag."""
        # Add a local route — the force_local_only filter now uses is_local.
        local_provider = {
            "name": "ollama:local",
            "vendor": "ollama",
            "route": "local",
            "client": AsyncMock(),
            "adapter": Mock(),
            "model": "llama3.2:3b",
            "is_cloud": False,
            "is_local": True,
            "base_url": None,
            "selection_hints": [],
        }
        local_provider["adapter"].create_messages = Mock(return_value=[])
        local_provider["adapter"].get_response = AsyncMock(
            return_value=LLMResponse(content="Local response", input_tokens=5, output_tokens=5)
        )
        llm_service.providers.append(local_provider)

        response = await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Test prompt",
            force_local_only=True,
        )

        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_get_response_with_structured_output(self, llm_service, mock_adapter):
        """Test get_response with Pydantic response format."""
        class TestOutput(BaseModel):
            result: str
            confidence: float

        mock_adapter.get_response = AsyncMock(return_value=LLMResponse(
            content='{"result": "test", "confidence": 0.9}',
            input_tokens=10,
            output_tokens=8,
        ))

        response = await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Test prompt",
            response_format=TestOutput,
        )

        assert isinstance(response, LLMResponse)

    @pytest.mark.asyncio
    async def test_get_response_with_model(self, llm_service):
        """Test get_response_with_model using specific model."""
        response = await llm_service.get_response_with_model(
            model_id="gpt-5-mini",
            system_prompt="Test",
            user_prompt="Hello",
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_get_audit_response(self, llm_service, mock_adapter):
        """Test get_audit_response returns structured audit result without a dedicated audit model."""
        # Mock JSON response
        mock_adapter.get_response = AsyncMock(return_value=LLMResponse(
            content='{"risk_level": 1, "reasoning": "Normal response"}',
            input_tokens=50,
            output_tokens=20,
        ))

        audit_result = await llm_service.get_audit_response(
            text_to_audit="This is a normal helpful response."
        )

        assert isinstance(audit_result, dict)
        assert "risk_level" in audit_result
        assert "reasoning" in audit_result

    @pytest.mark.asyncio
    async def test_generate_basic(self, llm_service):
        """Test generate method (lower-level generation)."""
        response = await llm_service.generate(
            system_prompt="You are helpful",
            user_prompt="Say hello",
        )

        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_generate_with_messages(self, llm_service):
        """Test generate_with_messages for multi-turn conversations."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        response = await llm_service.generate_with_messages(messages=messages)

        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_generate_with_model_mandate(self, llm_service):
        """Test that model mandate preference is respected in generate_with_messages."""
        llm_service.set_model_preference("claude-sonnet-4-5", vendor="anthropic")

        messages = [{"role": "user", "content": "Test"}]
        response = await llm_service.generate_with_messages(messages=messages)

        assert response is not None


class TestMeteringAndCost:
    """#1804 — generate_with_messages must meter (it previously bypassed the
    metering callback). #1806 — the metering callback may receive the
    provider-reported per-call cost (e.g. OpenRouter usage.cost)."""

    @pytest.mark.asyncio
    async def test_generate_with_messages_meters(self, llm_service):
        """A callback set via set_metering_callback fires for the non-stream
        message path with provider/model/token counts (#1804)."""
        calls = []

        async def _cb(**kwargs):
            calls.append(kwargs)

        llm_service.set_metering_callback(_cb)
        llm_service.set_observability_context(companion_id="c1", user_id="u1")

        out = await llm_service.generate_with_messages(
            messages=[{"role": "user", "content": "hi"}]
        )

        assert isinstance(out, str)
        assert len(calls) == 1
        assert calls[0]["provider"] == "openai:api"
        assert calls[0]["prompt_tokens"] == 10
        assert calls[0]["completion_tokens"] == 8

    @pytest.mark.asyncio
    async def test_cost_forwarded_to_callback(self, llm_service, mock_adapter):
        """When the provider reports a per-call cost (OpenRouter usage.cost),
        a cost-aware callback receives it (#1806)."""
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=8, cost=0.0042)
        raw = SimpleNamespace(usage=usage)
        mock_adapter.get_response = AsyncMock(return_value=LLMResponse(
            content="hi", input_tokens=10, output_tokens=8, total_tokens=18, raw=raw,
        ))
        seen = []

        async def _cb(*, companion_id, user_id, provider, model,
                      prompt_tokens, completion_tokens, cost=None):
            seen.append(cost)

        llm_service.set_metering_callback(_cb)
        llm_service.set_observability_context(companion_id="c1", user_id="u1")

        await llm_service.generate_with_messages(
            messages=[{"role": "user", "content": "hi"}]
        )

        assert seen == [0.0042]

    @pytest.mark.asyncio
    async def test_legacy_callback_without_cost_still_called(self, llm_service):
        """A callback written against the original signature (no cost kwarg)
        is not handed an unexpected kwarg — backward compatible (#1806)."""
        calls = []

        async def _legacy(*, companion_id, user_id, provider, model,
                          prompt_tokens, completion_tokens):
            calls.append((provider, prompt_tokens, completion_tokens))

        llm_service.set_metering_callback(_legacy)
        assert llm_service._metering_callback_accepts_cost is False
        llm_service.set_observability_context(companion_id="c1", user_id="u1")

        await llm_service.generate_with_messages(
            messages=[{"role": "user", "content": "hi"}]
        )

        assert calls == [("openai:api", 10, 8)]

    def test_extract_provider_cost_variants(self):
        """_extract_provider_cost reads usage.cost, model_extra, and the
        streaming raw dict; returns None when absent."""
        f = LLMService._extract_provider_cost
        # provider usage object
        assert f(LLMResponse(raw=SimpleNamespace(
            usage=SimpleNamespace(cost=0.01)))) == 0.01
        # openai v2 model_extra fallback
        assert f(LLMResponse(raw=SimpleNamespace(
            usage=SimpleNamespace(model_extra={"cost": 0.02})))) == 0.02
        # streaming raw dict
        assert f(LLMResponse(raw={"cost": 0.03})) == 0.03
        # absent
        assert f(LLMResponse(raw=None)) is None
        assert f(LLMResponse(raw=SimpleNamespace(usage=None))) is None


class TestAutoRoutingSentinel:
    """Regression tests for #1408 — the literal string "auto" must never
    reach a provider client. It expresses routing intent ("pick whatever
    each route's default is"), not a model identity. Every vendor 404s
    when handed a model id of "auto", which is how the bug surfaced in
    frinz's story extraction pipeline."""

    @pytest.mark.asyncio
    async def test_get_response_with_auto_override_succeeds(self, llm_service):
        """get_response(model_override="auto", ...) succeeds — the sentinel
        is normalized at the router boundary and each route falls back to
        its own configured default."""
        response = await llm_service.get_response(
            system_prompt="You are a helpful assistant.",
            user_prompt="hi",
            model_override="auto",
        )
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_auto_never_reaches_adapter(self, llm_service, mock_adapter):
        """The string "auto" never appears as the ``model`` kwarg in any
        adapter.get_response call, regardless of how it arrives at the
        router (explicit override, mandate default, prompt mandate)."""
        await llm_service.get_response(
            system_prompt="x",
            user_prompt="y",
            model_override="auto",
        )

        assert mock_adapter.get_response.called, "adapter.get_response was not invoked"
        for call in mock_adapter.get_response.call_args_list:
            model_used = call.kwargs.get("model")
            assert model_used != "auto", (
                f"Provider adapter received model='auto' — sentinel leaked "
                f"past the router boundary. Call kwargs: {call.kwargs!r}"
            )
            # And the model used must actually be a concrete provider default.
            assert model_used in {"gpt-5-mini", "claude-sonnet-4-5"}, (
                f"Unexpected model {model_used!r} — expected one of the "
                f"mock provider defaults."
            )

    @pytest.mark.asyncio
    async def test_resolve_provider_routing_strips_auto(self, llm_service):
        """The router contract: when model_override=='auto', target_model
        must come back as None so _try_single_provider uses each route's
        own provider['model']."""
        _providers, target_model = llm_service.resolve_provider_routing(
            model_override="auto",
        )
        assert target_model is None

    @pytest.mark.asyncio
    async def test_misconfigured_auto_route_is_skipped(
        self, llm_service, mock_adapter, caplog
    ):
        """Strict-refusal contract: a route configured ``model="auto"``
        where ``resolve_provider_default`` cannot resolve (empty discovery)
        is *skipped* via ``ModelNotAvailableForRoute`` so the fallback
        loop tries the next route. ``"auto"`` must NEVER reach a provider
        client even when the caller didn't pass an override — that is
        precisely the #1408 bug. A soft passthrough would re-leak it."""
        # First route is misconfigured; second route has a concrete default
        # the fallback chain can use.
        llm_service.providers[0]["model"] = "auto"
        llm_service.providers[1]["model"] = "claude-sonnet-4-5"

        with patch(
            "kestrel_sovereign.llm.model_selection.resolve_provider_default",
            side_effect=ValueError("no discovery cache"),
        ), caplog.at_level("WARNING", logger="kestrel_sovereign.llm.service"):
            response = await llm_service.get_response(
                system_prompt="x",
                user_prompt="y",
            )

        assert isinstance(response, str)
        assert any(
            "discovery cache is empty" in r.message for r in caplog.records
        ), "expected the skip warning to be logged"
        # And nothing the mock received was "auto" — the misconfigured
        # route was skipped, the fallback ran with its concrete model.
        for call in mock_adapter.get_response.call_args_list:
            assert call.kwargs.get("model") != "auto"

    @pytest.mark.asyncio
    async def test_all_routes_misconfigured_auto_surfaces_clear_error(
        self, llm_service
    ):
        """Worst-case: every route is misconfigured ``model="auto"`` and
        discovery is empty. Caller gets ``LLMAllProvidersFailedError``
        listing each route's reason — not a 404 from the wire."""
        for p in llm_service.providers:
            p["model"] = "auto"

        with patch(
            "kestrel_sovereign.llm.model_selection.resolve_provider_default",
            side_effect=ValueError("no discovery cache"),
        ):
            with pytest.raises(LLMAllProvidersFailedError):
                await llm_service.get_response(
                    system_prompt="x",
                    user_prompt="y",
                )

    @pytest.mark.asyncio
    async def test_scrub_auto_helper_callable_via_self(self, llm_service):
        """Pins the contract that ``_scrub_auto`` is callable as a bound
        instance method (``self._scrub_auto(x)``). Used directly by the
        remote-GPU fast paths in ``streaming.py`` and ``service.py.generate``
        — a regression that drops the ``@staticmethod`` or accidentally
        passes ``self`` would break every REMOTE_GPU session."""
        assert llm_service._scrub_auto("auto") is None
        assert llm_service._scrub_auto(None) is None
        assert llm_service._scrub_auto("claude-haiku-4-5") == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_route_with_auto_default_is_lazy_resolved(
        self, llm_service, mock_adapter
    ):
        """Fresh quickstart case: a route is configured with model="auto"
        and discovery hasn't populated yet. _try_single_provider must
        lazy-resolve to a concrete model rather than send "auto" downstream."""
        # Simulate the quickstart shape: both routes still on the sentinel.
        for p in llm_service.providers:
            p["model"] = "auto"

        # Make resolve_provider_default deterministic regardless of disk
        # cache state — return a concrete model for each route.
        with patch(
            "kestrel_sovereign.llm.model_selection.resolve_provider_default",
            side_effect=lambda name: {
                "openai:api": "gpt-5-mini",
                "anthropic:api": "claude-sonnet-4-5",
            }.get(name, "gpt-5-mini"),
        ):
            await llm_service.get_response(
                system_prompt="x",
                user_prompt="y",
            )

        for call in mock_adapter.get_response.call_args_list:
            assert call.kwargs.get("model") != "auto", (
                f"Provider adapter received model='auto' from a route whose "
                f"own configured default was 'auto'. Call kwargs: {call.kwargs!r}"
            )


# =============================================================================
# Priority 3 — Backend and Lifecycle Tests
# =============================================================================


class TestBackendLifecycle:
    """Tests for backend switching and lifecycle methods."""

    @pytest.mark.asyncio
    async def test_switch_backend_to_cloud(self, llm_service):
        """Test switching to cloud backend."""
        llm_service.switch_backend(BackendType.CLOUD)

        status = llm_service.get_backend_status()
        assert status["current_backend"] == "cloud"

    @pytest.mark.asyncio
    async def test_switch_backend_to_local(self, llm_service):
        """Test switching to local backend."""
        llm_service.switch_backend(BackendType.LOCAL)

        status = llm_service.get_backend_status()
        assert status["current_backend"] == "local"

    @pytest.mark.asyncio
    async def test_switch_backend_to_remote_gpu(self, llm_service):
        """Test switching to remote GPU backend."""
        config = {
            "base_url": "https://gpu.example.com/v1",
            "model": "llama-70b",
            "api_key": "test-key",
        }

        llm_service.switch_backend(BackendType.REMOTE_GPU, config=config)

        status = llm_service.get_backend_status()
        assert status["current_backend"] == "remote_gpu"
        assert status["remote_active"] is True

    @pytest.mark.asyncio
    async def test_switch_backend_remote_gpu_requires_config(self, llm_service):
        """Test that remote GPU backend requires configuration."""
        with pytest.raises(LLMServiceError, match="requires configuration"):
            llm_service.switch_backend(BackendType.REMOTE_GPU)

    @pytest.mark.asyncio
    async def test_get_backend_status_structure(self, llm_service):
        """Test backend status returns correct structure."""
        status = llm_service.get_backend_status()

        assert "current_backend" in status
        assert "default_backend" in status
        assert "remote_active" in status
        assert "remote_metadata" in status
        assert "last_remote_error" in status

    @pytest.mark.asyncio
    async def test_close_providers(self, llm_service, mock_openai_client):
        """Test that close() properly closes all provider clients."""
        await llm_service.close()

        # Verify close was called on clients
        assert mock_openai_client.close.called

    @pytest.mark.asyncio
    async def test_close_handles_exceptions_gracefully(self, llm_service):
        """Test that close() handles exceptions during cleanup."""
        # Add a provider with a broken close method
        broken_client = AsyncMock()
        broken_client.close = AsyncMock(side_effect=Exception("Close failed"))

        llm_service.providers.append({
            "name": "broken",
            "client": broken_client,
            "adapter": Mock(),
            "model": "test",
        })

        # Should not raise
        await llm_service.close()

    @pytest.mark.asyncio
    async def test_close_accepts_sync_provider_close(self, llm_service, caplog):
        """Provider clients may expose synchronous close() methods."""
        sync_client = Mock()
        sync_client.close = Mock(return_value=None)

        llm_service.providers = [{
            "name": "sync_provider",
            "client": sync_client,
            "adapter": Mock(),
            "model": "test",
        }]

        caplog.set_level("WARNING")
        await llm_service.close()

        sync_client.close.assert_called_once()
        assert "Unexpected error closing sync_provider client" not in caplog.text

    @pytest.mark.asyncio
    async def test_set_observability_store(self, llm_service):
        """Test setting observability store."""
        mock_store = Mock()
        llm_service.set_observability_store(mock_store)

        assert llm_service._observability_store is mock_store

    @pytest.mark.asyncio
    async def test_set_observability_context(self, llm_service):
        """Test setting observability context."""
        llm_service.set_observability_context(
            session_id="sess-123",
            companion_id="comp-456",
            user_id="user-789",
        )

        assert llm_service._observability_context["session_id"] == "sess-123"
        assert llm_service._observability_context["companion_id"] == "comp-456"
        assert llm_service._observability_context["user_id"] == "user-789"

    @pytest.mark.asyncio
    async def test_observability_logging_on_success(self, llm_service):
        """Test that successful LLM calls are logged to observability store."""
        mock_store = AsyncMock()
        llm_service.set_observability_store(mock_store)
        llm_service.set_observability_context(session_id="test-session")

        await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Hello",
        )

        # Verify observability store was called
        assert mock_store.log_llm_call.called


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_no_providers_initialized(self, mock_config, mock_mandate_config):
        """Test that RuntimeError is raised when no providers are initialized."""
        with patch("kestrel_sovereign.llm.service.load_config") as mock_load_config, \
             patch("kestrel_sovereign.llm.service.ProviderRegistry") as mock_registry_class:

            mock_load_config.side_effect = lambda path: (
                mock_config if "llm_config" in path else mock_mandate_config
            )

            # Mock empty providers
            mock_registry = Mock()
            mock_registry.initialize_providers = Mock(return_value=[])
            mock_registry_class.return_value = mock_registry

            service = LLMService()

            # Mock the usage tracking attributes
            service._usage_db = None
            service._db_initialized = False
            service._usage_database_url = None
            service._db_backend = "sqlite"

            with pytest.raises(RuntimeError, match="No LLM providers"):
                await service.get_response(
                    system_prompt="Test",
                    user_prompt="Test",
                )

            await service.close()

    @pytest.mark.asyncio
    async def test_model_mandate_keyword_trigger(self, llm_service):
        """Test that model mandate is triggered by keywords in prompt."""
        response = await llm_service.get_response(
            system_prompt="Test",
            user_prompt="Write some code for me",  # "code" should trigger mandate
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_remote_gpu_with_ttl_expiry(self, llm_service):
        """Test that expired remote GPU sessions are deactivated."""
        from datetime import datetime, timedelta, timezone

        config = {
            "base_url": "https://gpu.example.com/v1",
            "model": "llama-70b",
            "expires_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        }

        llm_service.switch_backend(BackendType.REMOTE_GPU, config=config)

        # Try to use expired backend
        with pytest.raises(LLMServiceError, match="expired"):
            llm_service._ensure_remote_active()

    @pytest.mark.asyncio
    async def test_generate_with_remote_gpu_fallback(self, llm_service, mock_adapter):
        """Test that generate falls back to providers when remote GPU fails."""
        # Setup remote GPU backend
        config = {
            "base_url": "https://gpu.example.com/v1",
            "model": "llama-70b",
        }
        llm_service.switch_backend(BackendType.REMOTE_GPU, config=config)

        # Mock remote adapter to fail
        llm_service._remote_adapter.get_response = AsyncMock(
            side_effect=Exception("Remote GPU unavailable")
        )

        # Should fall back to regular providers
        response = await llm_service.generate(
            system_prompt="Test",
            user_prompt="Hello",
        )

        assert isinstance(response, str)
        # Backend should be deactivated after failure
        status = llm_service.get_backend_status()
        assert not status["remote_active"]


# =============================================================================
# Priority 5 — Streaming Mandate Preference Tests
# =============================================================================


class TestStreamingMandatePreference:
    """Tests that streaming methods respect mandate preference."""

    @pytest.mark.asyncio
    async def test_get_streaming_response_uses_mandate(self, llm_service, mock_adapter):
        """Test that get_streaming_response respects mandate preference."""
        # Set mandate to anthropic/claude-sonnet-4-5
        llm_service.set_model_preference("claude-sonnet-4-5", vendor="anthropic")

        # Mock streaming response
        async def mock_streaming(*args, **kwargs):
            yield "Hello "
            yield "world"
        mock_adapter.get_streaming_response = Mock(return_value=mock_streaming())

        chunks = []
        async for chunk in llm_service.get_streaming_response(
            system_prompt="Test",
            user_prompt="Hello",
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Verify the adapter was called with the mandated model
        call_args = mock_adapter.get_streaming_response.call_args
        assert call_args is not None
        assert call_args.kwargs.get("model") == "claude-sonnet-4-5" or \
               (len(call_args.args) > 1 and call_args.args[1] == "claude-sonnet-4-5")

    @pytest.mark.asyncio
    async def test_stream_with_messages_uses_mandate(self, llm_service, mock_adapter):
        """Test that stream_with_messages respects mandate preference."""
        # Set mandate to anthropic/claude-sonnet-4-5
        llm_service.set_model_preference("claude-sonnet-4-5", vendor="anthropic")

        # Mock streaming response
        async def mock_streaming(*args, **kwargs):
            yield "Streamed "
            yield "response"
        mock_adapter.get_streaming_response = Mock(return_value=mock_streaming())

        messages = [{"role": "user", "content": "Test"}]
        chunks = []
        async for chunk in llm_service.stream_with_messages(
            messages=messages,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Verify the adapter was called with the mandated model
        call_args = mock_adapter.get_streaming_response.call_args
        assert call_args is not None
        assert call_args.kwargs.get("model") == "claude-sonnet-4-5" or \
               (len(call_args.args) > 1 and call_args.args[1] == "claude-sonnet-4-5")

    @pytest.mark.asyncio
    async def test_stream_with_messages_passes_cancel_token(self, llm_service, mock_adapter):
        """Request cancellation token is forwarded to adapter streams."""
        async def mock_streaming(*args, **kwargs):
            yield "response"

        mock_adapter.get_streaming_response = Mock(return_value=mock_streaming())
        token = Mock(return_value=False)

        chunks = []
        async for chunk in llm_service.stream_with_messages(
            messages=[{"role": "user", "content": "Test"}],
            cancel_token=token,
        ):
            chunks.append(chunk)

        assert chunks == ["response"]
        assert mock_adapter.get_streaming_response.call_args.kwargs[
            "cancel_token"
        ] is token

    @pytest.mark.asyncio
    async def test_generate_with_messages_passes_cancel_token(self, llm_service, mock_adapter):
        """Request cancellation token is forwarded to non-streaming calls."""
        token = Mock(return_value=False)

        await llm_service.generate_with_messages(
            messages=[{"role": "user", "content": "Test"}],
            cancel_token=token,
        )

        assert mock_adapter.get_response.await_args.kwargs["cancel_token"] is token

    @pytest.mark.asyncio
    async def test_stream_with_tool_detection_passes_cancel_token(self, llm_service, mock_adapter):
        """Request cancellation token is forwarded to tool-aware streams."""
        async def mock_streaming(*args, **kwargs):
            yield LLMResponse(content="done", input_tokens=1, output_tokens=1)

        mock_adapter.get_streaming_response_with_tools = Mock(
            return_value=mock_streaming()
        )
        token = Mock(return_value=False)

        items = []
        async for item in llm_service.stream_with_tool_detection(
            messages=[{"role": "user", "content": "Test"}],
            tools=[{"type": "function", "function": {
                "name": "noop",
                "parameters": {"type": "object", "properties": {}},
            }}],
            cancel_token=token,
        ):
            items.append(item)

        assert isinstance(items[-1], LLMResponse)
        assert mock_adapter.get_streaming_response_with_tools.call_args.kwargs[
            "cancel_token"
        ] is token

    @pytest.mark.asyncio
    async def test_streaming_without_mandate_uses_default(self, llm_service, mock_adapter):
        """Test that streaming without mandate uses provider default model."""
        # Ensure no mandate is set
        llm_service.clear_model_preference()

        async def mock_streaming(*args, **kwargs):
            yield "Default response"
        mock_adapter.get_streaming_response = Mock(return_value=mock_streaming())

        messages = [{"role": "user", "content": "Test"}]
        chunks = []
        async for chunk in llm_service.stream_with_messages(
            messages=messages,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Should use the provider's default model (gpt-5-mini for first provider)
        call_args = mock_adapter.get_streaming_response.call_args
        assert call_args is not None
        model_used = call_args.kwargs.get("model") or (call_args.args[1] if len(call_args.args) > 1 else None)
        assert model_used == "gpt-5-mini"

    @pytest.mark.asyncio
    async def test_streaming_override_takes_precedence_over_mandate(self, llm_service, mock_adapter):
        """Test that explicit model_override takes precedence over mandate."""
        # Set mandate to anthropic
        llm_service.set_model_preference("claude-sonnet-4-5", vendor="anthropic")

        async def mock_streaming(*args, **kwargs):
            yield "Override response"
        mock_adapter.get_streaming_response = Mock(return_value=mock_streaming())

        messages = [{"role": "user", "content": "Test"}]
        chunks = []
        async for chunk in llm_service.stream_with_messages(
            messages=messages,
            model_override="openai/gpt-5-mini",
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Should use the override model, not the mandate
        # stream_with_messages resolves "openai/gpt-5-mini" → model "gpt-5-mini"
        call_args = mock_adapter.get_streaming_response.call_args
        assert call_args is not None
        model_used = call_args.kwargs.get("model") or (call_args.args[1] if len(call_args.args) > 1 else None)
        assert model_used in ("gpt-5-mini", "openai/gpt-5-mini")
