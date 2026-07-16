"""
Unit tests for LLMService core methods.

Tests the unified LLM provider manager with mocked dependencies.
No real API calls are made.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext


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
         patch("kestrel_sovereign.llm.service.load_section") as mock_load_section, \
         patch("kestrel_sovereign.llm.service.ProviderRegistry") as mock_registry_class:

        # Setup config mocking
        mock_load_config.side_effect = lambda path: (
            mock_config if "llm_config" in path else mock_mandate_config
        )
        # ``service.config`` comes from ``load_section("llm")`` — pin it to the
        # fixture's two-vendor setup instead of leaking the real on-disk
        # kestrel.toml. The fixture wires an openai+anthropic provider chain,
        # so both vendors must appear in ``route_priority`` for cross-vendor
        # fallback in those tests to be a legitimate *configured* fallback
        # (not the blind cross-vendor swap blocked by #no-blind-fallbacks).
        mock_load_section.return_value = {
            "route_priority": ["openai:api", "anthropic:api"],
        }

        # Setup registry mocking
        mock_registry_class.return_value = mock_provider_registry

        service = LLMService()

        # Mock the usage tracking and constitutional profiles (which set attributes)
        service._usage_db = None
        service._db_initialized = False
        service._usage_database_url = None
        service._db_backend = "sqlite"

        # The shared model cache is a PROCESS-WIDE singleton. Tests here assume
        # a cold cache (e.g. an unknown model_override is permitted when
        # discovery hasn't run — _model_available_for_route returns True on an
        # empty cache). Another test on the same xdist worker can leave the
        # singleton warm with a different vendor catalog, which silently flips
        # those assumptions depending on worker assignment. Clear it so every
        # test using this fixture is hermetic regardless of execution order.
        from kestrel_sovereign.llm.model_cache import get_shared_model_cache
        get_shared_model_cache().clear()

        yield service

        # Cleanup — also clear so this module doesn't leak cache state to the
        # next test on the worker.
        get_shared_model_cache().clear()
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
        """Route narrows routing to the exact <vendor>:<route> entry.

        The triple must be a configured route serving the model — an explicit
        ``{vendor, route, model}`` is now validated against discovery (#1946),
        so this test wires a real ``anthropic:plan`` route + catalog entry
        rather than relying on the old silent-accept loophole.
        """
        from unittest.mock import MagicMock, patch
        from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo

        llm_service.providers.append(
            {"name": "anthropic:plan", "vendor": "anthropic", "route": "plan", "model": "auto"}
        )
        cache = MagicMock()
        cache.get_any = MagicMock(return_value=[
            ModelInfo(
                id="claude-sonnet-4-6", provider="anthropic",
                display_name="claude-sonnet-4-6", category=ModelCategory.CHAT,
                supports_tools=True,
            ),
        ])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
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
    async def test_explicit_route_not_gated_by_catalog(self, llm_service, mock_adapter):
        """Regression for #2352: feature-subagent dispatch on an explicitly
        pinned ``vendor:route/model`` must NOT be rejected by the vendor-catalog
        gate just because discovery hasn't cached the model.

        The streaming chat path never calls ``_model_available_for_route``, so a
        brand-new OpenRouter slug (``openai/gpt-5.4-mini``) streams fine there.
        The non-streaming ``get_response`` path (used by
        ``Feature.execute_as_subagent`` via ``generate``) used to gate on the
        catalog, raise ``ModelNotAvailableForRoute``, and surface it as
        "All providers failed / Provider unknown" for the exact same route. When
        the route is explicitly pinned there is no blind-cascade risk, so the
        call must go through and hit the adapter.
        """
        from unittest.mock import MagicMock, patch
        from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo

        # Wire an OpenRouter route into the service. Its configured model is a
        # DIFFERENT slug, and discovery only knows about that other slug — so
        # the catalog gate WOULD reject ``openai/gpt-5.4-mini`` if consulted.
        llm_service.providers.append({
            "name": "openrouter:api",
            "vendor": "openrouter",
            "route": "api",
            "client": AsyncMock(),
            "adapter": mock_adapter,
            "model": "openai/gpt-5-mini",
            "is_cloud": True,
            "is_local": False,
            "base_url": None,
            "selection_hints": [],
        })

        warm_cache = MagicMock()
        warm_cache.get_any = MagicMock(return_value=[
            ModelInfo(
                id="openai/gpt-5-mini", provider="openrouter",
                display_name="openai/gpt-5-mini", category=ModelCategory.CHAT,
                supports_tools=True,
            ),
        ])

        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=warm_cache,
        ):
            response = await llm_service.get_response(
                system_prompt="You are the visual identity subagent.",
                user_prompt="Task: send a selfie",
                model_override="openrouter:api/openai/gpt-5.4-mini",
            )

        # The adapter was actually called with the pinned model — no
        # ModelNotAvailableForRoute / "All providers failed" fallthrough.
        assert isinstance(response, str)
        mock_adapter.get_response.assert_awaited()
        called_model = mock_adapter.get_response.await_args.kwargs["model"]
        assert called_model == "openai/gpt-5.4-mini"

    @pytest.mark.asyncio
    async def test_blind_fallback_still_gated_by_catalog(self, llm_service, mock_adapter):
        """The catalog gate must STILL fire on non-explicit (blind fallback)
        routing — the #2352 fix only relaxes it for explicitly pinned routes.

        A bare model override with no vendor/route prefix is not an explicit
        selection, so a model that no configured route serves must be skipped
        (the cross-vendor cheap-model cascade guard), ending in
        ``LLMAllProvidersFailedError`` rather than a wrong-vendor call.
        """
        from unittest.mock import MagicMock, patch
        from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo

        warm_cache = MagicMock()
        warm_cache.get_any = MagicMock(return_value=[
            ModelInfo(
                id="gpt-5-mini", provider="openai",
                display_name="gpt-5-mini", category=ModelCategory.CHAT,
                supports_tools=True,
            ),
            ModelInfo(
                id="claude-sonnet-4-5", provider="anthropic",
                display_name="claude-sonnet-4-5", category=ModelCategory.CHAT,
                supports_tools=True,
            ),
        ])

        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=warm_cache,
        ):
            with pytest.raises(LLMAllProvidersFailedError):
                await llm_service.get_response(
                    system_prompt="Test",
                    user_prompt="Test prompt",
                    model_override="model-no-route-serves",
                )

    @staticmethod
    def _plan_paid_pair():
        """Build a [openai:plan(raises 429), openai:api(records+ok)] provider
        list in the internal dict format, so a strict-mode test can assert the
        paid :api route is never attempted after a plan throttle."""
        attempted: list[str] = []

        def _adapter(name, *, raise_exc=None):
            ad = Mock()
            ad.create_messages = Mock(return_value=[{"role": "user", "content": "hi"}])

            async def _get_response(*a, **k):
                attempted.append(name)
                if raise_exc is not None:
                    raise raise_exc
                return LLMResponse(content="ok", input_tokens=1, output_tokens=1,
                                   total_tokens=2)
            ad.get_response = _get_response
            return ad

        def _prov(name, route, adapter):
            vendor = name.split(":", 1)[0]
            return {
                "name": name, "vendor": vendor, "route": route,
                "client": AsyncMock(), "adapter": adapter, "model": "m",
                "is_cloud": True, "is_local": False, "base_url": None,
            }

        plan = _prov("openai:plan", "plan",
                     _adapter("openai:plan",
                              raise_exc=LLMProviderError("openai", "429 rate limit")))
        api = _prov("openai:api", "api", _adapter("openai:api"))
        return [plan, api], attempted

    @pytest.mark.asyncio
    async def test_generate_with_messages_refuses_plan_to_paid_downgrade(
        self, llm_service
    ):
        """#2074 regression: with allow_paid_fallback=False, a plan throttle in
        the NON-streaming agentic loop (generate_with_messages) must NOT silently
        fall through to the metered :api route. Before the fix this loop only had
        _skip_unconfigured_route (same-vendor :api passes) so it billed the API."""
        providers, attempted = self._plan_paid_pair()
        llm_service.providers = providers
        llm_service.config = {
            "route_priority": ["openai:plan", "openai:api"],
            "allow_paid_fallback": False,
        }
        llm_service.mandate_config = {"defaults": {}}

        with pytest.raises(Exception):
            await llm_service.generate_with_messages(
                messages=[{"role": "user", "content": "hi"}],
            )
        # plan was tried and raised; the paid :api route was refused, not billed.
        assert attempted == ["openai:plan"], attempted

    @pytest.mark.asyncio
    async def test_get_response_refuses_plan_to_paid_downgrade(self, llm_service):
        """#2074 regression, sibling loop: get_response must likewise refuse the
        silent plan->paid downgrade under allow_paid_fallback=False."""
        providers, attempted = self._plan_paid_pair()
        llm_service.providers = providers
        llm_service.config = {
            "route_priority": ["openai:plan", "openai:api"],
            "allow_paid_fallback": False,
        }
        llm_service.mandate_config = {"defaults": {}}

        with pytest.raises(Exception):
            await llm_service.get_response(
                system_prompt="s", user_prompt="hi",
            )
        assert attempted == ["openai:plan"], attempted

    @pytest.mark.asyncio
    async def test_generate_with_messages_allows_plan_to_paid_when_permitted(
        self, llm_service
    ):
        """Control: with allow_paid_fallback=True (default), the same plan
        throttle DOES fall through to :api — the guard is opt-in, not a behavior
        change for existing deployments."""
        providers, attempted = self._plan_paid_pair()
        llm_service.providers = providers
        llm_service.config = {
            "route_priority": ["openai:plan", "openai:api"],
            "allow_paid_fallback": True,
        }
        llm_service.mandate_config = {"defaults": {}}

        result = await llm_service.generate_with_messages(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert attempted == ["openai:plan", "openai:api"], attempted
        assert result is not None

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


class TestInvocationBoundary:
    """#2510 — successful provider paths finalize telemetry exactly once."""

    @staticmethod
    def _activate_fake_remote(llm_service, outcome):
        adapter = Mock()
        adapter.create_messages = Mock(
            return_value=[{"role": "user", "content": "hello"}]
        )
        adapter.get_response = AsyncMock()
        if isinstance(outcome, BaseException):
            adapter.get_response.side_effect = outcome
        else:
            adapter.get_response.return_value = outcome
        llm_service._backend = BackendType.REMOTE_GPU
        llm_service._remote_client = AsyncMock()
        llm_service._remote_config = RemoteGPUConfig(
            base_url="https://gpu.example.test/v1",
            model="remote-model",
        )
        llm_service._remote_adapter = adapter
        return adapter

    @pytest.mark.asyncio
    async def test_partial_explicit_context_inherits_task_local_identity(
        self, llm_service
    ):
        llm_service.set_observability_context(
            session_id="ambient-session",
            companion_id="ambient-companion",
            user_id="ambient-user",
        )

        context = llm_service._resolve_invocation_context(
            LLMInvocationContext(correlation_id="explicit-correlation"),
            session_id="explicit-session",
        )

        assert context == LLMInvocationContext(
            session_id="explicit-session",
            companion_id="ambient-companion",
            user_id="ambient-user",
            correlation_id="explicit-correlation",
        )

    def test_legacy_ambient_context_is_service_local(self, llm_service):
        """Two services in one task must never share billing identity."""

        other = object.__new__(LLMService)
        llm_service.set_observability_context(
            companion_id="first-companion", user_id="first-user"
        )

        assert llm_service._resolve_invocation_context().companion_id == (
            "first-companion"
        )
        assert other._resolve_invocation_context() == LLMInvocationContext()
        other._stamp_response_identity(None, model="bare-model", provider="bare")
        assert other.get_last_response_identity() == {
            "model": "bare-model",
            "provider": "bare",
        }

    @pytest.mark.asyncio
    async def test_scoped_context_restores_parent_and_child_keeps_snapshot(
        self, llm_service
    ):
        llm_service.set_observability_context(companion_id="outer")
        child_started = asyncio.Event()
        release_child = asyncio.Event()

        async def inherited_child():
            child_started.set()
            await release_child.wait()
            return llm_service._resolve_invocation_context()

        with llm_service.observability_context(companion_id="inner"):
            with llm_service.observability_context(companion_id="leaf"):
                assert (
                    llm_service._resolve_invocation_context().companion_id
                    == "leaf"
                )
            assert (
                llm_service._resolve_invocation_context().companion_id == "inner"
            )
            child = asyncio.create_task(inherited_child())
            await child_started.wait()

        assert llm_service._resolve_invocation_context().companion_id == "outer"
        llm_service.reset_observability_context()
        release_child.set()
        assert (await child).companion_id == "inner"

        async def unrelated_child():
            return llm_service._resolve_invocation_context()

        assert await asyncio.create_task(unrelated_child()) == LLMInvocationContext()

    @pytest.mark.asyncio
    async def test_remote_fallback_keeps_entry_context_after_ambient_mutation(
        self, llm_service
    ):
        adapter = self._activate_fake_remote(
            llm_service, LLMResponse(content="unused")
        )

        async def mutate_then_fail(**_kwargs):
            llm_service.set_observability_context(
                companion_id="late-companion", user_id="late-user"
            )
            raise ConnectionError("remote failed late")

        adapter.get_response = AsyncMock(side_effect=mutate_then_fail)
        llm_service.set_observability_context(
            companion_id="entry-companion", user_id="entry-user"
        )
        store = AsyncMock()
        llm_service.set_observability_store(store)

        await llm_service.generate(system_prompt="system", user_prompt="hello")

        assert store.log_llm_call.await_count == 2
        for call in store.log_llm_call.await_args_list:
            assert call.kwargs["companion_id"] == "entry-companion"
            assert call.kwargs["user_id"] == "entry-user"

    @pytest.mark.asyncio
    async def test_nested_audit_cannot_change_persisted_visible_identity(
        self, llm_service, mock_adapter
    ):
        """Persistence must attribute a string response to its visible call."""

        mock_adapter.get_response = AsyncMock(
            return_value=LLMResponse(
                content="visible answer", input_tokens=4, output_tokens=2
            )
        )
        visible = await llm_service.get_response(
            system_prompt="system", user_prompt="hello"
        )
        assert visible == "visible answer"
        assert llm_service.get_last_response_identity() == {
            "model": "gpt-5-mini",
            "provider": "openai:api",
        }

        audit_provider = dict(llm_service.providers[0])
        audit_provider.update(
            name="audit:api",
            vendor="audit",
            route="api",
            model="audit-model",
        )
        llm_service.providers = [audit_provider]
        mock_adapter.provider_capabilities = Mock(
            return_value=SimpleNamespace(supports_structured_output=True)
        )
        mock_adapter.get_response = AsyncMock(
            return_value=LLMResponse(
                content='{"risk_level": 1, "reasoning": "safe"}',
                input_tokens=3,
                output_tokens=1,
            )
        )

        assert (await llm_service.get_audit_response(visible))["risk_level"] == 1
        assert llm_service.get_last_response_identity() == {
            "model": "gpt-5-mini",
            "provider": "openai:api",
        }

        from kestrel_sovereign.kestrel_agent import KestrelAgent

        agent = object.__new__(KestrelAgent)
        agent.llm_service = llm_service
        agent.privacy_agent = SimpleNamespace(add_conversation=AsyncMock())
        await agent._persist_assistant_conversation(visible, response=visible)

        persisted = agent.privacy_agent.add_conversation.await_args
        assert persisted.args == ("assistant", "visible answer")
        assert persisted.kwargs["model"] == "gpt-5-mini"
        assert persisted.kwargs["provider"] == "openai:api"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("broken_sink", ["usage_db", "store", "meter"])
    async def test_ordinary_sink_failure_does_not_suppress_siblings(
        self, llm_service, broken_sink, caplog
    ):
        tracker = AsyncMock()
        store = AsyncMock()
        meter_calls = []

        if broken_sink == "usage_db":
            tracker.side_effect = RuntimeError("usage db down")
        if broken_sink == "store":
            store.log_llm_call.side_effect = OSError("store down")

        async def meter(**kwargs):
            meter_calls.append(kwargs)
            if broken_sink == "meter":
                raise RuntimeError("meter down")

        llm_service._track_model_usage = tracker
        llm_service.set_observability_store(store)
        llm_service.set_metering_callback(meter)
        llm_service.set_observability_context(
            companion_id="sink-companion", user_id="sink-user"
        )

        with caplog.at_level("INFO", logger="kestrel_sovereign.llm.service"):
            result = await llm_service.get_response(
                system_prompt="system", user_prompt="hello"
            )

        assert isinstance(result, str)
        tracker.assert_awaited_once()
        store.log_llm_call.assert_awaited_once()
        assert len(meter_calls) == 1
        assert sum(
            record.message.startswith("llm.usage: ") for record in caplog.records
        ) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cancelled_sink", ["usage_db", "store", "meter"])
    async def test_sink_cancellation_propagates_unchanged(
        self, llm_service, cancelled_sink
    ):
        cancellation = asyncio.CancelledError(f"cancelled in {cancelled_sink}")
        tracker = AsyncMock()
        store = AsyncMock()
        if cancelled_sink == "usage_db":
            tracker.side_effect = cancellation
        if cancelled_sink == "store":
            store.log_llm_call.side_effect = cancellation

        async def meter(**_kwargs):
            if cancelled_sink == "meter":
                raise cancellation

        llm_service._track_model_usage = tracker
        llm_service.set_observability_store(store)
        llm_service.set_metering_callback(meter)
        llm_service.set_observability_context(
            companion_id="cancel-companion", user_id="cancel-user"
        )

        with pytest.raises(asyncio.CancelledError) as caught:
            await llm_service.get_response(
                system_prompt="system", user_prompt="hello"
            )

        assert caught.value is cancellation

    @pytest.mark.asyncio
    async def test_missing_provider_usage_is_unknown_and_never_zero_billed(
        self, llm_service, mock_adapter
    ):
        mock_adapter.get_response = AsyncMock(
            return_value=LLMResponse(content="answer without usage")
        )
        llm_service._track_model_usage = AsyncMock()
        store = AsyncMock()
        llm_service.set_observability_store(store)
        meter_calls = []

        async def meter(**kwargs):
            meter_calls.append(kwargs)

        llm_service.set_metering_callback(meter)
        llm_service.set_observability_context(
            companion_id="unknown-companion", user_id="unknown-user"
        )

        assert (
            await llm_service.get_response(
                system_prompt="system", user_prompt="hello"
            )
            == "answer without usage"
        )

        llm_service._track_model_usage.assert_not_awaited()
        assert meter_calls == []
        logged = store.log_llm_call.await_args.kwargs
        assert logged["input_tokens"] is None
        assert logged["output_tokens"] is None
        assert logged["metadata"]["usage_available"] is False

    @pytest.mark.asyncio
    async def test_total_only_usage_is_tracked_but_not_zero_billed(
        self, llm_service, mock_adapter
    ):
        mock_adapter.get_response = AsyncMock(
            return_value=LLMResponse(content="answer", total_tokens=9)
        )
        llm_service._track_model_usage = AsyncMock()
        meter_calls = []

        async def meter(**kwargs):
            meter_calls.append(kwargs)

        llm_service.set_metering_callback(meter)
        llm_service.set_observability_context(
            companion_id="total-companion", user_id="total-user"
        )

        await llm_service.get_response(system_prompt="system", user_prompt="hello")

        llm_service._track_model_usage.assert_awaited_once_with(
            "gpt-5-mini", "openai:api", tokens=9
        )
        assert meter_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("entrypoint", "expected_path"),
        [
            ("get_response", "get_response"),
            ("generate", "get_response"),
            ("generate_with_messages", "generate_with_messages"),
            ("get_response_with_model", "get_response_with_model"),
            ("get_audit_response", "get_audit_response"),
            ("get_response_with_tools", "get_response"),
            ("get_response_structured", "get_response"),
        ],
    )
    async def test_standard_success_entrypoints_finalize_once(
        self,
        llm_service,
        mock_adapter,
        entrypoint,
        expected_path,
    ):
        class StructuredResult(BaseModel):
            answer: str

        if entrypoint == "get_audit_response":
            content = '{"risk_level": 1, "reasoning": "safe"}'
            tool_calls = None
        elif entrypoint == "get_response_structured":
            content = '{"answer": "ok"}'
            tool_calls = None
        elif entrypoint == "get_response_with_tools":
            content = None
            tool_calls = [ToolCall(id="call-1", name="lookup", arguments={})]
        else:
            content = "ok"
            tool_calls = None
        mock_adapter.get_response = AsyncMock(
            return_value=LLMResponse(
                content=content,
                tool_calls=tool_calls,
                input_tokens=5,
                output_tokens=3,
                total_tokens=8,
            )
        )
        finalizer = AsyncMock(wraps=llm_service._finalize_successful_invocation)
        llm_service._finalize_successful_invocation = finalizer

        if entrypoint == "get_response":
            await llm_service.get_response(system_prompt="system", user_prompt="hello")
        elif entrypoint == "generate":
            await llm_service.generate(system_prompt="system", user_prompt="hello")
        elif entrypoint == "generate_with_messages":
            await llm_service.generate_with_messages(
                messages=[{"role": "user", "content": "hello"}]
            )
        elif entrypoint == "get_response_with_model":
            await llm_service.get_response_with_model(
                model_id="gpt-5-mini",
                system_prompt="system",
                user_prompt="hello",
            )
        elif entrypoint == "get_audit_response":
            await llm_service.get_audit_response("hello")
        elif entrypoint == "get_response_with_tools":
            await llm_service.get_response(
                system_prompt="system",
                user_prompt="hello",
                tools=[{"type": "function", "function": {"name": "lookup"}}],
            )
        else:
            await llm_service.get_response(
                system_prompt="system",
                user_prompt="hello",
                response_format=StructuredResult,
            )

        finalizer.assert_awaited_once()
        assert finalizer.await_args.kwargs["path"] == expected_path

    @pytest.mark.asyncio
    async def test_remote_prompt_success_records_complete_telemetry_once(
        self, llm_service, caplog
    ):
        response = LLMResponse(
            content="remote answer",
            input_tokens=17,
            output_tokens=9,
            total_tokens=26,
            raw=SimpleNamespace(usage=SimpleNamespace(cost=0.0125)),
        )
        self._activate_fake_remote(llm_service, response)
        llm_service._track_model_usage = AsyncMock()
        store = AsyncMock()
        llm_service.set_observability_store(store)
        meter_calls = []

        async def meter(**kwargs):
            meter_calls.append(kwargs)

        llm_service.set_metering_callback(meter)
        context = LLMInvocationContext(
            session_id="session-remote",
            companion_id="companion-remote",
            user_id="user-remote",
            correlation_id="correlation-remote",
        )

        with caplog.at_level("INFO", logger="kestrel_sovereign.llm.service"):
            result = await llm_service.generate(
                system_prompt="system",
                user_prompt="hello",
                invocation_context=context,
            )

        assert result == "remote answer"
        llm_service._track_model_usage.assert_awaited_once_with(
            "remote-model", "remote_gpu", tokens=26
        )
        store.log_llm_call.assert_awaited_once()
        logged = store.log_llm_call.await_args.kwargs
        assert (
            logged["provider"],
            logged["model"],
            logged["input_tokens"],
            logged["output_tokens"],
        ) == ("remote_gpu", "remote-model", 17, 9)
        assert logged["duration_ms"] >= 0
        assert (logged["session_id"], logged["companion_id"], logged["user_id"]) == (
            "session-remote",
            "companion-remote",
            "user-remote",
        )
        assert logged["metadata"] == {
            "path": "generate.remote_gpu",
            "force_local_only": False,
            "correlation_id": "correlation-remote",
            "provider_reported_cost_usd": 0.0125,
        }
        assert meter_calls == [
            {
                "companion_id": "companion-remote",
                "user_id": "user-remote",
                "provider": "remote_gpu",
                "model": "remote-model",
                "prompt_tokens": 17,
                "completion_tokens": 9,
                "cost": 0.0125,
            }
        ]

        import json as _json

        usage_lines = [
            record
            for record in caplog.records
            if record.message.startswith("llm.usage: ")
        ]
        assert len(usage_lines) == 1
        usage = _json.loads(usage_lines[0].message.removeprefix("llm.usage: "))
        assert (
            usage["provider"],
            usage["model"],
            usage["cost"],
            usage["session_id"],
            usage["correlation_id"],
        ) == (
            "remote_gpu",
            "remote-model",
            0.0125,
            "session-remote",
            "correlation-remote",
        )

    @pytest.mark.asyncio
    async def test_remote_failure_fallback_records_each_provider_attempt_once(
        self, llm_service
    ):
        self._activate_fake_remote(llm_service, ConnectionError("gpu unavailable"))
        llm_service._track_model_usage = AsyncMock()
        store = AsyncMock()
        llm_service.set_observability_store(store)

        result = await llm_service.generate(
            system_prompt="system",
            user_prompt="hello",
            invocation_context=LLMInvocationContext(session_id="fallback-session"),
        )

        assert isinstance(result, str)
        llm_service._track_model_usage.assert_awaited_once()
        assert store.log_llm_call.await_count == 2
        failed, succeeded = [
            call.kwargs for call in store.log_llm_call.await_args_list
        ]
        assert (failed["success"], failed["provider"]) == (False, "remote_gpu")
        assert failed["error_message"] == "gpu unavailable"
        assert (succeeded["success"], succeeded["provider"]) == (
            True,
            "openai:api",
        )
        assert failed["session_id"] == succeeded["session_id"] == "fallback-session"

    @pytest.mark.asyncio
    async def test_remote_tool_stream_finalizes_terminal_response_once(
        self, llm_service
    ):
        adapter = self._activate_fake_remote(llm_service, LLMResponse(content="unused"))

        async def remote_stream(**_kwargs):
            yield "remote chunk"
            yield LLMResponse(
                content="remote chunk",
                input_tokens=7,
                output_tokens=4,
                total_tokens=11,
            )

        adapter.get_streaming_response_with_tools = remote_stream
        llm_service._track_model_usage = AsyncMock()
        store = AsyncMock()
        llm_service.set_observability_store(store)
        meter_calls = []

        async def meter(**kwargs):
            meter_calls.append(kwargs)

        llm_service.set_metering_callback(meter)
        context = LLMInvocationContext(
            session_id="superseded-session",
            companion_id="stream-companion",
            user_id="stream-user",
            correlation_id="stream-correlation",
        )

        items = [
            item
            async for item in llm_service.stream_with_tool_detection(
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"type": "function", "function": {"name": "lookup"}}],
                session_id="stream-session",
                invocation_context=context,
            )
        ]

        assert items[0] == "remote chunk"
        assert isinstance(items[1], LLMResponse)
        llm_service._track_model_usage.assert_awaited_once_with(
            "remote-model", "remote_gpu", tokens=11
        )
        store.log_llm_call.assert_awaited_once()
        logged = store.log_llm_call.await_args.kwargs
        assert logged["session_id"] == "stream-session"
        assert logged["metadata"] == {
            "streamed": True,
            "path": "stream_with_tool_detection",
            "correlation_id": "stream-correlation",
        }
        assert len(meter_calls) == 1
        assert meter_calls[0]["companion_id"] == "stream-companion"
        assert meter_calls[0]["user_id"] == "stream-user"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("entrypoint", "expected_path"),
        [
            ("get_streaming_response", "get_streaming_response"),
            ("generate_stream", "generate_stream"),
            ("stream_with_messages", "stream_with_messages"),
        ],
    )
    async def test_plain_streams_meter_terminal_usage_once_without_exposing_it(
        self, llm_service, mock_adapter, entrypoint, expected_path
    ):
        attempts = 0

        async def usage_stream(**_kwargs):
            nonlocal attempts
            attempts += 1
            yield "visible chunk"
            yield LLMResponse(
                content="visible chunk",
                input_tokens=8,
                output_tokens=5,
                total_tokens=13,
            )

        mock_adapter.get_streaming_response_with_tools = usage_stream
        llm_service._track_model_usage = AsyncMock()
        store = AsyncMock()
        llm_service.set_observability_store(store)
        meter_calls = []

        async def meter(**kwargs):
            meter_calls.append(kwargs)

        llm_service.set_metering_callback(meter)
        context = LLMInvocationContext(
            companion_id="plain-companion",
            user_id="plain-user",
            correlation_id="plain-correlation",
        )
        if entrypoint == "get_streaming_response":
            stream = llm_service.get_streaming_response(
                system_prompt="system",
                user_prompt="hello",
                invocation_context=context,
            )
        elif entrypoint == "generate_stream":
            stream = llm_service.generate_stream(
                system_prompt="system",
                user_prompt="hello",
                invocation_context=context,
            )
        else:
            stream = llm_service.stream_with_messages(
                messages=[{"role": "user", "content": "hello"}],
                invocation_context=context,
            )

        assert [item async for item in stream] == ["visible chunk"]
        assert attempts == 1
        llm_service._track_model_usage.assert_awaited_once_with(
            "gpt-5-mini", "openai:api", tokens=13
        )
        store.log_llm_call.assert_awaited_once()
        logged = store.log_llm_call.await_args.kwargs
        assert logged["metadata"]["path"] == expected_path
        assert logged["metadata"]["correlation_id"] == "plain-correlation"
        assert len(meter_calls) == 1
        assert meter_calls[0]["prompt_tokens"] == 8
        assert meter_calls[0]["completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_unknown_plain_stream_usage_is_logged_but_never_billed(
        self, llm_service, mock_adapter, caplog
    ):
        async def basic_stream(**_kwargs):
            yield "legacy chunk"

        mock_adapter.get_streaming_response = Mock(return_value=basic_stream())
        llm_service._track_model_usage = AsyncMock()
        store = AsyncMock()
        llm_service.set_observability_store(store)
        meter_calls = []

        async def meter(**kwargs):
            meter_calls.append(kwargs)

        llm_service.set_metering_callback(meter)
        llm_service.set_observability_context(
            companion_id="legacy-companion", user_id="legacy-user"
        )

        with caplog.at_level("INFO", logger="kestrel_sovereign.llm.service"):
            items = [
                item
                async for item in llm_service.get_streaming_response(
                    system_prompt="system", user_prompt="hello"
                )
            ]

        assert items == ["legacy chunk"]
        llm_service._track_model_usage.assert_not_awaited()
        store.log_llm_call.assert_awaited_once()
        logged = store.log_llm_call.await_args.kwargs
        assert logged["input_tokens"] is None
        assert logged["output_tokens"] is None
        assert logged["metadata"]["usage_available"] is False
        assert meter_calls == []

        import json as _json

        usage_lines = [
            record.message.removeprefix("llm.usage: ")
            for record in caplog.records
            if record.message.startswith("llm.usage: ")
        ]
        assert len(usage_lines) == 1
        usage = _json.loads(usage_lines[0])
        assert usage["usage_available"] is False
        assert usage["input_tokens"] is None
        assert usage["output_tokens"] is None

    @pytest.mark.asyncio
    async def test_remote_message_success_finalizes_once(self, llm_service):
        self._activate_fake_remote(
            llm_service,
            LLMResponse(content="remote message", input_tokens=4, output_tokens=2),
        )
        finalizer = AsyncMock(wraps=llm_service._finalize_successful_invocation)
        llm_service._finalize_successful_invocation = finalizer

        result = await llm_service.generate_with_messages(
            messages=[{"role": "user", "content": "hello"}]
        )

        assert result == "remote message"
        finalizer.assert_awaited_once()
        assert (
            finalizer.await_args.kwargs["path"] == "generate_with_messages.remote_gpu"
        )

    @pytest.mark.asyncio
    async def test_remote_cancellation_preserves_exception_and_records_nothing(
        self, llm_service, mock_adapter
    ):
        self._activate_fake_remote(llm_service, asyncio.CancelledError())
        llm_service._track_model_usage = AsyncMock()
        store = AsyncMock()
        llm_service.set_observability_store(store)

        with pytest.raises(asyncio.CancelledError):
            await llm_service.generate(
                system_prompt="system",
                user_prompt="hello",
                invocation_context=LLMInvocationContext(session_id="cancelled-session"),
            )

        mock_adapter.get_response.assert_not_awaited()
        llm_service._track_model_usage.assert_not_awaited()
        store.log_llm_call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_standard_cancellation_preserves_exception_identity(
        self, llm_service, mock_adapter
    ):
        cancellation = asyncio.CancelledError("request stopped")
        mock_adapter.get_response = AsyncMock(side_effect=cancellation)
        finalizer = AsyncMock(wraps=llm_service._finalize_successful_invocation)
        llm_service._finalize_successful_invocation = finalizer

        with pytest.raises(asyncio.CancelledError) as caught:
            await llm_service.get_response(
                system_prompt="system",
                user_prompt="hello",
            )

        assert caught.value is cancellation
        finalizer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_standard_attempts_never_finalize_success(
        self, llm_service, mock_adapter
    ):
        mock_adapter.get_response = AsyncMock(
            side_effect=LLMProviderError("test-provider", "provider failed")
        )
        finalizer = AsyncMock(wraps=llm_service._finalize_successful_invocation)
        llm_service._finalize_successful_invocation = finalizer
        store = AsyncMock()
        llm_service.set_observability_store(store)

        with pytest.raises(LLMAllProvidersFailedError):
            await llm_service.get_response(
                system_prompt="system",
                user_prompt="hello",
            )

        assert mock_adapter.get_response.await_count == 2
        finalizer.assert_not_awaited()
        assert store.log_llm_call.await_count == 2
        assert all(
            call.kwargs["success"] is False
            for call in store.log_llm_call.await_args_list
        )
        assert all(
            call.kwargs["system_prompt"] == "system"
            and call.kwargs["user_prompt"] == "hello"
            for call in store.log_llm_call.await_args_list
        )
        assert [
            call.kwargs["provider"] for call in store.log_llm_call.await_args_list
        ] == ["openai:api", "anthropic:api"]

    @pytest.mark.asyncio
    async def test_concurrent_legacy_contexts_cannot_cross_contaminate(
        self, llm_service, mock_adapter
    ):
        both_started = asyncio.Event()
        started = []

        async def interleaved_response(*, session_id, **_kwargs):
            started.append(session_id)
            if len(started) == 2:
                both_started.set()
            await both_started.wait()
            if session_id == "session-a":
                await asyncio.sleep(0.01)
            return LLMResponse(
                content=f"answer-{session_id}",
                input_tokens=3,
                output_tokens=2,
                total_tokens=5,
            )

        mock_adapter.get_response = AsyncMock(side_effect=interleaved_response)
        llm_service._track_model_usage = AsyncMock()
        store = AsyncMock()
        llm_service.set_observability_store(store)
        meter_calls = []

        async def meter(**kwargs):
            meter_calls.append(kwargs)

        llm_service.set_metering_callback(meter)

        async def invoke(label):
            llm_service.set_observability_context(
                companion_id=f"companion-{label}", user_id=f"user-{label}"
            )
            return await llm_service.generate_with_messages(
                messages=[{"role": "user", "content": label}],
                session_id=f"session-{label}",
            )

        results = await asyncio.gather(invoke("a"), invoke("b"))

        assert set(results) == {"answer-session-a", "answer-session-b"}
        assert store.log_llm_call.await_count == 2
        by_session = {
            call.kwargs["session_id"]: call.kwargs
            for call in store.log_llm_call.await_args_list
        }
        assert (
            by_session["session-a"]["companion_id"],
            by_session["session-a"]["user_id"],
        ) == ("companion-a", "user-a")
        assert (
            by_session["session-b"]["companion_id"],
            by_session["session-b"]["user_id"],
        ) == ("companion-b", "user-b")
        assert {(call["companion_id"], call["user_id"]) for call in meter_calls} == {
            ("companion-a", "user-a"),
            ("companion-b", "user-b"),
        }


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

        context = llm_service._resolve_invocation_context()
        assert context.session_id == "sess-123"
        assert context.companion_id == "comp-456"
        assert context.user_id == "user-789"

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
