"""Contract tests for agent-specific backend routing (issue #425).

Updated for the vendor/route/model architecture (epic #688).

Provider names are composite ``"<vendor>:<route>"`` keys. Mandate preferences
carry ``{vendor, model, route?}`` rather than the old flat ``{model, provider}``.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kestrel_sovereign.llm.error_handling import LLMProviderUnavailableError
from kestrel_sovereign.llm.service import LLMService


def _make_provider_info(name: str, model: str = "auto", *, vendor=None, route=None):
    """Create a mock ProviderInfo dataclass instance.

    ``name`` is the composite ``vendor:route`` key. If ``vendor``/``route``
    aren't passed explicitly, they're parsed from ``name``.
    """
    info = Mock()
    info.name = name
    info.model = model
    info.client = AsyncMock()
    info.adapter = Mock()
    if vendor is None or route is None:
        if ":" in name:
            v, r = name.split(":", 1)
            info.vendor = vendor or v
            info.route = route or r
        else:
            info.vendor = vendor or name
            info.route = route or "api"
    else:
        info.vendor = vendor
        info.route = route
    info.is_cloud = True
    info.is_local = False
    info.base_url = None
    info.selection_hints = []
    return info


@pytest.fixture
def service_with_providers():
    """LLMService with three routes: openai:api, anthropic:api, anthropic:plan."""
    mock_config = {
        "route_priority": ["openai:api", "anthropic:api", "anthropic:plan"],
        "vendors": {
            "openai": {"routes": {"api": {"model": "gpt-5-mini", "adapter": "OpenAIAdapter"}}},
            "anthropic": {
                "routes": {
                    "api": {"model": "claude-sonnet-4-6", "adapter": "AnthropicAdapter"},
                    "plan": {"model": "claude-sonnet-4-6", "adapter": "ClaudeMaxAdapter"},
                },
            },
        },
    }
    mock_mandate_config = {"defaults": {}, "mandates": {}}

    openai_api = _make_provider_info("openai:api", "gpt-5-mini")
    anthropic_api = _make_provider_info("anthropic:api", "claude-sonnet-4-6")
    anthropic_plan = _make_provider_info("anthropic:plan", "claude-sonnet-4-6")

    mock_registry = Mock()
    mock_registry.initialize_providers = Mock(
        return_value=[openai_api, anthropic_api, anthropic_plan]
    )
    mock_registry.get_providers_with_pattern = Mock(return_value=[])
    mock_registry.get_local_providers = Mock(return_value=[])
    mock_registry.update_provider_client = Mock(return_value=True)

    with patch("kestrel_sovereign.llm.service.load_config") as mock_load, patch(
        "kestrel_sovereign.llm.service.ProviderRegistry"
    ) as mock_reg_cls:
        mock_load.side_effect = lambda path: (
            mock_config if "llm_config" in path else mock_mandate_config
        )
        mock_reg_cls.return_value = mock_registry
        svc = LLMService()
        svc._usage_db = None
        svc._db_initialized = False
        svc._usage_database_url = None
        svc._db_backend = "sqlite"
        return svc


@pytest.fixture
def service_with_openai_plan():
    """LLMService with two OpenAI routes: api + plan (Codex subscription)."""
    mock_config = {
        "route_priority": ["openai:api", "openai:plan"],
        "vendors": {
            "openai": {
                "routes": {
                    "api": {"model": "gpt-5.4", "adapter": "OpenAIAdapter"},
                    "plan": {"model": "gpt-5.4", "adapter": "CodexAdapter"},
                },
            },
        },
    }
    mock_mandate_config = {"defaults": {}, "mandates": {}}

    openai_api = _make_provider_info("openai:api", "gpt-5.4")
    openai_plan = _make_provider_info("openai:plan", "gpt-5.4")

    mock_registry = Mock()
    mock_registry.initialize_providers = Mock(
        return_value=[openai_api, openai_plan]
    )
    mock_registry.get_providers_with_pattern = Mock(return_value=[])
    mock_registry.get_local_providers = Mock(return_value=[])
    mock_registry.update_provider_client = Mock(return_value=True)

    with patch("kestrel_sovereign.llm.service.load_config") as mock_load, patch(
        "kestrel_sovereign.llm.service.ProviderRegistry"
    ) as mock_reg_cls:
        mock_load.side_effect = lambda path: (
            mock_config if "llm_config" in path else mock_mandate_config
        )
        mock_reg_cls.return_value = mock_registry
        svc = LLMService()
        svc._usage_db = None
        svc._db_initialized = False
        svc._usage_database_url = None
        svc._db_backend = "sqlite"
        return svc


class TestPreferenceRoundTrip:
    """Verify that set -> get -> routing stays coherent."""

    def test_set_vendor_route_and_model(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        pref = svc.get_model_preference()
        assert pref["model"] == "claude-sonnet-4-6"
        assert pref["vendor"] == "anthropic"
        assert pref["route"] == "plan"

    def test_preference_survives_get_active_model_id(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        active = svc.get_active_model_id()
        assert active == "claude-sonnet-4-6"

    def test_clear_preference_returns_to_default(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")
        svc.clear_model_preference()

        pref = svc.get_model_preference()
        assert pref["model"] is None
        assert pref["vendor"] is None

    def test_auto_model_is_ignored(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("auto", vendor="openai")

        pref = svc.get_model_preference()
        assert pref["model"] is None
        assert pref["vendor"] is None


class TestResolveProviderRouting:
    """Contract: resolve_provider_routing is the single routing authority."""

    def test_no_preference_returns_all_providers(self, service_with_providers):
        providers, target = service_with_providers.resolve_provider_routing()
        assert len(providers) == 3
        assert target is None

    def test_bare_model_preference_requires_vendor_resolution(self, service_with_providers):
        """A bare model with no vendor must resolve via discovery, not broadcast.

        Before #688 fix, set_model_preference("gpt-5-mini") persisted
        {vendor: None, model: "gpt-5-mini"} and resolve_provider_routing
        would return ALL providers with target_model="gpt-5-mini" — a
        broadcast cascade that ended up serving the request from whichever
        provider happened to recognize the id (OpenRouter → Gemini in one
        live incident). Now, a bare model requires the catalog to identify
        a single vendor, or set_model_preference raises ValueError.
        """
        svc = service_with_providers

        # With an empty discovery cache, the refusal bubbles up — no
        # vendor-less mandate can be persisted.
        from unittest.mock import MagicMock, patch as _patch
        empty_cache = MagicMock()
        empty_cache.get_any = MagicMock(return_value=None)
        with _patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=empty_cache,
        ):
            with pytest.raises(ValueError):
                svc.set_model_preference("gpt-5-mini")

        # Mandate is unchanged (defaults), so routing returns all providers
        # with no target — the normal "no mandate" path.
        providers, target = svc.resolve_provider_routing()
        assert len(providers) == 3
        assert target is None

    def test_vendor_route_preference_restricts_to_single_route(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "anthropic:plan"
        assert target == "claude-sonnet-4-6"

    def test_vendor_only_preference_keeps_all_routes_for_vendor(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic")

        providers, target = svc.resolve_provider_routing()

        # Both anthropic routes should be returned (api + plan).
        assert len(providers) == 2
        assert {p["name"] for p in providers} == {"anthropic:api", "anthropic:plan"}
        assert target == "claude-sonnet-4-6"

    def test_model_override_takes_precedence(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        providers, target = svc.resolve_provider_routing(
            model_override="openai/gpt-5-mini"
        )

        assert len(providers) == 1
        assert providers[0]["name"] == "openai:api"
        assert target == "gpt-5-mini"

    def test_vendor_route_override_pins_exact_route(self, service_with_providers):
        providers, target = service_with_providers.resolve_provider_routing(
            model_override="anthropic:plan/claude-sonnet-4-6"
        )

        assert len(providers) == 1
        assert providers[0]["name"] == "anthropic:plan"
        assert target == "claude-sonnet-4-6"

    def test_bare_model_override_keeps_all_providers(self, service_with_providers):
        providers, target = service_with_providers.resolve_provider_routing(
            model_override="gpt-5-mini"
        )

        assert len(providers) == 3
        assert target == "gpt-5-mini"


class TestUnavailableProviderFails:
    """Contract: requesting a provider that is not initialized raises clearly."""

    def test_mandate_preference_for_missing_route_raises(self, service_with_providers):
        svc = service_with_providers
        # openai:plan isn't initialized in this fixture.
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        assert "openai:plan" in str(exc_info.value)

    def test_model_override_for_missing_vendor_route_raises(self, service_with_providers):
        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            service_with_providers.resolve_provider_routing(
                model_override="openai:plan/gpt-5.4"
            )

        assert "openai:plan" in str(exc_info.value)

    def test_unavailable_with_fallbacks_degrades_gracefully(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")
        svc.add_fallback_model("gpt-5-mini", provider="openai")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "openai:api"
        assert target == "gpt-5-mini"


class TestEmptyProviderListRaisesClearly:
    """Regression: resolve_provider_routing must never return an empty list.

    The streaming code paths iterate the returned providers as their fallback
    chain. Zero providers means the loop runs zero times; the only error
    available to surface is ``last_error=None``, which the wrapper renders as
    "All providers failed: None" with no clue *why* nothing was tried — the
    exact symptom downstream users (frinz integration tests) reported after
    this refactor. resolve_provider_routing now raises a specific
    ``LLMServiceError`` for the two real reasons:

      1. Every initialized route was disabled this session by a permanent
         auth failure (401/403).
      2. No routes were configured at all.
    """

    def test_all_routes_disabled_raises_with_reasons(self, service_with_providers):
        from kestrel_sovereign.llm.service import LLMServiceError

        svc = service_with_providers
        # Simulate: every initialized route hit a 401/403 earlier this session.
        for p in svc.providers:
            svc._disabled_routes[p["name"]] = "401 invalid_api_key"

        with pytest.raises(LLMServiceError) as exc_info:
            svc.resolve_provider_routing()

        msg = str(exc_info.value)
        # Each disabled route is named in the error so the operator knows
        # *which* keys to rotate.
        for p in svc.providers:
            assert p["name"] in msg
        assert "Rotate keys" in msg

    def test_no_routes_configured_raises_with_config_hint(self, service_with_providers):
        from kestrel_sovereign.llm.service import LLMServiceError

        svc = service_with_providers
        # Pathological: provider initialization produced zero usable routes
        # (e.g. every vendor missing its env var).
        svc.providers = []

        with pytest.raises(LLMServiceError) as exc_info:
            svc.resolve_provider_routing()

        msg = str(exc_info.value)
        assert "No LLM routes" in msg
        assert "kestrel.toml [llm]" in msg

    def test_streaming_does_not_see_empty_provider_list(self, service_with_providers):
        """End-to-end shape: zero usable routes raises *before* the streaming
        loop, so callers never get the misleading "All providers failed: None"
        with ``last_error`` unset."""
        from kestrel_sovereign.llm.service import LLMServiceError

        svc = service_with_providers
        svc.providers = []

        # Whatever the caller asks for, resolution raises before returning.
        with pytest.raises(LLMServiceError):
            svc.resolve_provider_routing(model_override=None)


class TestAnthropicPlanVsApi:
    """Contract: anthropic:plan and anthropic:api are distinct routes."""

    def test_plan_route_is_selected_when_mandated(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        providers, _ = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "anthropic:plan"

    def test_api_route_is_selected_when_mandated(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="api")

        providers, _ = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "anthropic:api"


class TestOpenAIPlanRouting:
    """Contract: openai:plan is routable when registered."""

    def test_openai_plan_preference_routes_to_openai_plan(self, service_with_openai_plan):
        svc = service_with_openai_plan
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "openai:plan"
        assert target == "gpt-5.4"

    def test_openai_plan_override_routes_to_openai_plan(self, service_with_openai_plan):
        providers, target = service_with_openai_plan.resolve_provider_routing(
            model_override="openai:plan/gpt-5.4"
        )

        assert len(providers) == 1
        assert providers[0]["name"] == "openai:plan"
        assert target == "gpt-5.4"


class TestPreferencePersistence:
    """Contract: preference changes trigger the persistence callback."""

    def test_persistence_callback_called_on_set(self, service_with_providers):
        svc = service_with_providers
        callback = AsyncMock()
        svc.set_preference_persistence_callback(callback)

        async def _run():
            svc.set_model_preference(
                "claude-sonnet-4-6", vendor="anthropic", route="plan"
            )
            await asyncio.sleep(0.01)

        asyncio.run(_run())

        # Callback signature is (model, vendor, route).
        callback.assert_awaited_once_with("claude-sonnet-4-6", "anthropic", "plan")

    def test_persistence_callback_called_on_clear(self, service_with_providers):
        svc = service_with_providers
        callback = AsyncMock()
        svc.set_preference_persistence_callback(callback)

        async def _run():
            svc.set_model_preference("gpt-5", vendor="openai")
            await asyncio.sleep(0.01)
            svc.clear_model_preference()
            await asyncio.sleep(0.01)

        asyncio.run(_run())
        # Clear sends (None, None, None).
        assert callback.await_args_list[-1].args == (None, None, None)


class TestCodexAdapterContract:
    """Contract: the Codex adapter raises clearly and delegates discovery."""

    def test_codex_adapter_raises_without_client(self):
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        adapter = CodexAdapter()

        with pytest.raises(RuntimeError, match="requires an OAuth token"):
            asyncio.run(adapter.get_response(
                client=None, model="gpt-5.4", messages=[]
            ))

    def test_codex_adapter_does_not_list_models(self):
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        adapter = CodexAdapter()
        with pytest.raises(NotImplementedError, match="canonical openai"):
            asyncio.run(adapter.list_models())


class TestOpenAIPlanProviderRegistry:
    """Contract: openai:plan can be initialized via provider_registry."""

    def test_registry_initializes_openai_plan(self, monkeypatch):
        from kestrel_sovereign.llm.provider_registry import ProviderRegistry

        monkeypatch.setenv("CODEX_AUTH_TOKEN", "test-token")
        config = {
            "route_priority": ["openai:plan"],
            "vendors": {
                "openai": {
                    "is_cloud": True,
                    "routes": {
                        "plan": {
                            "adapter": "CodexAdapter",
                            "auth_token_env": "CODEX_AUTH_TOKEN",
                            "model": "gpt-5.4",
                        },
                    },
                },
            },
        }
        registry = ProviderRegistry(config)
        providers = registry.initialize_providers()

        assert len(providers) == 1
        provider = providers[0]
        assert provider.name == "openai:plan"
        assert provider.vendor == "openai"
        assert provider.route == "plan"
        assert provider.model == "gpt-5.4"

        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        assert isinstance(provider.adapter, CodexAdapter)
