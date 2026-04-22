"""Contract tests for agent-specific backend routing (issue #425)."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kestrel_sovereign.llm.error_handling import LLMProviderUnavailableError
from kestrel_sovereign.llm.service import LLMService


def _make_provider_info(name: str, model: str = "auto"):
    """Create a mock ProviderInfo dataclass instance."""
    info = Mock()
    info.name = name
    info.model = model
    info.client = AsyncMock()
    info.adapter = Mock()
    return info


@pytest.fixture
def service_with_providers():
    """Create an LLMService with canonical execution providers."""
    mock_config = {
        "provider_priority": ["openai", "anthropic", "claude_plan"],
        "openai": {"model": "gpt-5-mini"},
        "anthropic": {"model": "claude-sonnet-4-6"},
        "claude_plan": {"model": "claude-sonnet-4-6"},
    }
    mock_mandate_config = {"defaults": {}, "mandates": {}}

    openai_info = _make_provider_info("openai", "gpt-5-mini")
    anthropic_info = _make_provider_info("anthropic", "claude-sonnet-4-6")
    claude_plan_info = _make_provider_info("claude_plan", "claude-sonnet-4-6")

    mock_registry = Mock()
    mock_registry.initialize_providers = Mock(
        return_value=[openai_info, anthropic_info, claude_plan_info]
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
        svc = LLMService(config_path="llm_config.toml")
        svc._usage_db = None
        svc._db_initialized = False
        svc._usage_database_url = None
        svc._db_backend = "sqlite"
        return svc


@pytest.fixture
def service_with_openai_plan():
    """Create an LLMService with API and subscription-backed OpenAI providers."""
    mock_config = {
        "provider_priority": ["openai", "openai_plan"],
        "openai": {"model": "gpt-5.4"},
        "openai_plan": {"model": "gpt-5.4"},
    }
    mock_mandate_config = {"defaults": {}, "mandates": {}}

    openai_info = _make_provider_info("openai", "gpt-5.4")
    openai_plan_info = _make_provider_info("openai_plan", "gpt-5.4")

    mock_registry = Mock()
    mock_registry.initialize_providers = Mock(
        return_value=[openai_info, openai_plan_info]
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
        svc = LLMService(config_path="llm_config.toml")
        svc._usage_db = None
        svc._db_initialized = False
        svc._usage_database_url = None
        svc._db_backend = "sqlite"
        return svc


class TestPreferenceRoundTrip:
    """Verify that set -> get -> routing stays coherent."""

    def test_set_provider_and_model(self, service_with_providers):
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

    def test_model_only_preference_keeps_all_providers(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("gpt-5-mini")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 3
        assert target == "gpt-5-mini"

    def test_provider_preference_restricts_to_single_provider(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "claude_plan"
        assert target == "claude-sonnet-4-6"

    def test_model_override_takes_precedence(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        providers, target = svc.resolve_provider_routing(
            model_override="openai/gpt-5-mini"
        )

        assert len(providers) == 1
        assert providers[0]["name"] == "openai"
        assert target == "gpt-5-mini"

    def test_bare_model_override_keeps_all_providers(self, service_with_providers):
        providers, target = service_with_providers.resolve_provider_routing(
            model_override="gpt-5-mini"
        )

        assert len(providers) == 3
        assert target == "gpt-5-mini"


class TestUnavailableProviderFails:
    """Contract: requesting a provider that is not initialized raises clearly."""

    def test_mandate_preference_for_missing_provider_raises(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        assert "openai_plan" in str(exc_info.value)
        assert "openai" in str(exc_info.value)

    def test_model_override_for_missing_provider_raises(self, service_with_providers):
        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            service_with_providers.resolve_provider_routing(
                model_override="openai_plan/gpt-5.4"
            )

        assert "openai_plan" in str(exc_info.value)

    def test_unavailable_with_fallbacks_degrades_gracefully(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")
        svc.add_fallback_model("gpt-5-mini", provider="openai")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "openai"
        assert target == "gpt-5-mini"


class TestClaudePlanRouting:
    """Contract: claude_plan is distinct from anthropic."""

    def test_claude_plan_is_distinct_from_anthropic(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        providers, _ = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "claude_plan"
        assert providers[0]["name"] != "anthropic"

    def test_anthropic_preference_does_not_use_claude_plan(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic")

        providers, _ = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "anthropic"


class TestOpenAIPlanRouting:
    """Contract: openai_plan is routable when registered."""

    def test_openai_plan_preference_routes_to_openai_plan(self, service_with_openai_plan):
        svc = service_with_openai_plan
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "openai_plan"
        assert target == "gpt-5.4"

    def test_openai_plan_override_routes_to_openai_plan(self, service_with_openai_plan):
        providers, target = service_with_openai_plan.resolve_provider_routing(
            model_override="openai_plan/gpt-5.4"
        )

        assert len(providers) == 1
        assert providers[0]["name"] == "openai_plan"
        assert target == "gpt-5.4"


class TestPreferencePersistence:
    """Contract: preference changes trigger the persistence callback."""

    def test_persistence_callback_called_on_set(self, service_with_providers):
        svc = service_with_providers
        callback = AsyncMock()
        svc.set_preference_persistence_callback(callback)

        async def _run():
            svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")
            await asyncio.sleep(0.01)

        asyncio.run(_run())

        callback.assert_awaited_once_with("claude-sonnet-4-6", "claude_plan")

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
        assert callback.await_args_list[-1].args == (None, None)


class TestOpenAIPlanAdapter:
    """Contract: plan adapter raises clearly and delegates discovery."""

    def test_openai_plan_adapter_raises_without_client(self):
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        adapter = CodexAdapter()

        with pytest.raises(RuntimeError, match="requires an OAuth token"):
            asyncio.run(adapter.get_response(
                client=None, model="gpt-5.4", messages=[]
            ))

    def test_openai_plan_adapter_does_not_list_models(self):
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        adapter = CodexAdapter()
        with pytest.raises(NotImplementedError, match="canonical openai provider"):
            asyncio.run(adapter.list_models())


class TestOpenAIPlanProviderRegistry:
    """Contract: openai_plan can be initialized via provider_registry."""

    def test_registry_initializes_openai_plan(self, monkeypatch):
        from kestrel_sovereign.llm.provider_registry import ProviderRegistry

        monkeypatch.setenv("CODEX_AUTH_TOKEN", "test-token")
        config = {
            "provider_priority": ["openai_plan"],
            "openai_plan": {"model": "gpt-5.4"},
        }
        registry = ProviderRegistry(config)
        provider = registry._initialize_single_provider("openai_plan")

        assert provider is not None
        assert provider.name == "openai_plan"
        assert provider.model == "gpt-5.4"

        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        assert isinstance(provider.adapter, CodexAdapter)
