"""Contract tests for agent-specific backend routing (issue #425).

These tests verify:
- Per-agent provider preference survives set/get round-trip
- resolve_provider_routing() honours provider + model preferences
- Unavailable providers fail honestly (LLMProviderUnavailableError)
- Fallback models are used when configured
- claude_max and codex are explicitly routable
- Preference persistence round-trip (set → get → routing)
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from kestrel_sovereign.llm.error_handling import LLMProviderUnavailableError
from kestrel_sovereign.llm.service import LLMService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(name: str, model: str = "auto") -> dict:
    """Create a minimal provider dict for testing."""
    adapter = Mock()
    adapter.create_messages = Mock(return_value=[])
    adapter.get_response = AsyncMock()
    return {
        "name": name,
        "client": AsyncMock(),
        "adapter": adapter,
        "model": model,
    }


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
    """Create an LLMService with mocked providers."""
    mock_config = {
        "provider_priority": ["openai", "anthropic", "claude_max"],
        "openai": {"model": "gpt-5-mini"},
        "anthropic": {"model": "claude-sonnet-4-6"},
        "claude_max": {"model": "claude-sonnet-4-6"},
    }
    mock_mandate_config = {"defaults": {}, "mandates": {}}

    openai_info = _make_provider_info("openai", "gpt-5-mini")
    anthropic_info = _make_provider_info("anthropic", "claude-sonnet-4-6")
    claude_max_info = _make_provider_info("claude_max", "claude-sonnet-4-6")

    mock_registry = Mock()
    mock_registry.initialize_providers = Mock(
        return_value=[openai_info, anthropic_info, claude_max_info]
    )
    mock_registry.get_providers_with_pattern = Mock(return_value=[])
    mock_registry.get_local_providers = Mock(return_value=[])
    mock_registry.update_provider_client = Mock(return_value=True)

    with patch("kestrel_sovereign.llm.service.load_config") as mock_load, \
         patch("kestrel_sovereign.llm.service.ProviderRegistry") as mock_reg_cls:
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


# ===========================================================================
# 1. Provider preference round-trip
# ===========================================================================


class TestPreferenceRoundTrip:
    """Verify that set → get → routing is consistent."""

    def test_set_provider_and_model(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")

        pref = svc.get_model_preference()
        assert pref["model"] == "claude-sonnet-4-6"
        assert pref["provider"] == "claude_max"

    def test_preference_survives_get_active_model_id(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")

        active = svc.get_active_model_id()
        assert active == "claude-sonnet-4-6"

    def test_clear_preference_returns_to_default(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")
        svc.clear_model_preference()

        pref = svc.get_model_preference()
        assert pref["model"] is None
        assert pref["provider"] is None

    def test_auto_model_is_ignored(self, service_with_providers):
        """Setting model='auto' should not create a real preference."""
        svc = service_with_providers
        svc.set_model_preference("auto", provider="openai")

        pref = svc.get_model_preference()
        assert pref["model"] is None


# ===========================================================================
# 2. resolve_provider_routing — explicit provider selection
# ===========================================================================


class TestResolveProviderRouting:
    """Contract: resolve_provider_routing is the single routing authority."""

    def test_no_preference_returns_all_providers(self, service_with_providers):
        svc = service_with_providers
        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 3
        assert target is None

    def test_model_only_preference_keeps_all_providers(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("gpt-5-mini")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 3
        assert target == "gpt-5-mini"

    def test_provider_preference_restricts_to_single_provider(self, service_with_providers):
        """When provider is set, ONLY that provider should be returned."""
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "claude_max"
        assert target == "claude-sonnet-4-6"

    def test_model_override_takes_precedence(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")

        providers, target = svc.resolve_provider_routing(
            model_override="openai/gpt-5-mini"
        )

        assert len(providers) == 1
        assert providers[0]["name"] == "openai"
        assert target == "gpt-5-mini"

    def test_bare_model_override_keeps_all_providers(self, service_with_providers):
        svc = service_with_providers

        providers, target = svc.resolve_provider_routing(
            model_override="gpt-5-mini"
        )

        assert len(providers) == 3
        assert target == "gpt-5-mini"


# ===========================================================================
# 3. Unavailable provider — honest failure
# ===========================================================================


class TestUnavailableProviderFails:
    """Contract: requesting a provider that isn't initialized raises clearly."""

    def test_mandate_preference_for_missing_provider_raises(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("codex", provider="codex")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        assert "codex" in str(exc_info.value)
        assert "openai" in str(exc_info.value)  # shows available providers

    def test_model_override_for_missing_provider_raises(self, service_with_providers):
        svc = service_with_providers

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing(model_override="codex/codex")

        assert "codex" in str(exc_info.value)

    def test_unavailable_with_fallbacks_degrades_gracefully(self, service_with_providers):
        """When fallbacks are configured, use them instead of raising."""
        svc = service_with_providers
        svc.set_model_preference("codex", provider="codex")
        svc.add_fallback_model("gpt-5-mini", provider="openai")

        providers, target = svc.resolve_provider_routing()

        # Should fall back to openai, not raise
        assert len(providers) == 1
        assert providers[0]["name"] == "openai"


# ===========================================================================
# 4. claude_max routing
# ===========================================================================


class TestClaudeMaxRouting:
    """Contract: claude_max is a first-class provider distinct from anthropic."""

    def test_claude_max_is_distinct_from_anthropic(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "claude_max"
        # NOT anthropic — even though the model name is the same
        assert providers[0]["name"] != "anthropic"

    def test_anthropic_preference_does_not_use_claude_max(self, service_with_providers):
        svc = service_with_providers
        svc.set_model_preference("claude-sonnet-4-6", provider="anthropic")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "anthropic"


# ===========================================================================
# 5. codex routing (stub — provider registered but not yet functional)
# ===========================================================================


class TestCodexRouting:
    """Contract: codex provider is routable when registered."""

    @pytest.fixture
    def service_with_codex(self):
        """Service that includes codex in the provider list."""
        mock_config = {
            "provider_priority": ["openai", "codex"],
            "openai": {"model": "gpt-5-mini"},
            "codex": {"model": "codex"},
        }
        mock_mandate_config = {"defaults": {}, "mandates": {}}

        openai_info = _make_provider_info("openai", "gpt-5-mini")
        codex_info = _make_provider_info("codex", "codex")

        mock_registry = Mock()
        mock_registry.initialize_providers = Mock(
            return_value=[openai_info, codex_info]
        )
        mock_registry.get_providers_with_pattern = Mock(return_value=[])
        mock_registry.get_local_providers = Mock(return_value=[])

        with patch("kestrel_sovereign.llm.service.load_config") as mock_load, \
             patch("kestrel_sovereign.llm.service.ProviderRegistry") as mock_reg_cls:
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

    def test_codex_preference_routes_to_codex(self, service_with_codex):
        svc = service_with_codex
        svc.set_model_preference("codex", provider="codex")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "codex"
        assert target == "codex"

    def test_codex_override_routes_to_codex(self, service_with_codex):
        svc = service_with_codex

        providers, target = svc.resolve_provider_routing(
            model_override="codex/codex"
        )

        assert len(providers) == 1
        assert providers[0]["name"] == "codex"


# ===========================================================================
# 6. Preference persistence contract
# ===========================================================================


class TestPreferencePersistence:
    """Contract: preference changes trigger the persistence callback."""

    def test_persistence_callback_called_on_set(self, service_with_providers):
        svc = service_with_providers
        callback = AsyncMock()
        svc.set_preference_persistence_callback(callback)

        # We need a running event loop for the fire-and-forget task
        import asyncio

        async def _run():
            svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")
            # Give the fire-and-forget task a chance to run
            await asyncio.sleep(0.01)

        asyncio.run(_run())

        callback.assert_awaited_once_with("claude-sonnet-4-6", "claude_max")

    def test_persistence_callback_called_on_clear(self, service_with_providers):
        svc = service_with_providers
        callback = AsyncMock()
        svc.set_preference_persistence_callback(callback)

        import asyncio

        async def _run():
            svc.set_model_preference("gpt-5", provider="openai")
            await asyncio.sleep(0.01)
            svc.clear_model_preference()
            await asyncio.sleep(0.01)

        asyncio.run(_run())

        # Last call should be (None, None)
        assert callback.await_args_list[-1].args == (None, None)


# ===========================================================================
# 7. Codex adapter contract
# ===========================================================================


class TestCodexAdapter:
    """Contract: codex adapter raises clearly on inference, lists models."""

    def test_codex_adapter_raises_without_client(self):
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        adapter = CodexAdapter()
        import asyncio

        with pytest.raises(RuntimeError, match="requires an OpenAI client"):
            asyncio.run(adapter.get_response(
                client=None, model="codex", messages=[]
            ))

    def test_codex_adapter_lists_models(self):
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        adapter = CodexAdapter()
        import asyncio
        models = asyncio.run(adapter.list_models())

        assert len(models) >= 1
        assert all(m.provider == "codex" for m in models)
        # Should include at least the featured model
        model_ids = [m.id for m in models]
        assert "gpt-5.4" in model_ids


# ===========================================================================
# 8. Provider registry — codex initialization
# ===========================================================================


class TestCodexProviderRegistry:
    """Contract: codex can be initialized via provider_registry."""

    def test_registry_initializes_codex(self):
        from kestrel_sovereign.llm.provider_registry import ProviderRegistry

        config = {
            "provider_priority": ["codex"],
            "codex": {"model": "codex"},
        }
        registry = ProviderRegistry(config)
        provider = registry._initialize_single_provider("codex")

        assert provider is not None
        assert provider.name == "codex"
        assert provider.model == "codex"
        # Codex uses OpenAI Responses API via AsyncOpenAI client
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter
        assert isinstance(provider.adapter, CodexAdapter)
