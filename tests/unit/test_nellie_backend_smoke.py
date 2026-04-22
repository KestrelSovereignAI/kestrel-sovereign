"""Nellie backend smoke-proof tests (issue #427)."""

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


def _build_service(provider_names_and_models: dict) -> LLMService:
    """Build an LLMService with specified providers for Nellie testing."""
    priority = list(provider_names_and_models.keys())
    config = {"provider_priority": priority}
    for name, model in provider_names_and_models.items():
        config[name] = {"model": model}

    providers = [
        _make_provider_info(name, model)
        for name, model in provider_names_and_models.items()
    ]

    mock_registry = Mock()
    mock_registry.initialize_providers = Mock(return_value=providers)
    mock_registry.get_providers_with_pattern = Mock(return_value=[])
    mock_registry.get_local_providers = Mock(return_value=[])
    mock_registry.update_provider_client = Mock(return_value=True)

    mandate_config = {"defaults": {}, "mandates": {}}

    with patch("kestrel_sovereign.llm.service.load_config") as mock_load, patch(
        "kestrel_sovereign.llm.service.ProviderRegistry"
    ) as mock_reg_cls:
        mock_load.side_effect = lambda path: (
            config if "llm_config" in path else mandate_config
        )
        mock_reg_cls.return_value = mock_registry
        svc = LLMService(config_path="llm_config.toml")
        svc._usage_db = None
        svc._db_initialized = False
        svc._usage_database_url = None
        svc._db_backend = "sqlite"
        return svc


class TestNellieClaudePlan:
    """Prove Nellie can be pinned to claude_plan and the system tells the truth."""

    @pytest.fixture
    def nellie_service(self):
        return _build_service({
            "anthropic": "claude-sonnet-4-6",
            "claude_plan": "claude-sonnet-4-6",
            "openai": "gpt-5-mini",
        })

    def test_pin_nellie_to_claude_plan(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        pref = svc.get_model_preference()
        assert pref["vendor"] == "claude_plan"
        assert pref["model"] == "claude-sonnet-4-6"

    def test_routing_uses_only_claude_plan(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "claude_plan"
        assert target == "claude-sonnet-4-6"

    def test_active_model_id_matches_preference(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        active = svc.get_active_model_id()
        pref = svc.get_model_preference()

        assert active == pref["model"]

    def test_backend_identity_matches_runtime(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        pref = svc.get_model_preference()
        providers, target = svc.resolve_provider_routing()

        assert pref["vendor"] == providers[0]["name"]
        assert pref["model"] == target

    def test_claude_plan_not_confused_with_anthropic(self, nellie_service):
        svc = nellie_service

        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")
        providers_plan, _ = svc.resolve_provider_routing()

        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic")
        providers_api, _ = svc.resolve_provider_routing()

        assert providers_plan[0]["name"] == "claude_plan"
        assert providers_api[0]["name"] == "anthropic"
        assert providers_plan[0]["name"] != providers_api[0]["name"]

    def test_persistence_callback_fires_on_pin(self, nellie_service):
        svc = nellie_service
        callback = AsyncMock()
        svc.set_preference_persistence_callback(callback)

        async def _run():
            svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")
            await asyncio.sleep(0.01)

        asyncio.run(_run())
        callback.assert_awaited_once_with("claude-sonnet-4-6", "claude_plan")


class TestNellieOpenAIPlan:
    """Prove Nellie can be pinned to openai_plan and the system tells the truth."""

    @pytest.fixture
    def nellie_service(self):
        return _build_service({
            "openai": "gpt-5-mini",
            "openai_plan": "gpt-5.4",
        })

    def test_pin_nellie_to_openai_plan(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        pref = svc.get_model_preference()
        assert pref["vendor"] == "openai_plan"
        assert pref["model"] == "gpt-5.4"

    def test_routing_uses_only_openai_plan(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "openai_plan"
        assert target == "gpt-5.4"

    def test_active_model_id_matches_preference(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        active = svc.get_active_model_id()
        pref = svc.get_model_preference()

        assert active == pref["model"]

    def test_backend_identity_matches_runtime(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        pref = svc.get_model_preference()
        providers, target = svc.resolve_provider_routing()

        assert pref["vendor"] == providers[0]["name"]
        assert pref["model"] == target

    def test_openai_plan_not_confused_with_openai(self, nellie_service):
        svc = nellie_service

        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")
        providers_plan, _ = svc.resolve_provider_routing()

        svc.set_model_preference("gpt-5-mini", vendor="openai")
        providers_openai, _ = svc.resolve_provider_routing()

        assert providers_plan[0]["name"] == "openai_plan"
        assert providers_openai[0]["name"] == "openai"


class TestNellieFailureModes:
    """Prove that failure modes are legible, not silent."""

    @pytest.fixture
    def nellie_without_openai_plan(self):
        return _build_service({
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-5-mini",
        })

    @pytest.fixture
    def nellie_without_claude_plan(self):
        return _build_service({
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-5-mini",
        })

    def test_openai_plan_missing_raises_with_provider_name(self, nellie_without_openai_plan):
        svc = nellie_without_openai_plan
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        error_msg = str(exc_info.value)
        assert "openai_plan" in error_msg
        assert "anthropic" in error_msg or "openai" in error_msg

    def test_claude_plan_missing_raises_with_provider_name(self, nellie_without_claude_plan):
        svc = nellie_without_claude_plan
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        error_msg = str(exc_info.value)
        assert "claude_plan" in error_msg

    def test_error_message_lists_available_providers(self, nellie_without_openai_plan):
        svc = nellie_without_openai_plan
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        error_msg = str(exc_info.value)
        assert "anthropic" in error_msg
        assert "openai" in error_msg

    def test_openai_plan_adapter_requires_client(self):
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        adapter = CodexAdapter()

        with pytest.raises(RuntimeError, match="requires an OAuth token"):
            asyncio.run(adapter.get_response(
                client=None, model="gpt-5.4", messages=[{"role": "user", "content": "hi"}]
            ))

    def test_claude_plan_registry_requires_auth_token(self):
        from kestrel_sovereign.llm.provider_registry import ProviderRegistry

        registry = ProviderRegistry({
            "provider_priority": ["claude_plan"],
            "claude_plan": {"model": "claude-sonnet-4-6"},
        })

        with patch.dict("os.environ", {}, clear=False):
            env = {k: v for k, v in __import__("os").environ.items() if k != "ANTHROPIC_AUTH_TOKEN"}
            with patch.dict("os.environ", env, clear=True):
                with pytest.raises(ValueError, match="ANTHROPIC_AUTH_TOKEN"):
                    registry._initialize_single_provider("claude_plan")


class TestNellieBackendSwitch:
    """Prove Nellie can switch backends and the system stays consistent."""

    @pytest.fixture
    def nellie_service(self):
        return _build_service({
            "claude_plan": "claude-sonnet-4-6",
            "openai_plan": "gpt-5.4",
            "openai": "gpt-5-mini",
        })

    def test_switch_claude_plan_to_openai_plan(self, nellie_service):
        svc = nellie_service

        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")
        providers, _ = svc.resolve_provider_routing()
        assert providers[0]["name"] == "claude_plan"

        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")
        providers, target = svc.resolve_provider_routing()
        assert providers[0]["name"] == "openai_plan"
        assert target == "gpt-5.4"

    def test_switch_openai_plan_to_claude_plan(self, nellie_service):
        svc = nellie_service

        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")
        providers, _ = svc.resolve_provider_routing()
        assert providers[0]["name"] == "openai_plan"

        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")
        providers, target = svc.resolve_provider_routing()
        assert providers[0]["name"] == "claude_plan"
        assert target == "claude-sonnet-4-6"

    def test_clear_preference_returns_to_all_providers(self, nellie_service):
        svc = nellie_service

        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")
        providers, _ = svc.resolve_provider_routing()
        assert len(providers) == 1

        svc.clear_model_preference()
        providers, target = svc.resolve_provider_routing()
        assert len(providers) == 3
        assert target is None

    def test_identity_consistent_through_switch(self, nellie_service):
        svc = nellie_service

        for provider, model in [
            ("claude_plan", "claude-sonnet-4-6"),
            ("openai_plan", "gpt-5.4"),
            ("openai", "gpt-5-mini"),
        ]:
            svc.set_model_preference(model, vendor=provider)

            active = svc.get_active_model_id()
            pref = svc.get_model_preference()
            providers, target = svc.resolve_provider_routing()

            assert active == model
            assert pref["vendor"] == provider
            assert pref["model"] == model
            assert providers[0]["name"] == provider
            assert target == model
