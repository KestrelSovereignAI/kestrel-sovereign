"""Nellie backend smoke-proof tests (issue #427, updated for epic #688).

These tests prove that an agent pinned to a specific vendor/route combination
stays pinned, and that failure modes are legible rather than silent.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kestrel_sovereign.llm.error_handling import LLMProviderUnavailableError
from kestrel_sovereign.llm.service import LLMService


def _make_provider_info(name: str, model: str = "auto"):
    """Create a mock ProviderInfo for a composite ``vendor:route`` name."""
    info = Mock()
    info.name = name
    info.model = model
    info.client = AsyncMock()
    info.adapter = Mock()
    if ":" in name:
        info.vendor, info.route = name.split(":", 1)
    else:
        info.vendor = name
        info.route = "api"
    info.is_cloud = True
    info.is_local = False
    info.base_url = None
    info.selection_hints = []
    return info


def _build_service(provider_names_and_models: dict) -> LLMService:
    """Build an LLMService whose registry returns the given composite routes.

    Keys are ``vendor:route`` composite names (``"anthropic:plan"`` etc.);
    values are the default model for that route.
    """
    ordered = list(provider_names_and_models.keys())
    vendors_cfg: dict = {}
    for key in ordered:
        vendor, route = key.split(":", 1)
        vendors_cfg.setdefault(vendor, {"routes": {}})["routes"][route] = {
            "model": provider_names_and_models[key],
            "adapter": "OpenAIAdapter",
        }
    config = {"route_priority": ordered, "vendors": vendors_cfg}

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


class TestNellieAnthropicPlan:
    """Prove Nellie can be pinned to anthropic:plan and the system tells the truth."""

    @pytest.fixture
    def nellie_service(self):
        return _build_service({
            "anthropic:api": "claude-sonnet-4-6",
            "anthropic:plan": "claude-sonnet-4-6",
            "openai:api": "gpt-5-mini",
        })

    def test_pin_nellie_to_anthropic_plan(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        pref = svc.get_model_preference()
        assert pref["vendor"] == "anthropic"
        assert pref["route"] == "plan"
        assert pref["model"] == "claude-sonnet-4-6"

    def test_routing_uses_only_anthropic_plan(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "anthropic:plan"
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

        # Vendor:route identity ↔ resolved provider name.
        assert f"{pref['vendor']}:{pref['route']}" == providers[0]["name"]
        assert pref["model"] == target

    def test_plan_not_confused_with_api(self, nellie_service):
        svc = nellie_service

        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")
        providers_plan, _ = svc.resolve_provider_routing()

        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="api")
        providers_api, _ = svc.resolve_provider_routing()

        assert providers_plan[0]["name"] == "anthropic:plan"
        assert providers_api[0]["name"] == "anthropic:api"
        assert providers_plan[0]["name"] != providers_api[0]["name"]

    def test_persistence_callback_fires_on_pin(self, nellie_service):
        svc = nellie_service
        callback = AsyncMock()
        svc.set_preference_persistence_callback(callback)

        async def _run():
            svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")
            await asyncio.sleep(0.01)

        asyncio.run(_run())
        # New callback signature is (model, vendor, route).
        callback.assert_awaited_once_with("claude-sonnet-4-6", "anthropic", "plan")


class TestNellieOpenAIPlan:
    """Prove Nellie can be pinned to openai:plan and the system tells the truth."""

    @pytest.fixture
    def nellie_service(self):
        return _build_service({
            "openai:api": "gpt-5-mini",
            "openai:plan": "gpt-5.4",
        })

    def test_pin_nellie_to_openai_plan(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        pref = svc.get_model_preference()
        assert pref["vendor"] == "openai"
        assert pref["route"] == "plan"
        assert pref["model"] == "gpt-5.4"

    def test_routing_uses_only_openai_plan(self, nellie_service):
        svc = nellie_service
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "openai:plan"
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

        assert f"{pref['vendor']}:{pref['route']}" == providers[0]["name"]
        assert pref["model"] == target

    def test_openai_plan_not_confused_with_openai_api(self, nellie_service):
        svc = nellie_service

        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")
        providers_plan, _ = svc.resolve_provider_routing()

        svc.set_model_preference("gpt-5-mini", vendor="openai", route="api")
        providers_api, _ = svc.resolve_provider_routing()

        assert providers_plan[0]["name"] == "openai:plan"
        assert providers_api[0]["name"] == "openai:api"


class TestNellieFailureModes:
    """Prove that failure modes are legible, not silent."""

    @pytest.fixture
    def nellie_without_plan_routes(self):
        return _build_service({
            "anthropic:api": "claude-sonnet-4-6",
            "openai:api": "gpt-5-mini",
        })

    def test_openai_plan_missing_raises_with_route_name(self, nellie_without_plan_routes):
        svc = nellie_without_plan_routes
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        error_msg = str(exc_info.value)
        assert "openai:plan" in error_msg

    def test_anthropic_plan_missing_raises_with_route_name(self, nellie_without_plan_routes):
        svc = nellie_without_plan_routes
        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        error_msg = str(exc_info.value)
        assert "anthropic:plan" in error_msg

    def test_error_message_lists_available_routes(self, nellie_without_plan_routes):
        svc = nellie_without_plan_routes
        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        error_msg = str(exc_info.value)
        # The "available" list contains the composite names.
        assert "anthropic:api" in error_msg
        assert "openai:api" in error_msg

    def test_codex_adapter_requires_client(self):
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        adapter = CodexAdapter()

        with pytest.raises(RuntimeError, match="requires an OAuth token"):
            asyncio.run(adapter.get_response(
                client=None,
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            ))

    def test_anthropic_plan_route_requires_auth_token(self):
        """anthropic:plan needs either ANTHROPIC_AUTH_TOKEN or an inline auth_token."""
        from kestrel_sovereign.llm.provider_registry import (
            ProviderRegistry,
            ProviderInitializationError,
        )
        import os

        config = {
            "route_priority": ["anthropic:plan"],
            "vendors": {
                "anthropic": {
                    "is_cloud": True,
                    "routes": {
                        "plan": {
                            "adapter": "ClaudeMaxAdapter",
                            "auth_token_env": "ANTHROPIC_AUTH_TOKEN",
                            "model": "claude-sonnet-4-6",
                        },
                    },
                },
            },
        }

        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_AUTH_TOKEN"}
        # Also blank ANTHROPIC_API_KEY so the anthropic route can't fall back.
        env.pop("ANTHROPIC_API_KEY", None)
        with patch.dict("os.environ", env, clear=True):
            registry = ProviderRegistry(config)
            with pytest.raises(ProviderInitializationError):
                registry.initialize_providers()


class TestNellieBackendSwitch:
    """Prove Nellie can switch backends and the system stays consistent."""

    @pytest.fixture
    def nellie_service(self):
        return _build_service({
            "anthropic:plan": "claude-sonnet-4-6",
            "openai:plan": "gpt-5.4",
            "openai:api": "gpt-5-mini",
        })

    def test_switch_anthropic_plan_to_openai_plan(self, nellie_service):
        svc = nellie_service

        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")
        providers, _ = svc.resolve_provider_routing()
        assert providers[0]["name"] == "anthropic:plan"

        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")
        providers, target = svc.resolve_provider_routing()
        assert providers[0]["name"] == "openai:plan"
        assert target == "gpt-5.4"

    def test_switch_openai_plan_to_anthropic_plan(self, nellie_service):
        svc = nellie_service

        svc.set_model_preference("gpt-5.4", vendor="openai", route="plan")
        providers, _ = svc.resolve_provider_routing()
        assert providers[0]["name"] == "openai:plan"

        svc.set_model_preference("claude-sonnet-4-6", vendor="anthropic", route="plan")
        providers, target = svc.resolve_provider_routing()
        assert providers[0]["name"] == "anthropic:plan"
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

        cases = [
            ("anthropic", "plan", "claude-sonnet-4-6"),
            ("openai", "plan", "gpt-5.4"),
            ("openai", "api", "gpt-5-mini"),
        ]
        for vendor, route, model in cases:
            svc.set_model_preference(model, vendor=vendor, route=route)

            active = svc.get_active_model_id()
            pref = svc.get_model_preference()
            providers, target = svc.resolve_provider_routing()

            assert active == model
            assert pref["vendor"] == vendor
            assert pref["route"] == route
            assert pref["model"] == model
            assert providers[0]["name"] == f"{vendor}:{route}"
            assert target == model
