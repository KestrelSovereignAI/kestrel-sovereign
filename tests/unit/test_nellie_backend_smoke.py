"""Nellie backend smoke-proof tests (issue #427).

Concrete proof that an agent named Nellie can be pinned to either
claude_max or codex, and that:
1. The persisted preference matches the active provider/model path
2. resolve_provider_routing() returns ONLY the pinned provider
3. Failure modes are legible when the provider isn't available
4. Backend identity (what the system says) matches runtime reality
   (what resolve_provider_routing actually returns)
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kestrel_sovereign.llm.error_handling import LLMProviderUnavailableError
from kestrel_sovereign.llm.service import LLMService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider_info(name: str, model: str = "auto"):
    """Create a mock ProviderInfo dataclass instance."""
    info = Mock()
    info.name = name
    info.model = model
    info.client = AsyncMock()
    info.adapter = Mock()
    return info


def _build_service(provider_names_and_models: dict) -> LLMService:
    """Build an LLMService with specified providers for Nellie testing.

    Args:
        provider_names_and_models: dict of {provider_name: model_name}

    Returns:
        Configured LLMService with mocked providers
    """
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

    with patch("kestrel_sovereign.llm.service.load_config") as mock_load, \
         patch("kestrel_sovereign.llm.service.ProviderRegistry") as mock_reg_cls:
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


# ===========================================================================
# 1. Nellie pinned to claude_max
# ===========================================================================


class TestNellieClaudeMax:
    """Prove Nellie can be pinned to claude_max and the system tells the truth."""

    @pytest.fixture
    def nellie_service(self):
        return _build_service({
            "anthropic": "claude-sonnet-4-6",
            "claude_max": "claude-sonnet-4-6",
            "openai": "gpt-5-mini",
        })

    def test_pin_nellie_to_claude_max(self, nellie_service):
        """Pin Nellie to claude_max and verify the preference sticks."""
        svc = nellie_service
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")

        pref = svc.get_model_preference()
        assert pref["provider"] == "claude_max"
        assert pref["model"] == "claude-sonnet-4-6"

    def test_routing_uses_only_claude_max(self, nellie_service):
        """After pinning, routing returns ONLY claude_max — not anthropic."""
        svc = nellie_service
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "claude_max"
        assert target == "claude-sonnet-4-6"

    def test_active_model_id_matches_preference(self, nellie_service):
        """get_active_model_id() must agree with the persisted preference."""
        svc = nellie_service
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")

        active = svc.get_active_model_id()
        pref = svc.get_model_preference()

        assert active == pref["model"], (
            f"Backend identity mismatch: get_active_model_id()={active!r} "
            f"but preference says {pref['model']!r}"
        )

    def test_backend_identity_matches_runtime(self, nellie_service):
        """The provider returned by routing must match the stated preference.

        This is the core smoke-proof: what the system *says* (preference)
        matches what the system *does* (routing).
        """
        svc = nellie_service
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")

        pref = svc.get_model_preference()
        providers, target = svc.resolve_provider_routing()

        # Backend identity (preference) == runtime reality (routing)
        assert pref["provider"] == providers[0]["name"]
        assert pref["model"] == target

    def test_claude_max_not_confused_with_anthropic(self, nellie_service):
        """claude_max and anthropic are distinct providers even with same model."""
        svc = nellie_service

        # Pin to claude_max
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")
        providers_max, _ = svc.resolve_provider_routing()

        # Pin to anthropic
        svc.set_model_preference("claude-sonnet-4-6", provider="anthropic")
        providers_api, _ = svc.resolve_provider_routing()

        assert providers_max[0]["name"] == "claude_max"
        assert providers_api[0]["name"] == "anthropic"
        assert providers_max[0]["name"] != providers_api[0]["name"]

    def test_persistence_callback_fires_on_pin(self, nellie_service):
        """When Nellie is pinned, the persistence callback fires."""
        svc = nellie_service
        callback = AsyncMock()
        svc.set_preference_persistence_callback(callback)

        async def _run():
            svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")
            await asyncio.sleep(0.01)

        asyncio.run(_run())
        callback.assert_awaited_once_with("claude-sonnet-4-6", "claude_max")


# ===========================================================================
# 2. Nellie pinned to codex
# ===========================================================================


class TestNellieCodex:
    """Prove Nellie can be pinned to codex and the system tells the truth."""

    @pytest.fixture
    def nellie_service(self):
        return _build_service({
            "openai": "gpt-5-mini",
            "codex": "gpt-5.4",
        })

    def test_pin_nellie_to_codex(self, nellie_service):
        """Pin Nellie to codex and verify the preference sticks."""
        svc = nellie_service
        svc.set_model_preference("gpt-5.4", provider="codex")

        pref = svc.get_model_preference()
        assert pref["provider"] == "codex"
        assert pref["model"] == "gpt-5.4"

    def test_routing_uses_only_codex(self, nellie_service):
        """After pinning, routing returns ONLY codex — not openai."""
        svc = nellie_service
        svc.set_model_preference("gpt-5.4", provider="codex")

        providers, target = svc.resolve_provider_routing()

        assert len(providers) == 1
        assert providers[0]["name"] == "codex"
        assert target == "gpt-5.4"

    def test_active_model_id_matches_preference(self, nellie_service):
        """get_active_model_id() must agree with the persisted preference."""
        svc = nellie_service
        svc.set_model_preference("gpt-5.4", provider="codex")

        active = svc.get_active_model_id()
        pref = svc.get_model_preference()

        assert active == pref["model"]

    def test_backend_identity_matches_runtime(self, nellie_service):
        """Backend identity == runtime reality for codex."""
        svc = nellie_service
        svc.set_model_preference("gpt-5.4", provider="codex")

        pref = svc.get_model_preference()
        providers, target = svc.resolve_provider_routing()

        assert pref["provider"] == providers[0]["name"]
        assert pref["model"] == target

    def test_codex_not_confused_with_openai(self, nellie_service):
        """codex and openai are distinct providers."""
        svc = nellie_service

        svc.set_model_preference("gpt-5.4", provider="codex")
        providers_codex, _ = svc.resolve_provider_routing()

        svc.set_model_preference("gpt-5-mini", provider="openai")
        providers_openai, _ = svc.resolve_provider_routing()

        assert providers_codex[0]["name"] == "codex"
        assert providers_openai[0]["name"] == "openai"


# ===========================================================================
# 3. Failure modes — legible errors when auth/provider missing
# ===========================================================================


class TestNellieFailureModes:
    """Prove that failure modes are legible, not silent."""

    @pytest.fixture
    def nellie_without_codex(self):
        """Service where codex is NOT available (auth missing scenario)."""
        return _build_service({
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-5-mini",
        })

    @pytest.fixture
    def nellie_without_claude_max(self):
        """Service where claude_max is NOT available (token missing scenario)."""
        return _build_service({
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-5-mini",
        })

    def test_codex_missing_raises_with_provider_name(self, nellie_without_codex):
        """When codex isn't initialized, pinning to it raises clearly."""
        svc = nellie_without_codex
        svc.set_model_preference("gpt-5.4", provider="codex")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        error_msg = str(exc_info.value)
        assert "codex" in error_msg
        # Error should show what IS available
        assert "anthropic" in error_msg or "openai" in error_msg

    def test_claude_max_missing_raises_with_provider_name(self, nellie_without_claude_max):
        """When claude_max isn't initialized, pinning to it raises clearly."""
        svc = nellie_without_claude_max
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        error_msg = str(exc_info.value)
        assert "claude_max" in error_msg

    def test_error_message_lists_available_providers(self, nellie_without_codex):
        """The error message should list all available providers."""
        svc = nellie_without_codex
        svc.set_model_preference("codex", provider="codex")

        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            svc.resolve_provider_routing()

        error_msg = str(exc_info.value)
        # Both available providers should be mentioned
        assert "anthropic" in error_msg
        assert "openai" in error_msg

    def test_codex_adapter_requires_client(self):
        """CodexAdapter.get_response raises RuntimeError without a client."""
        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        adapter = CodexAdapter()

        with pytest.raises(RuntimeError, match="requires an OAuth token"):
            asyncio.run(adapter.get_response(
                client=None, model="gpt-5.4", messages=[{"role": "user", "content": "hi"}]
            ))

    def test_claude_max_registry_requires_auth_token(self):
        """Provider registry raises ValueError when ANTHROPIC_AUTH_TOKEN is missing."""
        from kestrel_sovereign.llm.provider_registry import ProviderRegistry

        registry = ProviderRegistry({
            "provider_priority": ["claude_max"],
            "claude_max": {"model": "claude-sonnet-4-6"},
        })

        with patch.dict("os.environ", {}, clear=False):
            # Remove the token if it exists
            env = {k: v for k, v in __import__("os").environ.items()
                   if k != "ANTHROPIC_AUTH_TOKEN"}
            with patch.dict("os.environ", env, clear=True):
                with pytest.raises(ValueError, match="ANTHROPIC_AUTH_TOKEN"):
                    registry._initialize_single_provider("claude_max")


# ===========================================================================
# 4. Backend switch — Nellie can be re-pinned at runtime
# ===========================================================================


class TestNellieBackendSwitch:
    """Prove Nellie can switch backends and the system stays consistent."""

    @pytest.fixture
    def nellie_service(self):
        return _build_service({
            "claude_max": "claude-sonnet-4-6",
            "codex": "gpt-5.4",
            "openai": "gpt-5-mini",
        })

    def test_switch_claude_max_to_codex(self, nellie_service):
        """Nellie can switch from claude_max to codex and routing follows."""
        svc = nellie_service

        # Start on claude_max
        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")
        providers, target = svc.resolve_provider_routing()
        assert providers[0]["name"] == "claude_max"

        # Switch to codex
        svc.set_model_preference("gpt-5.4", provider="codex")
        providers, target = svc.resolve_provider_routing()
        assert providers[0]["name"] == "codex"
        assert target == "gpt-5.4"

    def test_switch_codex_to_claude_max(self, nellie_service):
        """Nellie can switch from codex to claude_max and routing follows."""
        svc = nellie_service

        svc.set_model_preference("gpt-5.4", provider="codex")
        providers, _ = svc.resolve_provider_routing()
        assert providers[0]["name"] == "codex"

        svc.set_model_preference("claude-sonnet-4-6", provider="claude_max")
        providers, target = svc.resolve_provider_routing()
        assert providers[0]["name"] == "claude_max"
        assert target == "claude-sonnet-4-6"

    def test_clear_preference_returns_to_all_providers(self, nellie_service):
        """After clearing preference, all providers are available again."""
        svc = nellie_service

        svc.set_model_preference("gpt-5.4", provider="codex")
        providers, _ = svc.resolve_provider_routing()
        assert len(providers) == 1

        svc.clear_model_preference()
        providers, target = svc.resolve_provider_routing()
        assert len(providers) == 3
        assert target is None

    def test_identity_consistent_through_switch(self, nellie_service):
        """get_active_model_id and routing agree through backend switches."""
        svc = nellie_service

        for provider, model in [
            ("claude_max", "claude-sonnet-4-6"),
            ("codex", "gpt-5.4"),
            ("openai", "gpt-5-mini"),
        ]:
            svc.set_model_preference(model, provider=provider)

            active = svc.get_active_model_id()
            pref = svc.get_model_preference()
            providers, target = svc.resolve_provider_routing()

            assert active == model, f"active_model_id mismatch for {provider}"
            assert pref["provider"] == provider
            assert pref["model"] == model
            assert providers[0]["name"] == provider
            assert target == model
