"""Contracts for model preservation across privacy-mode transitions."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode


def _prepare_app(agent):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


class _FakeLLMService:
    def __init__(self):
        self.providers = [
            {"name": "openai", "model": "gpt-5-mini"},
            {"name": "ollama", "model": "llama3.2:3b"},
        ]
        self._pref = {"model": None, "provider": None}
        self._pre_ephemeral_preference = None
        self.calls = []

    def get_model_preference(self):
        return dict(self._pref)

    def set_model_preference(self, model, provider=None):
        self.calls.append((model, provider))
        self._pref = {"model": model, "provider": provider}

    def get_active_model_id(self):
        if self._pref.get("model"):
            return self._pref["model"]
        return self.providers[0]["model"]

    def _get_local_provider_names(self):
        return ["ollama"]


class _FakeStorage:
    def __init__(self):
        self.modes = []

    def set_privacy_mode(self, mode):
        self.modes.append(mode)


def _make_agent_with_privacy_transition(llm_service):
    agent = KestrelAgent.__new__(KestrelAgent)
    agent._privacy_mode = PrivacyMode.NORMAL
    agent.storage = _FakeStorage()
    agent.llm_service = llm_service
    agent.features = {}
    agent.privacy_agent = MagicMock()
    agent.privacy_agent.set_mode = MagicMock(return_value="Privacy mode changed from normal to isolated.")
    return agent


def test_privacy_mode_restores_default_cloud_model_after_local_only_transition():
    llm_service = _FakeLLMService()
    privacy_agent = MagicMock()
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    agent = MagicMock(llm_service=llm_service, privacy_agent=privacy_agent)
    agent.set_privacy_mode = AsyncMock()

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                isolated_response = client.post(
                    "/agent/privacy-mode",
                    headers={"X-API-Key": "test-key"},
                    json={"mode": "isolated"},
                )
                normal_response = client.post(
                    "/agent/privacy-mode",
                    headers={"X-API-Key": "test-key"},
                    json={"mode": "normal"},
                )
        assert isolated_response.status_code == 200
        assert isolated_response.json()["model_switched"] == {
            "provider": "ollama",
            "model": "llama3.2:3b",
        }
        assert normal_response.status_code == 200
        assert normal_response.json()["model_switched"] == {
            "provider": "openai",
            "model": "gpt-5-mini",
        }
        assert llm_service.calls == [
            ("llama3.2:3b", "ollama"),
            ("gpt-5-mini", "openai"),
        ]
    finally:
        _restore_app(app, original)


def test_privacy_mode_restores_explicit_cloud_preference_after_local_only_transition():
    llm_service = _FakeLLMService()
    llm_service._pref = {"model": "claude-sonnet-4-5", "provider": "anthropic"}
    privacy_agent = MagicMock()
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    agent = MagicMock(llm_service=llm_service, privacy_agent=privacy_agent)
    agent.set_privacy_mode = AsyncMock()

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                client.post(
                    "/agent/privacy-mode",
                    headers={"X-API-Key": "test-key"},
                    json={"mode": "isolated"},
                )
                response = client.post(
                    "/agent/privacy-mode",
                    headers={"X-API-Key": "test-key"},
                    json={"mode": "normal"},
                )
        assert response.status_code == 200
        assert response.json()["model_switched"] == {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
        }
        assert llm_service.calls[-1] == ("claude-sonnet-4-5", "anthropic")
    finally:
        _restore_app(app, original)


def test_agent_level_privacy_transition_switches_to_local_model():
    llm_service = _FakeLLMService()
    agent = _make_agent_with_privacy_transition(llm_service)

    import asyncio

    result = asyncio.run(agent.set_privacy_mode_with_effects(PrivacyMode.ISOLATED))

    assert result.message == "Privacy mode changed from normal to isolated."
    assert result.model_switched == {
        "provider": "ollama",
        "model": "llama3.2:3b",
    }
    assert llm_service.calls == [("llama3.2:3b", "ollama")]
    assert llm_service._pre_ephemeral_preference == {
        "provider": "openai",
        "model": "gpt-5-mini",
    }


def test_privacy_command_path_uses_agent_level_model_transition():
    llm_service = _FakeLLMService()
    agent = _make_agent_with_privacy_transition(llm_service)
    handler = CommandHandler(agent)

    import asyncio

    result = asyncio.run(handler._cmd_privacy("!privacy isolated"))

    assert result == "Privacy mode changed from normal to isolated."
    assert llm_service.calls == [("llama3.2:3b", "ollama")]
