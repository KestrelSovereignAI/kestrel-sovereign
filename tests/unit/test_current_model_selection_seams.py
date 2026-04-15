"""Current-model selection seam contracts across API, feature, and agent paths."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin
from kestrel_sovereign.features.model.feature import ModelAgent
from kestrel_sovereign.llm.service import LLMService


def _make_llm_service(preference=None):
    svc = LLMService.__new__(LLMService)
    svc.providers = [
        {"name": "openai", "model": "gpt-5-mini"},
        {"name": "anthropic", "model": "claude-sonnet-4-6"},
    ]
    svc._mandate_preference = preference or {"model": None, "provider": None}
    return svc


class _Agent(ModelPreferenceMixin):
    def __init__(self, llm_service):
        self.llm_service = llm_service


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


def test_llm_service_exposes_canonical_current_model_selection_for_provider_preference():
    svc = _make_llm_service({"model": "claude-sonnet-4-6", "provider": "anthropic"})

    selection = svc.get_active_model_selection()

    assert selection == {
        "model": "anthropic/claude-sonnet-4-6",
        "provider": "anthropic",
        "model_name": "claude-sonnet-4-6",
    }


def test_current_model_paths_agree_for_model_only_preference():
    svc = _make_llm_service({"model": "gpt-5-mini", "provider": None})
    model_agent = ModelAgent(MagicMock(llm_service=svc))
    model_agent.llm_service = svc
    agent = _Agent(svc)

    feature_result = asyncio.run(model_agent.get_current_model())
    mixin_result = agent.get_current_model()
    api_agent = MagicMock(llm_service=svc)
    app, original = _prepare_app(api_agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    "/api/model/current",
                    headers={"X-API-Key": "test-key"},
                )
    finally:
        _restore_app(app, original)

    assert response.status_code == 200
    assert feature_result["current_model"] == "gpt-5-mini"
    assert mixin_result == "gpt-5-mini"
    assert response.json() == {
        "model": "gpt-5-mini",
        "provider": None,
        "model_name": "gpt-5-mini",
    }
