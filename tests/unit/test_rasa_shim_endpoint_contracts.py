"""Contract tests for the Rasa webhook shim."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


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


def _api_headers():
    # The Rasa webhook now self-authenticates with a dedicated token (#1729),
    # since /webhooks/* is exempt from the host API-key middleware.
    return {"X-API-Key": "test-key", "X-Webhook-Token": "rasa-token"}


def test_rasa_webhook_does_not_force_hardcoded_model_override():
    agent = MagicMock()
    agent.process_input = AsyncMock(return_value="Take your blood pressure again in 10 minutes.")

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key", "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token"}):
            with TestClient(app) as client:
                response = client.post(
                    "/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-123", "message": "BP was 140/90"},
                )

        assert response.status_code == 200
        assert response.json() == [
            {
                "recipient_id": "patient-123",
                "text": "Take your blood pressure again in 10 minutes.",
            }
        ]

        agent.process_input.assert_awaited_once()
        _, kwargs = agent.process_input.await_args
        assert kwargs["session_id"] == "sms:patient-123"
        assert kwargs["include_memories"] is False
        assert "Patient says: BP was 140/90" in kwargs["user_input"]
        assert "model_override" not in kwargs
    finally:
        _restore_app(app, original)
