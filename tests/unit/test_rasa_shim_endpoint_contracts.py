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
                    headers={**_api_headers(), "X-Request-ID": "rasa-retry-2765"},
                    json={"sender": "patient-123", "message": "BP was 140/90"},
                )

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "rasa-retry-2765"
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
        assert kwargs["invocation_id"] == "rasa-retry-2765"
        assert (
            kwargs["invocation_provenance"].source_locator
            == "POST:/webhooks/rest/webhook"
        )
        assert kwargs["invocation_provenance"].actor == "rasa_webhook"
        assert "Patient says: BP was 140/90" in kwargs["user_input"]
        assert "model_override" not in kwargs
    finally:
        _restore_app(app, original)


def test_rasa_webhook_reports_cooperative_stop_as_conflict():
    from kestrel_sovereign.agent.invocation import InvocationCancelledError

    agent = MagicMock()
    agent.process_input = AsyncMock(
        side_effect=InvocationCancelledError("isolated turn stopped")
    )
    app, original = _prepare_app(agent)
    try:
        with patch.dict(
            "os.environ",
            {
                "KESTREL_API_KEY": "test-key",
                "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token",
            },
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/webhooks/rest/webhook",
                    headers={
                        **_api_headers(),
                        "X-Request-ID": "rasa-stopped-turn",
                    },
                    json={"sender": "patient-123", "message": "stop this"},
                )

        assert response.status_code == 409
        assert response.json()["detail"] == "Request stopped during execution."
        assert response.headers["X-Request-ID"] == "rasa-stopped-turn"
    finally:
        _restore_app(app, original)


def _prepare_multi_agent_app(agents, default=None):
    """Boot the real app in multi-agent mode with ``{name: agent}``.

    ``default`` is the host-default agent (``app.state.agent``), ``None`` for
    a host with no default. The real routing middleware resolves
    ``/api/agents/{name}/...`` through the fake manager, exactly as the
    deployed host does.
    """
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    manager = MagicMock()
    manager.list_agents = MagicMock(side_effect=lambda: dict(agents))
    manager.get_agent = MagicMock(side_effect=lambda name: agents.get(name))
    app.router.lifespan_context = noop_lifespan
    app.state.agent = default
    app.state.agent_manager = manager
    return app, original


def _rasa_agent(reply):
    agent = MagicMock()
    agent.process_input = AsyncMock(return_value=reply)
    return agent


def test_prefixed_rasa_alias_invokes_only_the_routed_agent():
    """#3220: ``/api/agents/B/webhooks/rest/webhook`` runs on B, never on A.

    The handler read ``app.state.agent`` (the host default) and ignored the
    agent the routing middleware pinned, so a message explicitly addressed
    to B executed on A. Two agents, no default: each prefixed request must
    reach only its own agent, with the routed agent's reply and a session
    keyed by the sender.
    """
    a = _rasa_agent("reply from a")
    b = _rasa_agent("reply from b")
    app, original = _prepare_multi_agent_app({"a": a, "b": b})
    try:
        with patch.dict(
            "os.environ",
            {"KESTREL_API_KEY": "test-key", "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token"},
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agents/b/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-1", "message": "hello b"},
                )
                assert response.status_code == 200, response.text
                assert response.json() == [
                    {"recipient_id": "patient-1", "text": "reply from b"}
                ]
                b.process_input.assert_awaited_once()
                a.process_input.assert_not_awaited()
                _, kwargs = b.process_input.await_args
                assert kwargs["session_id"] == "sms:patient-1"
                assert "hello b" in kwargs["user_input"]

                response = client.post(
                    "/api/agents/a/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-2", "message": "hello a"},
                )
                assert response.status_code == 200, response.text
                assert response.json() == [
                    {"recipient_id": "patient-2", "text": "reply from a"}
                ]
                a.process_input.assert_awaited_once()
                b.process_input.assert_awaited_once()
    finally:
        _restore_app(app, original)


def test_prefixed_rasa_alias_beats_the_host_default_agent():
    """#3220, the reported shape: A is the host default and the request names
    B. The old handler ran A. The routed agent must win over the default;
    the unprefixed form on the same host still reaches the default.
    """
    a = _rasa_agent("reply from a")
    b = _rasa_agent("reply from b")
    app, original = _prepare_multi_agent_app({"a": a, "b": b}, default=a)
    try:
        with patch.dict(
            "os.environ",
            {"KESTREL_API_KEY": "test-key", "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token"},
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agents/b/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-5", "message": "for b"},
                )
                assert response.status_code == 200, response.text
                assert response.json()[0]["text"] == "reply from b"
                b.process_input.assert_awaited_once()
                a.process_input.assert_not_awaited()

                response = client.post(
                    "/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-6", "message": "for the default"},
                )
                assert response.status_code == 200, response.text
                assert response.json()[0]["text"] == "reply from a"
                a.process_input.assert_awaited_once()
                b.process_input.assert_awaited_once()
    finally:
        _restore_app(app, original)


def test_unprefixed_rasa_webhook_on_a_multi_agent_host_without_a_default_refuses():
    """#3220 regression guard: with no host-default agent the unprefixed form
    has no target. It must 503 without invoking anyone — never pick an agent
    from the fleet.
    """
    a = _rasa_agent("reply from a")
    b = _rasa_agent("reply from b")
    app, original = _prepare_multi_agent_app({"a": a, "b": b})
    try:
        with patch.dict(
            "os.environ",
            {"KESTREL_API_KEY": "test-key", "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token"},
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-3", "message": "hello?"},
                )
        assert response.status_code == 503, response.text
        a.process_input.assert_not_awaited()
        b.process_input.assert_not_awaited()
    finally:
        _restore_app(app, original)


def test_prefixed_rasa_alias_still_requires_the_webhook_token():
    """Routing to a named agent does not relax the shim's own auth (#1729):
    the token check runs before any agent is resolved, so an unauthenticated
    prefixed request invokes nobody.
    """
    b = _rasa_agent("reply from b")
    app, original = _prepare_multi_agent_app({"b": b})
    try:
        with patch.dict(
            "os.environ",
            {"KESTREL_API_KEY": "test-key", "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token"},
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agents/b/webhooks/rest/webhook",
                    headers={"X-API-Key": "test-key"},
                    json={"sender": "patient-4", "message": "no token"},
                )
        assert response.status_code == 401, response.text
        b.process_input.assert_not_awaited()
    finally:
        _restore_app(app, original)
