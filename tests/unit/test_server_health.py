"""Focused tests for server health endpoint behavior."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


def test_health_returns_503_when_agent_missing():
    """Missing agent should not report an always-ok health state."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)

    app.router.lifespan_context = noop_lifespan
    app.state.agent = None
    app.state.agent_manager = None
    app.state.startup_error = "agent init failed"

    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


def test_health_detailed_uses_heartbeat_feature_from_feature_dict():
    """HeartbeatFeature lookup should iterate feature values, not dict keys."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)

    app.router.lifespan_context = noop_lifespan

    heartbeat_feature = MagicMock()
    heartbeat_feature.get_latest_heartbeat = AsyncMock(
        return_value={"status": "healthy", "checks": [{"status": "ok"}]}
    )
    heartbeat_feature.__class__.__name__ = "HeartbeatFeature"

    agent = MagicMock()
    agent.features = {"HeartbeatFeature": heartbeat_feature}

    app.state.agent = agent
    app.state.agent_manager = None
    app.state.startup_error = None

    try:
        with TestClient(app) as client:
            response = client.get("/health/detailed")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    heartbeat_feature.get_latest_heartbeat.assert_awaited_once()
