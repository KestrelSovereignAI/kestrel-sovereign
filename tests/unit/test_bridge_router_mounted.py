"""Regression test for F105: the bridge HTTP router must be mounted.

``BridgeFeature.get_router()`` returns the real ``/api/bridge/*`` router so
that the agent's ``_mount_feature_routers`` pass wires it into the app. Before
the fix nothing mounted the router built by
``kestrel_sovereign.features.bridge.router.get_router()``, so every bridge
endpoint 404'd. These tests boot the real app via ``TestClient`` and assert:

* with the bridge feature enabled, ``GET /api/bridge/health`` returns 200;
* with the feature absent, the route is not mounted (404).
"""

import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign.features.bridge.feature import BridgeFeature


API_KEY = "test-bridge-key"
pytestmark = pytest.mark.usefixtures("isolated_process_rate_limiter")


def _make_bridge_feature():
    """A real BridgeFeature whose status method is stubbed (no DB needed)."""
    agent = MagicMock()
    agent.did = "did:test:bridge"
    bridge = BridgeFeature(agent=agent)
    bridge.bridge_status = AsyncMock(
        return_value=SimpleNamespace(
            data={
                "uptime_seconds": 1,
                "active_sessions_memory": 0,
                "database_available": False,
            }
        )
    )
    return bridge


def _boot(features):
    """Boot the real app with a mock single agent exposing ``features``.

    Returns the app plus a restore callable to unwind the patched state.
    """
    from server import app
    from kestrel_sovereign.server import (
        _mount_feature_routers,
        _unmount_feature_routers,
    )

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)

    agent = MagicMock()
    agent.features = features

    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    _mount_feature_routers(app)

    def restore():
        _unmount_feature_routers(app)
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    return app, restore


def test_bridge_health_mounted_when_feature_enabled():
    os.environ["KESTREL_API_KEY"] = API_KEY
    bridge = _make_bridge_feature()
    app, restore = _boot({"BridgeFeature": bridge})
    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/bridge/health", headers={"X-API-Key": API_KEY}
            )
    finally:
        restore()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["bridge"] is True


def test_bridge_health_absent_when_feature_disabled():
    os.environ["KESTREL_API_KEY"] = API_KEY
    app, restore = _boot({})
    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/bridge/health", headers={"X-API-Key": API_KEY}
            )
    finally:
        restore()

    assert response.status_code == 404
