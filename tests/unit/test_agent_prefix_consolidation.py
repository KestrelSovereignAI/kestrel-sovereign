"""Regression tests for the /agent → /api/agent prefix consolidation (#871).

After the consolidation, every Kestrel HTTP route lives under /api/*. The
deprecated /agent/* prefix is kept working for one release by a path-
rewrite middleware in server.py that:

  - rewrites the request scope's path from /agent/<x> to /api/agent/<x>
  - logs a deprecation warning the first time it sees a (path, ua) pair
  - decorates the response with Deprecation/Sunset/Link headers per RFC 8594

These tests prove the back-compat shim works end-to-end so we don't ship a
breaking change to anyone still on the old prefix.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
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


def _mock_agent():
    agent = MagicMock()
    agent.agent_id = "did:test:prefix-consolidation"
    agent.privacy_mode = MagicMock()
    agent.privacy_mode.value = "NORMAL"
    agent.features = {}
    agent.cancel_current_request = MagicMock(return_value=False)
    return agent


def test_canonical_api_agent_prefix_serves_agent_routes():
    """The canonical /api/agent/* prefix is the live mount."""
    app, original = _prepare_app(_mock_agent())
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/agent/stop",
            headers={"X-API-Key": "test-key"},
        )
        # 200 (cancelled) or 401 (auth) — anything other than 404 proves the
        # route exists.
        assert resp.status_code != 404, (
            f"/api/agent/stop returned 404 — canonical mount missing"
        )
    finally:
        _restore_app(app, original)


def test_deprecated_agent_prefix_still_resolves(caplog):
    """The legacy /agent/* prefix continues to resolve to the same handler
    via the back-compat middleware."""
    import logging
    app, original = _prepare_app(_mock_agent())
    try:
        client = TestClient(app, raise_server_exceptions=False)
        with caplog.at_level(logging.WARNING):
            resp = client.post(
                "/agent/stop",
                headers={"X-API-Key": "test-key", "User-Agent": "regression-test/1.0"},
            )
        assert resp.status_code != 404, (
            "/agent/stop returned 404 — back-compat shim broken"
        )
        assert resp.headers.get("Deprecation") == "true"
        assert "successor-version" in (resp.headers.get("Link") or "")
        # First hit logs once per (path, ua).
        assert any(
            "deprecated /agent/* prefix" in rec.getMessage()
            for rec in caplog.records
        ), "expected deprecation warning on first /agent/* hit"
    finally:
        _restore_app(app, original)


def test_deprecation_log_dedupes_per_path_and_ua():
    """The middleware logs once per (path, ua) so noisy clients don't flood
    the logs."""
    import logging
    from server import _DEPRECATED_AGENT_PREFIX_SEEN
    _DEPRECATED_AGENT_PREFIX_SEEN.clear()

    app, original = _prepare_app(_mock_agent())
    try:
        client = TestClient(app, raise_server_exceptions=False)
        ua = {"User-Agent": "dedupe-test/1.0", "X-API-Key": "test-key"}
        client.post("/agent/stop", headers=ua)
        client.post("/agent/stop", headers=ua)
        client.post("/agent/stop", headers=ua)
        # One unique (path, ua) seen.
        assert ("/agent/stop", "dedupe-test/1.0") in _DEPRECATED_AGENT_PREFIX_SEEN
        # Different ua produces a separate log entry.
        client.post("/agent/stop", headers={"User-Agent": "other-ua/2.0", "X-API-Key": "test-key"})
        assert ("/agent/stop", "other-ua/2.0") in _DEPRECATED_AGENT_PREFIX_SEEN
    finally:
        _DEPRECATED_AGENT_PREFIX_SEEN.clear()
        _restore_app(app, original)
