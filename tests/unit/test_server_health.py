"""Focused tests for server health endpoint behavior."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import Request
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
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )

    app.router.lifespan_context = noop_lifespan
    app.state.agent = None
    app.state.agent_manager = None
    app.state.startup_error = "agent init failed at /private/must-not-leak"
    app.state.mandatory_feature_failures = []

    try:
        with TestClient(app, client=("203.0.113.10", 55000)) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": False,
    }
    assert "must-not-leak" not in response.text


def test_load_balancer_probe_reports_minimal_degraded_state():
    """A remote probe needs readiness, not deployment diagnostics."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )
    original_identity_failures = getattr(
        app.state, "identity_readiness_failures", None
    )

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = None
        app.state.startup_error = None
        app.state.mandatory_feature_failures = []
        app.state.identity_readiness_failures = []
        with TestClient(app, client=("198.51.100.20", 55000)) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures
        app.state.identity_readiness_failures = original_identity_failures

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "agent_initialized": False,
    }


def test_health_detailed_requires_auth_and_uses_feature_dict_with_api_key():
    """Remote callers get no details; API-key operators get the full result."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )

    app.router.lifespan_context = noop_lifespan

    health_feature = MagicMock()
    health_feature.get_latest = AsyncMock(
        return_value={
            "status": "healthy",
            "checks": [
                {
                    "status": "pass",
                    "message": "active: anthropic/claude-opus-secret-route",
                }
            ],
        }
    )
    health_feature.__class__.__name__ = "HealthFeature"

    agent = MagicMock()
    agent.features = {"HealthFeature": health_feature}

    app.state.agent = agent
    app.state.agent_manager = None
    app.state.startup_error = None
    app.state.mandatory_feature_failures = []

    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                unauthenticated = client.get("/health/detailed")
                query_key_attempt = client.get(
                    "/health/detailed",
                    params={"api_key": "test-key"},
                )
                response = client.get(
                    "/health/detailed",
                    headers={"X-API-Key": "test-key"},
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures

    assert unauthenticated.status_code == 401
    assert "claude-opus-secret-route" not in unauthenticated.text
    assert query_key_attempt.status_code == 401
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "claude-opus-secret-route" in response.text
    health_feature.get_latest.assert_awaited_once()


def test_multi_agent_prefixed_detailed_health_keeps_auth_then_routes():
    """Agent-prefix rewriting must happen only after operator auth succeeds."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)

    health_feature = MagicMock()
    health_feature.get_latest = AsyncMock(
        return_value={"status": "healthy", "checks": [{"name": "database"}]}
    )
    health_feature.__class__.__name__ = "HealthFeature"
    agent = MagicMock()
    agent.features = {"HealthFeature": health_feature}
    manager = MagicMock()
    manager.get_agent.side_effect = lambda name: agent if name == "Kite" else None
    manager.list_agents.return_value = {"Kite": agent}

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = manager
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                unauthenticated = client.get(
                    "/api/agents/Kite/health/detailed"
                )
                authenticated = client.get(
                    "/api/agents/Kite/health/detailed",
                    headers={"X-API-Key": "test-key"},
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert authenticated.json()["checks"] == [{"name": "database"}]
    health_feature.get_latest.assert_awaited_once()


def test_public_health_does_not_surface_llm_reachability():
    """The public probe stays stable even when detailed routing data exists."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )

    app.router.lifespan_context = noop_lifespan

    agent = MagicMock()
    agent.llm_service.reachability = [
        {
            "name": "ollama:local",
            "status": "unreachable",
            "is_local": True,
        }
    ]

    app.state.agent = agent
    app.state.agent_manager = None
    app.state.startup_error = None
    app.state.mandatory_feature_failures = []

    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "agent_initialized": True}
    assert "ollama" not in response.text


def test_public_health_hides_mandatory_failure_diagnostics():
    """Mandatory failures affect readiness without becoming public records."""
    from server import app
    from kestrel_sovereign.features import MandatoryFeatureReadinessError

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )

    try:
        try:
            raise RuntimeError("ANTHROPIC_API_KEY=must-not-leak")
        except RuntimeError as cause:
            error = MandatoryFeatureReadinessError(
                "SecurityFeature",
                "initialization",
                "could not initialize",
            )
            error.__cause__ = cause

        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = None
        app.state.startup_error = str(error)
        app.state.mandatory_feature_failures = [
            {
                "agent": "default",
                "feature": error.feature_name,
                "stage": error.stage,
                "error": str(error),
            }
        ]

        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": False,
    }
    assert "SecurityFeature" not in response.text
    assert "ANTHROPIC_API_KEY" not in response.text
    assert "must-not-leak" not in response.text


def test_health_rejects_partially_loaded_fleet_with_mandatory_failure():
    """One good agent must not hide another configured agent's security gap."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )

    manager = MagicMock()
    manager.list_agents.return_value = {"secure": MagicMock()}
    app.router.lifespan_context = noop_lifespan
    app.state.agent = None
    app.state.agent_manager = manager
    app.state.startup_error = None
    app.state.mandatory_feature_failures = [
        {
            "agent": "broken",
            "feature": "SecurityFeature",
            "stage": "initialization",
            "error": "Mandatory feature 'SecurityFeature' could not initialize",
        }
    ]

    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": True,
    }
    assert "broken" not in response.text


def test_public_health_hides_identity_custody_diagnostics():
    """A blocked root of trust affects readiness without public detail."""
    from server import app
    from kestrel_sovereign.identity.runtime_identity import (
        IdentityReadinessError,
    )
    from kestrel_sovereign.server import _identity_readiness_failure_record

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )
    original_identity_failures = getattr(
        app.state, "identity_readiness_failures", None
    )

    try:
        error = IdentityReadinessError(
            "custody",
            cause_type="DecryptionError",
        )
        error.__cause__ = RuntimeError(
            "/private/agent KESTREL_DATA_KEY=must-not-leak"
        )
        record = _identity_readiness_failure_record("default", error)

        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = None
        app.state.startup_error = str(error)
        app.state.mandatory_feature_failures = []
        app.state.identity_readiness_failures = [record]

        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures
        app.state.identity_readiness_failures = original_identity_failures

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": False,
    }
    assert "DecryptionError" not in response.text
    assert "identity_custody" not in response.text
    assert "/private/agent" not in response.text
    assert "must-not-leak" not in response.text


def test_health_rejects_partially_loaded_fleet_with_identity_failure():
    """Healthy peers stay available but cannot hide a broken identity."""
    from server import app
    from kestrel_sovereign.identity.runtime_identity import (
        IdentityReadinessError,
    )
    from kestrel_sovereign.server import _identity_readiness_failure_record

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )
    original_identity_failures = getattr(
        app.state, "identity_readiness_failures", None
    )

    manager = MagicMock()
    manager.list_agents.return_value = {"healthy": MagicMock()}
    error = IdentityReadinessError("binding", cause_type="ConfiguredDIDMismatch")
    app.router.lifespan_context = noop_lifespan
    app.state.agent = None
    app.state.agent_manager = manager
    app.state.startup_error = None
    app.state.mandatory_feature_failures = []
    app.state.identity_readiness_failures = [
        _identity_readiness_failure_record("broken", error)
    ]

    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures
        app.state.identity_readiness_failures = original_identity_failures

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": True,
    }
    assert "broken" not in response.text
    assert "binding" not in response.text


def test_public_health_hides_safe_mode_agent_state_and_reason():
    """Safe mode blocks readiness without exposing its cause or agent."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )
    original_identity_failures = getattr(
        app.state, "identity_readiness_failures", None
    )

    agent = MagicMock()
    agent._safe_mode = True
    agent._safe_mode_reason = "secret path /private/constitution must-not-leak"
    agent._safe_mode_entered_at = datetime(
        2026, 7, 17, 12, 0, tzinfo=timezone.utc
    )
    agent._constitution_state_load_error = None
    agent._agent_name = "Kite"

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = agent
        app.state.agent_manager = None
        app.state.startup_error = None
        app.state.mandatory_feature_failures = []
        app.state.identity_readiness_failures = []

        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures
        app.state.identity_readiness_failures = original_identity_failures

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": True,
    }
    assert "Kite" not in response.text
    assert "restricted" not in response.text
    assert "must-not-leak" not in response.text
    assert "/private/constitution" not in response.text


def test_health_rejects_fleet_when_one_loaded_agent_is_in_safe_mode():
    """A normal peer cannot hide another loaded agent's restriction."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )
    original_identity_failures = getattr(
        app.state, "identity_readiness_failures", None
    )

    healthy = MagicMock()
    healthy._safe_mode = False
    restricted = MagicMock()
    restricted._safe_mode = True
    restricted._safe_mode_entered_at = None
    restricted._constitution_state_load_error = "DatabaseError"
    manager = MagicMock()
    manager.list_agents.return_value = {
        "healthy": healthy,
        "restricted": restricted,
    }

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = manager
        app.state.startup_error = None
        app.state.mandatory_feature_failures = []
        app.state.identity_readiness_failures = []

        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                response = client.get("/health")
                unauthenticated_detailed = client.get("/health/detailed")
                detailed = client.get(
                    "/health/detailed",
                    headers={"Authorization": "Bearer test-key"},
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures
        app.state.identity_readiness_failures = original_identity_failures

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": True,
    }
    assert "state_unavailable" not in response.text
    assert unauthenticated_detailed.status_code == 401
    assert detailed.status_code == 503
    assert detailed.json()["status"] == "restricted"
    assert detailed.json()["constitution_safe_mode"][0]["agent"] == "restricted"


def test_health_reports_startup_audit_pending_as_restricted():
    from kestrel_sovereign.server import _constitution_safe_mode_record

    agent = MagicMock()
    agent._safe_mode = False
    agent._constitution_audit_pending = True
    agent._safe_mode_entered_at = None

    record = _constitution_safe_mode_record("Kite", agent)

    assert record["state"] == "audit_pending"
    assert record["error_code"] == "constitution_audit_pending"
    assert record["failure"] == "startup_audit_required"


def test_detailed_health_cannot_report_healthy_during_safe_mode():
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)

    agent = MagicMock()
    agent._safe_mode = True
    agent._safe_mode_entered_at = None
    agent._constitution_state_load_error = None
    agent._agent_name = "Kite"

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = agent
        app.state.agent_manager = None
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    "/health/detailed",
                    headers={"X-API-Key": "test-key"},
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert response.status_code == 503
    assert response.json()["status"] == "restricted"


def test_oauth_session_can_access_detailed_health():
    """A signed browser session retains operator diagnostic access."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)

    health_feature = MagicMock()
    health_feature.get_latest = AsyncMock(
        return_value={
            "status": "degraded",
            "checks": [{"message": "operator-only backend diagnostic"}],
        }
    )
    health_feature.__class__.__name__ = "HealthFeature"
    agent = MagicMock()
    agent.features = {"HealthFeature": health_feature}

    @app.get("/_test/health-session")
    async def _establish_health_session(request: Request):
        request.session["user_email"] = "operator@example.com"
        return {"ok": True}

    session_route = app.router.routes[-1]
    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = agent
        app.state.agent_manager = None
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                established = client.get(
                    "/_test/health-session",
                    headers={"X-API-Key": "test-key"},
                )
                response = client.get("/health/detailed")
    finally:
        app.router.routes.remove(session_route)
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert established.status_code == 200
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert "operator-only backend diagnostic" in response.text
