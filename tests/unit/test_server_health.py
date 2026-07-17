"""Focused tests for server health endpoint behavior."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

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
    app.state.startup_error = "agent init failed"
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

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


def test_health_detailed_uses_health_feature_from_feature_dict():
    """HealthFeature lookup should iterate feature values, not dict keys."""
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
        return_value={"status": "healthy", "checks": [{"status": "ok"}]}
    )
    health_feature.__class__.__name__ = "HealthFeature"

    agent = MagicMock()
    agent.features = {"HealthFeature": health_feature}

    app.state.agent = agent
    app.state.agent_manager = None
    app.state.startup_error = None
    app.state.mandatory_feature_failures = []

    try:
        with TestClient(app) as client:
            response = client.get("/health/detailed")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    health_feature.get_latest.assert_awaited_once()


def test_health_surfaces_llm_reachability():
    """Basic /health should expose startup probe results for operators."""
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
    assert response.json()["llm_reachability"][0]["name"] == "ollama:local"


def test_health_names_mandatory_failure_without_leaking_cause():
    """Readiness diagnostics expose the class/stage, never dependency secrets."""
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
    payload = response.json()
    assert payload["status"] == "unhealthy"
    failure = payload["mandatory_feature_failures"][0]
    assert failure["feature"] == "SecurityFeature"
    assert "SecurityFeature" in failure["error"]
    assert "ANTHROPIC_API_KEY" not in failure["error"]
    assert "must-not-leak" not in failure["error"]


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
    assert response.json()["agent_initialized"] is True
    assert response.json()["mandatory_feature_failures"][0]["agent"] == "broken"


def test_health_names_identity_custody_failure_without_leaking_detail():
    """A blocked root of trust is explicit, stable, and public-safe."""
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
    payload = response.json()
    assert payload["status"] == "unhealthy"
    assert payload["agent_initialized"] is False
    failure = payload["identity_readiness_failures"][0]
    assert failure["state"] == "blocked"
    assert failure["failure"] == "custody"
    assert failure["error_code"] == "identity_custody"
    assert failure["cause_type"] == "DecryptionError"
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
    payload = response.json()
    assert payload["agent_initialized"] is True
    assert payload["identity_readiness_failures"][0]["agent"] == "broken"
    assert payload["identity_readiness_failures"][0]["failure"] == "binding"


def test_health_reports_durable_constitution_safe_mode_as_restricted():
    """An initialized but constitutionally restricted agent is not healthy."""
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
    payload = response.json()
    assert payload["status"] == "restricted"
    assert payload["agent_initialized"] is True
    record = payload["constitution_safe_mode"][0]
    assert record["agent"] == "Kite"
    assert record["error_code"] == "constitution_safe_mode"
    assert record["failure"] == "integrity_restriction"
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

        with TestClient(app) as client:
            response = client.get("/health")
            detailed = client.get("/health/detailed")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures
        app.state.identity_readiness_failures = original_identity_failures

    assert response.status_code == 503
    record = response.json()["constitution_safe_mode"][0]
    assert record["agent"] == "restricted"
    assert record["failure"] == "state_unavailable"
    assert detailed.status_code == 503
    assert detailed.json()["status"] == "restricted"


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
        with TestClient(app) as client:
            response = client.get("/health/detailed")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert response.status_code == 503
    assert response.json()["status"] == "restricted"
