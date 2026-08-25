"""Focused tests for server health endpoint behavior."""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient


_MISSING_STATE = object()
_READINESS_STATE_BASELINE = {
    "startup_error": None,
    "mandatory_feature_failures": [],
    "identity_readiness_failures": [],
    "scheduler_cold_agent_failures": [],
    "scheduler_readiness_failures": [],
    "host_scheduler_runner": None,
    # HOST state, and these tests control only the agent. `server.app`'s real
    # lifespan starts whatever host features are installed, and one that fails
    # to start is recorded on the host context and correctly downgrades
    # /health/detailed to `degraded` (#3058) — so an environment with a
    # feature package installed made unrelated tests here see a degraded host.
    # That is the endpoint doing its job; the downgrade is asserted in
    # tests/unit/test_host_feature_contribution_runtime.py, not by leaking
    # into every test in this file.
    "host_context": None,
}


@pytest.fixture(autouse=True)
def _isolate_shared_health_readiness_state():
    """Give every test a healthy app-state baseline and restore its caller.

    ``server.app`` is a module singleton, so a readiness latch set by any
    earlier unit test can short-circuit an unrelated detailed-health test with
    a 503. Tests remain free to assert latching inside their own body; this
    fixture only establishes the pre-test baseline and restores the exact
    missing-versus-present state afterwards.
    """

    from server import app

    saved = {
        name: getattr(app.state, "_state", {}).get(name, _MISSING_STATE)
        for name in _READINESS_STATE_BASELINE
    }
    for name, value in _READINESS_STATE_BASELINE.items():
        setattr(app.state, name, list(value) if isinstance(value, list) else value)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is _MISSING_STATE:
                delattr(app.state, name)
            else:
                setattr(app.state, name, value)


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


def test_health_startup_error_dominates_retained_cleanup_manager():
    """A manager retained solely for rollback cleanup can never pass readiness."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_cleanup_manager = getattr(
        app.state, "startup_cleanup_agent_manager", None
    )
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )
    original_identity_failures = getattr(
        app.state, "identity_readiness_failures", None
    )

    retained = MagicMock()
    retained.list_agents.return_value = {"draining": MagicMock()}
    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        # Exercise the old failure mode too: even if an earlier rollback left
        # the manager on the public field, startup_error is authoritative.
        app.state.agent_manager = retained
        app.state.startup_cleanup_agent_manager = retained
        app.state.startup_error = "scheduler startup failed"
        app.state.mandatory_feature_failures = []
        app.state.identity_readiness_failures = []

        with TestClient(app, client=("203.0.113.10", 55000)) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_cleanup_agent_manager = original_cleanup_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures
        app.state.identity_readiness_failures = original_identity_failures

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": False,
    }


def test_health_latches_loaded_scheduler_runner_safety_failure():
    """A scoped runner outage cannot be masked by a healthy host manager."""
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
    original_scheduler_failures = getattr(
        app.state, "scheduler_readiness_failures", None
    )
    failed_runner = SimpleNamespace(
        readiness_failure=RuntimeError("postgres://operator:secret@db/internal")
    )
    failed_agent = SimpleNamespace(
        features={"SchedulerFeature": SimpleNamespace(_runner=failed_runner)}
    )
    manager = MagicMock()
    manager.list_agents.return_value = {"cold-tenant": failed_agent}

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = manager
        app.state.startup_error = None
        app.state.mandatory_feature_failures = []
        app.state.identity_readiness_failures = []
        app.state.scheduler_readiness_failures = []

        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                public = client.get("/health")
                detailed = client.get(
                    "/health/detailed", headers={"X-API-Key": "test-key"}
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures
        app.state.identity_readiness_failures = original_identity_failures
        app.state.scheduler_readiness_failures = original_scheduler_failures

    assert public.status_code == 503
    assert public.json() == {"status": "unhealthy", "agent_initialized": True}
    assert "secret" not in public.text
    assert detailed.status_code == 503
    records = detailed.json()["scheduler_readiness_failures"]
    assert records == [
        {
            "scope": "runtime",
            "state": "unavailable",
            "error_code": "scheduler_runtime_unavailable",
            "cause_type": "RuntimeError",
            "agent": "cold-tenant",
        }
    ]
    assert "secret" not in detailed.text


def test_health_fails_while_scheduler_supervisor_has_no_worker():
    """A dead polling task cannot be hidden by otherwise healthy process state."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    runner = SimpleNamespace(
        readiness_failure=None,
        _running=True,
        worker_available=False,
    )
    agent = SimpleNamespace(
        features={"SchedulerFeature": SimpleNamespace(_runner=runner)}
    )

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = agent
        app.state.agent_manager = None
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": True,
    }


def test_health_fails_for_enabled_standalone_scheduler_without_runner():
    """No-database scheduler mode cannot make an unpolled host look ready."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    agent = SimpleNamespace(
        features={
            "SchedulerFeature": SimpleNamespace(
                enabled=True,
                _runner=None,
                _polling_managed_by_host=False,
            )
        }
    )

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = agent
        app.state.agent_manager = None
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": True,
    }


def test_detailed_health_fails_for_enabled_scheduler_without_live_worker():
    """Cached subsystem health cannot mask a missing scheduler worker."""
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
            "status": "healthy",
            "overall_healthy": True,
            "checks": [
                {
                    "name": "database",
                    "status": "pass",
                    "message": "Database healthy",
                }
            ],
        }
    )
    health_feature.__class__.__name__ = "HealthFeature"
    agent = SimpleNamespace(
        features={
            "HealthFeature": health_feature,
            "SchedulerFeature": SimpleNamespace(
                enabled=True,
                _runner=None,
                _polling_managed_by_host=False,
            ),
        }
    )

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
                health_feature.get_latest.side_effect = RuntimeError(
                    "synthetic health failure"
                )
                failed_health_response = client.get(
                    "/health/detailed",
                    headers={"X-API-Key": "test-key"},
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert response.status_code == 503
    assert response.json()["overall_healthy"] is False
    assert response.json()["error"] == "Scheduler unavailable"
    assert response.json()["scheduler_readiness_failures"] == []
    assert response.json()["checks"] == [
        {
            "name": "database",
            "status": "pass",
            "message": "Database healthy",
        }
    ]
    assert failed_health_response.status_code == 503
    assert failed_health_response.json()["error"] == "Scheduler unavailable"
    assert failed_health_response.json()["checks"] == []
    assert health_feature.get_latest.await_count == 2


def test_host_managed_scheduler_without_scoped_runner_uses_host_worker():
    """A shared host runner backs features that intentionally omit local pollers."""
    from kestrel_sovereign.server import _active_scheduler_workers_available

    feature = SimpleNamespace(
        enabled=True,
        _runner=None,
        _polling_managed_by_host=True,
    )
    agent = SimpleNamespace(features={"SchedulerFeature": feature})
    host_runner = SimpleNamespace(worker_available=True)
    fake_app = SimpleNamespace(
        state=SimpleNamespace(host_scheduler_runner=host_runner)
    )

    assert _active_scheduler_workers_available(fake_app, agent, None) is True


def test_public_health_fails_while_scheduler_tick_is_stalled():
    """A fresh heartbeat cannot hide a polling tick beyond its hard bound."""
    from server import app
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    runner = SchedulerRunner(None, "agent-1", lambda *_: None)
    live_task = SimpleNamespace(done=lambda: False)
    runner._arm_requested = True
    runner._running = True
    runner._task = live_task
    runner._worker_task = live_task
    runner._tick_started_monotonic = (
        time.monotonic() - runner._tick_in_progress_limit_seconds - 1
    )
    agent = SimpleNamespace(
        features={"SchedulerFeature": SimpleNamespace(_runner=runner)}
    )

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = agent
        app.state.agent_manager = None
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert runner.tick_stalled is True
    assert runner.worker_available is False
    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": True,
    }


def test_health_fails_when_scheduler_arm_was_requested_but_never_started():
    """A swallowed on_agent_ready failure remains visible to the LB probe."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    runner = SimpleNamespace(
        readiness_failure=None,
        _arm_requested=True,
        _running=False,
        worker_available=False,
    )
    agent = SimpleNamespace(
        features={"SchedulerFeature": SimpleNamespace(_runner=runner)}
    )

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = agent
        app.state.agent_manager = None
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": True,
    }


def test_health_fails_when_scheduler_ready_hook_never_armed_runner():
    """A swallowed ready-hook failure cannot hide behind a private flag."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    runner = SimpleNamespace(
        readiness_failure=None,
        _arm_requested=False,
        _running=False,
        worker_available=False,
    )
    agent = SimpleNamespace(
        features={"SchedulerFeature": SimpleNamespace(_runner=runner)}
    )

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = agent
        app.state.agent_manager = None
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": True,
    }


@pytest.mark.parametrize(
    "feature_name, feature",
    [
        (
            "ThirdPartyFeature",
            SimpleNamespace(
                _runner=SimpleNamespace(
                    worker_available=False,
                    readiness_failure=RuntimeError("unrelated failure"),
                ),
            ),
        ),
        (
            "SchedulerFeature",
            SimpleNamespace(
                enabled=False,
                _runner=SimpleNamespace(
                    worker_available=False,
                    readiness_failure=RuntimeError("disabled failure"),
                ),
            ),
        ),
    ],
    ids=("unrelated-runner-contract", "disabled-scheduler"),
)
def test_health_ignores_non_active_scheduler_runners(feature_name, feature):
    """Only an enabled SchedulerFeature participates in scheduler readiness."""
    from kestrel_sovereign.server import (
        _active_scheduler_workers_available,
        _latch_active_scheduler_runner_failures,
    )

    agent = SimpleNamespace(features={feature_name: feature})
    fake_app = SimpleNamespace(
        state=SimpleNamespace(
            host_scheduler_runner=None,
            scheduler_readiness_failures=[],
        )
    )

    _latch_active_scheduler_runner_failures(fake_app, agent, None)

    assert _active_scheduler_workers_available(fake_app, agent, None) is True
    assert fake_app.state.scheduler_readiness_failures == []


def test_health_probe_fails_closed_when_agent_features_cannot_be_inspected():
    """A broken feature accessor returns controlled 503 instead of HTTP 500."""
    from server import app

    class BrokenFeatureInventory:
        @property
        def features(self):
            raise RuntimeError("synthetic feature inventory failure")

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_scheduler_failures = getattr(
        app.state, "scheduler_readiness_failures", None
    )
    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = BrokenFeatureInventory()
        app.state.agent_manager = None
        app.state.scheduler_readiness_failures = []
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.scheduler_readiness_failures = original_scheduler_failures

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "agent_initialized": True,
    }


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
    original_scheduler_failures = getattr(
        app.state, "scheduler_readiness_failures", None
    )
    original_host_scheduler_runner = getattr(
        app.state, "host_scheduler_runner", None
    )
    original_startup_error = getattr(app.state, "startup_error", None)
    original_mandatory_failures = getattr(
        app.state, "mandatory_feature_failures", None
    )
    original_identity_failures = getattr(
        app.state, "identity_readiness_failures", None
    )
    original_cold_agent_failures = getattr(
        app.state, "scheduler_cold_agent_failures", None
    )

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
        # This shared application may have a deliberately latched scheduler
        # failure from a prior test. This test owns a healthy mock fleet, so it
        # must not inherit that unrelated readiness state.
        app.state.startup_error = None
        app.state.mandatory_feature_failures = []
        app.state.identity_readiness_failures = []
        app.state.scheduler_cold_agent_failures = []
        app.state.scheduler_readiness_failures = []
        app.state.host_scheduler_runner = None
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
        app.state.scheduler_readiness_failures = original_scheduler_failures
        app.state.host_scheduler_runner = original_host_scheduler_runner
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures
        app.state.identity_readiness_failures = original_identity_failures
        app.state.scheduler_cold_agent_failures = original_cold_agent_failures

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


def test_safe_mode_still_names_the_feature_a_fleet_agent_refused():
    """The two conditions that co-occur when a host is already degraded.

    The safe-mode branch returns before the fleet walk that carries
    `features_not_loaded`, and it used to ask the SINGLETON agent — which is
    `None` on every multi-agent host, so the helper read through `getattr` and
    no-oped. The operator saw "restricted" and no reason to suspect a missing
    capability, in the one response whose whole job is to carry them (#3079).
    """
    from types import SimpleNamespace

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

    restricted = MagicMock()
    restricted._safe_mode = True
    restricted._safe_mode_entered_at = None
    restricted._constitution_state_load_error = None
    restricted.rejected_feature_contributions = (
        SimpleNamespace(
            feature_name="TalonCoordinatorFeature",
            reason=(
                "signal source already registered with a different contract: "
                "fleet_stalled_sweep"
            ),
        ),
    )
    manager = MagicMock()
    manager.list_agents.return_value = {"kite": restricted}

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None  # every multi-agent host
        app.state.agent_manager = manager
        app.state.startup_error = None
        app.state.mandatory_feature_failures = []
        app.state.identity_readiness_failures = []

        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
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

    body = detailed.json()
    assert detailed.status_code == 503
    assert body["status"] == "restricted"
    # The safe mode is still reported...
    assert body["constitution_safe_mode"][0]["agent"] == "kite"
    # ...and so is the capability the host is missing, naming WHICH agent lost
    # it: a host-level list is not scoped to one agent.
    assert body["features_not_loaded"] == [
        {
            "agent": "kite",
            "feature": "TalonCoordinatorFeature",
            "reason": (
                "signal source already registered with a different contract: "
                "fleet_stalled_sweep"
            ),
        }
    ]


def test_safe_mode_manufactures_no_key_when_nothing_was_refused():
    """No rejections must not invent an empty list in the response."""
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

    restricted = MagicMock()
    restricted._safe_mode = True
    restricted._safe_mode_entered_at = None
    restricted._constitution_state_load_error = None
    restricted.rejected_feature_contributions = ()
    manager = MagicMock()
    manager.list_agents.return_value = {"kite": restricted}

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = manager
        app.state.startup_error = None
        app.state.mandatory_feature_failures = []
        app.state.identity_readiness_failures = []

        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
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

    assert "features_not_loaded" not in detailed.json()


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


def _make_health_agent(status, checks=None):
    """Build a MagicMock agent whose HealthFeature reports ``status``."""
    health_feature = MagicMock()
    health_feature.get_latest = AsyncMock(
        return_value={"status": status, "checks": checks or []}
    )
    health_feature.__class__.__name__ = "HealthFeature"
    agent = MagicMock()
    agent.features = {"HealthFeature": health_feature}
    return agent


@pytest.mark.asyncio
async def test_detailed_health_fallback_matches_lock_warning_rollup():
    """The lock diagnosis has one severity with or without HealthFeature."""
    from kestrel_sovereign.server import _agent_detailed_health

    checks = [
        {"name": "database", "status": "pass"},
        {"name": "llm_service", "status": "pass"},
        {"name": "resource_locks", "status": "warn"},
    ]
    agent = SimpleNamespace(features={}, storage=None)

    with patch(
        "kestrel_sovereign.features.health.checks.run_standard_checks",
        new=AsyncMock(return_value=checks),
    ):
        result = await _agent_detailed_health(agent)

    assert result == {"status": "degraded", "checks": checks}


@pytest.mark.asyncio
async def test_detailed_health_fallback_uses_raw_db_behind_privacy_storage(
    tmp_path,
):
    """The no-feature path must not probe the privacy database facade."""
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.server import _agent_detailed_health
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    from kestrel_sovereign.storage.privacy_wrapper import (
        PrivacyEnforcingStorage,
    )

    raw_storage = AsyncStorage(
        str(tmp_path / "privacy-health.db"),
        agent_id="did:key:health-test",
    )
    await raw_storage.initialize()
    try:
        privacy_storage = PrivacyEnforcingStorage(
            raw_storage, PrivacyMode.NORMAL
        )
        agent = SimpleNamespace(
            features={},
            storage=privacy_storage,
            llm_service=SimpleNamespace(providers=[]),
        )

        # Deliberately run the real shared suite. ``storage.db.backend`` raises
        # PrivacyViolationError on this wrapper, so a passing database result
        # proves the fallback resolved the feature-internal raw database.
        result = await _agent_detailed_health(agent)
    finally:
        await raw_storage.close()

    database = next(
        check for check in result["checks"] if check["name"] == "database"
    )
    assert database["status"] == "pass"
    assert database["message"] == "Database connection healthy"


@pytest.mark.parametrize(
    "failed_check", ["bootstrap_state", "disk_space", "memory_system"]
)
@pytest.mark.asyncio
async def test_detailed_health_fallback_matches_noncritical_failure_rollup(
    failed_check,
):
    """Installing HealthFeature cannot change a non-critical severity."""
    from kestrel_sovereign.server import _agent_detailed_health

    checks = [
        {"name": "database", "status": "pass"},
        {"name": "llm_service", "status": "pass"},
        {"name": failed_check, "status": "fail"},
    ]
    agent = SimpleNamespace(features={}, storage=None)

    with patch(
        "kestrel_sovereign.features.health.checks.run_standard_checks",
        new=AsyncMock(return_value=checks),
    ):
        result = await _agent_detailed_health(agent)

    assert result == {"status": "degraded", "checks": checks}


def test_detailed_health_reports_fleet_when_no_singleton_agent():
    """Multi-agent hosts (app.state.agent is None) must resolve from the fleet.

    Regression for #2698: a running managed agent must not be reported as
    "No agent available".
    """
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)

    emma = _make_health_agent(
        "healthy", checks=[{"name": "database", "status": "pass"}]
    )
    manager = MagicMock()
    manager.list_agents.return_value = {"Emma": emma}

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = manager
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                response = client.get(
                    "/health/detailed",
                    headers={"X-API-Key": "test-key"},
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "No agent available" not in response.text
    assert body["agents"]["Emma"]["status"] == "healthy"
    assert body["agents"]["Emma"]["checks"] == [
        {"name": "database", "status": "pass"}
    ]
    # Tracing block stays present on the fleet branch (#2690).
    assert "tracing" in body


def test_healthy_fleet_with_unavailable_scheduler_identity_fails_readiness():
    """A scheduler authority gap is a 503, with redacted authenticated detail."""
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
    original_cold_failures = getattr(
        app.state, "scheduler_cold_agent_failures", None
    )
    original_scheduler_failures = getattr(
        app.state, "scheduler_readiness_failures", None
    )

    manager = MagicMock()
    manager.list_agents.return_value = {"Warm": _make_health_agent("healthy")}
    cold_failure = {
        "agent": "Unincepted",
        "scope": "identity",
        "state": "unavailable",
        "error_code": "scheduler_identity_unavailable",
        "cause_type": "RuntimeError",
    }
    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = manager
        app.state.startup_error = None
        app.state.mandatory_feature_failures = []
        app.state.identity_readiness_failures = []
        app.state.scheduler_cold_agent_failures = [cold_failure]
        app.state.scheduler_readiness_failures = [cold_failure]

        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                public = client.get("/health")
                detailed = client.get(
                    "/health/detailed", headers={"X-API-Key": "test-key"}
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.startup_error = original_startup_error
        app.state.mandatory_feature_failures = original_mandatory_failures
        app.state.identity_readiness_failures = original_identity_failures
        app.state.scheduler_cold_agent_failures = original_cold_failures
        app.state.scheduler_readiness_failures = original_scheduler_failures

    assert public.status_code == 503
    assert public.json() == {"status": "unhealthy", "agent_initialized": True}
    assert detailed.status_code == 503
    assert detailed.json()["status"] == "unhealthy"
    assert detailed.json()["overall_healthy"] is False
    assert detailed.json()["scheduler_readiness_failures"] == [cold_failure]
    assert "identity database" not in detailed.text


def test_detailed_health_fleet_mixed_failure_is_degraded():
    """Some healthy + some not rolls up to degraded, with per-agent detail."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)

    manager = MagicMock()
    manager.list_agents.return_value = {
        "Emma": _make_health_agent("healthy"),
        "Nellie": _make_health_agent(
            "unhealthy", checks=[{"name": "llm_service", "status": "fail"}]
        ),
    }

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = manager
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                response = client.get(
                    "/health/detailed",
                    headers={"X-API-Key": "test-key"},
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["agents"]["Emma"]["status"] == "healthy"
    assert body["agents"]["Nellie"]["status"] == "unhealthy"


def test_detailed_health_fleet_all_failing_is_unhealthy():
    """Zero healthy managed agents rolls up to unhealthy."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)

    manager = MagicMock()
    manager.list_agents.return_value = {
        "Emma": _make_health_agent("unhealthy"),
        "Nellie": _make_health_agent("degraded"),
    }

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = manager
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                response = client.get(
                    "/health/detailed",
                    headers={"X-API-Key": "test-key"},
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    assert response.status_code == 200
    assert response.json()["status"] == "unhealthy"


def test_detailed_health_no_agents_reports_no_agent_available():
    """Truly no agents (no singleton, empty/absent manager) keeps today's text."""
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)
    original_scheduler_failures = getattr(
        app.state, "scheduler_readiness_failures", None
    )

    try:
        app.router.lifespan_context = noop_lifespan
        app.state.agent = None
        app.state.agent_manager = None
        app.state.scheduler_readiness_failures = []
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                response = client.get(
                    "/health/detailed",
                    headers={"X-API-Key": "test-key"},
                )
                app.state.scheduler_readiness_failures = [
                    {"scope": "protocol", "state": "unavailable"}
                ]
                scheduler_response = client.get(
                    "/health/detailed",
                    headers={"X-API-Key": "test-key"},
                )
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager
        app.state.scheduler_readiness_failures = original_scheduler_failures

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["error"] == "No agent available"
    assert "tracing" in body
    assert scheduler_response.status_code == 503
    assert scheduler_response.json()["overall_healthy"] is False
    assert scheduler_response.json()["error"] == "Scheduler unavailable"


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
