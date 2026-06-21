"""Focused contract tests for security endpoints."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from kestrel_sovereign.features.security.approval_queue import DecisionResult


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
    return {"X-API-Key": "test-key"}


def _make_security_feature():
    permission_store = MagicMock()
    permission_store.get_permission_tree = AsyncMock(
        return_value=[
            SimpleNamespace(
                feature_name="files",
                rollup_state="mixed",
                tools=[
                    SimpleNamespace(tool_name="read", level=SimpleNamespace(value="allow")),
                    SimpleNamespace(tool_name="write", level=SimpleNamespace(value="ask")),
                ],
            )
        ]
    )
    permission_store.get_audit_log = AsyncMock(
        return_value=[
            {
                "feature": "files",
                "tool": "write",
                "action": "prompt",
                "decision": "approved",
                "user_choice": "session",
                "args_summary": "path=/tmp/demo",
                "timestamp": "2026-03-17T12:00:00+00:00",
            }
        ]
    )
    permission_store.set_permission = AsyncMock()
    permission_store.set_feature_permission = AsyncMock()
    permission_store.get_global_auto_mode = MagicMock(return_value=False)
    permission_store.set_global_auto_mode = MagicMock()
    permission_store.log_decision = AsyncMock()
    permission_store.clear_session_overrides = MagicMock()
    approval_queue = MagicMock(
        pending_requests=[
            SimpleNamespace(
                id="req-1",
                feature_name="files",
                tool_name="write",
                tool_args={"path": "/tmp/demo"},
                created_at=datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
            )
        ]
    )
    approval_queue.submit_decision = AsyncMock(
        return_value=DecisionResult(in_memory=True, persisted=True)
    )
    return MagicMock(permission_store=permission_store, approval_queue=approval_queue)


def test_security_tree_pending_and_audit_endpoints_serialize_expected_shapes():
    security_feature = _make_security_feature()
    agent = MagicMock(features={"SecurityFeature": security_feature})

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                tree_response = client.get("/api/security/permissions/tree", headers=_api_headers())
                pending_response = client.get("/api/security/pending", headers=_api_headers())
                audit_response = client.get("/api/security/audit?limit=25", headers=_api_headers())
        assert tree_response.status_code == 200
        assert tree_response.json()["tree"][0]["tools"][1]["level"] == "ask"
        assert pending_response.status_code == 200
        assert pending_response.json()["count"] == 1
        assert pending_response.json()["pending"][0]["id"] == "req-1"
        assert audit_response.status_code == 200
        assert audit_response.json()["logs"][0]["decision"] == "approved"
        security_feature.permission_store.get_audit_log.assert_awaited_once_with(25)
    finally:
        _restore_app(app, original)


def test_security_permission_mutation_endpoints_validate_levels_and_scope():
    security_feature = _make_security_feature()
    agent = MagicMock(features={"SecurityFeature": security_feature})
    # Make is_demo a real bool so the demo-isolation rail (#766) treats
    # this as a live agent. MagicMock would return a truthy MagicMock
    # for any unset attribute and accidentally bypass the rail.
    agent.is_demo = False

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                tool_response = client.post(
                    "/api/security/permissions",
                    headers=_api_headers(),
                    json={"feature": "files", "tool": "write", "level": "allow"},
                )
                auto_response = client.post(
                    "/api/security/permissions",
                    headers=_api_headers(),
                    json={"feature": "files", "tool": "read", "level": "auto"},
                )
                feature_response = client.post(
                    "/api/security/permissions/feature",
                    headers={
                        **_api_headers(),
                        # #766: bulk DENY runs through the rail. The UI
                        # attaches this header automatically; the test
                        # mirrors that behavior.
                        "X-Kestrel-Allow-Destructive": "test-bulk-deny",
                    },
                    json={"feature": "files", "level": "deny"},
                )
                invalid_level = client.post(
                    "/api/security/permissions",
                    headers=_api_headers(),
                    json={"feature": "files", "level": "forbidden"},
                )
                invalid_scope = client.post(
                    "/api/security/approve",
                    headers=_api_headers(),
                    json={"approval_id": "req-1", "approved": True, "scope": "forever"},
                )
        assert tool_response.status_code == 200
        assert auto_response.status_code == 200
        assert "constitutional" in auto_response.json()["warning"]
        assert feature_response.status_code == 200
        assert invalid_level.status_code == 400
        assert "Invalid level" in invalid_level.json()["detail"]
        assert invalid_scope.status_code == 400
        assert "Invalid scope" in invalid_scope.json()["detail"]
        assert security_feature.permission_store.set_permission.await_count == 2
        assert security_feature.permission_store.set_feature_permission.await_count == 1
    finally:
        _restore_app(app, original)


def test_security_approval_and_cancellation_endpoints_preserve_queue_contracts():
    security_feature = _make_security_feature()
    security_feature.approval_queue.submit_decision.return_value = DecisionResult(
        in_memory=True,
        persisted=True,
    )
    security_feature.approval_queue.cancel_request.return_value = True
    security_feature.approval_queue.cancel_all.return_value = 3
    agent = MagicMock(features={"SecurityFeature": security_feature})

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                approve_response = client.post(
                    "/api/security/approve",
                    headers=_api_headers(),
                    json={"approval_id": "req-1", "approved": True, "scope": "session"},
                )
                cancel_response = client.post(
                    "/api/security/cancel/req-1",
                    headers=_api_headers(),
                )
                cancel_all_response = client.post(
                    "/api/security/cancel-all",
                    headers=_api_headers(),
                )
                reset_response = client.post(
                    "/api/security/reset-session",
                    headers=_api_headers(),
                )
        assert approve_response.status_code == 200
        assert approve_response.json() == {
            "success": True,
            "approved": True,
            "scope": "session",
            "persisted": True,
        }
        security_feature.approval_queue.submit_decision.assert_awaited_once_with(
            "req-1", True, "session"
        )
        assert cancel_response.status_code == 200
        assert cancel_all_response.status_code == 200
        assert cancel_all_response.json()["cancelled"] == 3
        assert reset_response.status_code == 200
        security_feature.permission_store.clear_session_overrides.assert_called_once_with()
    finally:
        _restore_app(app, original)


def test_security_global_auto_mode_endpoints_are_session_scoped():
    security_feature = _make_security_feature()
    security_feature.permission_store.get_global_auto_mode.side_effect = [False, True]
    agent = MagicMock(features={"SecurityFeature": security_feature})

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                status_response = client.get(
                    "/api/security/auto-mode",
                    headers=_api_headers(),
                )
                enable_response = client.post(
                    "/api/security/auto-mode",
                    headers=_api_headers(),
                    json={"enabled": True},
                )
        assert status_response.status_code == 200
        assert status_response.json()["enabled"] is False
        assert enable_response.status_code == 200
        assert enable_response.json()["enabled"] is True
        assert "non-DENY tools" in enable_response.json()["warning"]
        security_feature.permission_store.set_global_auto_mode.assert_called_once_with(True)
        security_feature.permission_store.log_decision.assert_awaited_once()
    finally:
        _restore_app(app, original)


def test_security_endpoints_return_503_when_feature_is_unavailable():
    agent = MagicMock(features={})

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/security/permissions/tree", headers=_api_headers())
        assert response.status_code == 503
        assert response.json()["detail"] == "SecurityFeature not available"
    finally:
        _restore_app(app, original)
