"""Sovereign authority boundary for host-wide agent lifecycle endpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from kestrel_sovereign.auth import AuthMethod, CallerContext
from kestrel_sovereign.endpoints.models import router


def _app():
    app = FastAPI()
    manager = MagicMock()
    manager.create_agent = AsyncMock(
        return_value=SimpleNamespace(agent_id="did:test:new")
    )
    manager.remove_agent = AsyncMock(return_value=True)
    manager._created_configs = {}
    manager.list_agents.return_value = {}
    app.state.agent_manager = manager
    app.state.agent = None

    @app.middleware("http")
    async def bind_test_caller_and_rewrite(request: Request, call_next):
        authority = request.headers.get("x-test-authority")
        if authority == "api-key":
            request.state.caller = CallerContext.sovereign(AuthMethod.API_KEY)
        elif authority == "oauth":
            request.state.caller = CallerContext.authenticated(
                "user@example.com",
                AuthMethod.OAUTH_SESSION,
            )
        elif authority == "jwt":
            request.state.caller = CallerContext.authenticated(
                "did:test:user",
                AuthMethod.JWT,
            )
        if request.scope["path"] == "/api/agents/Claw/api/agents":
            request.state.agent = SimpleNamespace(agent_id="did:test:claw")
            request.scope["path"] = "/api/agents"
        return await call_next(request)

    app.include_router(router)
    return app, manager


@pytest.mark.parametrize("authority", [None, "oauth", "jwt"])
def test_non_sovereign_cannot_create_host_agent(authority):
    app, manager = _app()
    headers = {"x-test-authority": authority} if authority else {}

    response = TestClient(app).post(
        "/api/agents",
        json={"name": "NewAgent"},
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Sovereign authority is required."
    manager.create_agent.assert_not_awaited()


@pytest.mark.parametrize("offboard_runtime", [False, True])
def test_non_sovereign_cannot_withdraw_or_offboard_host_agent(offboard_runtime):
    app, manager = _app()

    response = TestClient(app).delete(
        "/api/agents/SecretTarget",
        params={"offboard_runtime": str(offboard_runtime).lower()},
        headers={"x-test-authority": "oauth"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Sovereign authority is required."
    manager.remove_agent.assert_not_awaited()


def test_per_agent_route_rewrite_does_not_bypass_host_lifecycle_authority():
    app, manager = _app()

    response = TestClient(app).post(
        "/api/agents/Claw/api/agents",
        json={"name": "SmuggledAgent"},
        headers={"x-test-authority": "jwt"},
    )

    assert response.status_code == 403
    manager.create_agent.assert_not_awaited()


def test_sovereign_api_key_can_create_and_withdraw_host_agents():
    app, manager = _app()
    client = TestClient(app)
    headers = {"x-test-authority": "api-key"}

    created = client.post(
        "/api/agents",
        json={"name": "NewAgent"},
        headers=headers,
    )
    removed = client.delete("/api/agents/NewAgent", headers=headers)

    assert created.status_code == 200
    assert removed.status_code == 200
    manager.create_agent.assert_awaited_once_with("NewAgent")
    manager.remove_agent.assert_awaited_once_with(
        "NewAgent",
        offboard_runtime=False,
    )


def test_read_only_agent_discovery_does_not_inherit_mutation_gate():
    app, _manager = _app()

    response = TestClient(app).get("/api/agents")

    assert response.status_code == 200
    assert response.json()["mode"] == "multi_agent"


@pytest.mark.parametrize(
    ("authority", "expected"),
    [("oauth", False), ("jwt", False), ("api-key", True)],
)
def test_agent_discovery_advertises_caller_scoped_lifecycle_authority(
    authority,
    expected,
):
    app, _manager = _app()

    response = TestClient(app).get(
        "/api/agents",
        headers={"x-test-authority": authority},
    )

    assert response.status_code == 200
    assert response.json()["can_create_agents"] is expected
    assert response.json()["can_delete_agents"] is expected
    assert response.headers["cache-control"] == "private, no-store"
    assert {
        value.strip().lower() for value in response.headers["vary"].split(",")
    } >= {"authorization", "cookie", "x-api-key"}
