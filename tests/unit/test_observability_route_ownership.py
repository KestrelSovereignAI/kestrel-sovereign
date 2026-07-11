"""Agent reads stay in core; fleet ingestion belongs to the host feature."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.endpoints.observability import router


def test_core_observability_events_is_read_only():
    app = FastAPI()
    app.include_router(router)
    app.state.agent = type("Agent", (), {"observability_store": None})()

    with TestClient(app) as client:
        response = client.post(
            "/api/observability/events",
            json={"agent_name": "a", "session_id": "s", "event_type": "metric"},
        )

    assert response.status_code == 405
    assert response.headers["allow"] == "GET"


def test_core_does_not_claim_host_fleet_observability_namespace():
    paths = {route.path for route in router.routes}

    assert "/api/observability/events" in paths
    assert not any(path.startswith("/api/host/observability") for path in paths)
